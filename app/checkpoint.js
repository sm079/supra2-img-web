// Reader for PyTorch checkpoints (torch.save's zip format) from a Blob.
//
// The zip holds "<name>/data.pkl" (the pickled object graph) and one uncompressed entry per
// tensor storage ("<name>/data/<key>"). The pickle is run by a small interpreter that knows the
// handful of torch functions a state dict uses (_rebuild_tensor_v2, OrderedDict, ...). It builds
// plain data only: every other global becomes an inert { __global, args } record, so nothing in
// the file is ever executed. Tensors come back as references ({ isTensor, dtype, shape, ... })
// whose bytes are read on demand.

import { toF32 } from "./weights.js";

const STORAGE_DTYPES = {
  FloatStorage: "F32", HalfStorage: "F16", BFloat16Storage: "BF16", DoubleStorage: "F64",
  LongStorage: "I64", IntStorage: "I32", ShortStorage: "I16", CharStorage: "I8", ByteStorage: "U8", BoolStorage: "BOOL",
};
const TORCH_DTYPES = {
  float32: "F32", float: "F32", float16: "F16", half: "F16", bfloat16: "BF16", float64: "F64", double: "F64",
  int64: "I64", long: "I64", int32: "I32", int: "I32", int16: "I16", int8: "I8", uint8: "U8", bool: "BOOL",
};
const DTYPE_BYTES = { F64: 8, F32: 4, BF16: 2, F16: 2, I64: 8, I32: 4, I16: 2, I8: 1, U8: 1, BOOL: 1 };

// ---------------------------------------------------------------------------- zip

async function readBytes(blob, start, end) {
  return new Uint8Array(await blob.slice(start, end).arrayBuffer());
}

async function zipEntries(blob) {
  const tailLen = Math.min(blob.size, 65536 + 22 + 20);
  const tail = await readBytes(blob, blob.size - tailLen, blob.size);
  const dv = new DataView(tail.buffer);
  let eocd = -1;
  for (let i = tail.length - 22; i >= 0; i--) if (dv.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  if (eocd < 0) throw new Error("not a PyTorch checkpoint (no zip directory found)");
  let count = dv.getUint16(eocd + 10, true);
  let cdSize = dv.getUint32(eocd + 12, true);
  let cdOffset = dv.getUint32(eocd + 16, true);
  if (count === 0xffff || cdSize === 0xffffffff || cdOffset === 0xffffffff) {
    // zip64: the locator sits right before the classic end record
    const loc = eocd - 20;
    if (loc < 0 || dv.getUint32(loc, true) !== 0x07064b50) throw new Error("zip64 locator missing");
    const recOff = Number(dv.getBigUint64(loc + 8, true));
    const rec = new DataView((await readBytes(blob, recOff, recOff + 56)).buffer);
    if (rec.getUint32(0, true) !== 0x06064b50) throw new Error("bad zip64 end record");
    count = Number(rec.getBigUint64(32, true));
    cdSize = Number(rec.getBigUint64(40, true));
    cdOffset = Number(rec.getBigUint64(48, true));
  }
  const cd = await readBytes(blob, cdOffset, cdOffset + cdSize);
  const cv = new DataView(cd.buffer);
  const dec = new TextDecoder();
  const entries = new Map();
  let p = 0;
  for (let i = 0; i < count; i++) {
    if (cv.getUint32(p, true) !== 0x02014b50) throw new Error("bad zip central directory");
    const method = cv.getUint16(p + 10, true);
    let comp = cv.getUint32(p + 20, true);
    let size = cv.getUint32(p + 24, true);
    const nameLen = cv.getUint16(p + 28, true);
    const extraLen = cv.getUint16(p + 30, true);
    const commentLen = cv.getUint16(p + 32, true);
    let lho = cv.getUint32(p + 42, true);
    const name = dec.decode(cd.subarray(p + 46, p + 46 + nameLen));
    // zip64 extra field: 64-bit values for whichever of the fields above overflowed
    for (let e = p + 46 + nameLen; e < p + 46 + nameLen + extraLen;) {
      const id = cv.getUint16(e, true);
      const len = cv.getUint16(e + 2, true);
      if (id === 1) {
        let q = e + 4;
        if (size === 0xffffffff) { size = Number(cv.getBigUint64(q, true)); q += 8; }
        if (comp === 0xffffffff) { comp = Number(cv.getBigUint64(q, true)); q += 8; }
        if (lho === 0xffffffff) { lho = Number(cv.getBigUint64(q, true)); q += 8; }
      }
      e += 4 + len;
    }
    entries.set(name, { name, method, comp, size, lho });
    p += 46 + nameLen + extraLen + commentLen;
  }
  return entries;
}

// Absolute offset of an entry's data (the local header's name/extra lengths can differ from the
// central directory's).
async function dataStart(blob, entry) {
  if (entry.start == null) {
    const h = new DataView((await readBytes(blob, entry.lho, entry.lho + 30)).buffer);
    if (h.getUint32(0, true) !== 0x04034b50) throw new Error(`bad zip local header for ${entry.name}`);
    entry.start = entry.lho + 30 + h.getUint16(26, true) + h.getUint16(28, true);
  }
  return entry.start;
}

async function entryBytes(blob, entry) {
  const s = await dataStart(blob, entry);
  const raw = await readBytes(blob, s, s + entry.comp);
  if (entry.method === 0) return raw;
  if (entry.method === 8) {
    const stream = new Blob([raw]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }
  throw new Error(`unsupported zip compression ${entry.method} for ${entry.name}`);
}

// ---------------------------------------------------------------------------- pickle

const MARK = Symbol("mark");

class Global {
  constructor(module, name) {
    this.module = module;
    this.name = name;
  }
  get full() {
    return `${this.module}.${this.name}`;
  }
}

function tensorRef(storage, offset, shape, stride, dtype) {
  return { isTensor: true, key: storage.key, dtype: dtype || storage.dtype, offset, shape: [...shape], stride: [...stride] };
}

function newDict(pairs = []) {
  const d = Object.create(null);
  for (const [k, v] of pairs) d[String(k)] = v;
  return d;
}

// Calls to globals: the few that build data are interpreted, everything else stays inert.
function call(fn, args) {
  if (!(fn instanceof Global)) return { __call: fn, args };
  switch (fn.full) {
    case "torch._utils._rebuild_tensor_v2":
      return tensorRef(args[0], args[1], args[2], args[3]);
    case "torch._utils._rebuild_tensor_v3": {
      const dt = args[6] instanceof Global ? TORCH_DTYPES[args[6].name] : null;
      return tensorRef(args[0], args[1], args[2], args[3], dt);
    }
    case "torch._utils._rebuild_parameter":
    case "torch._utils._rebuild_parameter_with_state":
      return args[0];
    case "collections.OrderedDict":
    case "builtins.dict":
      return newDict(args[0] || []);
    case "builtins.list":
    case "builtins.tuple":
      return [...(args[0] || [])];
    case "builtins.set":
    case "builtins.frozenset":
      return new Set(args[0] || []);
    case "torch.device":
      return String(args[0]);
    default:
      return { __global: fn.full, args };
  }
}

function unpickle(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const dec = new TextDecoder();
  let p = 0;
  let stack = [];
  const metastack = [];
  const memo = new Map();
  const u8 = () => bytes[p++];
  const u16 = () => { const v = dv.getUint16(p, true); p += 2; return v; };
  const u32 = () => { const v = dv.getUint32(p, true); p += 4; return v; };
  const i32 = () => { const v = dv.getInt32(p, true); p += 4; return v; };
  const u64 = () => { const v = Number(dv.getBigUint64(p, true)); p += 8; return v; };
  const take = (n) => { const v = bytes.subarray(p, p + n); p += n; return v; };
  const str = (n) => dec.decode(take(n));
  const line = () => { const e = bytes.indexOf(10, p); const s = dec.decode(bytes.subarray(p, e)); p = e + 1; return s; };
  const popMark = () => { const items = stack; stack = metastack.pop(); return items; };
  const long = (b) => {
    if (!b.length) return 0;
    let v = 0n;
    for (let i = b.length - 1; i >= 0; i--) v = (v << 8n) | BigInt(b[i]);
    if (b[b.length - 1] & 0x80) v -= 1n << BigInt(8 * b.length);
    return Number(v);
  };
  const setItems = (d, kv) => {
    if (d instanceof Map) { for (let i = 0; i < kv.length; i += 2) d.set(kv[i], kv[i + 1]); return; }
    for (let i = 0; i < kv.length; i += 2) d[String(kv[i])] = kv[i + 1];
  };

  for (;;) {
    const op = u8();
    switch (op) {
      case 0x80: p++; break; // PROTO
      case 0x95: p += 8; break; // FRAME
      case 0x2e: return stack.pop(); // STOP
      case 0x28: metastack.push(stack); stack = []; break; // MARK
      case 0x7d: stack.push(newDict()); break; // EMPTY_DICT
      case 0x5d: stack.push([]); break; // EMPTY_LIST
      case 0x29: stack.push([]); break; // EMPTY_TUPLE
      case 0x8f: stack.push(new Set()); break; // EMPTY_SET
      case 0x64: { const kv = popMark(); const d = newDict(); setItems(d, kv); stack.push(d); break; } // DICT
      case 0x6c: { const items = popMark(); stack.push(items); break; } // LIST
      case 0x74: { const items = popMark(); stack.push(items); break; } // TUPLE
      case 0x85: stack.push([stack.pop()]); break; // TUPLE1
      case 0x86: { const b = stack.pop(); const a = stack.pop(); stack.push([a, b]); break; } // TUPLE2
      case 0x87: { const c = stack.pop(); const b = stack.pop(); const a = stack.pop(); stack.push([a, b, c]); break; } // TUPLE3
      case 0x91: { const items = popMark(); stack.push(new Set(items)); break; } // FROZENSET
      case 0x90: { const items = popMark(); const s = stack.at(-1); for (const x of items) s.add(x); break; } // ADDITEMS
      case 0x61: { const v = stack.pop(); stack.at(-1).push(v); break; } // APPEND
      case 0x65: { const items = popMark(); stack.at(-1).push(...items); break; } // APPENDS
      case 0x73: { const v = stack.pop(); const k = stack.pop(); setItems(stack.at(-1), [k, v]); break; } // SETITEM
      case 0x75: { const kv = popMark(); setItems(stack.at(-1), kv); break; } // SETITEMS
      case 0x71: memo.set(u8(), stack.at(-1)); break; // BINPUT
      case 0x72: memo.set(u32(), stack.at(-1)); break; // LONG_BINPUT
      case 0x94: memo.set(memo.size, stack.at(-1)); break; // MEMOIZE
      case 0x70: memo.set(Number(line()), stack.at(-1)); break; // PUT
      case 0x68: stack.push(memo.get(u8())); break; // BINGET
      case 0x6a: stack.push(memo.get(u32())); break; // LONG_BINGET
      case 0x67: stack.push(memo.get(Number(line()))); break; // GET
      case 0x58: stack.push(str(u32())); break; // BINUNICODE
      case 0x8c: stack.push(str(u8())); break; // SHORT_BINUNICODE
      case 0x8d: stack.push(str(u64())); break; // BINUNICODE8
      case 0x56: stack.push(line()); break; // UNICODE (raw-unicode-escape; plain text in practice)
      case 0x42: stack.push(take(u32()).slice()); break; // BINBYTES
      case 0x43: stack.push(take(u8()).slice()); break; // SHORT_BINBYTES
      case 0x8e: stack.push(take(u64()).slice()); break; // BINBYTES8
      case 0x54: stack.push(str(u32())); break; // BINSTRING
      case 0x55: stack.push(str(u8())); break; // SHORT_BINSTRING
      case 0x4a: stack.push(i32()); break; // BININT
      case 0x4b: stack.push(u8()); break; // BININT1
      case 0x4d: stack.push(u16()); break; // BININT2
      case 0x8a: stack.push(long(take(u8()))); break; // LONG1
      case 0x8b: stack.push(long(take(u32()))); break; // LONG4
      case 0x49: { const s = line(); stack.push(s === "00" ? false : s === "01" ? true : Number(s)); break; } // INT
      case 0x4c: stack.push(Number(line().replace(/L$/, ""))); break; // LONG
      case 0x47: { const v = dv.getFloat64(p, false); p += 8; stack.push(v); break; } // BINFLOAT
      case 0x46: stack.push(Number(line())); break; // FLOAT
      case 0x4e: stack.push(null); break; // NONE
      case 0x88: stack.push(true); break; // NEWTRUE
      case 0x89: stack.push(false); break; // NEWFALSE
      case 0x63: { const m = line(); const n = line(); stack.push(new Global(m, n)); break; } // GLOBAL
      case 0x93: { const n = stack.pop(); const m = stack.pop(); stack.push(new Global(m, n)); break; } // STACK_GLOBAL
      case 0x51: { // BINPERSID: ('storage', storage_type, key, location, numel)
        const pid = stack.pop();
        if (!Array.isArray(pid) || pid[0] !== "storage") throw new Error("unsupported persistent id in checkpoint: " + JSON.stringify(pid));
        const type = pid[1] instanceof Global ? pid[1].name : String(pid[1]);
        const dtype = STORAGE_DTYPES[type];
        if (!dtype) throw new Error(`unsupported storage type ${type}`);
        stack.push({ key: String(pid[2]), dtype, numel: pid[4] });
        break;
      }
      case 0x52: { const args = stack.pop(); const fn = stack.pop(); stack.push(call(fn, args)); break; } // REDUCE
      case 0x81: { const args = stack.pop(); const cls = stack.pop(); stack.push(call(cls, args)); break; } // NEWOBJ
      case 0x92: { stack.pop(); const args = stack.pop(); const cls = stack.pop(); stack.push(call(cls, args)); break; } // NEWOBJ_EX
      case 0x62: { // BUILD
        const state = stack.pop();
        const obj = stack.at(-1);
        if (obj && typeof obj === "object" && !Array.isArray(obj) && !obj.isTensor) {
          if (state && typeof state === "object" && !Array.isArray(state)) Object.assign(obj, state);
          else obj.__state = state;
        }
        break;
      }
      case 0x30: stack.pop(); break; // POP
      case 0x31: popMark(); break; // POP_MARK
      case 0x32: stack.push(stack.at(-1)); break; // DUP
      default:
        throw new Error(`unsupported pickle opcode 0x${op.toString(16)} at ${p - 1}`);
    }
  }
}

// ---------------------------------------------------------------------------- checkpoint

export async function readCheckpoint(blob) {
  const entries = await zipEntries(blob);
  const pkl = [...entries.keys()].find((n) => n === "data.pkl" || n.endsWith("/data.pkl"));
  if (!pkl) throw new Error("not a PyTorch checkpoint (no data.pkl)");
  const prefix = pkl.slice(0, -"data.pkl".length);
  const obj = unpickle(await entryBytes(blob, entries.get(pkl)));

  const byteorder = entries.get(prefix + "byteorder");
  if (byteorder && new TextDecoder().decode(await entryBytes(blob, byteorder)).trim() !== "little") {
    throw new Error("big-endian checkpoints are not supported");
  }

  // Tensor -> Float32Array (contiguous copy, any strides)
  async function tensorF32(t) {
    const entry = entries.get(`${prefix}data/${t.key}`);
    if (!entry) throw new Error(`checkpoint storage ${t.key} is missing`);
    if (entry.method !== 0) throw new Error("compressed tensor storages are not supported");
    const es = DTYPE_BYTES[t.dtype];
    const numel = t.shape.reduce((a, b) => a * b, 1);
    const contiguous = t.shape.every((_, i) => t.shape[i] === 1 || t.stride[i] === t.shape.slice(i + 1).reduce((a, b) => a * b, 1));
    const start = await dataStart(blob, entry);
    if (contiguous) {
      const a = start + t.offset * es;
      return toF32(await readBytes(blob, a, a + numel * es), t.dtype);
    }
    // strided view: read the spanned range of the storage, then gather
    const span = 1 + t.shape.reduce((a, n, i) => a + (n - 1) * t.stride[i], 0);
    const a = start + t.offset * es;
    const src = toF32(await readBytes(blob, a, a + span * es), t.dtype);
    const out = new Float32Array(numel);
    const idx = new Array(t.shape.length).fill(0);
    for (let i = 0; i < numel; i++) {
      let off = 0;
      for (let d = 0; d < idx.length; d++) off += idx[d] * t.stride[d];
      out[i] = src[off];
      for (let d = idx.length - 1; d >= 0; d--) { if (++idx[d] < t.shape[d]) break; idx[d] = 0; }
    }
    return out;
  }

  return { obj, tensorF32 };
}
