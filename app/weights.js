// Safetensors files read from Blobs (the OPFS cache): has(name), info(name) -> { dtype, shape },
// f32(name) -> Float32Array. uploadLinear / uploadConv3 turn tensors into GPU weights.

const BYTES = { F64: 8, F32: 4, BF16: 2, F16: 2, I64: 8, I32: 4, I16: 2, I8: 1, U8: 1, BOOL: 1 };

export function bf16ToF32(u8) {
  const u16 = new Uint16Array(u8.buffer, u8.byteOffset, u8.byteLength / 2);
  const out = new Float32Array(u16.length);
  const o32 = new Uint32Array(out.buffer);
  for (let i = 0; i < u16.length; i++) o32[i] = u16[i] << 16;
  return out;
}

function f16ToF32(u8) {
  const u16 = new Uint16Array(u8.buffer, u8.byteOffset, u8.byteLength / 2);
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 31;
    const m = h & 1023;
    out[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return out;
}

// Raw little-endian bytes of `dtype` -> Float32Array (u8 must own its buffer from offset 0).
export function toF32(u8, dtype) {
  if (dtype === "F32") return new Float32Array(u8.buffer, u8.byteOffset, u8.byteLength / 4);
  if (dtype === "BF16") return bf16ToF32(u8);
  if (dtype === "F16") return f16ToF32(u8);
  const Other = { F64: Float64Array, I64: BigInt64Array, I32: Int32Array, I16: Int16Array, I8: Int8Array, U8: Uint8Array, BOOL: Uint8Array }[dtype];
  if (!Other) throw new Error(`unsupported dtype ${dtype}`);
  const buf = u8.byteOffset % 8 ? u8.slice().buffer : u8.buffer; // typed-array views need an aligned offset
  return Float32Array.from(new Other(buf, u8.byteOffset % 8 ? 0 : u8.byteOffset, u8.byteLength / BYTES[dtype]), Number);
}

export class SafeTensors {
  static async open(blob) {
    const head = new DataView(await blob.slice(0, 8).arrayBuffer());
    const n = Number(head.getBigUint64(0, true));
    const header = JSON.parse(new TextDecoder().decode(await blob.slice(8, 8 + n).arrayBuffer()));
    const metadata = header.__metadata__ || {};
    delete header.__metadata__;
    return new SafeTensors(blob, header, 8 + n, metadata);
  }

  constructor(blob, header, dataStart, metadata = {}) {
    this.blob = blob;
    this.metadata = metadata;
    this.header = header;
    this.dataStart = dataStart;
  }

  has(name) {
    return name in this.header;
  }

  info(name) {
    const t = this.header[name];
    if (!t) throw new Error(`missing tensor ${name}`);
    return t;
  }

  async f32(name) {
    const t = this.info(name);
    const [a, b] = t.data_offsets;
    return toF32(new Uint8Array(await this.blob.slice(this.dataStart + a, this.dataStart + b).arrayBuffer()), t.dtype);
  }

  // Gather rows of a 2D table (embeddings) without reading the whole tensor.
  async rows(name, ids) {
    const t = this.info(name);
    const [, dim] = t.shape;
    const rowBytes = dim * BYTES[t.dtype];
    const out = new Float32Array(ids.length * dim);
    await Promise.all(ids.map(async (id, i) => {
      const off = this.dataStart + t.data_offsets[0] + id * rowBytes;
      out.set(toF32(new Uint8Array(await this.blob.slice(off, off + rowBytes).arrayBuffer()), t.dtype), i * dim);
    }));
    return out;
  }
}

export async function uploadVector(gpu, src, name) {
  return gpu.upload(await src.f32(name));
}

// Linear layer "<base>weight" [N, K] (+ "<base>bias").
export async function uploadLinear(gpu, src, base) {
  const w = src.info(base + "weight");
  const N = w.shape[0];
  const K = w.shape.slice(1).reduce((a, b) => a * b, 1);
  const bias = src.has(base + "bias") ? await uploadVector(gpu, src, base + "bias") : null;
  return { kind: "f32", N, K, buf: gpu.upload(await src.f32(base + "weight")), bias };
}

// Weight from a Float32Array already laid out [N, K].
export function linearFromArray(gpu, data, N, K, bias = null) {
  return { kind: "f32", N, K, buf: gpu.upload(data), bias: bias ? gpu.upload(bias) : null };
}

// 3x3 conv "<base>weight" [cout, cin, 3, 3] -> [cout, 3, 3, cinPad] for the implicit-GEMM conv
// (input channels zero-padded to cinPad, which must be a multiple of 8).
export async function uploadConv3(gpu, src, base, cinPad) {
  const [cout, cin, kh, kw] = src.info(base + "weight").shape;
  if (kh !== 3 || kw !== 3) throw new Error(`${base}: expected a 3x3 conv`);
  const cp = cinPad || cin;
  const w = await src.f32(base + "weight");
  const out = new Float32Array(cout * 9 * cp);
  for (let o = 0; o < cout; o++) {
    for (let i = 0; i < cin; i++) {
      for (let t = 0; t < 9; t++) out[(o * 9 + t) * cp + i] = w[(o * cin + i) * 9 + t];
    }
  }
  const bias = src.has(base + "bias") ? await uploadVector(gpu, src, base + "bias") : null;
  return { kind: "f32", N: cout, K: 9 * cp, buf: gpu.upload(out), bias };
}
