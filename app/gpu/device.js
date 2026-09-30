// WebGPU runtime: device setup, buffer pool, pipeline cache and batched dispatch.
//
// Dispatches are recorded into one compute pass and flushed in batches. Per-dispatch
// parameters live in a uniform ring that is uploaded right before each submit; that is
// safe because queue.writeBuffer is ordered after previously submitted work.

const RING_SLOT = 256;
const RING_SLOTS = 8192;

export class Tensor {
  constructor(gpu, buf, shape, bytes) {
    this.gpu = gpu;
    this.buf = buf;
    this.shape = shape;
    this.bytes = bytes;
  }
  get size() {
    return this.shape.reduce((a, b) => a * b, 1);
  }
  release() {
    if (this.buf) this.gpu.pool.put(this.buf, this.bytes);
    this.buf = null;
  }
}

class BufferPool {
  constructor(device) {
    this.device = device;
    this.free = new Map(); // rounded size -> GPUBuffer[]
    this.allocated = 0;
    this.peak = 0;
  }
  static round(bytes) {
    // bucket sizes to limit fragmentation: 1/8-power-of-two steps
    const b = Math.max(256, bytes);
    const p = 2 ** Math.floor(Math.log2(b));
    const step = Math.max(256, p / 8);
    return Math.ceil(b / step) * step;
  }
  get(bytes) {
    const size = BufferPool.round(bytes);
    const list = this.free.get(size);
    if (list && list.length) return { buf: list.pop(), size };
    const buf = this.device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.allocated += size;
    this.peak = Math.max(this.peak, this.allocated);
    return { buf, size };
  }
  put(buf, size) {
    if (!this.free.has(size)) this.free.set(size, []);
    this.free.get(size).push(buf);
  }
  trim() {
    for (const list of this.free.values()) for (const b of list) { b.destroy(); this.allocated -= b.size; }
    this.free.clear();
  }
}

export class GPU {
  // profile: time every dispatch with timestamp queries (diagnostics only; tools/profile.html)
  static async create({ profile = false } = {}) {
    if (!navigator.gpu) throw new Error("WebGPU is not available in this browser.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter found.");
    const L = adapter.limits;
    const timing = profile && adapter.features.has("timestamp-query");
    if (profile && !timing) console.warn("timestamp-query not supported: profiling disabled");
    const device = await adapter.requestDevice({
      requiredFeatures: timing ? ["timestamp-query"] : [],
      requiredLimits: {
        maxBufferSize: L.maxBufferSize,
        maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
        maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
        maxStorageBuffersPerShaderStage: Math.min(10, L.maxStorageBuffersPerShaderStage),
      },
    });
    const info = adapter.info || {};
    const gpu = new GPU(device, info);
    if (timing) gpu.profiler = new Profiler(device);
    return gpu;
  }

  constructor(device, info) {
    this.profiler = null;
    this.device = device;
    this.info = info;
    this.limits = device.limits;
    this.maxBinding = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    this.pool = new BufferPool(device);
    this.pipelines = new Map();
    this.ring = device.createBuffer({ size: RING_SLOT * RING_SLOTS, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.ringData = new ArrayBuffer(RING_SLOT * RING_SLOTS);
    this.ringUsed = 0;
    this.encoder = null;
    this.pass = null;
    this.lost = null;
    device.lost.then((e) => { this.lost = e; console.error("WebGPU device lost:", e.message); });
    this.errors = [];
    device.addEventListener?.("uncapturederror", (e) => {
      console.error("WebGPU error:", e.error?.message);
      this.errors.push(e.error?.message || String(e.error));
    });
  }

  // ---------------------------------------------------------------- tensors

  empty(shape) {
    const bytes = shape.reduce((a, b) => a * b, 1) * 4;
    const { buf, size } = this.pool.get(bytes);
    return new Tensor(this, buf, shape, size);
  }

  fromArray(arr, shape) {
    const t = this.empty(shape);
    // queue.writeBuffer runs ahead of dispatches that are recorded but not yet submitted;
    // a pooled buffer may still be read by one of those, so submit them first.
    this.flush();
    this.write(t.buf, arr);
    return t;
  }

  write(buf, arr, offset = 0) {
    // writeBuffer needs 4-byte multiples
    let data = arr;
    if (data.byteLength % 4) {
      const padded = new Uint8Array(Math.ceil(data.byteLength / 4) * 4);
      padded.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
      data = padded;
    }
    this.device.queue.writeBuffer(buf, offset, data.buffer, data.byteOffset, data.byteLength);
  }

  // Persistent (non-pooled) buffer for weights.
  upload(arr) {
    const size = Math.max(16, Math.ceil(arr.byteLength / 16) * 16); // vec4-bindable
    const buf = this.device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.write(buf, arr);
    return buf;
  }

  async read(t, count = t.size) {
    this.flush();
    const bytes = count * 4;
    const staging = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(t.buf, 0, staging, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    this.checkErrors();
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  }

  // ---------------------------------------------------------------- dispatch

  pipeline(code) {
    let p = this.pipelines.get(code);
    if (!p) {
      const module = this.device.createShaderModule({ code });
      p = this.device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
      this.pipelines.set(code, p);
    }
    return p;
  }

  // params: array of [type, value] with type 'u32' | 'f32' | 'i32', packed in order.
  // meta: { name, flops } for the profiler.
  dispatch(code, buffers, params, groups, meta) {
    if (this.lost) throw new Error("WebGPU device lost: " + this.lost.message);
    const pipe = this.pipeline(code);
    if (this.ringUsed >= RING_SLOTS) this.flush();
    const slot = this.ringUsed++;
    const dv = new DataView(this.ringData, slot * RING_SLOT, RING_SLOT);
    params.forEach(([type, v], i) => {
      if (type === "f32") dv.setFloat32(i * 4, v, true);
      else if (type === "i32") dv.setInt32(i * 4, v, true);
      else dv.setUint32(i * 4, v >>> 0, true);
    });
    const entries = [{ binding: 0, resource: { buffer: this.ring, offset: slot * RING_SLOT, size: RING_SLOT } }];
    buffers.forEach((b, i) => entries.push({ binding: i + 1, resource: { buffer: b.buf || b } }));
    const bg = this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    const [x, y = 1, z = 1] = groups;
    if (this.profiler) {
      // one timed pass per dispatch
      if (this.profiler.full()) this.flush();
      if (!this.encoder) this.encoder = this.device.createCommandEncoder();
      const pass = this.encoder.beginComputePass({ timestampWrites: this.profiler.next(meta) });
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(x, y, z);
      pass.end();
      this.pass = true; // marks pending work
      return;
    }
    if (!this.pass) {
      this.encoder = this.device.createCommandEncoder();
      this.pass = this.encoder.beginComputePass();
    }
    this.pass.setPipeline(pipe);
    this.pass.setBindGroup(0, bg);
    this.pass.dispatchWorkgroups(x, y, z);
  }

  flush() {
    if (!this.pass) return;
    if (this.profiler) this.profiler.resolve(this.encoder);
    else this.pass.end();
    this.device.queue.writeBuffer(this.ring, 0, this.ringData, 0, this.ringUsed * RING_SLOT);
    this.device.queue.submit([this.encoder.finish()]);
    this.profiler?.collect();
    this.pass = null;
    this.encoder = null;
    this.ringUsed = 0;
  }

  async sync() {
    this.flush();
    await this.device.queue.onSubmittedWorkDone();
    this.checkErrors();
  }

  // Validation errors are reported asynchronously; surface them instead of producing zeros.
  checkErrors() {
    if (this.errors.length) {
      const msg = this.errors.join(" | ");
      this.errors = [];
      throw new Error("WebGPU validation error: " + msg);
    }
  }
}

// Per-dispatch GPU timings aggregated by kernel name.
class Profiler {
  constructor(device, capacity = 2048) {
    this.device = device;
    this.capacity = capacity;
    this.set = device.createQuerySet({ type: "timestamp", count: capacity * 2 });
    this.resolveBuf = device.createBuffer({ size: capacity * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    this.metas = [];
    this.pending = [];
    this.stats = new Map();
  }
  full() {
    return this.metas.length >= this.capacity;
  }
  next(meta) {
    const i = this.metas.length;
    this.metas.push(meta || { name: "?", flops: 0 });
    return { querySet: this.set, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 };
  }
  resolve(encoder) {
    const n = this.metas.length;
    if (!n) return;
    this.readBuf = this.device.createBuffer({ size: n * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    encoder.resolveQuerySet(this.set, 0, n * 2, this.resolveBuf, 0);
    encoder.copyBufferToBuffer(this.resolveBuf, 0, this.readBuf, 0, n * 16);
  }
  collect() {
    const metas = this.metas;
    const buf = this.readBuf;
    this.metas = [];
    this.readBuf = null;
    if (!buf) return;
    this.pending.push(buf.mapAsync(GPUMapMode.READ).then(() => {
      const t = new BigInt64Array(buf.getMappedRange());
      metas.forEach((m, i) => {
        const ms = Number(t[2 * i + 1] - t[2 * i]) / 1e6;
        const s = this.stats.get(m.name) || { name: m.name, count: 0, ms: 0, flops: 0 };
        s.count++;
        s.ms += ms;
        s.flops += m.flops || 0;
        this.stats.set(m.name, s);
      });
      buf.unmap();
      buf.destroy();
    }));
  }
  reset() {
    this.stats.clear();
  }
  async report() {
    await Promise.all(this.pending);
    this.pending = [];
    return [...this.stats.values()].sort((a, b) => b.ms - a.ms);
  }
}

// Split a linear workgroup count into a 2D grid within the 65535 per-dimension limit.
export function grid(n) {
  if (n <= 65535) return [n, 1];
  const y = Math.ceil(n / 65535);
  return [Math.ceil(n / y), y];
}
