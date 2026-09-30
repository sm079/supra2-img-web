// Page-side handle on the engine. Runs the pipeline in a Web Worker (worker.js) so the UI stays
// responsive; falls back to running it in the page when WebGPU isn't exposed to workers.

import { SupraPipeline } from "./pipeline.js";
import { requestPersistence } from "./store.js";

class WorkerEngine {
  constructor(worker) {
    this.worker = worker;
    this.next = 1;
    this.calls = new Map();
    this.ready = false;
    this.gpu = { name: "", peak: 0 };
    worker.onmessage = (e) => {
      const m = e.data;
      const call = this.calls.get(m.id);
      if (!call) return;
      if (m.type === "status") call.onStatus?.(m.status);
      else if (m.type === "progress") call.onProgress?.(m.progress);
      else if (m.type === "preview") call.onPreview?.({ width: m.width, height: m.height, data: m.data }, m.step);
      else {
        this.calls.delete(m.id);
        if (m.type === "error") {
          const err = new Error(m.message);
          err.name = m.name;
          err.stack = m.stack;
          call.reject(err);
        } else call.resolve(m);
      }
    };
  }

  call(msg, hooks = {}) {
    const id = this.next++;
    return new Promise((resolve, reject) => {
      this.calls.set(id, { resolve, reject, ...hooks });
      if (hooks.signal) {
        if (hooks.signal.aborted) this.worker.postMessage({ type: "cancel", target: id });
        hooks.signal.addEventListener("abort", () => this.worker.postMessage({ type: "cancel", target: id }), { once: true });
      }
      this.worker.postMessage({ id, ...msg });
    });
  }

  async load({ onStatus, signal, modelsBase } = {}) {
    const r = await this.call({ type: "load", modelsBase }, { onStatus, signal });
    this.ready = true;
    this.gpu = r.gpu;
  }

  async generate({ onProgress, onPreview, signal, ...opts }) {
    const r = await this.call({ type: "generate", opts }, { onProgress, onPreview, signal });
    this.gpu = r.gpu;
    return { image: { width: r.width, height: r.height, data: r.data }, timings: r.timings };
  }

  unload() {
    this.ready = false;
    return this.call({ type: "unload" });
  }
}

class LocalEngine {
  constructor() {
    this.pipe = new SupraPipeline();
  }
  get ready() { return this.pipe.ready; }
  get gpu() {
    const i = this.pipe.gpu?.info || {};
    return { name: [i.vendor, i.architecture, i.description].filter(Boolean).join(" "), peak: this.pipe.gpu?.pool.peak || 0 };
  }
  load(opts) { return this.pipe.load(opts); }
  generate(opts) { return this.pipe.generate(opts); }
  unload() { this.pipe.unload(); }
}

// inPage: force the in-page engine (debugging; ?engine=page)
export async function createEngine({ inPage = false } = {}) {
  requestPersistence(); // Window-only API; the worker can't ask for it
  if (inPage) return new LocalEngine();
  try {
    const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    const engine = new WorkerEngine(worker);
    const { ok } = await Promise.race([
      engine.call({ type: "init" }),
      new Promise((_, rej) => setTimeout(() => rej(new Error("worker did not start")), 10000)),
    ]);
    if (ok) {
      engine.inWorker = true;
      return engine;
    }
    worker.terminate();
    console.warn("WebGPU is not available in workers here; running the engine on the page");
  } catch (e) {
    console.warn("engine worker unavailable, running on the page:", e.message);
  }
  return new LocalEngine();
}
