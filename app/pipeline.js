// Supra2-IMG text-to-image pipeline: cached model files -> WebGPU models -> Euler flow sampling
// with classifier-free guidance -> VAE. Follows SupraLabs' inference.py.

import { GPU } from "./gpu/device.js";
import { SafeTensors } from "./weights.js";
import { FILES, cachedFile, requestPersistence } from "./store.js";
import { Tokenizer } from "./vendor/tokenizers.min.mjs";
import { T5Encoder } from "./models/t5.js";
import { SupraDiT, CFG, patchify, unpatchify } from "./models/dit.js";
import { VAEDecoder } from "./models/vae.js";
import { TorchGenerator } from "./rng.js";

export const IMAGE_SIZE = CFG.latentSize * 8; // 256
const EOS = 1; // T5 </s>
const LATENT = CFG.latentCh * CFG.latentSize * CFG.latentSize;

// ComfyUI's SD1.x latent -> RGB projection, for step previews without running the VAE
const RGB_FACTORS = [[0.3512, 0.2297, 0.3227], [0.325, 0.4974, 0.235], [-0.2829, 0.1762, 0.2721], [-0.212, -0.2616, -0.7177]];

function latentPreview(z) {
  const S = CFG.latentSize;
  const hw = S * S;
  const img = new Uint8ClampedArray(hw * 4);
  for (let i = 0; i < hw; i++) {
    for (let c = 0; c < 3; c++) {
      let v = 0;
      for (let k = 0; k < 4; k++) v += z[k * hw + i] * RGB_FACTORS[k][c];
      img[i * 4 + c] = ((v + 1) / 2) * 255;
    }
    img[i * 4 + 3] = 255;
  }
  return { data: img, width: S, height: S };
}

export class SupraPipeline {
  // gpuOptions: passed to GPU.create (e.g. { profile: true })
  constructor(gpuOptions = {}) {
    this.gpuOptions = gpuOptions;
    this.gpu = null;
    this.ready = false;
    this.conds = new Map(); // prompt text -> prepared cross-attention K/V (most recent few)
  }

  // Downloads (first time only) and loads everything onto the GPU. modelsBase: URL of the folder
  // holding the model files.
  async load({ onStatus = () => {}, signal, modelsBase } = {}) {
    if (this.ready) { onStatus({ phase: "ready" }); return; }
    await requestPersistence();
    this.gpu = this.gpu || (await GPU.create(this.gpuOptions));
    const gpu = this.gpu;

    const keys = ["tokenizer", "tokenizerConfig", "te", "vae", "dit"];
    const got = Object.fromEntries(keys.map((k) => [k, 0]));
    const total = () => keys.reduce((a, k) => a + Math.max(got[k], FILES[k].bytes), 0);
    const files = {};
    for (const k of keys) {
      files[k] = await cachedFile(FILES[k], modelsBase, (done) => {
        got[k] = done;
        onStatus({ phase: "download", file: FILES[k].label, done: keys.reduce((a, p) => a + got[p], 0), total: total() });
      }, signal);
    }

    const stage = (what) => (frac) => onStatus({ phase: "load", what, frac });
    stage("tokenizer")(0);
    this.tokenizer = new Tokenizer(JSON.parse(await files.tokenizer.text()), JSON.parse(await files.tokenizerConfig.text()));

    stage("text encoder")(0);
    this.te = await T5Encoder.load(gpu, await SafeTensors.open(files.te), stage("text encoder"));

    stage("image model")(0);
    const st = await SafeTensors.open(files.dit);
    let config = {};
    try { config = JSON.parse(st.metadata.config || "{}"); } catch { /* keep defaults */ }
    if (config.patch != null && config.patch !== CFG.patch) throw new Error(`checkpoint patch size ${config.patch} is not supported`);
    this.ctxLen = Number.isFinite(config.ctx_len) ? config.ctx_len : 128;
    this.dit = await SupraDiT.load(gpu, st, stage("image model"));
    // the unconditional text states stored with the checkpoint (unmasked rows only); without
    // them the empty prompt is encoded instead (inference.py's fallback)
    this.uncondCtx = st.has("uncond_text") ? { data: await st.f32("uncond_text"), L: st.info("uncond_text").shape[0] } : null;

    stage("image decoder")(0);
    this.vae = await VAEDecoder.load(gpu, await SafeTensors.open(files.vae));
    stage("image decoder")(1);
    await gpu.sync();
    this.ready = true;
    onStatus({ phase: "ready" });
  }

  // Flan-T5 ids with </s>, truncated to the checkpoint's context length like the HF tokenizer.
  tokenize(text) {
    let ids = this.tokenizer.encode(text, { add_special_tokens: false }).ids;
    if (ids.length > this.ctxLen - 1) ids = ids.slice(0, this.ctxLen - 1);
    return [...ids, EOS];
  }

  // Text -> prepared cross-attention K/V. key "\0uncond": the stored unconditional states.
  async condition(text, key = text) {
    if (this.conds.has(key)) {
      const c = this.conds.get(key);
      this.conds.delete(key); // move to most recent
      this.conds.set(key, c);
      return c;
    }
    const gpu = this.gpu;
    let ctx;
    if (key === "\0uncond" && this.uncondCtx) ctx = gpu.fromArray(this.uncondCtx.data, [this.uncondCtx.L, CFG.ctxDim]);
    else ctx = await this.te.encode(this.tokenize(text));
    const c = this.dit.prepareContext(ctx);
    ctx.release();
    this.conds.set(key, c);
    while (this.conds.size > 4) {
      const [k, old] = this.conds.entries().next().value;
      old.release();
      this.conds.delete(k);
    }
    return c;
  }

  // Text -> Float32Array [L, 768] T5 states (for tools/check.html)
  async encodeText(text) {
    const t = await this.te.encode(this.tokenize(text));
    const out = await this.gpu.read(t);
    t.release();
    return out;
  }

  // opts: { prompt, negative, steps, cfg, seed, onProgress, onPreview, signal }
  // Euler integration of the flow from t = 0 (noise) to t = 1 (image), as in inference.py.
  async generate(opts) {
    const { prompt, negative = "", steps = 50, cfg = 3, seed = 0, onProgress = () => {}, onPreview, signal } = opts;
    const gpu = this.gpu;
    const check = () => { if (signal?.aborted) throw new DOMException("Generation cancelled", "AbortError"); };
    const t0 = performance.now();
    onProgress({ phase: "encode" });
    const useCfg = cfg > 1;
    const conds = [await this.condition(prompt)];
    if (useCfg) conds.push(negative.trim() ? await this.condition(negative) : await this.condition("", "\0uncond"));
    check();

    const f = Math.fround;
    const dt = 1 / steps;
    const ts = Array.from({ length: steps }, (_, i) => f(i * dt));
    let z = opts.noise || new TorchGenerator(seed).randn(LATENT);
    const tSample = performance.now();
    const mod = this.dit.modulations(ts);
    let image, tVae;
    try {
      const dtf = f(dt);
      const cf = f(cfg);
      for (let i = 0; i < steps; i++) {
        check();
        const out = await this.dit.forward(patchify(z), conds, mod, i, async (b) => {
          if (b % 7 === 6) {
            await gpu.sync(); // keeps the queue short so cancel and progress stay responsive
            check();
            onProgress({ phase: "sample", step: i, steps, frac: (i + (b + 1) / CFG.depth) / steps });
          }
        });
        const vc = unpatchify(out, 0);
        let v = vc;
        if (useCfg) {
          const vu = unpatchify(out, vc.length);
          v = new Float32Array(vc.length);
          for (let j = 0; j < v.length; j++) v[j] = vu[j] + f(cf * f(vc[j] - vu[j]));
        }
        const zn = new Float32Array(LATENT);
        for (let j = 0; j < LATENT; j++) zn[j] = z[j] + f(dtf * v[j]);
        z = zn;
        onProgress({ phase: "sample", step: i, steps, frac: (i + 1) / steps });
        if (onPreview) {
          // the current guess of the final latent: x1 = z_t + (1 - t) * v
          const left = f(1 - f((i + 1) * dt));
          const x1 = new Float32Array(LATENT);
          for (let j = 0; j < LATENT; j++) x1[j] = z[j] + left * v[j];
          onPreview(latentPreview(x1), i);
        }
      }
      check();
      tVae = performance.now();
      onProgress({ phase: "decode", frac: 0 });
      image = await this.vae.decode(z, CFG.latentSize, CFG.latentSize, (fr) => onProgress({ phase: "decode", frac: fr }));
    } finally {
      mod.release();
      // free pooled activations so idle VRAM is just the weights; also runs after a cancel
      await gpu.sync().catch(() => {});
      gpu.pool.trim();
    }
    const t1 = performance.now();
    return {
      image,
      latent: z,
      timings: { encode: tSample - t0, sample: tVae - tSample, decode: t1 - tVae, total: t1 - t0, perStep: (tVae - tSample) / steps },
    };
  }

  unload() {
    const destroy = (o) => {
      if (!o || typeof o !== "object") return;
      if (o instanceof GPUBuffer) { o.destroy(); return; }
      for (const v of Array.isArray(o) ? o : o instanceof Map ? [...o.values()] : Object.values(o)) if (v && typeof v === "object" && v !== this.gpu && !(v instanceof Float32Array) && !(v instanceof SafeTensors)) destroy(v);
    };
    for (const c of this.conds.values()) c.release();
    this.conds.clear();
    for (const k of ["te", "dit", "vae"]) { destroy(this[k]); this[k] = null; }
    this.gpu?.pool.trim();
    this.ready = false;
  }
}
