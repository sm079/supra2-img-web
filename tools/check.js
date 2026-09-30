import { GPU } from "../app/gpu/device.js";
import { SafeTensors } from "../app/weights.js";
import { FILES, cachedFile } from "../app/store.js";
import { T5Encoder } from "../app/models/t5.js";
import { VAEDecoder } from "../app/models/vae.js";
import { SupraPipeline } from "../app/pipeline.js";
import { Tokenizer } from "../app/vendor/tokenizers.min.mjs";
import { TorchGenerator } from "../app/rng.js";

const q = new URLSearchParams(location.search);
const dumpDir = new URL(q.get("dump") || "../out/dump/", location.href);
const logEl = document.getElementById("log");
const log = (s, cls) => {
  const span = document.createElement("span");
  if (cls) span.className = cls;
  span.textContent = s + "\n";
  logEl.append(span);
  console.log(s);
};
window.__checkDone = false;
window.__checkResults = {};

async function loadDump(name) {
  const r = await fetch(new URL(name + ".bin", dumpDir));
  if (!r.ok) throw new Error(`missing dump ${name}`);
  return new Float32Array(await r.arrayBuffer());
}

function compare(name, got, ref, tol) {
  let maxAbs = 0, num = 0, den = 0;
  for (let i = 0; i < ref.length; i++) {
    const d = got[i] - ref[i];
    maxAbs = Math.max(maxAbs, Math.abs(d));
    num += d * d;
    den += ref[i] * ref[i];
  }
  const rel = Math.sqrt(num / Math.max(den, 1e-30));
  const ok = rel < tol && Number.isFinite(rel) && got.length === ref.length;
  window.__checkResults[name] = { rel, maxAbs, ok };
  log(`${ok ? "PASS" : "FAIL"} ${name}: rel L2 ${rel.toExponential(3)}  max|d| ${maxAbs.toExponential(3)}  (n=${ref.length}${got.length !== ref.length ? `, got ${got.length}` : ""}, tol ${tol})`, ok ? "ok" : "bad");
  return ok;
}

function show(label, rgbHWC, W, H) {
  const c = document.createElement("canvas");
  c.width = W; c.height = H; c.title = label;
  const img = new ImageData(W, H);
  for (let i = 0; i < W * H; i++) {
    for (let k = 0; k < 3; k++) img.data[i * 4 + k] = Math.floor(((Math.min(1, Math.max(-1, rgbHWC[i * 3 + k])) + 1) / 2) * 255 + 0.5);
    img.data[i * 4 + 3] = 255;
  }
  c.getContext("2d").putImageData(img, 0, 0);
  document.getElementById("images").append(c);
}

const progress = (label) => {
  let last = -1;
  return (done, total) => {
    const pct = Math.floor((100 * done) / total);
    if (pct !== last && pct % 20 === 0) { last = pct; log(`download ${label} ${pct}%`); }
  };
};

try {
  const meta = await (await fetch(new URL("index.json", dumpDir))).json();
  const only = (q.get("only") || (meta.dit ? "tok,noise,te,vae,dit,full" : "tok,noise,te,vae")).split(",");
  const gpu = await GPU.create();
  log(`adapter: ${[gpu.info.vendor, gpu.info.architecture, gpu.info.description].filter(Boolean).join(" ")}`);

  if (only.includes("noise")) {
    const ref = await loadDump("noise");
    compare("noise (torch.randn CPU)", new TorchGenerator(meta.tensors.noise.seed).randn(ref.length), ref, 1e-5);
  }

  let tokenizer = null;
  const needTok = only.some((o) => ["tok", "te"].includes(o));
  if (needTok) {
    const tj = await cachedFile(FILES.tokenizer, progress("tokenizer"));
    const tc = await cachedFile(FILES.tokenizerConfig);
    tokenizer = new Tokenizer(JSON.parse(await tj.text()), JSON.parse(await tc.text()));
  }
  const ids = (text) => [...tokenizer.encode(text, { add_special_tokens: false }).ids.slice(0, 127), 1];
  if (only.includes("tok")) {
    for (const p of meta.prompts) {
      const got = ids(p.text);
      const ok = JSON.stringify(got) === JSON.stringify(p.ids);
      window.__checkResults[`tok ${JSON.stringify(p.text)}`] = { ok };
      log(`${ok ? "PASS" : "FAIL"} tokenizer ${JSON.stringify(p.text)}${ok ? "" : `: got ${got} want ${p.ids}`}`, ok ? "ok" : "bad");
    }
  }

  if (only.includes("te")) {
    const t0 = performance.now();
    const te = await T5Encoder.load(gpu, await SafeTensors.open(await cachedFile(FILES.te, progress("text encoder"))));
    log(`text encoder loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    for (let i = 0; i < meta.prompts.length; i++) {
      const out = await te.encode(meta.prompts[i].ids);
      const got = await gpu.read(out);
      out.release();
      compare(`T5 hidden (${meta.prompts[i].ids.length} tokens)`, got, await loadDump(`te_${i}`), 1e-4);
    }
  }

  if (only.includes("vae")) {
    const vae = await VAEDecoder.load(gpu, await SafeTensors.open(await cachedFile(FILES.vae, progress("vae"))));
    const lat = await loadDump("vae_latent");
    const t0 = performance.now();
    const img = await vae.decode(lat, 32, 32);
    log(`vae decode ${(performance.now() - t0).toFixed(0)} ms`);
    const ref = await loadDump("vae_image");
    compare("VAE decode", img.pixels, ref, 1e-4);
    show("webgpu vae", img.pixels, 256, 256);
    show("reference vae", ref, 256, 256);
  }

  if (meta.dit && (only.includes("dit") || only.includes("full"))) {
    const pipe = new SupraPipeline();
    pipe.gpu = gpu;
    await pipe.load({ modelsBase: new URL(q.get("models") || "../models/", location.href).href, onStatus: (s) => { if (s.phase === "load" && s.frac === 0) log(`loading ${s.what}`); } });
    const noise = await loadDump("noise");
    if (only.includes("dit")) {
      const { patchify, unpatchify } = await import("../app/models/dit.js");
      const cond = await pipe.condition(meta.dit.prompt);
      const unc = await pipe.condition("", "\0uncond");
      const mod = pipe.dit.modulations([0]);
      const out = await pipe.dit.forward(patchify(noise), [cond, unc], mod, 0);
      mod.release();
      compare("DiT v (step 0, cond)", unpatchify(out, 0), await loadDump("dit_v0_cond"), 1e-4);
      compare("DiT v (step 0, uncond)", unpatchify(out, 1024 * 4), await loadDump("dit_v0_uncond"), 1e-4);
    }
    if (only.includes("full")) {
      const t0 = performance.now();
      const res = await pipe.generate({ prompt: meta.dit.prompt, steps: meta.dit.steps, cfg: meta.dit.cfg, seed: meta.dit.seed, noise });
      log(`full sample: ${(performance.now() - t0).toFixed(0)} ms (${res.timings.perStep.toFixed(1)} ms/step, decode ${res.timings.decode.toFixed(0)} ms)`);
      compare(`final latent (${meta.dit.steps} steps)`, res.latent, await loadDump("dit_final_latent"), 1e-2);
      const ref = await loadDump("dit_image");
      compare("final image", res.image.pixels, ref, 2e-2);
      show("webgpu", res.image.pixels, 256, 256);
      show("reference", ref, 256, 256);
    }
  }
  log("done");
} catch (e) {
  log("ERROR " + (e.stack || e.message), "bad");
  window.__checkError = String(e.message || e);
}
window.__checkDone = true;
