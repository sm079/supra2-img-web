// Model files: where they come from, and the one-time download into the Origin Private File
// System (OPFS).
//
// Everything is fetched straight from the original Hugging Face repos, pinned to a commit, so
// there is no converted copy to host. From the two large safetensors files only the needed
// tensors are downloaded (Flan-T5's encoder, the VAE's decoder): the app fetches the file's JSON
// header first, then just those byte ranges, and saves them as a smaller safetensors file.
//
// Files are written to "<name>.part" in place (download.js), so an interrupted download resumes
// with an HTTP Range request. A finished file is renamed to its final name and reused on later
// visits. Names include the pinned commit, so a future update never mixes old and new files.

import { downloadToOPFS, OPFS_DIR as DIR, planSize } from "./download.js";

const HF = "https://huggingface.co";
const T5_REV = "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2";

// bytes: download size (for extracts, the kept tensor data; the new header adds a few KB)
export const FILES = {
  dit: {
    label: "image model", repo: "SupraLabs/Supra2-IMG", rev: "b22ffe6c85983a63a535ff9bfd40caacc428dddf",
    path: "model_final_ema.pt", bytes: 416651529,
  },
  te: {
    label: "text encoder", repo: "google/flan-t5-base", rev: T5_REV, path: "model.safetensors", bytes: 438514176,
    keep: (n) => n.startsWith("encoder.") || n === "shared.weight",
  },
  vae: {
    label: "image decoder", repo: "stabilityai/sd-vae-ft-mse", rev: "31f26fdeee1355a5c34592e401dd41e45d25a493",
    path: "diffusion_pytorch_model.safetensors", bytes: 197960796,
    keep: (n) => n.startsWith("decoder.") || n.startsWith("post_quant_conv."),
  },
  tokenizer: { label: "tokenizer", repo: "google/flan-t5-base", rev: T5_REV, path: "tokenizer.json", bytes: 2424064 },
  tokenizerConfig: { label: "tokenizer", repo: "google/flan-t5-base", rev: T5_REV, path: "tokenizer_config.json", bytes: 2537 },
};
export const TOTAL_BYTES = Object.values(FILES).reduce((a, f) => a + f.bytes, 0);

export const fileUrl = (f) => `${HF}/${f.repo}/resolve/${f.rev}/${f.path}`;
// OPFS name: repo, commit and what was kept
export const cacheName = (f) => `${f.repo.replace("/", "--")}--${f.rev.slice(0, 8)}--${f.keep ? "extract--" : ""}${f.path}`;

const IN_WORKER = typeof WorkerGlobalScope !== "undefined" && self instanceof WorkerGlobalScope;

async function dir() {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle(DIR, { create: true });
}

async function tryFile(d, name) {
  try {
    return await (await d.getFileHandle(name)).getFile();
  } catch {
    return null;
  }
}

export async function requestPersistence() {
  try {
    return await navigator.storage.persist?.();
  } catch {
    return false;
  }
}

async function fetchRange(url, start, end, signal) {
  const res = await fetch(url, { headers: { Range: `bytes=${start}-${end - 1}` }, signal });
  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText} for ${url}`);
  const buf = new Uint8Array(await res.arrayBuffer());
  // a server that ignores Range sends the whole file
  return res.status === 206 ? buf : buf.subarray(start, end);
}

// What to download: a whole file, or a new safetensors header + the kept tensors' byte ranges.
async function plan(f, signal) {
  if (!f.keep) return { prefix: new Uint8Array(0), pieces: [[0, f.bytes]] };
  const url = fileUrl(f);
  const n = Number(new DataView((await fetchRange(url, 0, 8, signal)).buffer).getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(await fetchRange(url, 8, 8 + n, signal)));
  const base = 8 + n;
  const kept = Object.entries(header)
    .filter(([name]) => name !== "__metadata__" && f.keep(name))
    .sort((a, b) => a[1].data_offsets[0] - b[1].data_offsets[0]);
  const out = {};
  const pieces = [];
  let off = 0;
  for (const [name, t] of kept) {
    const [a, b] = t.data_offsets;
    out[name] = { dtype: t.dtype, shape: t.shape, data_offsets: [off, off + b - a] };
    off += b - a;
    const last = pieces.at(-1);
    if (last && last[1] === base + a) last[1] = base + b; else pieces.push([base + a, base + b]);
  }
  let json = JSON.stringify({ __metadata__: { source: `${f.repo}@${f.rev}/${f.path}` }, ...out });
  json += " ".repeat((8 - (json.length % 8)) % 8); // keep the data 8-byte aligned
  const hb = new TextEncoder().encode(json);
  const prefix = new Uint8Array(8 + hb.length);
  new DataView(prefix.buffer).setBigUint64(0, BigInt(hb.length), true);
  prefix.set(hb, 8);
  return { prefix, pieces };
}

function download(args, onProgress, signal) {
  if (IN_WORKER) return downloadToOPFS({ ...args, signal, onProgress });
  return new Promise((resolve, reject) => {
    const w = new Worker(new URL("./download-worker.js", import.meta.url), { type: "module" });
    const stop = () => { w.terminate(); reject(new DOMException("Download cancelled", "AbortError")); };
    signal?.addEventListener("abort", stop, { once: true });
    w.onmessage = (e) => {
      const m = e.data;
      if (m.type === "progress") onProgress?.(m.done);
      else {
        signal?.removeEventListener("abort", stop);
        w.terminate();
        if (m.type === "done") resolve();
        else reject(new Error(m.message));
      }
    };
    w.onerror = (e) => { w.terminate(); reject(new Error(e.message || "download worker failed")); };
    w.postMessage(args);
  });
}

// Returns the cached File for `f`, downloading it first if needed. onProgress(bytesDone, bytesTotal)
export async function cachedFile(f, onProgress, signal) {
  const d = await dir();
  const name = cacheName(f);
  const done = await tryFile(d, name);
  if (done) {
    onProgress?.(done.size, done.size);
    return done;
  }
  const p = await plan(f, signal);
  const size = planSize(p);
  const part = name + ".part";
  const partFile = await tryFile(d, part);
  if (!(partFile && partFile.size === size)) {
    await download({ url: fileUrl(f), part, prefix: p.prefix, pieces: p.pieces }, (n) => onProgress?.(n, size), signal);
  }
  const ph = await d.getFileHandle(part);
  const got = (await ph.getFile()).size;
  if (got !== size) throw new Error(`size mismatch for ${name}: got ${got}, expected ${size}`);
  if (ph.move) {
    await ph.move(name);
    return (await d.getFileHandle(name)).getFile();
  }
  return ph.getFile(); // no rename support: the complete .part is used as is
}

// Bytes already on disk per file key (finished or partial), for progress and the settings view.
export async function cachedBytes() {
  const d = await dir();
  const out = {};
  for (const [k, f] of Object.entries(FILES)) {
    const done = await tryFile(d, cacheName(f));
    const part = done ? null : await tryFile(d, cacheName(f) + ".part");
    out[k] = { done: !!done, bytes: (done || part)?.size || 0 };
  }
  return out;
}

export async function storageBytes() {
  const d = await dir();
  let n = 0;
  for await (const [, h] of d.entries()) if (h.kind === "file") n += (await h.getFile()).size;
  return n;
}

export async function clearCache() {
  const root = await navigator.storage.getDirectory();
  try { await root.removeEntry(DIR, { recursive: true }); } catch { /* nothing saved */ }
}
