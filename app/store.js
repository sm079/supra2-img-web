// Model files and the one-time download into the Origin Private File System (OPFS).
//
// The files live in a Hugging Face model repo (built by tools/build_models.py). Files are
// written to "<name>.part" in place (download.js), so an interrupted download resumes with an
// HTTP Range request. A finished file is renamed to its final name and reused on later visits.

import { downloadToOPFS, OPFS_DIR as DIR } from "./download.js";

export const FILES = {
  dit: { label: "image model", path: "supra2-img-ema.safetensors", bytes: 416405320 },
  te: { label: "text encoder", path: "flan-t5-base-encoder.safetensors", bytes: 438527456 },
  vae: { label: "image decoder", path: "sd-vae-ft-mse-decoder.safetensors", bytes: 197976468 },
  tokenizer: { label: "tokenizer", path: "tokenizer.json", bytes: 2424064 },
  tokenizerConfig: { label: "tokenizer", path: "tokenizer_config.json", bytes: 2537 },
};
export const TOTAL_BYTES = Object.values(FILES).reduce((a, f) => a + f.bytes, 0);

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

// Returns the cached File for `f`, downloading it from `base` first if needed.
// onProgress(bytesDone, bytesTotal)
export async function cachedFile(f, base, onProgress, signal) {
  const d = await dir();
  const done = await tryFile(d, f.path);
  if (done && done.size === f.bytes) {
    onProgress?.(f.bytes, f.bytes);
    return done;
  }
  if (done) await d.removeEntry(f.path);
  const part = f.path + ".part";
  const partFile = await tryFile(d, part);
  if (!(partFile && partFile.size === f.bytes)) {
    await download({ url: new URL(f.path, base).href, part, size: f.bytes }, (n) => onProgress?.(n, f.bytes), signal);
  }
  const ph = await d.getFileHandle(part);
  const got = (await ph.getFile()).size;
  if (got !== f.bytes) throw new Error(`size mismatch for ${f.path}: got ${got}, expected ${f.bytes}`);
  if (ph.move) {
    await ph.move(f.path);
    return (await d.getFileHandle(f.path)).getFile();
  }
  return ph.getFile(); // no rename support: the complete .part is used as is
}

// Bytes already on disk per file key (finished or partial), for the first-run screen.
export async function cachedBytes() {
  const d = await dir();
  const out = {};
  for (const [k, f] of Object.entries(FILES)) {
    const done = await tryFile(d, f.path);
    const ok = done?.size === f.bytes;
    const part = ok ? null : await tryFile(d, f.path + ".part");
    out[k] = { done: ok, bytes: ok ? f.bytes : part?.size || 0 };
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
