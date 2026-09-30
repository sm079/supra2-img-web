// Downloads one file into OPFS with a FileSystemSyncAccessHandle (worker-only API).
//
// The output is `prefix` followed by byte ranges ("pieces") of the remote file, in order. A whole
// file is one piece; an extract of a large safetensors file is a new header plus the ranges of
// the tensors it keeps. Nearby pieces are fetched with one Range request (the bytes between them
// are skipped), so skipping a few small tensors doesn't cost extra round trips.
// Writes go in place, so resuming an interrupted download is just "continue at the current size",
// and flush() makes progress durable without copying.

export const OPFS_DIR = "supra2-img-web";
const FLUSH_EVERY = 64 * 2 ** 20;
const MAX_GAP = 8 * 2 ** 20; // fetch across gaps smaller than this instead of starting a new request

export const planSize = ({ prefix, pieces }) => (prefix?.byteLength || 0) + pieces.reduce((a, [s, e]) => a + e - s, 0);

// Consecutive pieces joined into request groups: { from, to, first, last } (piece indices).
function groups(pieces) {
  const out = [];
  pieces.forEach(([s, e], i) => {
    const g = out.at(-1);
    if (g && s >= g.to && s - g.to <= MAX_GAP) { g.to = e; g.last = i; } else out.push({ from: s, to: e, first: i, last: i });
  });
  return out;
}

export async function downloadToOPFS({ url, part, prefix = new Uint8Array(0), pieces, headers = {}, onProgress, signal }) {
  const size = planSize({ prefix, pieces });
  // output offset of each piece
  const outStart = [];
  let acc = prefix.byteLength;
  for (const [s, e] of pieces) { outStart.push(acc); acc += e - s; }

  const root = await navigator.storage.getDirectory();
  const dir = await root.getDirectoryHandle(OPFS_DIR, { create: true });
  const fh = await dir.getFileHandle(part, { create: true });
  const handle = await fh.createSyncAccessHandle();
  try {
    let have = handle.getSize();
    if (have > size) { handle.truncate(0); have = 0; }
    if (have < prefix.byteLength) {
      handle.write(prefix, { at: 0 });
      have = prefix.byteLength;
    }
    let retries = 0;
    for (const g of groups(pieces)) {
      while (have < outStart[g.last] + (pieces[g.last][1] - pieces[g.last][0])) {
        if (signal?.aborted) throw new DOMException("Download cancelled", "AbortError");
        // first piece of the group that isn't complete, and where in the remote file to resume
        let k = g.first;
        while (have >= outStart[k] + pieces[k][1] - pieces[k][0]) k++;
        let src = pieces[k][0] + (have - outStart[k]);
        const whole = src === 0 && g.to === pieces.at(-1)[1] && pieces.length === 1;
        let res;
        try {
          res = await fetch(url, { headers: whole ? headers : { ...headers, Range: `bytes=${src}-${g.to - 1}` }, cache: "no-store", signal });
        } catch (err) {
          if (err.name === "AbortError" || ++retries > 5) throw err;
          await new Promise((r) => setTimeout(r, 1000 * retries));
          continue;
        }
        if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText} for ${url}`);
        if (!whole && res.status !== 206) {
          if (src !== 0) throw new Error(`the server ignored a Range request for ${url}`);
        }
        const reader = res.body.getReader();
        let sinceFlush = 0;
        try {
          for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            // value covers remote bytes [src, src + len): write the parts inside pieces
            let off = 0;
            while (off < value.byteLength && k <= g.last) {
              const [ps, pe] = pieces[k];
              if (src < ps) { const skip = Math.min(ps - src, value.byteLength - off); off += skip; src += skip; continue; }
              const n = Math.min(pe - src, value.byteLength - off);
              handle.write(value.subarray(off, off + n), { at: outStart[k] + (src - ps) });
              off += n; src += n; have += n; sinceFlush += n;
              if (src >= pe) k++;
            }
            if (sinceFlush >= FLUSH_EVERY) { handle.flush(); sinceFlush = 0; }
            onProgress?.(have);
            if (k > g.last) { reader.cancel().catch(() => {}); break; }
          }
        } catch (err) {
          handle.flush();
          if (err.name === "AbortError" || ++retries > 5) throw err;
          continue; // connection dropped: resume from `have`
        }
        handle.flush();
        const groupEnd = outStart[g.last] + pieces[g.last][1] - pieces[g.last][0];
        if (have < groupEnd && ++retries > 5) throw new Error(`download ended early at ${have} of ${size} bytes`);
      }
    }
    if (have !== size) throw new Error(`download ended early at ${have} of ${size} bytes`);
    return have;
  } finally {
    try { handle.flush(); handle.close(); } catch { /* ignore */ }
  }
}
