// Downloads one file into OPFS with a FileSystemSyncAccessHandle (worker-only API).
// Writes go in place, so resuming an interrupted download is just "continue at the current
// size" with an HTTP Range request, and flush() makes progress durable without copying.

export const OPFS_DIR = "supra2-img-web";
const FLUSH_EVERY = 64 * 2 ** 20;

export async function downloadToOPFS({ url, part, size, headers = {}, onProgress, signal }) {
  const root = await navigator.storage.getDirectory();
  const dir = await root.getDirectoryHandle(OPFS_DIR, { create: true });
  const fh = await dir.getFileHandle(part, { create: true });
  const handle = await fh.createSyncAccessHandle();
  try {
    let have = handle.getSize();
    if (have > size) { handle.truncate(0); have = 0; }
    let retries = 0;
    while (have < size) {
      if (signal?.aborted) throw new DOMException("Download cancelled", "AbortError");
      let res;
      try {
        res = await fetch(url, { headers: have ? { ...headers, Range: `bytes=${have}-` } : headers, cache: "no-store", signal });
      } catch (err) {
        if (err.name === "AbortError" || ++retries > 5) throw err;
        await new Promise((r) => setTimeout(r, 1000 * retries));
        continue;
      }
      if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText} for ${url}`);
      if (have && res.status !== 206) { handle.truncate(0); have = 0; } // range ignored: restart
      const reader = res.body.getReader();
      let sinceFlush = 0;
      try {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          handle.write(value, { at: have });
          have += value.byteLength;
          sinceFlush += value.byteLength;
          if (sinceFlush >= FLUSH_EVERY) { handle.flush(); sinceFlush = 0; }
          onProgress?.(have);
        }
      } catch (err) {
        handle.flush();
        if (err.name === "AbortError" || ++retries > 5) throw err;
        continue; // connection dropped: resume from `have`
      }
      handle.flush();
      if (have < size && ++retries > 5) throw new Error(`download ended early at ${have} of ${size} bytes`);
    }
    return have;
  } finally {
    try { handle.flush(); handle.close(); } catch { /* ignore */ }
  }
}
