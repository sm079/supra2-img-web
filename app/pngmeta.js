// Generation settings stored inside saved PNGs, as an iTXt chunk (UTF-8 text) whose keyword names
// the app. Each app only reads its own keyword, so an image made by another app is ignored.

const SIG = [137, 80, 78, 71, 13, 10, 26, 10];
const enc = new TextEncoder();
const dec = new TextDecoder();

let crcTable;
function crc32(bytes) {
  if (!crcTable) {
    crcTable = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      crcTable[n] = c >>> 0;
    }
  }
  let c = 0xffffffff;
  for (const b of bytes) c = crcTable[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

const isPng = (b) => b.length > 8 && SIG.every((v, i) => b[i] === v);

// Returns a copy of the PNG blob with `data` (JSON) stored under `keyword`, right after the header.
export async function writeMeta(blob, keyword, data) {
  const png = new Uint8Array(await blob.arrayBuffer());
  if (!isPng(png)) return blob;
  // keyword, null, compression flag 0, method 0, empty language tag, null, empty translated keyword, null, text
  const body = new Uint8Array([...enc.encode(keyword), 0, 0, 0, 0, 0, ...enc.encode(JSON.stringify(data))]);
  const chunk = new Uint8Array(12 + body.length);
  const view = new DataView(chunk.buffer);
  view.setUint32(0, body.length);
  chunk.set(enc.encode("iTXt"), 4);
  chunk.set(body, 8);
  view.setUint32(8 + body.length, crc32(chunk.subarray(4, 8 + body.length)));
  const ihdrEnd = 8 + 12 + new DataView(png.buffer).getUint32(8);
  return new Blob([png.subarray(0, ihdrEnd), chunk, png.subarray(ihdrEnd)], { type: "image/png" });
}

// Reads the JSON stored under `keyword`, or null when the file isn't a PNG from this app.
export async function readMeta(blob, keyword) {
  const png = new Uint8Array(await blob.arrayBuffer());
  if (!isPng(png)) return null;
  const view = new DataView(png.buffer);
  for (let p = 8; p + 12 <= png.length;) {
    const len = view.getUint32(p);
    const type = dec.decode(png.subarray(p + 4, p + 8));
    if (type === "IDAT" || type === "IEND") break; // ours sits before the image data
    if (type === "iTXt") {
      const body = png.subarray(p + 8, p + 8 + len);
      const k = body.indexOf(0);
      if (k > 0 && dec.decode(body.subarray(0, k)) === keyword && body[k + 1] === 0) {
        let q = body.indexOf(0, k + 3); // end of language tag
        q = body.indexOf(0, q + 1); // end of translated keyword
        try { return JSON.parse(dec.decode(body.subarray(q + 1))); } catch { return null; }
      }
    }
    p += 12 + len;
  }
  return null;
}

// Reads this app's settings from a drop: an image file, or one of this page's own images dragged
// from the gallery (a blob: URL).
export async function readDropped(dt, keyword) {
  const file = [...(dt.files || [])].find((f) => f.type === "image/png" || /\.png$/i.test(f.name));
  if (file) return readMeta(file, keyword);
  const url = droppedBlobUrl(dt);
  if (!url) return null;
  try { return readMeta(await (await fetch(url)).blob(), keyword); } catch { return null; }
}

// Whether a drop carries an image (rather than text), so the browser's own handling should be skipped.
export function isImageDrop(dt) {
  return [...(dt.items || [])].some((i) => i.kind === "file") || !!droppedBlobUrl(dt);
}

function droppedBlobUrl(dt) {
  const url = (dt.getData("text/uri-list") || "").split(/\r?\n/).find((l) => l && !l.startsWith("#")) || "";
  return url.startsWith(`blob:${location.origin}/`) ? url : null;
}
