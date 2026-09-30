// Module worker wrapper around downloadToOPFS, for downloads started from the main thread.
import { downloadToOPFS } from "./download.js";

self.onmessage = async (e) => {
  try {
    const got = await downloadToOPFS({ ...e.data, onProgress: (done) => self.postMessage({ type: "progress", done }) });
    self.postMessage({ type: "done", size: got });
  } catch (err) {
    self.postMessage({ type: "error", message: String(err?.message || err) });
  }
};
