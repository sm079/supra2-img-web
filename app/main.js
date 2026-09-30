import { createEngine } from "./engine.js";
import { FILES, TOTAL_BYTES, cachedBytes, storageBytes, clearCache } from "./store.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
// Model files live on Hugging Face, pinned to one commit: browsers cache files by name, so new
// uploads only reach visitors when this points at the new commit. ?models=./models/ uses a local
// build from tools/build_models.py.
const MODELS_URL = "https://huggingface.co/sm079/supra2-img-web/resolve/7bdda2b40e98d7a71590dd19b5ce4c72a1d4dfc9/";
const BASE = new URL(params.get("models") || MODELS_URL, location.href);
if (!BASE.pathname.endsWith("/")) BASE.pathname += "/";

// ------------------------------------------------------------------ choices (plain language)

const QUALITY = [
  { id: "draft", label: "Draft", steps: 20, tip: "Quickest (20 steps)" },
  { id: "good", label: "Good", steps: 30, tip: "A bit more refined (30 steps)" },
  { id: "best", label: "Best", steps: 50, tip: "SupraLabs' recommended setting (50 steps)" },
];
const EXAMPLES = [
  "a sea jellyfish floating in the pitch-black ocean depths",
  "a red fox sitting in a snowy forest at sunrise, soft golden light, detailed fur",
  "a cozy wooden cabin by a lake at night, warm light in the windows, stars reflected in the water",
  "a bowl of ramen with a soft-boiled egg, steam rising, top-down photo on a dark table",
  "an astronaut riding a horse across a desert, dramatic clouds, cinematic lighting",
  "a watercolor painting of a lighthouse on a rocky coast during a storm",
  "a small robot watering plants in a sunny greenhouse, cute, 3d render",
];
const DEFAULTS = { steps: 50, cfg: 3 };
const FILE_ROWS = [
  ["dit", "Image model", "Supra2-IMG"],
  ["te", "Text encoder", "Flan-T5-Base (encoder only)"],
  ["vae", "Image decoder", "SD VAE ft-MSE (decoder only)"],
  ["tokenizer", "Tokenizer", "Flan-T5"],
];
const ICONS = {
  dl: '<svg viewBox="0 0 24 24"><path d="M12 4v11M7 10l5 5 5-5M5 20h14"/></svg>',
  reuse: '<svg viewBox="0 0 24 24"><path d="M4 12a8 8 0 0 1 14-5.3L20 9M20 4v5h-5M20 12a8 8 0 0 1-14 5.3L4 15M4 20v-5h5"/></svg>',
};

// ------------------------------------------------------------------ state

const store = {
  get(k, d) { try { const v = localStorage.getItem("supra2-img-web." + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("supra2-img-web." + k, JSON.stringify(v)); } catch { /* private mode */ } },
};
const ui = { ...DEFAULTS, ...store.get("ui", {}) };
let pipe = null;
let phase = "starting"; // starting | welcome | loading | ready | generating (queue running)
let loadAbort = null;
let exampleIdx = 0;
let current = null; // the finished image being viewed
let live = false; // viewer follows the image being made
let pinned = false; // user picked an older image during this queue run; don't jump to new ones
let galleryOpen = false;
const gallery = [];
const jobs = []; // waiting
let running = null; // { ...job, abort, frac, thumb, hasPreview }

const saveUi = () => store.set("ui", ui);
const fmtGB = (n) => (n >= 2 ** 30 ? `${(n / 2 ** 30).toFixed(2)} GB` : n >= 2 ** 20 ? `${(n / 2 ** 20).toFixed(n >= 100 * 2 ** 20 ? 0 : 1)} MB` : n > 0 ? `${Math.max(1, Math.round(n / 1024))} KB` : "0 KB");
const fmtTime = (ms) => {
  if (ms < 950) return `${(ms / 1000).toFixed(1)} s`;
  const s = Math.max(1, Math.round(ms / 1000));
  return s >= 60 ? `${Math.floor(s / 60)} min ${s % 60 ? `${s % 60} s` : ""}`.trim() : `${s} s`;
};
// Download speed averaged over the last 20 s, so the time-left estimate doesn't swing with bursty progress.
function rateMeter(windowMs = 20000) {
  const samples = [];
  return (done) => {
    const now = performance.now();
    samples.push([now, done]);
    while (samples.length > 2 && now - samples[0][0] > windowMs) samples.shift();
    const [t0, d0] = samples[0];
    return now - t0 > 1500 ? ((done - d0) / (now - t0)) * 1000 : 0;
  };
}
const newSeed = () => Math.floor(Math.random() * 2 ** 32);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pressed = (id) => $(id).getAttribute("aria-pressed") === "true";

// ------------------------------------------------------------------ small UI helpers

function setStatus(text, kind = "idle") {
  $("statusText").textContent = text;
  $("statusDot").className = "dot " + ({ ok: "ok", busy: "busy", err: "err" }[kind] || "");
}

function showPanel(which) {
  for (const id of ["welcome", "loading", "empty"]) $(id).hidden = id !== which;
  $("canvas").hidden = which !== "canvas";
  $("stageFoot").hidden = which === "welcome" || which === "loading";
  if (which === "welcome" || which === "loading") openGallery(false);
}

function showError(msg) {
  const el = $("errorCard");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(showError.t);
  showError.t = setTimeout(() => { el.hidden = true; }, 9000);
}

function friendlyError(e) {
  const m = String(e?.message || e);
  if (/device lost|out of memory|OOM|allocation/i.test(m)) return "Your graphics card ran out of memory. Close other tabs or apps that use it and reload the page.";
  if (/HTTP|fetch|network|Failed to fetch/i.test(m)) return "The download was interrupted. Check your connection and try again.";
  if (/quota|storage|space/i.test(m)) return "Not enough storage space in this browser.";
  return "Something went wrong: " + m;
}

function radioGroup(el, items, isOn, onPick) {
  el.innerHTML = "";
  for (const it of items) {
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(isOn(it)));
    if (it.tip) b.title = it.tip;
    b.textContent = it.label;
    b.onclick = () => onPick(it);
    el.append(b);
  }
}

// ------------------------------------------------------------------ sampling controls

function renderControls() {
  radioGroup($("quality"), QUALITY, (q) => q.steps === ui.steps, (q) => { ui.steps = q.steps; saveUi(); renderControls(); });
  $("steps").value = ui.steps;
  $("stepsOut").textContent = ui.steps;
  $("stepsHint").textContent = `${ui.steps} steps`;
  $("cfg").value = ui.cfg;
  $("cfgOut").textContent = ui.cfg.toFixed(1);
  $("cfgHint").textContent = ui.cfg <= 1 ? "guidance off · 2× faster" : ui.cfg === DEFAULTS.cfg ? "recommended" : ui.cfg > 6 ? "strong: may look harsh" : "";
  renderGo();
}

// per-step time measured on this device (with and without guidance)
function estimateMs(steps = ui.steps, cfg = ui.cfg) {
  const s = store.get("speed", null);
  if (!s) return null;
  const perStep = cfg > 1 ? s.stepCfg ?? s.step * 1.8 : s.step ?? s.stepCfg / 1.8;
  return perStep * steps + s.overhead;
}

function renderGo() {
  const busy = phase === "generating";
  const n = +$("batch").value;
  $("goLabel").textContent = busy ? (n > 1 ? `Queue ${n} more` : "Add to queue") : n > 1 ? `Generate ${n}` : "Generate";
  const est = phase === "ready" || busy ? estimateMs() : null;
  $("estimate").textContent = est ? `about ${fmtTime(est * n)}` : "";
  $("stopBtn").hidden = !busy;
  $("goBtn").disabled = !(phase === "ready" || busy);
  if (busy) setStatus(jobs.length ? `Creating image… (${jobs.length} queued)` : "Creating image…", "busy");
}

function setRandom(on) {
  $("randomBtn").setAttribute("aria-pressed", String(on));
  $("randomBtn").title = on ? "A new random seed for every image (click to keep the seed)" : "Keeping this seed (click for a new random seed every image)";
  store.set("randomSeed", on);
}

// ------------------------------------------------------------------ loading

async function ensureModel() {
  if (pipe.ready) { toReady(); return; }
  const cached = await cachedBytes();
  if (Object.values(cached).every((c) => c.done)) return loadModel();
  showWelcome(cached);
}

function showWelcome(cached) {
  phase = "welcome";
  showPanel("welcome");
  renderGo();
  setStatus("Download the model to get started");
  let need = 0;
  const rows = FILE_ROWS.map(([k, label, what]) => {
    const bytes = k === "tokenizer" ? FILES.tokenizer.bytes + FILES.tokenizerConfig.bytes : FILES[k].bytes;
    const done = cached[k].done && (k !== "tokenizer" || cached.tokenizerConfig.done);
    if (!done) need += bytes - (cached[k]?.bytes || 0);
    return `<div class="dl-row"><span><b>${esc(label)}</b> · ${esc(what)}</span><span${done ? ' class="saved"' : ""}>${done ? "✓ saved" : fmtGB(bytes)}</span></div>`;
  });
  rows.push(`<div class="dl-row total"><span>Total</span><span>${fmtGB(TOTAL_BYTES)}</span></div>`);
  $("dlCard").innerHTML = rows.join("");
  $("welcomeGo").textContent = need < TOTAL_BYTES * 0.99 ? `Download the remaining ${fmtGB(need)} & start` : `Download ${fmtGB(TOTAL_BYTES)} & start`;
}

async function loadModel() {
  phase = "loading";
  showPanel("loading");
  renderGo();
  $("loadingCancel").hidden = true;
  $("loadingTitle").textContent = "Getting ready…";
  $("loadingText").textContent = "";
  $("loadingBar").style.width = "0%";
  loadAbort = new AbortController();
  const meter = rateMeter();
  try {
    await pipe.load({
      modelsBase: BASE.href,
      signal: loadAbort.signal,
      onStatus: (s) => {
        if (s.phase === "download") {
          $("loadingCancel").hidden = false;
          const rate = meter(s.done);
          $("loadingTitle").textContent = "Downloading the model";
          $("loadingBar").style.width = `${(100 * s.done) / s.total}%`;
          const left = rate > 0 ? ` · about ${fmtTime(((s.total - s.done) / rate) * 1000)} left` : "";
          $("loadingText").textContent = `${fmtGB(s.done)} of ${fmtGB(s.total)} · ${s.file}${left}`;
          setStatus(`Downloading ${Math.floor((100 * s.done) / s.total)}%`, "busy");
        } else if (s.phase === "load") {
          $("loadingCancel").hidden = true;
          $("loadingTitle").textContent = "Loading onto your graphics card";
          $("loadingBar").style.width = `${s.frac * 100}%`;
          $("loadingText").textContent = `Preparing the ${s.what}…`;
          setStatus("Loading model…", "busy");
        }
      },
    });
    toReady();
  } catch (e) {
    phase = "welcome";
    if (e.name === "AbortError") { ensureModel(); return; }
    console.error(e);
    setStatus("Couldn't load the model", "err");
    showWelcome(await cachedBytes());
    showError(friendlyError(e));
  }
}

function toReady() {
  phase = "ready";
  setStatus("Ready", "ok");
  renderGpuChip();
  if (current) showItem(current); else showPanel("empty");
  renderGo();
  if (jobs.length) runQueue();
}

// ------------------------------------------------------------------ queue & generation

// The viewer shows either a finished image (`current`) or, while `live`, the image being made.
// Every draw bumps the token so a slow image decode can't paint over a newer choice.
let drawToken = 0;

function draw(img) {
  drawToken++;
  const c = $("canvas");
  if (c.width !== img.width || c.height !== img.height) { c.width = img.width; c.height = img.height; }
  c.getContext("2d").putImageData(new ImageData(img.data, img.width, img.height), 0, 0);
  showPanel("canvas");
}

async function showItem(item) {
  const tok = ++drawToken;
  const im = new Image();
  im.src = item.url;
  try { await im.decode(); } catch { return; }
  if (tok !== drawToken) return;
  const c = $("canvas");
  c.width = im.naturalWidth;
  c.height = im.naturalHeight;
  c.getContext("2d").drawImage(im, 0, 0);
  showPanel("canvas");
}

// Scale the small (32x32) preview kept on the running job up to the canvas.
function drawLive() {
  if (!running?.hasPreview) return;
  drawToken++;
  const c = $("canvas");
  if (c.width !== 256 || c.height !== 256) { c.width = 256; c.height = 256; }
  const ctx = c.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(running.thumb, 0, 0, 256, 256);
  showPanel("canvas");
}

function onPreview(job, img) {
  const t = job.thumb;
  if (t.width !== img.width || t.height !== img.height) { t.width = img.width; t.height = img.height; }
  t.getContext("2d").putImageData(new ImageData(img.data, img.width, img.height), 0, 0);
  job.hasPreview = true;
  for (const c of document.querySelectorAll(".tile.running canvas")) {
    if (c.width !== t.width || c.height !== t.height) { c.width = t.width; c.height = t.height; }
    c.getContext("2d").drawImage(t, 0, 0);
    c.parentElement.classList.add("has-preview");
  }
  if (live) drawLive();
}

// Snapshot everything at click time, so the settings can keep changing while jobs wait.
function enqueue() {
  if (!(phase === "ready" || phase === "generating")) return;
  const n = +$("batch").value;
  const random = pressed("randomBtn");
  if ($("seed").value.trim() === "") $("seed").value = newSeed();
  const base = Math.min(2 ** 32 - 1, Math.max(0, Math.floor(Number($("seed").value) || 0)));
  const text = $("prompt").value.trim();
  const negative = $("negBox").open ? $("negative").value.trim() : "";
  for (let k = 0; k < n; k++) {
    // random mode: a fresh seed per image; fixed seed + batch: consecutive seeds
    const seed = random ? newSeed() : (base + k) >>> 0;
    jobs.push({ text, negative, ui: { ...ui }, seed });
    if (k === 0 || random) $("seed").value = seed;
  }
  store.set("seed", $("seed").value);
  store.set("prompt", $("prompt").value);
  renderStrip();
  if (!running) runQueue();
  else renderGo();
}

async function runQueue() {
  if (running || (phase !== "ready" && phase !== "generating")) return;
  phase = "generating";
  renderGo();
  while (jobs.length) {
    const job = jobs.shift();
    running = { ...job, abort: new AbortController(), frac: 0, thumb: document.createElement("canvas"), hasPreview: false };
    // follow the new image unless the user is looking at an older one
    if (!pinned) live = true;
    renderStrip();
    renderInfo();
    await runJob(running);
    running = null;
    renderStrip();
  }
  phase = "ready";
  live = false;
  pinned = false;
  renderInfo();
  $("progress").hidden = true;
  setStatus("Ready", "ok");
  renderGo();
  renderGpuChip();
}

async function runJob(job) {
  const { steps, cfg } = job.ui;
  $("errorCard").hidden = true;
  $("progress").hidden = !live;
  $("progressBar").style.width = "0%";
  const queued = () => (jobs.length ? ` · ${jobs.length} more queued` : "");
  $("progressText").textContent = `Reading your prompt…${queued()}`;
  setStatus(jobs.length ? `Creating image… (${jobs.length} queued)` : "Creating image…", "busy");
  let backgrounded = document.visibilityState === "hidden";
  const onVis = () => { if (document.visibilityState === "hidden") backgrounded = true; };
  document.addEventListener("visibilitychange", onVis);
  const t0 = performance.now();
  let sampleStart = 0;
  try {
    const res = await pipe.generate({
      prompt: job.text, negative: job.negative, steps, cfg, seed: job.seed,
      signal: job.abort.signal,
      onPreview: (img) => onPreview(job, img),
      onProgress: (p) => {
        if (p.phase === "sample") {
          if (!sampleStart) sampleStart = performance.now();
          const el = performance.now() - sampleStart;
          const left = p.frac > 0.1 && el > 1500 ? ` · about ${fmtTime((el / p.frac) * (1 - p.frac))} left` : "";
          job.frac = p.frac * 0.95;
          $("progressBar").style.width = `${job.frac * 100}%`;
          $("progressText").textContent = `Step ${p.step + 1} of ${p.steps}${left}${queued()}`;
        } else if (p.phase === "decode") {
          job.frac = 0.95 + p.frac * 0.05;
          $("progressBar").style.width = `${job.frac * 100}%`;
          $("progressText").textContent = `Finishing up…${queued()}`;
        }
        updateRunningTile();
      },
    });
    if (live) draw(res.image);
    if (!backgrounded) {
      const s = store.get("speed", {}) || {};
      s[cfg > 1 ? "stepCfg" : "step"] = res.timings.perStep;
      s.overhead = res.timings.decode + res.timings.encode;
      store.set("speed", s);
    }
    const record = { text: job.text, negative: job.negative, ui: job.ui, seed: job.seed };
    await addToGallery(res.image, record, { ...res.timings, total: performance.now() - t0, backgrounded }, live);
  } catch (e) {
    if (e.name !== "AbortError") {
      console.error(e);
      showError(friendlyError(e));
      jobs.length = 0; // don't keep hammering a failing GPU
    }
    if (live) { if (current) showItem(current); else showPanel("empty"); }
  } finally {
    document.removeEventListener("visibilitychange", onVis);
  }
}

function stopAll() {
  jobs.length = 0;
  running?.abort.abort();
  renderStrip();
}

function renderGpuChip() {
  const g = pipe.gpu || {};
  $("gpuChip").hidden = !g.name;
  $("gpuChip").textContent = g.name ? g.name.split(" ")[0] : "";
  $("gpuChip").title = `${g.name || "Graphics card"}${g.peak ? ` · up to ${fmtGB(g.peak)} of working memory used` : ""}`;
}

function toBlobURL(img) {
  const c = document.createElement("canvas");
  c.width = img.width;
  c.height = img.height;
  c.getContext("2d").putImageData(new ImageData(img.data, img.width, img.height), 0, 0);
  return new Promise((r) => c.toBlob((b) => r(URL.createObjectURL(b)), "image/png"));
}

// Finished images live as PNG blobs (not raw pixels), so a long session stays light on memory.
const GALLERY_MAX = 300;

async function addToGallery(img, record, timings, follow) {
  const url = await toBlobURL(img);
  const item = { url, width: img.width, height: img.height, record, timings };
  gallery.unshift(item);
  if (gallery.length > GALLERY_MAX) {
    const old = gallery.pop();
    URL.revokeObjectURL(old.url);
    if (current === old) current = null;
  }
  if (follow) current = item;
  renderInfo();
  renderStrip();
}

function select(item) {
  current = item;
  if (running) { live = false; pinned = true; $("progress").hidden = true; }
  showItem(item);
  openGallery(false);
  renderInfo();
  renderStrip();
}

function selectLive() {
  if (!running) return;
  live = true;
  pinned = false;
  $("progress").hidden = false;
  drawLive();
  openGallery(false);
  renderInfo();
  renderStrip();
}

function metaLine(parts, note) {
  const el = document.createElement("span");
  el.className = "meta";
  el.innerHTML = parts.map((p, i) => (i ? '<span class="sep">·</span>' : "") + p).join("") + (note ? `<span class="sep">·</span><span class="note">${esc(note)}</span>` : "");
  return el;
}

function settingsBits(r) {
  const bits = [`${r.ui.steps} steps`, r.ui.cfg > 1 ? `guidance ${r.ui.cfg}` : "no guidance", `seed ${r.seed}`];
  if (r.negative) bits.push("avoid: " + esc(r.negative.length > 30 ? r.negative.slice(0, 30) + "…" : r.negative));
  return bits;
}

function renderInfo() {
  const el = $("info");
  el.innerHTML = "";
  if (live && running) {
    const m = metaLine(["<b>Creating</b>", ...settingsBits(running)]);
    m.title = running.text;
    el.append(m);
    return;
  }
  if (!current) return;
  const { record: r, timings: t } = current;
  const m = metaLine([`<b>${esc(r.text || "(no prompt)")}</b>`, ...settingsBits(r), `made in ${fmtTime(t.total)}`], t.backgrounded ? "slower: the tab was in the background" : "");
  m.title = r.text;
  const actions = document.createElement("span");
  actions.className = "actions";
  const dl = document.createElement("a");
  dl.href = current.url;
  dl.download = `supra2-img-${r.seed}.png`;
  dl.title = "Save this image";
  dl.innerHTML = `${ICONS.dl}<span>Save</span>`;
  const reuse = document.createElement("button");
  reuse.type = "button";
  reuse.innerHTML = `${ICONS.reuse}<span>Use these settings</span>`;
  reuse.title = "Restore this image's prompt, settings and seed";
  reuse.onclick = () => {
    $("prompt").value = r.text;
    store.set("prompt", r.text);
    $("negative").value = r.negative || "";
    $("negBox").open = !!r.negative;
    store.set("negative", $("negative").value);
    Object.assign(ui, r.ui);
    saveUi();
    $("seed").value = r.seed;
    store.set("seed", String(r.seed));
    setRandom(false);
    renderControls();
  };
  actions.append(dl, reuse);
  el.append(m, actions);
}

function ringSVG(frac) {
  const c = 2 * Math.PI * 14;
  return `<svg class="ring" viewBox="0 0 34 34"><circle class="bg" cx="17" cy="17" r="14"/><circle class="fg" cx="17" cy="17" r="14" stroke-dasharray="${c}" stroke-dashoffset="${c * (1 - frac)}"/></svg>`;
}

function updateRunningTile() {
  if (!running) return;
  for (const t of document.querySelectorAll(".tile.running")) {
    t.querySelector(".fg")?.setAttribute("stroke-dashoffset", String(2 * Math.PI * 14 * (1 - running.frac)));
    t.querySelector(".tile-bar i").style.width = `${running.frac * 100}%`;
  }
}

// Tiles in display order: waiting jobs (next one nearest), the one being made, then finished images (newest first).
function renderTiles(el) {
  el.innerHTML = "";
  jobs.forEach((job, i) => {
    const b = document.createElement("div");
    b.className = "tile queued";
    b.title = `Waiting: ${job.text}`;
    b.innerHTML = `<span>${i === 0 ? "Next" : `#${i + 1}`}</span>`;
    const x = document.createElement("button");
    x.type = "button";
    x.className = "x";
    x.title = "Remove from queue";
    x.setAttribute("aria-label", "Remove from queue");
    x.textContent = "×";
    x.onclick = () => { const k = jobs.indexOf(job); if (k >= 0) jobs.splice(k, 1); renderStrip(); };
    b.append(x);
    el.prepend(b); // last queued on the far left
  });
  if (running) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "tile running" + (running.hasPreview ? " has-preview" : "");
    b.title = `Being made: ${running.text}`;
    b.setAttribute("aria-current", String(live));
    const c = document.createElement("canvas");
    if (running.hasPreview) {
      c.width = running.thumb.width;
      c.height = running.thumb.height;
      c.getContext("2d").drawImage(running.thumb, 0, 0);
    }
    b.append(c);
    b.insertAdjacentHTML("beforeend", `${ringSVG(running.frac)}<span class="tile-bar"><i style="width:${running.frac * 100}%"></i></span>`);
    b.onclick = selectLive;
    el.append(b);
  }
  for (const item of gallery) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "tile";
    b.title = item.record.text;
    b.setAttribute("aria-current", String(!live && item === current));
    // reuse the strip's decoded <img> so re-rendering doesn't flash empty tiles
    let im = el.id === "strip" ? item.stripImg : null;
    if (!im) {
      im = document.createElement("img");
      im.src = item.url;
      im.alt = item.record.text;
      if (el.id === "strip") item.stripImg = im;
    }
    b.append(im);
    b.onclick = () => select(item);
    el.append(b);
  }
}

function renderStrip() {
  renderTiles($("strip"));
  if (galleryOpen) renderTiles($("galleryGrid"));
  const n = gallery.length;
  $("allCount").textContent = n ? `All ${n}` : "All";
  $("allBtn").disabled = !n && !running && !jobs.length;
  $("galleryTitle").textContent = n === 1 ? "1 image" : `${n} images`;
  renderGo();
}

function openGallery(on) {
  galleryOpen = on;
  $("gallery").hidden = !on;
  $("allBtn").setAttribute("aria-pressed", String(on));
  if (on) renderTiles($("galleryGrid"));
  else $("galleryGrid").innerHTML = "";
}

// ------------------------------------------------------------------ settings dialog

function setScaling(v) {
  $("canvasBox").classList.toggle("pixel", v === "pixel");
  store.set("scaling", v);
}

async function openSettings() {
  const bytes = await storageBytes();
  $("storageText").textContent = bytes ? `Model files ${fmtGB(bytes)}` : "Nothing downloaded yet";
  const busy = phase === "generating" || phase === "loading";
  $("clearBtn").disabled = !bytes || busy;
  $("scaling").value = store.get("scaling", "smooth");
  $("settings").showModal();
}

// ------------------------------------------------------------------ init

async function init() {
  if (!navigator.gpu) {
    $("fatal").hidden = false;
    $("fatal").textContent = "This browser can't run Supra2-Img Web.\n\nIt needs WebGPU: use a recent version of Chrome, Edge or Safari, or Firefox 141+ on Windows.";
    return;
  }
  setStatus("Starting…", "busy");
  pipe = await createEngine({ inPage: params.get("engine") === "page" });

  // prompt
  $("prompt").value = store.get("prompt", EXAMPLES[0]);
  $("prompt").oninput = () => store.set("prompt", $("prompt").value);
  $("negative").value = store.get("negative", "");
  $("negBox").open = !!$("negative").value;
  $("negative").oninput = () => store.set("negative", $("negative").value);
  $("exampleBtn").onclick = () => {
    exampleIdx = (exampleIdx + 1) % EXAMPLES.length;
    $("prompt").value = EXAMPLES[exampleIdx];
    store.set("prompt", $("prompt").value);
    $("prompt").focus();
  };

  // sampling
  $("steps").oninput = () => { ui.steps = +$("steps").value; saveUi(); renderControls(); };
  $("cfg").oninput = () => { ui.cfg = +$("cfg").value; saveUi(); renderControls(); };
  $("cfg").ondblclick = () => { ui.cfg = DEFAULTS.cfg; saveUi(); renderControls(); };
  $("cfg").title = "How closely to follow the prompt (double-click to reset to 3)";

  // seed
  setRandom(store.get("randomSeed", true));
  $("seed").value = store.get("seed", String(newSeed()));
  $("randomBtn").onclick = () => setRandom(!pressed("randomBtn"));
  $("seed").oninput = () => {
    $("seed").value = $("seed").value.replace(/\D/g, "").slice(0, 10);
    setRandom(false); // typing a seed means "use this one"
    store.set("seed", $("seed").value);
  };
  $("rollBtn").onclick = () => {
    $("seed").value = newSeed();
    setRandom(false); // a rolled seed should be the one that's used
    store.set("seed", $("seed").value);
  };

  // generate / queue
  $("batch").value = String(store.get("batch", 1));
  if (!$("batch").value) $("batch").value = "1";
  $("batch").onchange = () => { store.set("batch", +$("batch").value); renderGo(); };
  $("goBtn").onclick = enqueue;
  $("stopBtn").onclick = stopAll;
  $("allBtn").onclick = () => openGallery(!galleryOpen);
  $("galleryClose").onclick = () => openGallery(false);
  renderStrip();
  renderControls();
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); enqueue(); }
    if (e.key === "Escape" && galleryOpen) openGallery(false);
  });

  // model & settings
  setScaling(store.get("scaling", "smooth"));
  $("scaling").onchange = () => setScaling($("scaling").value);
  $("settingsBtn").onclick = openSettings;
  $("welcomeGo").onclick = loadModel;
  $("loadingCancel").onclick = () => loadAbort?.abort();
  $("clearBtn").onclick = async () => {
    if (!confirm("Remove the downloaded model files from this browser? You'll need to download them again to generate images.")) return;
    $("settings").close();
    await pipe.unload();
    await clearCache();
    ensureModel();
  };

  await ensureModel();
}

init();
