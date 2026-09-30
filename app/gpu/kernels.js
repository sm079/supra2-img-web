// WGSL kernel generators. Every kernel binds its parameters as a uniform at binding 0.

const cache = new Map();
const memo = (key, fn) => {
  if (!cache.has(key)) cache.set(key, fn());
  return cache.get(key);
};

// Exact (erf) GELU; erf via Abramowitz-Stegun 7.1.26 (|err| < 1.5e-7).
const GELU = /* wgsl */ `
fn erf_(x: f32) -> f32 {
  let s = sign(x);
  let a = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return s * y;
}
fn gelu(x: f32) -> f32 { return 0.5 * x * (1.0 + erf_(x * 0.7071067811865476)); }
fn silu(x: f32) -> f32 { return x / (1.0 + exp(-x)); }
`;

// GEMM lives in gemm.js
export { matmulShader } from "./gemm.js";

// ----------------------------------------------------------------------------- row kernels

// Row-wise softmax in place: S[row][0..cols) * scale (+ BIAS[row][col] when `bias`), optional
// causal mask (col > row % rowsPerHead).
export const softmaxShader = (bias = false) => memo("softmax" + bias, () => /* wgsl */ `
struct Params { rows: u32, cols: u32, causal: u32, rowsPerHead: u32, nx: u32, scale: f32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> S: array<f32>;
${bias ? "@group(0) @binding(2) var<storage, read> BIAS: array<f32>;" : ""}
fn score(i: u32) -> f32 { return S[i] * P.scale${bias ? " + BIAS[i]" : ""}; }
var<workgroup> red: array<f32, 256>;
fn reduceMax(t: u32, v: f32) -> f32 {
  red[t] = v; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { red[t] = max(red[t], red[t + s]); } workgroupBarrier(); }
  let r = red[0]; workgroupBarrier(); return r;
}
fn reduceSum(t: u32, v: f32) -> f32 {
  red[t] = v; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { red[t] = red[t] + red[t + s]; } workgroupBarrier(); }
  let r = red[0]; workgroupBarrier(); return r;
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let row = wg.y * P.nx + wg.x;
  if (row >= P.rows) { return; }
  let base = row * P.cols;
  var limit = P.cols;
  if (P.causal == 1u) { limit = min(P.cols, row % P.rowsPerHead + 1u); }
  var mx = -3.0e38;
  for (var c = t; c < limit; c += 256u) { mx = max(mx, score(base + c)); }
  mx = reduceMax(t, mx);
  var sm = 0.0;
  for (var c = t; c < limit; c += 256u) { let e = exp(score(base + c) - mx); S[base + c] = e; sm += e; }
  sm = reduceSum(t, sm);
  let inv = 1.0 / sm;
  for (var c = t; c < P.cols; c += 256u) {
    if (c < limit) { S[base + c] = S[base + c] * inv; } else { S[base + c] = 0.0; }
  }
}`);

// RMSNorm over rows of length `cols` with a weight vector of length `cols` (T5LayerNorm):
//   y = x * rsqrt(mean(x^2) + eps) * w
export const rmsnormShader = () => memo("rms", () => /* wgsl */ `
struct Params { rows: u32, cols: u32, nx: u32, eps: f32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> Wt: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let row = wg.y * P.nx + wg.x;
  if (row >= P.rows) { return; }
  let base = row * P.cols;
  var ss = 0.0;
  for (var c = t; c < P.cols; c += 64u) { let v = X[base + c]; ss += v * v; }
  red[t] = ss; workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) { if (t < s) { red[t] = red[t] + red[t + s]; } workgroupBarrier(); }
  let r = inverseSqrt(red[0] / f32(P.cols) + P.eps);
  for (var c = t; c < P.cols; c += 64u) { Y[base + c] = X[base + c] * r * Wt[c]; }
}`);

// LayerNorm without affine parameters, optionally followed by adaLN modulation:
//   y = LN(x) * (1 + scale) + shift, with scale and shift read from MOD at element offsets.
export const layernormShader = (mod) => memo("ln" + mod, () => /* wgsl */ `
struct Params { rows: u32, cols: u32, shiftOff: u32, scaleOff: u32, nx: u32, eps: f32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
${mod ? "@group(0) @binding(2) var<storage, read> MOD: array<f32>;" : ""}
@group(0) @binding(${mod ? 3 : 2}) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 256>;
fn rsum(t: u32, v: f32) -> f32 {
  red[t] = v; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { red[t] = red[t] + red[t + s]; } workgroupBarrier(); }
  let r = red[0]; workgroupBarrier(); return r;
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let row = wg.y * P.nx + wg.x;
  if (row >= P.rows) { return; }
  let base = row * P.cols;
  var s = 0.0;
  for (var c = t; c < P.cols; c += 256u) { s += X[base + c]; }
  let mean = rsum(t, s) / f32(P.cols);
  var v = 0.0;
  for (var c = t; c < P.cols; c += 256u) { let d = X[base + c] - mean; v += d * d; }
  let r = inverseSqrt(rsum(t, v) / f32(P.cols) + P.eps);
  for (var c = t; c < P.cols; c += 256u) {
    Y[base + c] = (X[base + c] - mean) * r${mod ? " * (1.0 + MOD[P.scaleOff + c]) + MOD[P.shiftOff + c]" : ""};
  }
}`);

// GroupNorm statistics over an NHWC image X[pixels][C], one workgroup per group of C/G channels.
// Two passes (mean, then centered variance) for accuracy. STATS[g] = (mean, rstd).
export const groupnormStatsShader = () => memo("gnstats", () => /* wgsl */ `
struct Params { pixels: u32, C: u32, cg: u32, eps: f32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> STATS: array<f32>;
var<workgroup> red: array<f32, 256>;
fn rsum(t: u32, v: f32) -> f32 {
  red[t] = v; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { red[t] = red[t] + red[t + s]; } workgroupBarrier(); }
  let r = red[0]; workgroupBarrier(); return r;
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let g = wg.x;
  let n = P.pixels * P.cg;
  let c0 = g * P.cg;
  var s = 0.0;
  for (var i = t; i < n; i += 256u) { s += X[(i / P.cg) * P.C + c0 + i % P.cg]; }
  let mean = rsum(t, s) / f32(n);
  var v = 0.0;
  for (var i = t; i < n; i += 256u) { let d = X[(i / P.cg) * P.C + c0 + i % P.cg] - mean; v += d * d; }
  let var_ = rsum(t, v) / f32(n);
  if (t == 0u) { STATS[2u * g] = mean; STATS[2u * g + 1u] = inverseSqrt(var_ + P.eps); }
}`);

// GroupNorm apply: y = (x - mean_g) * rstd_g * gamma[c] + beta[c], optionally followed by SiLU.
export const groupnormApplyShader = (silu) => memo("gnapply" + silu, () => /* wgsl */ `
struct Params { n: u32, C: u32, cg: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> STATS: array<f32>;
@group(0) @binding(3) var<storage, read> GAMMA: array<f32>;
@group(0) @binding(4) var<storage, read> BETA: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i >= P.n) { return; }
  let c = i % P.C;
  let g = c / P.cg;
  var y = (X[i] - STATS[2u * g]) * STATS[2u * g + 1u] * GAMMA[c] + BETA[c];
  ${silu ? "y = y / (1.0 + exp(-y));" : ""}
  Y[i] = y;
}`);

// Fused ("flash") attention, one head x 64 queries per workgroup of 128 threads, keys streamed in
// blocks of 32 with an online softmax; nothing proportional to Lq*Lk touches memory.
// Head dim D in {64, 128}. 16 KB of workgroup memory, as a vec4 array SH[1024]:
//   [0, 512)    P    probabilities of the current key block, [32 keys][64 rows]
//   [512, 768)  Qs   Q chunk [16 dims][64 rows]        (phase 1)
//   [768, 896)  Ks   K chunk [16 dims][32 keys]        (phase 1), then row-sum partials
//   [896, 1024) row-max partials [64 rows][8]
//   [512, 1024) Vs   V chunk [32 keys][64 dims]         (phase 3, reuses the above)
// Thread (sr = t / 8, sc = t % 8) owns rows sr*4..+3, keys sc*4..+3 of the score tile and
// columns c*32 + sc*4..+3 (per 64-wide V chunk) of the output rows; all register arrays use
// constant indices only.
export const flashAttentionShader = (D) => memo("flash" + D, () => {
  const DC = D / 16; // phase-1 dim chunks
  const VC = D / 64; // phase-3 value chunks
  const NO = 2 * VC; // output vec4 columns per row
  const r4 = [0, 1, 2, 3];
  const o = [];
  for (let i = 0; i < 4; i++) for (let c = 0; c < NO; c++) o.push(`var o${i}_${c} = vec4<f32>();`);
  const rescale = [];
  for (let i = 0; i < 4; i++) for (let c = 0; c < NO; c++) rescale.push(`o${i}_${c} = o${i}_${c} * alpha.${"xyzw"[i]};`);
  const pvFma = (vc) => {
    const out = [];
    for (let i = 0; i < 4; i++) {
      for (let c = 0; c < 2; c++) out.push(`o${i}_${vc * 2 + c} += pv.${"xyzw"[i]} * v${c};`);
    }
    return out.join("\n        ");
  };
  const store = [];
  for (let i = 0; i < 4; i++) {
    store.push(`{ let q = q0 + sr * 4u + ${i}u; if (q < P.Lq) { let inv = 1.0 / l.${"xyzw"[i]}; let ob = q * P.ldo + P.oOff + h * ${D}u;`);
    for (let vc = 0; vc < VC; vc++) {
      for (let c = 0; c < 2; c++) {
        const col = `${vc * 64 + c * 32}u + sc * 4u`;
        store.push(`  { let v = o${i}_${vc * 2 + c} * inv; let b = ob + ${col}; O[b] = v.x; O[b + 1u] = v.y; O[b + 2u] = v.z; O[b + 3u] = v.w; }`);
      }
    }
    store.push("} }");
  }
  return /* wgsl */ `
struct Params { Lq: u32, Lk: u32, ldq: u32, ldk: u32, ldv: u32, ldo: u32, qOff: u32, kOff: u32, vOff: u32, oOff: u32, scale: f32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> K: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> V: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;
var<workgroup> SH: array<vec4<f32>, 1024>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let h = wg.y;
  let q0 = wg.x * 64u;
  let sr = t / 8u;
  let sc = t % 8u;
  ${o.join("\n  ")}
  var m = vec4<f32>(-1e30);
  var l = vec4<f32>(0.0);

  for (var k0 = 0u; k0 < P.Lk; k0 += 32u) {
    // ---- phase 1: S = Q K^T for the 64 x 32 tile
    var s0 = vec4<f32>(); var s1 = vec4<f32>(); var s2 = vec4<f32>(); var s3 = vec4<f32>();
    for (var dc = 0u; dc < ${DC}u; dc++) {
      {
        // Q chunk: row t/2, dims (t%2)*8..+7 of this chunk
        let qr = t / 2u;
        let d8 = (t % 2u) * 8u;
        var a = vec4<f32>(); var b = vec4<f32>();
        if (q0 + qr < P.Lq) {
          let gi = ((q0 + qr) * P.ldq + P.qOff + h * ${D}u + dc * 16u + d8) >> 2u;
          a = Q[gi]; b = Q[gi + 1u];
        }
        let lane = qr & 3u;
        let col = qr >> 2u;
        SH[512u + (d8 + 0u) * 16u + col][lane] = a.x; SH[512u + (d8 + 1u) * 16u + col][lane] = a.y;
        SH[512u + (d8 + 2u) * 16u + col][lane] = a.z; SH[512u + (d8 + 3u) * 16u + col][lane] = a.w;
        SH[512u + (d8 + 4u) * 16u + col][lane] = b.x; SH[512u + (d8 + 5u) * 16u + col][lane] = b.y;
        SH[512u + (d8 + 6u) * 16u + col][lane] = b.z; SH[512u + (d8 + 7u) * 16u + col][lane] = b.w;
      }
      {
        // K chunk: key t/4, dims (t%4)*4..+3
        let kr = t / 4u;
        let d4 = (t % 4u) * 4u;
        var a = vec4<f32>();
        if (k0 + kr < P.Lk) { a = K[((k0 + kr) * P.ldk + P.kOff + h * ${D}u + dc * 16u + d4) >> 2u]; }
        let lane = kr & 3u;
        let col = kr >> 2u;
        SH[768u + (d4 + 0u) * 8u + col][lane] = a.x; SH[768u + (d4 + 1u) * 8u + col][lane] = a.y;
        SH[768u + (d4 + 2u) * 8u + col][lane] = a.z; SH[768u + (d4 + 3u) * 8u + col][lane] = a.w;
      }
      workgroupBarrier();
      for (var d = 0u; d < 16u; d++) {
        let qa = SH[512u + d * 16u + sr];
        let kb = SH[768u + d * 8u + sc];
        s0 += qa.x * kb; s1 += qa.y * kb; s2 += qa.z * kb; s3 += qa.w * kb;
      }
      workgroupBarrier();
    }

    // ---- phase 2: online softmax
    let kmask = vec4<f32>(select(vec4<f32>(0.0), vec4<f32>(-1e30),
      vec4<u32>(k0 + sc * 4u) + vec4<u32>(0u, 1u, 2u, 3u) >= vec4<u32>(P.Lk)));
    s0 = s0 * P.scale + kmask; s1 = s1 * P.scale + kmask; s2 = s2 * P.scale + kmask; s3 = s3 * P.scale + kmask;
    let pm = vec4<f32>(max(max(s0.x, s0.y), max(s0.z, s0.w)), max(max(s1.x, s1.y), max(s1.z, s1.w)),
                       max(max(s2.x, s2.y), max(s2.z, s2.w)), max(max(s3.x, s3.y), max(s3.z, s3.w)));
    // row-max partials: rows sr*4+i, slot sc  -> float index (sr*4+i)*8 + sc
    ${r4.map((i) => `SH[896u + ((sr * 4u + ${i}u) * 8u + sc) / 4u][sc % 4u] = pm.${"xyzw"[i]};`).join("\n    ")}
    workgroupBarrier();
    var bm = vec4<f32>(-1e30);
    ${r4.map((i) => `{ let a = SH[896u + (sr * 4u + ${i}u) * 2u]; let b = SH[896u + (sr * 4u + ${i}u) * 2u + 1u];
      bm.${"xyzw"[i]} = max(max(max(a.x, a.y), max(a.z, a.w)), max(max(b.x, b.y), max(b.z, b.w))); }`).join("\n    ")}
    let mn = max(m, bm);
    let e0 = exp(s0 - mn.x); let e1 = exp(s1 - mn.y); let e2 = exp(s2 - mn.z); let e3 = exp(s3 - mn.w);
    let ps = vec4<f32>(dot(e0, vec4<f32>(1.0)), dot(e1, vec4<f32>(1.0)), dot(e2, vec4<f32>(1.0)), dot(e3, vec4<f32>(1.0)));
    ${r4.map((i) => `SH[768u + ((sr * 4u + ${i}u) * 8u + sc) / 4u][sc % 4u] = ps.${"xyzw"[i]};`).join("\n    ")}
    // probabilities, stored [key][row] so phase 3 reads 4 rows as one vec4
    SH[(sc * 4u + 0u) * 16u + sr] = vec4<f32>(e0.x, e1.x, e2.x, e3.x);
    SH[(sc * 4u + 1u) * 16u + sr] = vec4<f32>(e0.y, e1.y, e2.y, e3.y);
    SH[(sc * 4u + 2u) * 16u + sr] = vec4<f32>(e0.z, e1.z, e2.z, e3.z);
    SH[(sc * 4u + 3u) * 16u + sr] = vec4<f32>(e0.w, e1.w, e2.w, e3.w);
    workgroupBarrier();
    var bs = vec4<f32>(0.0);
    ${r4.map((i) => `{ let a = SH[768u + (sr * 4u + ${i}u) * 2u]; let b = SH[768u + (sr * 4u + ${i}u) * 2u + 1u];
      bs.${"xyzw"[i]} = dot(a, vec4<f32>(1.0)) + dot(b, vec4<f32>(1.0)); }`).join("\n    ")}
    let alpha = exp(m - mn);
    l = l * alpha + bs;
    m = mn;
    ${rescale.join("\n    ")}
    workgroupBarrier(); // partials consumed before the V chunk overwrites them

    // ---- phase 3: O += P V, V streamed in 64-wide chunks
    ${[...Array(VC).keys()].map((vc) => `{
      let kr = t / 4u;
      let d16 = (t % 4u) * 16u;
      var a = vec4<f32>(); var b = vec4<f32>(); var c = vec4<f32>(); var e = vec4<f32>();
      if (k0 + kr < P.Lk) {
        let gi = ((k0 + kr) * P.ldv + P.vOff + h * ${D}u + ${vc * 64}u + d16) >> 2u;
        a = V[gi]; b = V[gi + 1u]; c = V[gi + 2u]; e = V[gi + 3u];
      }
      let base = 512u + kr * 16u + d16 / 4u;
      SH[base] = a; SH[base + 1u] = b; SH[base + 2u] = c; SH[base + 3u] = e;
      workgroupBarrier();
      for (var k = 0u; k < 32u; k++) {
        let pv = SH[k * 16u + sr];
        let v0 = SH[512u + k * 16u + sc];
        let v1 = SH[512u + k * 16u + 8u + sc];
        ${pvFma(vc)}
      }
      workgroupBarrier();
    }`).join("\n    ")}
  }

  ${store.join("\n  ")}
}`;
});

// Elementwise ops over n elements.
export const elementwiseShader = (op) => memo("ew" + op, () => {
  const body = {
    add: "Y[i] = A[i] + B[i];",
    mul: "Y[i] = A[i] * B[i];",
    silu: "Y[i] = silu(A[i]);",
    copy: "Y[i] = A[i];",
  }[op];
  // "auto" layouts drop unreferenced bindings, so B is only declared for binary ops
  const binary = op === "add" || op === "mul";
  return /* wgsl */ `
struct Params { n: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> A: array<f32>;
${binary ? "@group(0) @binding(2) var<storage, read> B: array<f32>;" : ""}
@group(0) @binding(${binary ? 3 : 2}) var<storage, read_write> Y: array<f32>;
${GELU}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i >= P.n) { return; }
  ${body}
}`;
});

export const zeroShader = () => memo("zero", () => /* wgsl */ `
struct Params { n: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i < P.n) { Y[i] = 0.0; }
}`);
