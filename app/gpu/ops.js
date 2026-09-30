// Tensor ops built on the WGSL kernels. Activations are f32 Tensors (row-major).
// Weights are { kind: "f32", N, K, buf, bias? } with the matrix stored [N, K] (PyTorch layout).

import { grid } from "./device.js";
import * as K from "./kernels.js";

const mmParams = (o) => [
  ["u32", o.M], ["u32", o.N], ["u32", o.K], ["f32", o.alpha ?? 1],
  ["u32", o.lda ?? o.K], ["u32", o.aBatch ?? 0], ["u32", o.aOff ?? 0], ["u32", o.ldb ?? o.K],
  ["u32", o.bBatch ?? 0], ["u32", o.bDiv ?? 1], ["u32", o.bOff ?? 0], ["u32", o.ldc ?? o.N],
  ["u32", o.cBatch ?? 0], ["u32", o.cOff ?? 0], ["u32", o.gOff ?? 0], ["u32", o.cin ?? 0],
  ["u32", o.ch ?? 0], ["u32", o.cw ?? 0], ["u32", o.up ? 1 : 0], ["u32", 0],
];

function runMatmul(gpu, { a = "rows", A, W, C, bias, act = "none", resid = false, gate, batch = 1, name, ...o }) {
  const b = W.kind;
  // big 128x128 tiles unless the problem is too small to fill them
  const R = o.M >= 256 && o.N >= 128 ? 8 : 4;
  const T = 16 * R;
  // vec4 loads need whole 8-wide k chunks and 4-aligned offsets/strides
  const al = (...xs) => xs.every((x) => (x ?? 0) % 4 === 0);
  const k8 = o.K % 8 === 0;
  const vecA = k8 && (a === "conv3" ? o.cin % 8 === 0 : al(o.lda ?? o.K, o.aOff, o.aBatch));
  const vecB = k8 && b === "f32" && al(o.ldb ?? o.K, o.bOff, o.bBatch);
  const code = K.matmulShader({ a, b, bias: !!bias, act, resid, gate: !!gate, R, vecA, vecB });
  const bufs = [A, W.buf, C];
  if (bias) bufs.push(bias);
  if (gate) bufs.push(gate);
  const meta = { name: name || (a === "conv3" ? "conv3x3" : "gemm"), flops: 2 * o.M * o.N * o.K * batch };
  gpu.dispatch(code, bufs, mmParams(o), [Math.ceil(o.N / T), Math.ceil(o.M / T), batch], meta);
}

// y = x @ W^T (+ bias) (act). opts.out + opts.resid: out += (gate[gOff+n] *) y, in place.
// opts.rows / opts.aOff: use `rows` rows of x starting at element offset aOff.
export function linear(gpu, x, W, opts = {}) {
  const M = opts.rows ?? x.size / W.K;
  const out = opts.out || gpu.empty([M, W.N]);
  runMatmul(gpu, {
    A: x, W, C: out, M, N: W.N, K: W.K, aOff: opts.aOff,
    bias: opts.bias || W.bias, act: opts.act, resid: !!opts.resid, gate: opts.gate, gOff: opts.gOff, name: opts.name,
  });
  return out;
}

export function zero(gpu, t) {
  const n = t.size;
  const [nx, ny] = grid(Math.ceil(n / 256));
  gpu.dispatch(K.zeroShader(), [t], [["u32", n], ["u32", nx]], [nx, ny], { name: "zero" });
}

// 3x3 conv (pad 1) on an NHWC image [h*w, cin] -> [H*W, cout], weights laid out [cout, 3, 3, cin]
// (cin % 8 == 0); up=true fuses a 2x nearest upsample of the input (output is then 2h x 2w).
export function conv3x3(gpu, x, W, h, w, up = false) {
  const H = up ? h * 2 : h;
  const Wd = up ? w * 2 : w;
  const cin = W.K / 9;
  const out = gpu.empty([H * Wd, W.N]);
  runMatmul(gpu, { a: "conv3", A: x, W, C: out, M: H * Wd, N: W.N, K: W.K, cin, ch: H, cw: Wd, up, bias: W.bias, resid: false });
  return out;
}

export function rmsnorm(gpu, x, weight, cols, eps = 1e-6) {
  const rows = x.size / cols;
  const y = gpu.empty(x.shape);
  const [nx, ny] = grid(rows);
  gpu.dispatch(K.rmsnormShader(), [x, weight, y], [["u32", rows], ["u32", cols], ["u32", nx], ["f32", eps]], [nx, ny], { name: "rmsnorm" });
  return y;
}

// LayerNorm without affine (eps 1e-6); with `mod`, adaLN: LN(x) * (1 + MOD[scaleOff..]) + MOD[shiftOff..].
export function layernorm(gpu, x, cols, mod = null, shiftOff = 0, scaleOff = 0) {
  const rows = x.size / cols;
  const y = gpu.empty(x.shape);
  const [nx, ny] = grid(rows);
  gpu.dispatch(K.layernormShader(!!mod), mod ? [x, mod, y] : [x, y],
    [["u32", rows], ["u32", cols], ["u32", shiftOff], ["u32", scaleOff], ["u32", nx], ["f32", 1e-6]], [nx, ny],
    { name: mod ? "layernorm_mod" : "layernorm" });
  return y;
}

// GroupNorm over an NHWC image [pixels, C] (+ SiLU).
export function groupnorm(gpu, x, gamma, beta, C, groups, silu, eps = 1e-6) {
  const pixels = x.size / C;
  const cg = C / groups;
  const stats = gpu.empty([groups, 2]);
  gpu.dispatch(K.groupnormStatsShader(), [x, stats], [["u32", pixels], ["u32", C], ["u32", cg], ["f32", eps]], [groups], { name: "groupnorm.stats" });
  const y = gpu.empty(x.shape);
  const n = x.size;
  const [nx, ny] = grid(Math.ceil(n / 256));
  gpu.dispatch(K.groupnormApplyShader(silu), [x, stats, gamma, beta, y], [["u32", n], ["u32", C], ["u32", cg], ["u32", nx]], [nx, ny], { name: "groupnorm.apply" });
  stats.release();
  return y;
}

export function elementwise(gpu, op, a, b, out) {
  const y = out || gpu.empty(a.shape);
  const n = y.size;
  const [nx, ny] = grid(Math.ceil(n / 256));
  gpu.dispatch(K.elementwiseShader(op), b ? [a, b, y] : [a, y], [["u32", n], ["u32", nx]], [nx, ny], { name: "elementwise." + op });
  return y;
}

// Multi-head attention.
//  q: [Lq, ldq] with head h at column qOff + h*D; k, v: [Lk, ldk]/[Lk, ldv] with head h at
//  kOff/vOff + h*D. Output rows [Lq, H*D] written at element offset oOff of `out`.
//  scale defaults to 1/sqrt(D). bias: additive score bias [H, Lq, Lk] (T5 relative positions).
// Without a bias and with D in {64, 128}, the fused kernel runs (no score matrix in memory);
// otherwise scores are materialized per chunk of (heads x query rows).
export function attention(gpu, { q, k, v, Lq, Lk, H, D, ldq, ldk, ldv, qOff = 0, kOff = 0, vOff = 0, oOff = 0, scale = 1 / Math.sqrt(D), bias = null, out }) {
  const o = out || gpu.empty([Lq, H * D]);
  const aligned = [ldq, ldk, ldv, qOff, kOff, vOff, oOff].every((x) => x % 4 === 0);
  if (!bias && (D === 64 || D === 128) && aligned) {
    gpu.dispatch(K.flashAttentionShader(D), [q, k, v, o],
      [["u32", Lq], ["u32", Lk], ["u32", ldq], ["u32", ldk], ["u32", ldv], ["u32", H * D],
        ["u32", qOff], ["u32", kOff], ["u32", vOff], ["u32", oOff], ["f32", scale]],
      [Math.ceil(Lq / 64), H], { name: "attn.flash", flops: 4 * Lq * Lk * D * H });
    return o;
  }
  const budget = Math.min(gpu.maxBinding, 256 * 2 ** 20) / 4; // floats per score chunk
  const rows = Math.min(Lq, Math.max(64, Math.floor(budget / Lk / 64) * 64));
  const heads = Math.max(1, Math.min(H, Math.floor(budget / (rows * Lk))));
  if (bias && (rows < Lq || heads < H)) throw new Error("biased attention must fit in one chunk");
  for (let h0 = 0; h0 < H; h0 += heads) {
    const nh = Math.min(heads, H - h0);
    for (let r0 = 0; r0 < Lq; r0 += rows) {
      const nr = Math.min(rows, Lq - r0);
      const S = gpu.empty([nh, nr, Lk]);
      // S = Q K^T  (batched over heads)
      runMatmul(gpu, {
        A: q, W: { kind: "f32", buf: k.buf }, C: S, batch: nh, M: nr, N: Lk, K: D,
        lda: ldq, aBatch: D, aOff: r0 * ldq + qOff + h0 * D,
        ldb: ldk, bBatch: D, bOff: kOff + h0 * D,
        ldc: Lk, cBatch: nr * Lk, name: "attn.qk",
      });
      const [nx, ny] = grid(nh * nr);
      gpu.dispatch(K.softmaxShader(!!bias), bias ? [S, bias] : [S],
        [["u32", nh * nr], ["u32", Lk], ["u32", 0], ["u32", nr], ["u32", nx], ["f32", scale]], [nx, ny], { name: "attn.softmax" });
      // O = P V
      runMatmul(gpu, {
        A: S, W: { kind: "f32t", buf: v.buf }, C: o, batch: nh, M: nr, N: D, K: Lk,
        lda: Lk, aBatch: nr * Lk, aOff: 0,
        ldb: ldv, bBatch: D, bOff: vOff + h0 * D,
        ldc: H * D, cBatch: D, cOff: oOff + r0 * H * D + h0 * D, name: "attn.pv",
      });
      S.release();
    }
  }
  return o;
}
