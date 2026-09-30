// SupraDiT: the ~104M-parameter diffusion transformer of Supra2-IMG (SupraLabs inference.py).
//
// 4x32x32 SD latent -> 2x2 patches -> 256 tokens of width 576; 14 blocks of AdaLN-Zero
// self-attention, cross-attention to projected Flan-T5 states (no modulation) and a tanh-GELU
// MLP; learned absolute position embedding; conditioned only on the timestep. Predicts the
// rectified-flow velocity. Classifier-free guidance runs the conditional and unconditional passes
// as one batch: every linear sees both, attention runs per sample.

import * as ops from "../gpu/ops.js";
import { uploadLinear, uploadVector, linearFromArray } from "../weights.js";

export const CFG = { latentCh: 4, latentSize: 32, patch: 2, dim: 576, depth: 14, heads: 9, headDim: 64, freqDim: 256, ctxDim: 768 };
const TOKENS = (CFG.latentSize / CFG.patch) ** 2; // 256
const PATCH_DIM = CFG.latentCh * CFG.patch * CFG.patch; // 16
const MOD_BLOCK = 6 * CFG.dim; // shift/scale/gate for attention and MLP
const MOD_ALL = CFG.depth * MOD_BLOCK + 2 * CFG.dim; // + final layer shift/scale

// Latent [C, 32, 32] -> tokens [256, C*P*P] (x.view(B,C,h,P,w,P).permute(0,2,4,1,3,5))
export function patchify(z) {
  const { latentCh: C, latentSize: S, patch: P } = CFG;
  const g = S / P;
  const out = new Float32Array(TOKENS * PATCH_DIM);
  for (let i = 0; i < g; i++) {
    for (let j = 0; j < g; j++) {
      for (let c = 0; c < C; c++) {
        for (let p = 0; p < P; p++) {
          for (let q = 0; q < P; q++) out[(i * g + j) * PATCH_DIM + (c * P + p) * P + q] = z[(c * S + i * P + p) * S + j * P + q];
        }
      }
    }
  }
  return out;
}

// Inverse of patchify for rows [256, 16] starting at element `off`.
export function unpatchify(x, off = 0) {
  const { latentCh: C, latentSize: S, patch: P } = CFG;
  const g = S / P;
  const z = new Float32Array(C * S * S);
  for (let i = 0; i < g; i++) {
    for (let j = 0; j < g; j++) {
      for (let c = 0; c < C; c++) {
        for (let p = 0; p < P; p++) {
          for (let q = 0; q < P; q++) z[(c * S + i * P + p) * S + j * P + q] = x[off + (i * g + j) * PATCH_DIM + (c * P + p) * P + q];
        }
      }
    }
  }
  return z;
}

// Sinusoidal timestep features [cos | sin] of t * 1000 (TimestepEmbedder._sinusoidal), float32.
export function timestepFeatures(t) {
  const f = Math.fround;
  const half = CFG.freqDim / 2;
  const out = new Float32Array(CFG.freqDim);
  for (let i = 0; i < half; i++) {
    const freq = f(Math.exp(f(f(f(-Math.log(10000)) * i) / half)));
    const arg = f(f(f(t) * freq) * 1000);
    out[i] = Math.cos(arg);
    out[half + i] = Math.sin(arg);
  }
  return out;
}

export class SupraDiT {
  // ck: SafeTensors with the EMA weights (tools/convert_dit.py)
  static async load(gpu, ck, onProgress) {
    const m = new SupraDiT(gpu);
    const D = CFG.dim;
    m.xEmbed = await uploadLinear(gpu, ck, "x_embed.");
    m.pos = await ck.f32("pos_embed"); // [1, 256, 576]
    m.t1 = await uploadLinear(gpu, ck, "t_embed.mlp.0.");
    m.t2 = await uploadLinear(gpu, ck, "t_embed.mlp.2.");
    m.ctxProj = await uploadLinear(gpu, ck, "ctx_proj.");
    // every adaLN linear stacked into one [MOD_ALL, 576] matrix: one GEMM per sampling run
    const adaW = new Float32Array(MOD_ALL * D);
    const adaB = new Float32Array(MOD_ALL);
    const stack = async (base, row) => {
      const w = await ck.f32(base + "weight");
      adaW.set(w, row * D);
      adaB.set(await ck.f32(base + "bias"), row);
      return w.length / D;
    };
    m.blocks = [];
    for (let i = 0; i < CFG.depth; i++) {
      const p = `blocks.${i}.`;
      await stack(p + "adaln.1.", i * MOD_BLOCK);
      m.blocks.push({
        qkv: await uploadLinear(gpu, ck, p + "self_attn.qkv."),
        proj: await uploadLinear(gpu, ck, p + "self_attn.proj."),
        q: await uploadLinear(gpu, ck, p + "cross_attn.q."),
        kv: await uploadLinear(gpu, ck, p + "cross_attn.kv."),
        cproj: await uploadLinear(gpu, ck, p + "cross_attn.proj."),
        fc1: await uploadLinear(gpu, ck, p + "mlp.0."),
        fc2: await uploadLinear(gpu, ck, p + "mlp.2."),
      });
      onProgress?.((i + 1) / CFG.depth);
    }
    await stack("final.adaln.1.", CFG.depth * MOD_BLOCK);
    m.ada = linearFromArray(gpu, adaW, MOD_ALL, D, adaB);
    m.final = await uploadLinear(gpu, ck, "final.linear.");
    return m;
  }

  constructor(gpu) {
    this.gpu = gpu;
    this.posTiled = new Map(); // batch size -> pos_embed repeated per sample
  }

  posEmbed(B) {
    if (!this.posTiled.has(B)) {
      const t = new Float32Array(B * this.pos.length);
      for (let b = 0; b < B; b++) t.set(this.pos, b * this.pos.length);
      this.posTiled.set(B, { buf: this.gpu.upload(t), size: t.length });
    }
    return this.posTiled.get(B);
  }

  // Timestep conditioning for a whole run: ts -> Tensor [ts.length, MOD_ALL] of adaLN outputs.
  modulations(ts) {
    const gpu = this.gpu;
    const n = ts.length;
    const feats = new Float32Array(n * CFG.freqDim);
    ts.forEach((t, i) => feats.set(timestepFeatures(t), i * CFG.freqDim));
    const f = gpu.fromArray(feats, [n, CFG.freqDim]);
    const h = ops.linear(gpu, f, this.t1, { act: "silu" });
    f.release();
    const c = ops.linear(gpu, h, this.t2, { act: "silu" }); // adaLN starts with SiLU(c)
    h.release();
    const mod = ops.linear(gpu, c, this.ada);
    c.release();
    return mod;
  }

  // Prompt conditioning: T5 states [L, 768] -> per-block cross-attention keys/values [L, 1152].
  prepareContext(ctx) {
    const gpu = this.gpu;
    const L = ctx.shape[0];
    const p = ops.linear(gpu, ctx, this.ctxProj);
    const kv = this.blocks.map((b) => ops.linear(gpu, p, b.kv));
    p.release();
    return { kv, L, release() { for (const t of kv) t.release(); } };
  }

  // One forward pass for B = conds.length samples of the same latent.
  //  tokens: Float32Array [256, 16] (patchified latent); mod: Tensor from modulations();
  //  step: row of `mod` to use. Returns Float32Array [B * 256, 16] (patchified velocity).
  async forward(tokens, conds, mod, step, onBlock) {
    const gpu = this.gpu;
    const { dim: D, heads: H, headDim: HD } = CFG;
    const B = conds.length;
    const R = B * TOKENS;
    const inp = new Float32Array(R * PATCH_DIM);
    for (let b = 0; b < B; b++) inp.set(tokens, b * TOKENS * PATCH_DIM);
    const xin = gpu.fromArray(inp, [R, PATCH_DIM]);
    const e = ops.linear(gpu, xin, this.xEmbed);
    xin.release();
    const x = ops.elementwise(gpu, "add", e, this.posEmbed(B));
    e.release();
    const m0 = step * MOD_ALL;
    for (let i = 0; i < this.blocks.length; i++) {
      const blk = this.blocks[i];
      const o = m0 + i * MOD_BLOCK;
      // self-attention (AdaLN-Zero)
      let h = ops.layernorm(gpu, x, D, mod, o, o + D);
      const qkv = ops.linear(gpu, h, blk.qkv);
      h.release();
      const a = gpu.empty([R, D]);
      for (let b = 0; b < B; b++) {
        const r = b * TOKENS * 3 * D;
        ops.attention(gpu, { q: qkv, k: qkv, v: qkv, Lq: TOKENS, Lk: TOKENS, H, D: HD, ldq: 3 * D, ldk: 3 * D, ldv: 3 * D, qOff: r, kOff: r + D, vOff: r + 2 * D, oOff: b * TOKENS * D, out: a });
      }
      qkv.release();
      ops.linear(gpu, a, blk.proj, { out: x, resid: true, gate: mod, gOff: o + 2 * D });
      a.release();
      // cross-attention to the prompt (plain LayerNorm, no gate)
      h = ops.layernorm(gpu, x, D);
      const q = ops.linear(gpu, h, blk.q);
      h.release();
      const c = gpu.empty([R, D]);
      for (let b = 0; b < B; b++) {
        const kv = conds[b].kv[i];
        ops.attention(gpu, { q, k: kv, v: kv, Lq: TOKENS, Lk: conds[b].L, H, D: HD, ldq: D, ldk: 2 * D, ldv: 2 * D, qOff: b * TOKENS * D, kOff: 0, vOff: D, oOff: b * TOKENS * D, out: c });
      }
      q.release();
      ops.linear(gpu, c, blk.cproj, { out: x, resid: true });
      c.release();
      // MLP (AdaLN-Zero)
      h = ops.layernorm(gpu, x, D, mod, o + 3 * D, o + 4 * D);
      const f = ops.linear(gpu, h, blk.fc1, { act: "gelu_tanh" });
      h.release();
      ops.linear(gpu, f, blk.fc2, { out: x, resid: true, gate: mod, gOff: o + 5 * D });
      f.release();
      await onBlock?.(i);
    }
    const o = m0 + CFG.depth * MOD_BLOCK;
    const h = ops.layernorm(gpu, x, D, mod, o, o + D);
    x.release();
    const out = ops.linear(gpu, h, this.final);
    h.release();
    const v = await gpu.read(out);
    out.release();
    return v;
  }
}
