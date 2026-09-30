// Flan-T5-Base encoder (last_hidden_state), fp32.
//
// Pre-norm blocks with T5LayerNorm (RMS, no mean), unscaled attention with a learned relative
// position bias (block 0's table, shared by all blocks) and a gated-GELU feed-forward. Prompts run
// at their true token count: with an attention mask, padding tokens change nothing for the real
// tokens, and the DiT only attends to real tokens.

import * as ops from "../gpu/ops.js";
import { uploadLinear, uploadVector } from "../weights.js";

const CFG = { layers: 12, dim: 768, heads: 12, headDim: 64, buckets: 32, maxDistance: 128 };

// HF T5Attention._relative_position_bucket (bidirectional), in float32 like PyTorch
export function relativeBucket(rel, buckets = CFG.buckets, maxDistance = CFG.maxDistance) {
  const nb = buckets / 2;
  let bucket = rel > 0 ? nb : 0;
  const n = Math.abs(rel);
  const maxExact = nb / 2;
  if (n < maxExact) return bucket + n;
  const f = Math.fround;
  const large = maxExact + Math.trunc(f(f(f(Math.log(f(n / maxExact))) / f(Math.log(maxDistance / maxExact))) * (nb - maxExact)));
  return bucket + Math.min(large, nb - 1);
}

export class T5Encoder {
  static async load(gpu, st, onProgress) {
    const m = new T5Encoder(gpu, st);
    m.layers = [];
    for (let i = 0; i < CFG.layers; i++) {
      const p = `encoder.block.${i}.layer.`;
      m.layers.push({
        ln1: await uploadVector(gpu, st, p + "0.layer_norm.weight"),
        q: await uploadLinear(gpu, st, p + "0.SelfAttention.q."),
        k: await uploadLinear(gpu, st, p + "0.SelfAttention.k."),
        v: await uploadLinear(gpu, st, p + "0.SelfAttention.v."),
        o: await uploadLinear(gpu, st, p + "0.SelfAttention.o."),
        ln2: await uploadVector(gpu, st, p + "1.layer_norm.weight"),
        wi0: await uploadLinear(gpu, st, p + "1.DenseReluDense.wi_0."),
        wi1: await uploadLinear(gpu, st, p + "1.DenseReluDense.wi_1."),
        wo: await uploadLinear(gpu, st, p + "1.DenseReluDense.wo."),
      });
      onProgress?.((i + 1) / CFG.layers);
    }
    m.relBias = await st.f32("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"); // [buckets, heads]
    m.norm = await uploadVector(gpu, st, "encoder.final_layer_norm.weight");
    return m;
  }

  constructor(gpu, st) {
    this.gpu = gpu;
    this.st = st; // embedding rows are gathered from the file on demand
  }

  // [heads, L, L] additive score bias
  positionBias(L) {
    const { heads: H, buckets } = CFG;
    const out = new Float32Array(H * L * L);
    for (let i = 0; i < L; i++) {
      for (let j = 0; j < L; j++) {
        const b = relativeBucket(j - i, buckets);
        for (let h = 0; h < H; h++) out[(h * L + i) * L + j] = this.relBias[b * H + h];
      }
    }
    return out;
  }

  // ids -> Tensor [L, 768]
  async encode(ids) {
    const gpu = this.gpu;
    const { dim: D, heads: H, headDim: HD } = CFG;
    const L = ids.length;
    let x = gpu.fromArray(await this.st.rows("shared.weight", ids), [L, D]);
    const bias = gpu.fromArray(this.positionBias(L), [H, L, L]);
    for (const l of this.layers) {
      const h = ops.rmsnorm(gpu, x, l.ln1, D);
      const q = ops.linear(gpu, h, l.q);
      const k = ops.linear(gpu, h, l.k);
      const v = ops.linear(gpu, h, l.v);
      h.release();
      const a = ops.attention(gpu, { q, k, v, Lq: L, Lk: L, H, D: HD, ldq: D, ldk: D, ldv: D, scale: 1, bias });
      q.release(); k.release(); v.release();
      ops.linear(gpu, a, l.o, { out: x, resid: true });
      a.release();
      const h2 = ops.rmsnorm(gpu, x, l.ln2, D);
      const g = ops.linear(gpu, h2, l.wi0, { act: "gelu_tanh" });
      const u = ops.linear(gpu, h2, l.wi1);
      h2.release();
      const gu = ops.elementwise(gpu, "mul", g, u);
      g.release(); u.release();
      ops.linear(gpu, gu, l.wo, { out: x, resid: true });
      gu.release();
    }
    bias.release();
    const out = ops.rmsnorm(gpu, x, this.norm, D);
    x.release();
    return out;
  }
}
