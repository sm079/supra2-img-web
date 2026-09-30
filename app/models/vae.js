// Stable Diffusion VAE decoder (stabilityai/sd-vae-ft-mse, diffusers AutoencoderKL), NHWC on the
// GPU. 4-channel latent at 1/8 resolution -> RGB.
//
// conv_in -> mid (resnet, single-head attention, resnet) -> 4 up blocks of 3 resnets (512, 512,
// 256, 128 channels; the first three end in a 2x nearest upsample + conv) -> GroupNorm, SiLU,
// conv_out. Every norm is GroupNorm(32 groups, eps 1e-6).

import * as ops from "../gpu/ops.js";
import { uploadLinear, uploadVector, uploadConv3 } from "../weights.js";

export const VAE_SCALE = 0.18215;
const GROUPS = 32;
const CIN_PAD = 8; // conv_in's 4 latent channels padded for the implicit-GEMM conv

export class VAEDecoder {
  static async load(gpu, st) {
    const v = new VAEDecoder(gpu);
    const norm = async (p) => ({ g: await uploadVector(gpu, st, p + ".weight"), b: await uploadVector(gpu, st, p + ".bias") });
    const res = async (p) => ({
      n1: await norm(p + ".norm1"),
      c1: await uploadConv3(gpu, st, p + ".conv1."),
      n2: await norm(p + ".norm2"),
      c2: await uploadConv3(gpu, st, p + ".conv2."),
      sc: st.has(p + ".conv_shortcut.weight") ? await uploadLinear(gpu, st, p + ".conv_shortcut.") : null, // 1x1 conv
    });
    // post_quant_conv (1x1, 4 -> 4) runs on the CPU
    v.pqW = await st.f32("post_quant_conv.weight");
    v.pqB = await st.f32("post_quant_conv.bias");
    v.convIn = await uploadConv3(gpu, st, "decoder.conv_in.", CIN_PAD);
    v.mid0 = await res("decoder.mid_block.resnets.0");
    const a = "decoder.mid_block.attentions.0.";
    // older (query/key/value/proj_attn) and newer (to_q/to_k/to_v/to_out.0) diffusers names
    const n = st.has(a + "query.weight") ? { q: "query.", k: "key.", v: "value.", o: "proj_attn." } : { q: "to_q.", k: "to_k.", v: "to_v.", o: "to_out.0." };
    v.attn = {
      norm: await norm(a + "group_norm"),
      q: await uploadLinear(gpu, st, a + n.q),
      k: await uploadLinear(gpu, st, a + n.k),
      v: await uploadLinear(gpu, st, a + n.v),
      o: await uploadLinear(gpu, st, a + n.o),
    };
    v.mid1 = await res("decoder.mid_block.resnets.1");
    v.ups = [];
    for (let i = 0; i < 4; i++) {
      const p = `decoder.up_blocks.${i}`;
      const resnets = [];
      for (let j = 0; st.has(`${p}.resnets.${j}.conv1.weight`); j++) resnets.push(await res(`${p}.resnets.${j}`));
      const up = st.has(`${p}.upsamplers.0.conv.weight`) ? await uploadConv3(gpu, st, `${p}.upsamplers.0.conv.`) : null;
      v.ups.push({ resnets, up });
    }
    v.normOut = await norm("decoder.conv_norm_out");
    v.convOut = await uploadConv3(gpu, st, "decoder.conv_out.");
    return v;
  }

  constructor(gpu) {
    this.gpu = gpu;
  }

  resblock(x, r, h, w) {
    const gpu = this.gpu;
    const cin = x.shape[1];
    let t = ops.groupnorm(gpu, x, r.n1.g, r.n1.b, cin, GROUPS, true);
    const a = ops.conv3x3(gpu, t, r.c1, h, w);
    t.release();
    t = ops.groupnorm(gpu, a, r.n2.g, r.n2.b, r.c1.N, GROUPS, true);
    a.release();
    const b = ops.conv3x3(gpu, t, r.c2, h, w);
    t.release();
    const skip = r.sc ? ops.linear(gpu, x, r.sc) : x;
    const out = ops.elementwise(gpu, "add", skip, b);
    b.release();
    if (skip !== x) skip.release();
    x.release();
    return out;
  }

  attention(x, hw) {
    const gpu = this.gpu;
    const C = x.shape[1];
    const A = this.attn;
    const n = ops.groupnorm(gpu, x, A.norm.g, A.norm.b, C, GROUPS, false);
    const q = ops.linear(gpu, n, A.q);
    const k = ops.linear(gpu, n, A.k);
    const v = ops.linear(gpu, n, A.v);
    n.release();
    const a = ops.attention(gpu, { q, k, v, Lq: hw, Lk: hw, H: 1, D: C, ldq: C, ldk: C, ldv: C });
    q.release(); k.release(); v.release();
    const o = ops.linear(gpu, a, A.o);
    a.release();
    const out = ops.elementwise(gpu, "add", x, o);
    o.release();
    x.release();
    return out;
  }

  // latent: Float32Array [4, h, w] as sampled (scaled by VAE_SCALE) -> RGBA Uint8ClampedArray
  async decode(latent, h, w, onStage) {
    const gpu = this.gpu;
    const hw = h * w;
    // z / scale, post_quant_conv, NCHW -> NHWC padded to CIN_PAD channels
    const z = new Float32Array(hw * CIN_PAD);
    for (let i = 0; i < hw; i++) {
      for (let o = 0; o < 4; o++) {
        let s = this.pqB[o];
        for (let c = 0; c < 4; c++) s += this.pqW[o * 4 + c] * (latent[c * hw + i] / VAE_SCALE);
        z[i * CIN_PAD + o] = s;
      }
    }
    const zin = gpu.fromArray(z, [hw, CIN_PAD]);
    let x = ops.conv3x3(gpu, zin, this.convIn, h, w);
    zin.release();
    x = this.resblock(x, this.mid0, h, w);
    x = this.attention(x, hw);
    x = this.resblock(x, this.mid1, h, w);
    let H = h, W = w;
    for (let i = 0; i < this.ups.length; i++) {
      const u = this.ups[i];
      for (const r of u.resnets) x = this.resblock(x, r, H, W);
      if (u.up) {
        const y = ops.conv3x3(gpu, x, u.up, H, W, true);
        x.release();
        x = y;
        H *= 2; W *= 2;
      }
      await gpu.sync();
      onStage?.((i + 1) / this.ups.length);
    }
    const t = ops.groupnorm(gpu, x, this.normOut.g, this.normOut.b, x.shape[1], GROUPS, true);
    x.release();
    const rgb = ops.conv3x3(gpu, t, this.convOut, H, W);
    t.release();
    const px = await gpu.read(rgb);
    rgb.release();
    // torchvision save_image: clamp(-1, 1) -> [0, 1] -> round(x * 255)
    const img = new Uint8ClampedArray(H * W * 4);
    for (let i = 0; i < H * W; i++) {
      for (let c = 0; c < 3; c++) {
        const v = (Math.min(1, Math.max(-1, px[i * 3 + c])) + 1) / 2;
        img[i * 4 + c] = Math.floor(v * 255 + 0.5);
      }
      img[i * 4 + 3] = 255;
    }
    return { data: img, width: W, height: H, pixels: px };
  }
}
