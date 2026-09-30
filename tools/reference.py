"""PyTorch reference outputs for tools/check.html.

Runs the same models the web app runs, on the CPU in float32, and dumps inputs and outputs as raw
little-endian float32 files plus an index.json:

  python tools/reference.py --out out/dump                # text encoder, VAE, noise
  python tools/reference.py --out out/dump --dit          # + a full Supra2-IMG sample

- Text encoder: Hugging Face transformers T5EncoderModel (google/flan-t5-base) and its tokenizer.
- VAE: diffusers AutoencoderKL (stabilityai/sd-vae-ft-mse) decoding a fixed latent.
- Noise: torch.randn on the CPU generator, which app/rng.js reproduces.
- DiT (--dit): an independent implementation of the architecture in SupraLabs' inference.py,
  reading the converted weights from tools/convert_dit.py. It samples one image with the app's
  defaults (unconditional states from the stored embedding, as inference.py does).

Files come from the Hugging Face cache (downloaded on first use), pinned to the same commits as
app/store.js.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

T5_REPO, T5_REV = "google/flan-t5-base", "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
VAE_REPO, VAE_REV = "stabilityai/sd-vae-ft-mse", "31f26fdeee1355a5c34592e401dd41e45d25a493"
PROMPTS = [
    "a sea jellyfish floating in the pitch-black ocean depths",
    "A cozy cabin in a snowy forest at night, warm light in the windows, (aurora) in the sky!",
    "",
]
VAE_SCALE = 0.18215


class Dump:
    def __init__(self, out):
        self.out = out
        self.index = {}
        os.makedirs(out, exist_ok=True)

    def __call__(self, name, t, **meta):
        a = t.detach().float().contiguous().cpu().numpy() if torch.is_tensor(t) else np.asarray(t, dtype=np.float32)
        a.astype("<f4").tofile(os.path.join(self.out, name + ".bin"))
        self.index[name] = {"shape": list(a.shape), **meta}

    def save(self, **extra):
        with open(os.path.join(self.out, "index.json"), "w") as f:
            json.dump({**extra, "tensors": self.index}, f, indent=1)


# --------------------------------------------------------------------------- SupraDiT (independent)


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


def layernorm(x):
    return F.layer_norm(x, (x.shape[-1],), eps=1e-6)


def linear(sd, p, x):
    return F.linear(x, sd[p + ".weight"], sd.get(p + ".bias"))


def attention(q, k, v, heads, mask=None):
    B, N, C = q.shape
    M = k.shape[1]
    q = q.view(B, N, heads, -1).transpose(1, 2)
    k = k.view(B, M, heads, -1).transpose(1, 2)
    v = v.view(B, M, heads, -1).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=None if mask is None else mask.bool()[:, None, None, :])
    return o.transpose(1, 2).reshape(B, N, C)


def dit_forward(sd, z, t, ctx, mask, depth=14, heads=9, dim=576, P=2):
    B, C, H, W = z.shape
    h, w = H // P, W // P
    x = z.view(B, C, h, P, w, P).permute(0, 2, 4, 1, 3, 5).reshape(B, h * w, C * P * P)
    x = linear(sd, "x_embed", x) + sd["pos_embed"]
    half = 128
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half) / half)
    args = t[:, None].float() * freqs[None] * 1000.0
    c = torch.cat([torch.cos(args), torch.sin(args)], -1)
    c = linear(sd, "t_embed.mlp.2", F.silu(linear(sd, "t_embed.mlp.0", c)))
    ctx = linear(sd, "ctx_proj", ctx)
    for i in range(depth):
        p = f"blocks.{i}."
        s1, c1, g1, s2, c2, g2 = linear(sd, p + "adaln.1", F.silu(c)).chunk(6, 1)
        qkv = linear(sd, p + "self_attn.qkv", modulate(layernorm(x), s1, c1)).view(B, -1, 3, dim)
        x = x + g1[:, None] * linear(sd, p + "self_attn.proj", attention(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], heads))
        q = linear(sd, p + "cross_attn.q", layernorm(x))
        kv = linear(sd, p + "cross_attn.kv", ctx).view(B, -1, 2, dim)
        x = x + linear(sd, p + "cross_attn.proj", attention(q, kv[:, :, 0], kv[:, :, 1], heads, mask))
        m = linear(sd, p + "mlp.2", F.gelu(linear(sd, p + "mlp.0", modulate(layernorm(x), s2, c2)), approximate="tanh"))
        x = x + g2[:, None] * m
    shift, scale = linear(sd, "final.adaln.1", F.silu(c)).chunk(2, 1)
    x = linear(sd, "final.linear", modulate(layernorm(x), shift, scale))
    return x.view(B, h, w, C, P, P).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)


# --------------------------------------------------------------------------- main


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/dump")
    ap.add_argument("--dit", action="store_true", help="also sample an image with the diffusion model")
    ap.add_argument("--dit-file", default="models/supra2-img-ema.safetensors", help="converted model (tools/convert_dit.py)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--vae-seed", type=int, default=1234)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    dump = Dump(args.out)

    from transformers import AutoTokenizer, T5EncoderModel
    from diffusers import AutoencoderKL

    tok = AutoTokenizer.from_pretrained(T5_REPO, revision=T5_REV)
    te = T5EncoderModel.from_pretrained(T5_REPO, revision=T5_REV, torch_dtype=torch.float32).eval()
    prompts = []
    for i, p in enumerate(PROMPTS):
        ids = tok(p, truncation=True, max_length=128)["input_ids"]
        hidden = te(input_ids=torch.tensor([ids])).last_hidden_state[0]
        dump(f"te_{i}", hidden)
        prompts.append({"text": p, "ids": ids})
    print("text encoder:", [len(p["ids"]) for p in prompts], "tokens")

    # noise: torch.randn(1, 4, 32, 32) after torch.manual_seed(seed), as inference.py on a CPU
    torch.manual_seed(args.seed)
    dump("noise", torch.randn(1, 4, 32, 32)[0], seed=args.seed)

    vae = AutoencoderKL.from_pretrained(VAE_REPO, revision=VAE_REV, torch_dtype=torch.float32).eval()
    g = torch.Generator().manual_seed(args.vae_seed)
    lat = torch.randn(1, 4, 32, 32, generator=g) * 0.8  # a plausible scaled latent
    dump("vae_latent", lat[0])
    img = vae.decode(lat / VAE_SCALE).sample[0]  # [3, 256, 256] in [-1, 1] (unclamped)
    dump("vae_image", img.permute(1, 2, 0))  # HWC like the app's decoder output
    print("vae: decoded", tuple(img.shape))

    meta = {"prompts": prompts}
    if args.dit:
        from safetensors.torch import load_file

        sd = load_file(args.dit_file)
        uncond = sd.pop("uncond_text", None)
        prompt = PROMPTS[0]
        ids = prompts[0]["ids"]
        ctx = te(input_ids=torch.tensor([ids])).last_hidden_state
        mask = torch.ones(1, len(ids))
        if uncond is not None:
            uctx = uncond.float()[None]
            umask = torch.ones(1, uctx.shape[1])
        else:
            uid = tok("", truncation=True, max_length=128)["input_ids"]
            uctx = te(input_ids=torch.tensor([uid])).last_hidden_state
            umask = torch.ones(1, len(uid))
        # the app runs the prompt at its true length; pad the two contexts to a common length
        L = max(ctx.shape[1], uctx.shape[1])
        pad = lambda c, m: (F.pad(c, (0, 0, 0, L - c.shape[1])), F.pad(m, (0, L - m.shape[1])))
        ctx, mask = pad(ctx, mask)
        uctx, umask = pad(uctx, umask)
        torch.manual_seed(args.seed)
        z = torch.randn(1, 4, 32, 32)
        dt = 1.0 / args.steps
        for i in range(args.steps):
            t = torch.full((2,), i * dt)
            v = dit_forward(sd, torch.cat([z, z]), t, torch.cat([ctx, uctx]), torch.cat([mask, umask]))
            vc, vu = v.chunk(2)
            if i == 0:
                dump("dit_v0_cond", vc[0])
                dump("dit_v0_uncond", vu[0])
            v = vu + args.cfg * (vc - vu)
            z = z + dt * v
        dump("dit_final_latent", z[0], prompt=prompt, seed=args.seed, steps=args.steps, cfg=args.cfg)
        img = vae.decode(z / VAE_SCALE).sample[0]
        dump("dit_image", img.permute(1, 2, 0))
        meta["dit"] = {"prompt": prompt, "seed": args.seed, "steps": args.steps, "cfg": args.cfg}
        print("dit: sampled", args.steps, "steps")
    dump.save(**meta)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
