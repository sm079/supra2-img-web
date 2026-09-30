"""Builds the model files the web app downloads (models/), from the original Hugging Face repos.

  python tools/build_models.py        ->  models/

| file | from |
|---|---|
| supra2-img-ema.safetensors      | SupraLabs/Supra2-IMG model_final_ema.pt, converted |
| flan-t5-base-encoder.safetensors | google/flan-t5-base model.safetensors, encoder tensors only |
| sd-vae-ft-mse-decoder.safetensors | stabilityai/sd-vae-ft-mse, decoder tensors only |
| tokenizer.json, tokenizer_config.json | google/flan-t5-base |

Supra2-IMG is published as a pickled PyTorch checkpoint. It is read with
torch.load(weights_only=True), PyTorch's restricted unpickler, which only rebuilds tensors and
plain containers and never runs code from the file. The output keeps the EMA weights under their
original names (float32, as trained) and adds the stored unconditional text embedding as
"uncond_text" [L, 768] with only the unmasked rows (the DiT ignores masked context rows). Plain
config values go into the safetensors metadata. All weights stay float32.

Upload the folder (plus tools/hf_model_card.md as README.md) with
  hf upload <user>/<repo> models/ .
and point MODELS_URL in app/main.js at the new commit.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

SUPRA = ("SupraLabs/Supra2-IMG", "b22ffe6c85983a63a535ff9bfd40caacc428dddf")
T5 = ("google/flan-t5-base", "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2")
VAE = ("stabilityai/sd-vae-ft-mse", "31f26fdeee1355a5c34592e401dd41e45d25a493")


def source(repo, file):
    return f"{repo[0]}@{repo[1]}/{file}"


def build_dit(out):
    path = hf_hub_download(SUPRA[0], "model_final_ema.pt", revision=SUPRA[1])
    state = torch.load(path, map_location="cpu", weights_only=True)
    cfg = state.get("config", {}) if isinstance(state, dict) else {}
    sd = state["ema"] if "ema" in state else state.get("model", state)
    tensors = {k: v.detach().to(torch.float32).contiguous() for k, v in sd.items()}
    n = sum(v.numel() for v in tensors.values())
    if "uncond_text" in cfg:
        text = cfg["uncond_text"].float()
        mask = cfg.get("uncond_mask", torch.ones(text.shape[0])).float()
        tensors["uncond_text"] = text[mask > 0].contiguous()
    plain = {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool)) or v is None}
    save_file(tensors, out, metadata={"source": source(SUPRA, "model_final_ema.pt"), "config": json.dumps(plain)})
    print(f"{out}: {len(sd)} tensors, {n / 1e6:.1f}M parameters")


def extract(repo, file, keep, out):
    path = hf_hub_download(repo[0], file, revision=repo[1])
    with safe_open(path, "pt") as f:
        tensors = {k: f.get_tensor(k).contiguous() for k in f.keys() if keep(k)}
    save_file(tensors, out, metadata={"source": source(repo, file)})
    print(f"{out}: {len(tensors)} tensors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    build_dit(os.path.join(args.out, "supra2-img-ema.safetensors"))
    extract(T5, "model.safetensors", lambda k: k.startswith("encoder.") or k == "shared.weight", os.path.join(args.out, "flan-t5-base-encoder.safetensors"))
    extract(VAE, "diffusion_pytorch_model.safetensors", lambda k: k.startswith(("decoder.", "post_quant_conv.")), os.path.join(args.out, "sd-vae-ft-mse-decoder.safetensors"))
    for f in ["tokenizer.json", "tokenizer_config.json"]:
        shutil.copyfile(hf_hub_download(T5[0], f, revision=T5[1]), os.path.join(args.out, f))
    for f in sorted(os.listdir(args.out)):
        print(f"  {f}  {os.path.getsize(os.path.join(args.out, f))}")


if __name__ == "__main__":
    main()
