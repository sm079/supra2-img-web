"""Converts SupraLabs' Supra2-IMG checkpoint (model_final_ema.pt) to safetensors for the web app.

  python tools/convert_dit.py            ->  models/supra2-img-ema.safetensors

The .pt is read with torch.load(weights_only=True), PyTorch's restricted unpickler: it only
rebuilds tensors and plain containers and never runs code from the file. The output keeps the
EMA weights under their original names (float32, as trained) and adds the stored unconditional
text embedding as "uncond_text" [L, 768] with only the unmasked rows kept (the DiT ignores masked
context rows). Plain config values go into the safetensors metadata.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

REPO, REV = "SupraLabs/Supra2-IMG", "b22ffe6c85983a63a535ff9bfd40caacc428dddf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/supra2-img-ema.safetensors")
    args = ap.parse_args()
    path = hf_hub_download(REPO, "model_final_ema.pt", revision=REV)
    state = torch.load(path, map_location="cpu", weights_only=True)
    cfg = state.get("config", {}) if isinstance(state, dict) else {}
    sd = state["ema"] if "ema" in state else state.get("model", state)
    print(f"checkpoint keys: {list(state.keys()) if isinstance(state, dict) else type(state)}")
    print(f"config: { {k: (tuple(v.shape), str(v.dtype)) if torch.is_tensor(v) else v for k, v in cfg.items()} }")

    out = {k: v.detach().to(torch.float32).contiguous() for k, v in sd.items()}
    n = sum(v.numel() for v in out.values())
    if "uncond_text" in cfg:
        text = cfg["uncond_text"].float()
        mask = cfg.get("uncond_mask", torch.ones(text.shape[0])).float()
        out["uncond_text"] = text[mask > 0].contiguous()
        print(f"uncond_text: {tuple(text.shape)} -> {tuple(out['uncond_text'].shape)} unmasked rows")
    meta = {"source": f"{REPO}@{REV}/model_final_ema.pt", "format": "supra2-img-ema"}
    plain = {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool)) or v is None}
    meta["config"] = json.dumps(plain)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_file(out, args.out, metadata=meta)
    print(f"wrote {args.out}: {len(sd)} tensors, {n / 1e6:.1f}M parameters, {os.path.getsize(args.out)} bytes")


if __name__ == "__main__":
    main()
