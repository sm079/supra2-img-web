"""Writes a synthetic PyTorch checkpoint shaped like Supra2-IMG's (an "ema" state dict plus a
"config" dict with tensors and plain values) and the expected values, for tools/test_checkpoint.mjs.

  python tools/make_test_checkpoint.py out/   ->  out/synthetic.pt, out/synthetic.json
"""

import argparse
import collections
import json
import os
import sys

import torch

out = sys.argv[1] if len(sys.argv) > 1 else "out"
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)
ema = collections.OrderedDict()
ema["x_embed.weight"] = torch.randn(576, 16)
ema["pos_embed"] = torch.randn(1, 256, 576)
big = torch.randn(64, 48)
ema["noncontig"] = big.t()[:, 5:40]            # strided view into a shared storage
ema["shared_a"] = big[3]                         # storage offset
cfg = {"patch": 2, "ctx_len": 128, "lr": 1e-4, "name": "supra", "betas": (0.9, 0.99), "flags": [True, False, None],
       "uncond_text": torch.randn(128, 768), "uncond_mask": torch.zeros(128, dtype=torch.long), "big": 2**40, "neg": -5,
       "bf": torch.randn(7).to(torch.bfloat16), "half": torch.randn(9).half(), "bool": torch.tensor([True, False, True]),
       "ns": argparse.Namespace(a=1, b="x")}
cfg["uncond_mask"][:1] = 1
torch.save({"ema": ema, "config": cfg, "step": 12345}, os.path.join(out, "synthetic.pt"))
ref = {k: v.contiguous().float().flatten().tolist()[:50] for k, v in list(ema.items()) + [(k, cfg[k]) for k in ["uncond_text", "uncond_mask", "bf", "half", "bool"]]}
ref["_shapes"] = {k: list(v.shape) for k, v in list(ema.items())}
json.dump(ref, open(os.path.join(out, "synthetic.json"), "w"))
