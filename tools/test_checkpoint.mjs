// Checks app/checkpoint.js against a synthetic checkpoint (tools/make_test_checkpoint.py).
//   python tools/make_test_checkpoint.py out && node tools/test_checkpoint.mjs out
import { readFile } from "node:fs/promises";
import { TorchCheckpoint } from "../app/weights.js";

const dir = process.argv[2] || "out";
const blob = new Blob([await readFile(`${dir}/synthetic.pt`)]);
const ref = JSON.parse(await readFile(`${dir}/synthetic.json`, "utf8"));
const ck = await TorchCheckpoint.open(blob, (o) => o.ema);
let fail = 0;
const check = (name, ok, extra = "") => { console.log(`${ok ? "PASS" : "FAIL"} ${name}${extra ? " " + extra : ""}`); if (!ok) fail++; };

for (const [k, shape] of Object.entries(ref._shapes)) {
  check(`${k} shape`, JSON.stringify(ck.info(k).shape) === JSON.stringify(shape), JSON.stringify(ck.info(k).shape));
}
const close = (a, b) => a.length >= b.length && b.every((v, i) => Math.abs(a[i] - v) <= 1e-6 * Math.max(1, Math.abs(v)));
for (const k of Object.keys(ref._shapes)) check(`${k} values`, close(await ck.f32(k), ref[k]));
const cfg = ck.obj.config;
for (const k of ["uncond_text", "uncond_mask", "bf", "half", "bool"]) check(`config.${k}`, close(await ck.ck.tensorF32(cfg[k]), ref[k]));
check("plain values", cfg.patch === 2 && cfg.ctx_len === 128 && cfg.lr === 1e-4 && cfg.name === "supra" && cfg.big === 2 ** 40 && cfg.neg === -5 && ck.obj.step === 12345,
  JSON.stringify({ patch: cfg.patch, ctx: cfg.ctx_len, lr: cfg.lr, big: cfg.big, neg: cfg.neg }));
check("tuples/lists", JSON.stringify(cfg.betas) === "[0.9,0.99]" && JSON.stringify(cfg.flags) === "[true,false,null]");
check("unknown globals stay inert", cfg.ns?.__global === "argparse.Namespace", JSON.stringify(cfg.ns));
process.exit(fail ? 1 : 0);
