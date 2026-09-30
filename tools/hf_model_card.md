---
license: other
license_name: apache-2.0-and-mit
license_link: https://huggingface.co/sm079/supra2-img-web/blob/main/LICENSE-Apache-2.0.txt
library_name: safetensors
pipeline_tag: text-to-image
tags:
- webgpu
- supra
base_model:
- SupraLabs/Supra2-IMG
- google/flan-t5-base
- stabilityai/sd-vae-ft-mse
---

# Supra2-Img Web model files

Model files for [Supra2-Img Web](https://github.com/sm079/supra2-img-web), which runs
[SupraLabs' Supra2-IMG](https://huggingface.co/SupraLabs/Supra2-IMG) text-to-image in the browser
on WebGPU. They are **unofficial, modified** repackagings of the original models, made with
[`tools/build_models.py`](https://github.com/sm079/supra2-img-web/blob/main/tools/build_models.py):

| file | contents | source | license |
|---|---|---|---|
| `supra2-img-ema.safetensors` | the Supra2-IMG diffusion transformer (104M parameters, float32), converted from the pickled checkpoint to safetensors; adds the checkpoint's stored unconditional text embedding as `uncond_text` (unmasked row only) | [SupraLabs/Supra2-IMG](https://huggingface.co/SupraLabs/Supra2-IMG) `model_final_ema.pt` @ `b22ffe6c` | Apache-2.0 |
| `flan-t5-base-encoder.safetensors` | the Flan-T5-Base encoder only (decoder tensors removed), float32, unchanged values | [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) `model.safetensors` @ `7bcac572` | Apache-2.0 |
| `sd-vae-ft-mse-decoder.safetensors` | the VAE decoder and `post_quant_conv` only (encoder removed), float32, unchanged values | [stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse) @ `31f26fde` | MIT |
| `tokenizer.json`, `tokenizer_config.json` | the Flan-T5 tokenizer, unchanged | google/flan-t5-base @ `7bcac572` | Apache-2.0 |

The weights keep their original licenses: Supra2-IMG and Flan-T5-Base are licensed under the
Apache License 2.0 (copy in `LICENSE-Apache-2.0.txt`); sd-vae-ft-mse is licensed under the MIT
license by Stability AI. All credit for the models goes to their authors: SupraLabs, Google and
Stability AI. This repository is not affiliated with them.
