# Quantization

Which weight formats the checkpoints use, at how many bits, and what that costs in perplexity.

## Formats

* **EXL3** (trellis): the `exl3` conversions of turboderp's exllamav3. `bits` is the average over the tensors; the head, the MTP head and the vision tower have their own widths. Loaded as they are, decoded in the GEMV / GEMM kernels.
* **NVFP4** (modelopt): E2M1 values with an E4M3 scale per 16 and one FP32 scale per matrix. Software-unpacked on Volta (no FP4 hardware). Either read from a modelopt checkpoint (the routed experts), or made at load by the engine's quantizer from BF16 tensors (round to nearest with a +-2 code search per block, or the calibrated tensors from `tests/tools/nvfp4_ldlq.py`). FP8 sources (the Flash-Next MTP experts: E4M3 with 128 x 128 block scales) are dequantized first.

### Models

This document tries its best to continuously reference a model by its quantization, but providers also matter to an extent. Below is a collated table mapping their names to their source:
* [thelastspark/Swift-Qwen3.8-Flash-Next-exl3 (4.00bpw)](https://huggingface.co/thelastspark/Swift-Qwen3.8-Flash-Next-exl3)
* [turboderp/GLM-5.3-Flash-exl3 (3.05bpw)](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3)
* [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) (self-quantized as `*-exl3-calibrated-v2`)
* [Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3 (5.0bpw)](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw)
* [turboderp/Qwen3-VL-32B-Instruct-exl3 (6bpw)](https://huggingface.co/turboderp/Qwen3-VL-32B-Instruct-exl3)
* [QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4)
* [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
* [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)

## Volta tile order (EXL3, build option)

A 16x16 weight tile is stored in the order of the Ampere `m16n8k16` fragments, so the Volta GEMM moves each lane's decoded values to the lanes that need them with shuffles. A build with `-DENGINE_EXL3_VOLTA_LAYOUT=ON` (CMake option `ENGINE_EXL3_VOLTA_LAYOUT`, off by default) instead reads tiles in an order where the eight values a lane decodes are exactly its `m8n8k4` operands, and the GEMM kernels and `reconstruct` use them without moving anything (two octets share each output block and their partial sums are added once per column).

* Such a build only runs checkpoints written in that order. The GEMM kernels, the tensor-core and dp4a int8 GEMVs, `reconstruct`, the fused MoE decode tile (tensor-core and int8) and the persistent MoE kernel read it; the CPU and ROCm paths are not converted.
* `tests/tools/exl3_volta_layout.py <in> <out>` re-encodes a checkpoint (exllamav3 venv; `--claim` runs one process per GPU, `--max-layers N` cuts the model to its first N layers). Half-integer rates go through the fractional-rate quantizer of upstream exllamav3, built from its sources on first use (`tests/tools/exl3_frac_ext.cu`); the quantizer runs at about 16 thousand tiles per second per GPU at 2.5 bpw, so a whole GLM takes hours. The order is a tile permutation applied before the trellis encoder, which does not depend on it; re-encoding already quantized weights adds about one quantization's worth of error, so a conversion from the BF16 weights is the way to produce a real checkpoint.
* `tests/tools/exl3_quantize.py --src <bf16 dir> --template <exl3 dir> --out <dir> [--order volta|stock] [--hess map.json]` quantizes BF16 weights with upstream exllamav3's quantizer (regularization, global scale search, block-LDL error feedback, trellis encoder; integer and half-integer rates) in either tile order, so a Volta-order checkpoint made this way carries no re-encode error. Upstream's Python is loaded from `EXLLAMAV3_UPSTREAM_DIR` and the encoder it calls is built on first use from its sources (`tests/tools/exl3_quant_ext.cpp`, sm_70); only the tile permutation is replaced. The template is an existing EXL3 directory of the same model: it names the tensors to quantize, their rates and codebooks, and supplies everything that stays as it is. `--hess` maps tensor names (the Hessians come from `test_ppl --calibrate` with `ENGINE_HESSIAN_DIR`, or from a served run; sites per layer: 0 attention input, 1 MLP input, 2 o_proj / out_proj input, 3 down_proj input, (0, 4) the LM head input) (or `prefix*`) to Hessian files of the `ENGINE_HESSIAN_DIR` format; a tensor without one is quantized without calibration. `tests/tools/exl3_dequant_check.py` decodes a checkpoint in plain torch and prints the relative error against the BF16 weights.
* `test_trellis_layout` checks the dequantizer against a golden decode (`tests/tools/trellis_layout_golden.py --layout stock|volta`); `test_prefill_linear` compares the fused GEMM and the GEMV with reconstruct plus cuBLAS from 1 to 64 rows; `test_moe` with `MOE_SYNTH=1` runs the MoE kernels on random experts of any widths in either order. `ENGINE_DEBUG_GLM_LAYERS=N` cuts the GLM to its first N layers, so a short `--max-layers N` conversion can be compared with the original through `diag_logits_dump` and `compare_logits.py`.

## Bit widths of the EXL3 checkpoints

| Checkpoint | Average bits | Head | MTP head | Vision | Notes |
|---|---|---|---|---|---|
| Swift-Qwen3.8-Flash-Next-exl3-4.00bpw | 4.0 | 6 | 4 | 6 | attention and GDN projections 7 / 8 bit, routed and shared experts 3 / 4 bit (layers 5, 8, 45 and 46 are 3-bit); router, `shared_expert_gate` and the GDN `in_proj_a/b` FP16 |
| GLM-5.3-Flash-exl3-3.05bpw | 3.05 | 6 | | 6 | |
| Qwen3.8-27B-DFlash2-EXL3-5.0bpw (drafter) | 5.0 | 6 | | | |
| Qwen3-VL-32B-Instruct-exl3-6bpw (MiniMax-H3 text tower) | 6.0 | 6 | | | `mcg` codebook, the others `mul1`; each conversion calibrated on 250 rows of 2048 tokens |

The NVFP4 checkpoints: `Qwen3.8-Flash-Next-NVFP4` is modelopt `MIXED_PRECISION`: routed experts NVFP4 (group 16), the MTP head's experts FP8 E4M3 (128 x 128 blocks), the n-gram table FP8, everything else BF16. `GLM-5.3-Flash-NVFP4` is the same idea for GLM (experts NVFP4, the rest BF16); its layer 45, the MTP head, is all BF16 (experts included) and is quantized to NVFP4 at load when `speculator=mtp`; the EXL3 conversion has no head. The engine uses the NVFP4 experts as they are and quantizes the dense tensors it is told to (`nvfp4_dense`) from the BF16 ones.

## Perplexity

Perplexity values are reported in their mean NLL forms.

### Flash-Next

`test_ppl --flash --qbench <checkpoint>/qbench_prompts.md`: nine traced prompts of the converter's qbench set, 20,487 scored response tokens. Lower is better; the model card gives 1.3825 for BF16.

| Configuration | Perplexity | Over BF16 |
|---|---|---|
| BF16 (model card) | 1.3825 | |
| EXL3 4.00 bpw | 1.4247 | +3.1% |
| EXL3 4.00 bpw, routed experts from the NVFP4 checkpoint | 1.4282 | +3.3% |
| + recommended dense groups in NVFP4 (`gdn,attn,idx,head,shared_gate,shared_up`, calibrated) | 1.4530 | +5.1% |
| + all dense groups in NVFP4 (calibrated) | 1.4815 | +7.2% |
| the NVFP4 checkpoint on its own (`configs/flash_next_nvfp4_only.json`: all dense groups calibrated, FP8 n-gram table) | 1.4813 | +7.1% |

Per group, on top of the NVFP4 experts, round to nearest against calibrated: `gdn` 1.4394 / 1.4382, `attn` 1.4714 / 1.4341, `head` 1.4434 / 1.4348, `shared` 1.4479, `idx` 1.4283, all groups 1.5322 / 1.4815. `shared_down` costs 0.016 in every variant tried (round to nearest, calibrated, even the BF16 original as FP16), so the recommended setting leaves it on the EXL3 tensor.

The speed side of the same table is in [FLASH_NEXT.md](FLASH_NEXT.md#nvfp4) and [PERFORMANCE.md](PERFORMANCE.md).

### GLM-5.3-Flash

`test_ppl --glm --chunk 8 --file <text> --tokens 12000` on held-out text, with the routed experts at EXL3 3.05 bpw (NVFP4 experts for GLM do not fit the arena).

| Dense groups | Perplexity |
|---|---|
| none (EXL3 3.05 bpw) | 1.798 |
| `dense` (the native NVFP4 MLP layers 0 to 2), round to nearest | 1.799 |
| `idx` | 1.798 |
| `head` | 1.799 |
| `mla` | 1.800 |
| `shared` | 1.805 |
| `kda` | 1.808 |
| all of `kda,mla,idx,head,dense,shared`, round to nearest | 1.812 |
| all, calibrated (`nvfp4_ldlq.py --model glm`) | 1.804 |

The calibrated tensors are the ones `configs/glm_nvfp4_dense.json` uses. GLM is sensitive to rounding-level noise whatever the format (a different prefill chunking moves its logits by 0.2 to 0.7), so differences of a few thousandths between runs are within what the sampling and the router flips produce.

## Other quality checks

* KV cache formats: FP8 and the int8 / NVFP4 caches against FP16 on the same prompts (cosine of the logits: FP8 0.999998, K8V4 0.9998, NVFP4 0.99995 on Qwen3.8-27B at tensor-parallel 4; the 8K needle passes on all of them on GLM). See [TESTS.md](TESTS.md).
* MiniMax-H3 text tower: EXL3 6 bpw 0.999999 cosine against the BF16 hidden states (0.999957 with an image), the NVFP4 build 0.999913 text-only and 0.9737 with an image ([video.md](video.md)).
