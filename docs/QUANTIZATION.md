# Quantization

The weight formats the engine reads, the checkpoints at how many bits and what that costs in quality, and the engine's own tools that calibrate and quantize them.

## Formats

* **EXL3** (trellis): the `exl3` conversions of turboderp's exllamav3, or the engine's own (`volta_quantize`, below). `bits` is the average over the tensors; the
  head, the MTP head and the vision tower have their own widths. Loaded as they are, decoded in the GEMV / GEMM kernels, in either tile order (below).
  Flash-Next's 6.05 bpw checkpoints (6-bit experts) run their prefill on the persistent MoE kernel's uniform 6-bit instantiation (gate, up and down of one width);
  DeepSeek-V4.1 packs may give each of an expert's three projections its own width (1 to 4 bits).
* **NVFP4** (modelopt, compressed-tensors): E2M1 values with an E4M3 scale per 16 and one FP32 scale per matrix. Software-unpacked on Volta (no FP4 hardware). Either
  read from a checkpoint (the routed experts, or a whole NVFP4 checkpoint), or made at load by the engine's quantizer from BF16 tensors (round to nearest with a +-2
  code search per block, or the calibrated tensors from `volta_nvfp4`). FP8 sources (the Flash-Next MTP experts: E4M3 with 128 x 128 block scales)
  are dequantized first. See [NVFP4](#nvfp4).
* **GGUF** (llama.cpp's block formats): see [GGUF](#gguf).
* **FP8** with block scales: DeepSeek-V4.1 packs that keep the original FP8 attention, shared-expert and indexer weights (E4M3 with E8M0 scales) hold them FP8 in
  device memory (`fp8_resident`). Original FP8 checkpoints of the DeepSeek-V3 style (E4M3 with F32 128 x 128 block scales, `<base>.weight_scale_inv`, the
  multiplier of the decoded value; GLM-5.3-Flash) run as they are: the routed experts stay FP8 in `FP8MoE`, whose tiles decode the E4M3 bytes to FP16 exactly
  (the value times 2^-8) and scale each 128-wide input step's FP32 sum by its block scale; every other FP8 tensor is dequantized to FP16 at load.

BF16 tensors are converted to FP16 at load: V100 has no BF16 arithmetic ([HARDWARE_SM70.md](HARDWARE_SM70.md)).

### Checkpoints

The checkpoints this document names, and where they come from:
* [thelastspark/Swift-Qwen3.8-Flash-Next-exl3 (4.00bpw)](https://huggingface.co/thelastspark/Swift-Qwen3.8-Flash-Next-exl3)
* [turboderp/GLM-5.3-Flash-exl3 (3.05bpw)](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3)
* [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) (self-quantized as `*-exl3-calibrated-v2`)
* [Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3 (5.0bpw)](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw)
* [turboderp/Qwen3-VL-32B-Instruct-exl3 (6bpw)](https://huggingface.co/turboderp/Qwen3-VL-32B-Instruct-exl3)
* [QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4)
* [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
* [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)
* [turboderp/DeepSeek-V4.1-Flash-1017-exl3 (1.68bpw)](https://huggingface.co/turboderp/DeepSeek-V4.1-Flash-1017-exl3): routed experts at 1.5 and 2 bpw
* [ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)
* [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017): the original; its DSpark drafter (`mtp.*`) self-quantized as
  `*-DSpark-EXL3-e3` (experts at 3 bits)
* [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it): the original of the Gemma 4 conversions
* [z-lab/gemma-4-31B-it-DFlash](https://huggingface.co/z-lab/gemma-4-31B-it-DFlash): Gemma 4's DFlash (v1) drafter

### GGUF

A model directory holding a `.gguf` file, or every file of a split one (`-00001-of-0000N.gguf` ...), plus the original `config.json`, tokenizer files and chat
template loads like a safetensors one (`src/loader/gguf.cpp`, GGUF versions 2 and 3). Architectures (`general.architecture`): `qwen35` (Qwen3.8-27B), `qwen4exp`
(Flash-Next), `glm5-next`, `deepseek41`, `gemma4`, `dflash` (a DFlash2 drafter, next to the release's `config.json`) and `clip` (a vision projector `mmproj` file of
type `gemma4v`, `glm5v`, `deepseek41v` or `qwen3vl_merger`). The adapters (`src/loader/gguf_*.cpp`) map the GGUF names to the Hugging Face names the models load and
undo llama.cpp's converter transforms on host copies: norm weights stored as 1 + w, `A_log` stored as -exp(A_log), linear-attention V heads in tiled order,
per-head matrices stored transposed, a patch embedding stored as a conv kernel, experts stacked into one tensor, tensors split into parts.

| Backend | Block types | Kernels |
|---|---|---|
| CUDA | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_0, Q2_K to Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS | llama.cpp's vendored `vec_dot` on q8_1 activations up to 8 rows, a tile decoded into shared memory and multiplied on the tensor cores for 9 to 16 rows, dequantize + the FP16 GEMM above; dense Q8_0 weights repacked at load into Q8_0R for a dp4a GEMV (`gguf_q8r`) |
| CPU | the same | llama.cpp's CPU dot products (the activations quantized to q8_0, q8_1 or q8_K by type), ik_llama.cpp's iqk GEMM for several rows |
| Vulkan | the same but Q2_0 | Slang decoders, GEMV and tiled GEMM against FP16 activations (no activation quantization), llama.cpp's mat-vec and matmul shaders |

* Norms, embeddings and other small tensors (F32, F16, BF16) are dequantized once at load. Routed experts (Flash-Next, GLM, DeepSeek) keep their blocks: route by
  route in decode and verify blocks, prefill chunks of more than 64 rows on tensor-core tiles that decode each expert's weights once per pass, and on CUDA they take
  part in the expert tier; on CPU and Vulkan they run expert by expert, all resident.
* Flash-Next's n-gram table stays memory-mapped and is decoded row by row on the host; so are DeepSeek-V4.1's Engram rows (IQ4_NL blocks).
* The 27B's MTP head is read from its GGUF file; the MTP blocks of Flash-Next and GLM are not read (Flash-Next with `speculator=mtp` takes its head from an NVFP4
  checkpoint, `mtp_head_dir`).
* The backends round the activations differently (llama.cpp's CUDA kernels to q8_1, its CPU kernels to q8_K, the Vulkan shaders not at all), so cross-backend
  checks of GGUF weights take a cosine bound of 0.999 ([TESTS.md](TESTS.md)).

## Volta tile order (EXL3)

A 16x16 weight tile is stored in the order of the Ampere `m16n8k16` fragments, so the Volta GEMM moves each lane's decoded values to the lanes that need them with
shuffles. A checkpoint in the Volta order (`"tile_layout": "volta"` in its `quantization_config.json`) stores the tiles so that the eight values a lane decodes are
exactly its `m8n8k4` operands, and the kernels use them without moving anything (two octets share each output block and their partial sums are added once per
column). In that order a lane's decoded values are the operands of two k-chunks rather than two row groups, so a lane keeps a partial sum per row half: the trellis
MoE prefill takes 128-column tiles in that order, which hold as many accumulators as the stock order's 256-column tiles.

* The order is a property of the checkpoint, read at load time and carried by every weight (`LinearWeight::volta_tiles`), so a target and its drafter may differ.
  SM_70 CUDA builds compile every translation unit whose kernels decode tiles twice, the second copy with `EXL3_VOLTA_LAYOUT=1` in namespace `exl3_vl`
  (`src/kernels/cuda/trellis/exl3_vl.cuh`): the GEMM kernels, the tensor-core and dp4a int8 GEMVs, `reconstruct`, the fused MoE decode tile (tensor-core and int8)
  and the persistent MoE kernel. The CPU and Vulkan builds read both orders in their reconstruction. ROCm builds and CUDA devices of SM_80 or later read the stock
  order only, and the loader refuses a Volta-order checkpoint there; the `cpu_experts` kernel reads the stock order only as well. `configs/glm_volta.json` is the
  GLM configuration with model and drafter in that order.
* The order is not visible in the tensors themselves (the other order decodes to noise without any error), so the entry has to be right; no entry means the stock
  order. Weights a checkpoint keeps in the stock order inside a Volta-order checkpoint are listed by name substring in `"tile_layout_stock"` (e.g. `["model.visual"]`:
  the vision tower, whose widths are not multiples of 128, keeps its template's stock order), and the loader reads the weights those match in the stock order;
  without the entry such a tower decodes as noise.
* `volta_quantize --order volta` (the default, below) writes a Volta-order checkpoint from the original weights. The order is a tile permutation applied before the
  trellis encoder, which does not depend on it.
* `test_trellis_order` decodes the same random codes in both orders (the Volta reconstruction must equal the permuted stock one bit for bit) and checks the GEMV, GEMM
  and reconstruct paths of each order against a double-precision product from 1 to 48 rows; `test_trellis_layout` checks the dequantizer against a golden decode
  (`tests/tools/trellis_layout_golden.py --layout stock|volta`); `test_prefill_linear` compares the fused GEMM and the GEMV with reconstruct plus the tensor-core GEMM
  from 1 to 64 rows; `test_moe` with `MOE_SYNTH=1` runs the MoE kernels on random experts of any widths in either order. `ENGINE_DEBUG_GLM_LAYERS=N` cuts the GLM
  to its first N layers, so a conversion cut to N layers can be compared with the original through `diag_logits_dump` and `compare_logits.py`.

## Bit widths of the EXL3 checkpoints

| Checkpoint | Average bits | Head | MTP head | Vision | Notes |
|---|---|---|---|---|---|
| Swift-Qwen3.8-Flash-Next-exl3-4.00bpw | 4.0 | 6 | 4 | 6 | attention and GDN projections 7 / 8 bit, routed and shared experts 3 / 4 bit (layers 5, 8, 45 and 46 are 3-bit); router, `shared_expert_gate` and the GDN `in_proj_a/b` FP16 |
| GLM-5.3-Flash-exl3-3.05bpw | 3.05 | 6 | | 6 | |
| Qwen3.8-27B-DFlash2-EXL3-5.0bpw (drafter) | 5.0 | 6 | | | |
| Qwen3-VL-32B-Instruct-exl3-6bpw (MiniMax-H3 text tower) | 6.0 | 6 | | | `mcg` codebook, the others `mul1`; each conversion calibrated on 250 rows of 2048 tokens |

The NVFP4 checkpoints: `Qwen3.8-Flash-Next-NVFP4` is modelopt `MIXED_PRECISION`: routed experts NVFP4 (group 16; `weight` U8 packed, `weight_scale` E4M3 per 16
values, `weight_scale_2`, `input_scale`), the MTP head's experts FP8 E4M3 (128 x 128 blocks), the n-gram table FP8, everything else BF16. `GLM-5.3-Flash-NVFP4` is the
same idea for GLM (experts NVFP4, the rest BF16); its layer 45, the MTP head, is all BF16 (experts included) and is quantized to NVFP4 at load when
`speculator=mtp`; the EXL3 conversion has no head. The engine uses the NVFP4 experts as they are and quantizes the dense tensors it is told to (`nvfp4_dense`) from
the BF16 ones. `Qwen3.8-27B-QUASAR-NVFP4` runs as it is (`configs/qwen_nvfp4.json`); `nvfp4_dense=head` quantizes its FP16 head too.

## NVFP4

### Flash-Next

* `nvfp4_experts=<dir>` takes the routed experts from the modelopt checkpoint with the EXL3 directory as the model (the MTP experts stay EXL3; activations stay
  FP16, `input_scale` is unused). An expert takes 2.76 MB against the 4.00 bpw conversion's 2.46 MB.
* `nvfp4_dense=<groups>` quantizes dense groups at load (`gdn`, `attn`, `idx`, `head`, `shared_gate`, `shared_up`, `shared_down`, `shared`, `all`) from the BF16
  tensors of the NVFP4 checkpoint: per 16 values the E4M3 scale nearest to amax / 6 and its +-2 code neighbours are tried, one global scale per tensor.
  `nvfp4_dense_dir=<dir>` loads calibrated tensors instead, for the groups they cover.
* With an EXL3 model directory and `nvfp4_experts`: `nvfp4_dense=gdn,attn,idx,head,shared_gate,shared_up` with the calibrated directory. `shared_down` stays on
  the 8-bit trellis tensor: it costs +0.016 perplexity in every variant tried (round to nearest, calibrated, the BF16 original as FP16).
* MTP head (`mtp_nvfp4=true`, needs `nvfp4_experts`): read from the NVFP4 checkpoint instead of the model directory. Its 512 experts are FP8 E4M3 with 128 x 128
  block scales (`model-fp8-mtp-ple.safetensors`) and everything else is BF16; the experts are dequantized and quantized to NVFP4 at load (round to nearest), the
  dense tensors go through the same quantizer as the trunk's dense groups. `mtp_head_dir=<dir>` takes the head from such a checkpoint while the model's own
  weights, routed experts included, stay as they are.
* Only the NVFP4 checkpoint (nothing from a conversion): `model_dir` and `nvfp4_experts` both point at `Qwen3.8-Flash-Next-NVFP4`, with `nvfp4_dense=all`,
  `nvfp4_dense_dir=Qwen3.8-Flash-Next-NVFP4-dense` (the calibrated tensors, below) and `mtp_nvfp4=true` (`configs/flash_next_nvfp4_only.json`; the MTP head's
  dense tensors are rounded to nearest). The small tensors (embeddings, norms,
  router, gated-residual weights, PLE projections) are the same bytes as in the conversions, the dense groups are quantized to NVFP4 at load and the router is
  converted from BF16. The n-gram (PLE) table is read in place from the FP8 shards of `model-fp8-mtp-ple.safetensors` (51 GB against the conversion's 39 GB; 128
  shards of 160-wide rows, one BF16 `weight_scale`, no bias), memory-mapped like the conversion's file.

### GLM-5.3-Flash

* `nvfp4_dense_src=<GLM-5.3-Flash-NVFP4>` with `nvfp4_dense=kda,mla,idx,head,dense,shared` (groups: `kda` q / k / v stacked per rank + o_proj, `mla` q_a / q_b /
  kv_a / o_proj, `idx`, `shared` or its three parts, `dense` the native NVFP4 MLP layers 0-2, `head`); the routed experts stay EXL3.
* `nvfp4_expert_layers=lo-hi` (with `nvfp4_dense_src`) loads the routed experts of those layers from the NVFP4 checkpoint instead; they stay resident, outside the
  expert tier.
* `speculator=mtp` quantizes layer 45 of the NVFP4 checkpoint (the MTP head, all BF16) to NVFP4 at load (`configs/glm_mtp.json`).

### Calibration

`ENGINE_HESSIAN_DIR=<dir> test_ppl --model qwen4_exp|glm5_next --calibrate --file <text> --tokens N` (with `ENGINE_MAX_SEQ` above N; GLM a third of the layers per pass with
`ENGINE_HESSIAN_LAYERS=lo-hi`) accumulates x^T x of every dense projection input and writes one file per site (the capture is described below).
`volta_nvfp4 --src <NVFP4 checkpoint> --hess <dir> --out <dir> [--model flash|glm] [--groups ...] [--layers lo-hi] [--tp 4] [--devices 0,1,2,3] [--rtn]` then runs
block LDLQ on the GPUs (`engine/quant/nvfp4_quantize.h`: damping 2.5% of the mean diagonal, a Cholesky factor in double, 16-row blocks from the last to the first
with the propagated error, each column's E4M3 scale searched over the two codes either side of the nearest to amax / 6, no Hadamard rotation; row-parallel inputs
use the per-rank Hessians as a block diagonal; GLM's q / k / v share one global scale) and writes a `*-NVFP4-dense` directory of modelopt-named tensors (8 layers
per file: the loader opens at most 128 shard files), which `nvfp4_dense_dir` loads. `test_nvfp4_ldlq` holds the quantizer against a host restatement.

`Qwen3.8-Flash-Next-NVFP4-dense` (the directory the Flash-Next NVFP4 configurations name) holds every dense group of `Qwen3.8-Flash-Next-NVFP4`, calibrated on
262,144 tokens in 2048-token rows of exllamav3's calibration texts (c4, code, multilingual, technical, wiki, tiny) interleaved with C++ sources, documentation
and an agent transcript; the 313 tensors take under a minute on four GPUs. The proxy error falls 1.8 to 2 times against round to nearest on most projections,
3.7 times on the GDN `out_proj` and 16 times on the attention `o_proj` (means over the layers).

### Flash-Next dense groups against FP16

`test_ear --model qwen4_exp --compare` with `Qwen3.8-Flash-Next-NVFP4` as the reference (its dense tensors FP16, the default load): the reference's own answers
to `tests/data/ear_prompts.json` (30 samples, 20,792 tokens; T 1.0, top-p 0.95, top-k 20, seed 1), 16 rows per teacher-forced forward. M=1 decode is
`bench_decode` at tensor-parallel 4 (FP16 dense: 10.4 ms).

| `nvfp4_dense` | Tensors | EAR | KLD | Top-1 | M=1 decode |
|---|---|---|---|---|---|
| `shared` | round to nearest | 0.9844 | 0.0030 | 98.2% | 10.3 ms |
| `head` | round to nearest | 0.9789 | 0.0052 | 97.4% | 10.3 ms |
| `gdn` | round to nearest | 0.9722 | 0.0099 | 97.0% | 9.8 ms |
| `attn` | round to nearest | 0.9559 | 0.0253 | 95.3% | 10.3 ms |
| `gdn,shared,head` | round to nearest | 0.9621 | 0.0174 | 95.9% | 9.2 ms |
| `gdn,shared,head` | calibrated | 0.9722 | 0.0092 | 96.9% | 9.2 ms |
| `all` | round to nearest | 0.9424 | 0.0418 | 93.7% | 8.8-9.2 ms |
| `all` | calibrated | 0.9691 | 0.0115 | 96.5% | 8.8-9.2 ms |
| `all` but `shared_down` | calibrated | 0.9695 | 0.0112 | 96.7% | 8.9-9.3 ms |

By kind, `all` round to nearest against calibrated: agentic 0.9274 / 0.9617, chat 0.9187 / 0.9566, code 0.9481 / 0.9741, math 0.9797 / 0.9891, multilingual
0.9421 / 0.9658. The decode gains do not add up group by group: `shared` and `head` alone save under 1% each, together 4.5%.

## Perplexity

Perplexity values are reported in their mean NLL forms.

### Flash-Next

`test_ppl --model qwen4_exp --qbench <checkpoint>/qbench_prompts.md`: nine traced prompts of the converter's qbench set, 20,487 scored response tokens. Lower is better; the
model card gives 1.3825 for BF16.

| Configuration | Perplexity | Over BF16 |
|---|---|---|
| BF16 (model card) | 1.3825 | |
| EXL3 4.00 bpw | 1.4247 | +3.1% |
| EXL3 4.00 bpw, routed experts from the NVFP4 checkpoint | 1.4282 | +3.3% |
| + recommended dense groups in NVFP4 (`gdn,attn,idx,head,shared_gate,shared_up`, calibrated) | 1.4530 | +5.1% |
| + all dense groups in NVFP4 (calibrated) | 1.4815 | +7.2% |
| the NVFP4 checkpoint on its own (all dense groups calibrated through `nvfp4_dense_dir`, FP8 n-gram table) | 1.4813 | +7.1% |

Per group, on top of the NVFP4 experts, round to nearest against calibrated: `gdn` 1.4394 / 1.4382, `attn` 1.4714 / 1.4341, `head` 1.4434 / 1.4348, `shared`
1.4479, `idx` 1.4283, all groups 1.5322 / 1.4815. `shared_down` costs 0.016 in every variant tried (round to nearest, calibrated, even the BF16 original as FP16),
so the recommended setting leaves it on the EXL3 tensor.

The speed side of the same configurations is in [PERFORMANCE.md](PERFORMANCE.md).

### GLM-5.3-Flash

`test_ppl --model glm5_next --chunk 8 --file <text> --tokens 12000` on held-out text, with the routed experts at EXL3 3.05 bpw, run with
`ENGINE_EXPERT_CACHE=0.9 ENGINE_VRAM_RESERVE_MIB=3500 ENGINE_MAX_IMAGE_TOKENS=0` to leave room for the dequantization workspace.

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
| all, calibrated (`--model glm`, `ENGINE_HESSIAN_LAYERS=lo-hi` in three passes) | 1.804 |

Against the original FP8 checkpoint (`test_ppl --model glm5_next --tokens 8192 --chunk 512` on the default text, the docs/ folder as of commit c233440, prefill windows
of 8 chunks, `glm_logit_rows=4096`): the FP8 model 1.3846 NLL, the EXL3 3.05 bpw copy 1.3979, KL( FP8 || 3.05 ) 0.0414 per token over the FP8 model's top 128.

The calibrated tensors are the ones the GLM configurations use (`configs/glm_256k.json`). GLM is sensitive to rounding-level noise whatever the format (a different
prefill chunking moves its logits by 0.2 to 0.7), so differences of a few thousandths between runs are within what the sampling and the router flips produce.

## Other quality checks

* KV cache formats: FP8 and the int8 / NVFP4 caches against FP16 on the same prompts (cosine of the logits: FP8 0.999998, K8V4 0.9998, NVFP4 0.99995 on
  Qwen3.8-27B at tensor-parallel 4; the 8K needle passes on all of them on GLM). See [TESTS.md](TESTS.md).
* The expert tier does not change results: Flash-Next with NVFP4 experts scores 1.4282 at `expert_cache` 0.5 and 0.25 as fully resident.
* MiniMax-H3 text tower: EXL3 6 bpw 0.999999 cosine against the BF16 hidden states (0.999957 with an image), the NVFP4 build 0.999913 text-only and 0.9737 with an
  image ([VIDEO.md](VIDEO.md)).
* `test_ear` measures the expected acceptance rate of a quantized model against a reference model on the reference's own generations, per kind of text
  ([TESTS.md](TESTS.md)); `test_ppl --kl-teacher` the KL, the expected acceptance rate and the top-1 agreement against a reference's stored
  distributions over every position of a text.

## Quantizing EXL3 with the engine

The engine quantizes EXL3 checkpoints itself: `volta_quantize` (`src/tools/volta_quantize.cpp`) with the engine's EXL3 quantizer (`engine_quant`,
`src/kernels/cuda/trellis/quant`: the algorithm of exllamav3 1.6.7's quantizer, MIT, with the trellis encoder ported to SM_70 and decoding through the inference
codebook), calibrated by Hessians the engine captures while it runs the model. The tools are CUDA-build targets. The steps, top to bottom:

1. **Calibration text**: the token sequences the model sees as it serves (below).
2. **Hessians**: `test_ppl --calibrate` with `hessian_dir` replays the text and writes the input Hessians per site; `route_dump` records the routed experts' input
   rows for MoE models.
3. **Mix**: `hess_compare merge` sums capture directories of several kinds of text.
4. **Quantize**: `volta_quantize` with a template EXL3 checkpoint of the same model.
5. **Tune** (optional): `volta_tune` (routed experts' scales against the original experts) or `volta_e2e` (every scale against a teacher's distributions, Flash-Next).
6. **Score**: `test_ppl --prompts <dump dir> --select hold`, `test_ear`, `hess_compare score` / `moe`.

### Calibration text

* Started with `--set dump_requests=<dir>`, the server writes every request body and the rendered prompt (`req_N.prompt.txt`); those are the token sequences the
  model saw. `tests/agentic/bench.py` ([TESTS.md](TESTS.md#agentic-benchmark-testsagentic)) drives a server with agent sessions (`gen_tasks.py` generates the
  tasks).
* `volta_sessions --out <dir> [--max-chars 120000] [--split] SESSION.jsonl ...` turns recorded Claude Code sessions into Anthropic Messages request bodies
  (main-thread turns in order, consecutive entries of one role joined, thinking, tool calls and tool results as recorded, cut at a message boundary past
  `--max-chars`); `--split` writes the rest of each session too, as further bodies of up to `--max-chars` each, cut before a user turn and opening with the
  session's first message (a chat template wants a user query), numbered in order of the parts, so that sessions longer than the server's context
  (`count_tokens` refuses a prompt over `max_seq`) render whole. Posted to a server's `/v1/messages/count_tokens` with `dump_requests` set, each is rendered by the
  model's chat template exactly as the live conversation was, without a forward pass.
* Answers of another server (a hosted copy of the model) become calibration text the same way: `volta_sessions --requests PROMPTS.json --out <dir> --model NAME
  [--max-tokens 6000] [--temperature 1.0] [--top-p 0.95]` writes chat completion bodies for a prompt list (`test_ear`'s format), and `--conversations PROMPTS.json
  --responses <dir> --out <dir>` turns the replies into Anthropic bodies of the exchange, the reasoning as a thinking block (a reply cut inside its reasoning keeps
  the reasoning alone). `test_ear --generate --text-out <dir>` writes the local model's own answers as `sN.prompt.txt`.
* `tests/tools/prose_corpus.py --src <dir of <title>/whisper.json> --out prose.txt` samples a contiguous stretch from the middle of several book transcripts for
  the non-agent share of a calibration set; `test_ppl --calibrate --file prose.txt --seq 2048 --tokens N` replays it as independent rows.

### Hessian capture

`ENGINE_HESSIAN_DIR=<dir> [ENGINE_HESSIAN_LAYERS=<lo-hi>] test_ppl [--model qwen4_exp | glm5_next] --calibrate --prompts <dump dir> [--select train] [--tokens N] [--chunk 512]`
replays the last request of each session (the rows a session shares with an earlier one, its system prompt and tool schemas, run without adding, so every distinct
token counts once; `--select train` leaves out every tenth session for scoring, `--stride K` counts every K-th chunk past the shared prefix), and `--file <text>
--seq 2048` replays plain text. Files are `L<layer>_s<site>_r<rank>.f32` (x^T x and the row count), written at shutdown; `tests/tools/hess_check.py` checks a
directory (symmetric, positive diagonal, one row count). Layer slices (`hessian_layers`) keep the accumulators beside the weights. Sites per layer:

| Model | Sites |
|---|---|
| Qwen3.5-family dense (`qwen3_5`) | 0 attention input (q / k / v, or `in_proj_qkv` and `in_proj_z` of a linear-attention layer), 1 MLP input, 2 `o_proj` / `out_proj` input (per rank), 3 `down_proj` input (per rank); layer 0 site 4 the LM head input |
| Flash-Next (`qwen4_exp`) | 0 attention / GDN input (with the QSA indexer's `index_qk_proj`), 1 MLP input (routed experts' and shared expert's gate / up), 2 `o_proj` / `out_proj` input (per rank), 3 the shared expert's down input; layer 0 site 4 the LM head input; with `ENGINE_CALIB_MTP=1` the MTP head runs after every forward on the same rows and its block is written as layer 48 (the five sites of a decoder layer) and its input projections as layer 49 (site 0 `fc_embedding`, site 1 `fc_hidden`) |
| GLM (`glm5_next`) | 0 attention input (`qkv_proj`, `q_a_proj`, `kv_a_proj_with_mqa`), 1 FFN input, 2 `o_proj` input (per rank), 3 the shared expert's down input (layers 0 to 2: the dense MLP's), 4 the q-lora norm output (`q_b_proj` and `indexer.wq_b`); layer 0 site 5 the LM head input; the replicated sites live on rank `layer % tp` and are written as `_r0` |

The routed experts' gate and up projections share the block-input site (every expert sees every row); their down projections need each expert's own activations,
which `volta_quantize` computes from the router-input rows that the debug option `route_dump` records (`ENGINE_DEBUG_ROUTE_DUMP=<file>` on the same replay; the
routing weights go to `<file>.w`; `ENGINE_DEBUG_ROUTE_DUMP_FROM=48` limits the dump to Flash-Next's MTP block). GLM is captured with `ENGINE_EXPERT_CACHE=0.95
ENGINE_MAX_SEQ=32768 ENGINE_MAX_SLOTS=1` in slices of 15 layers (a capture keeps the dense MLPs and the shared expert unsliced, which is why the experts are not all
resident); its replay restores a snapshot of the prefix that a family of sessions shares instead of running it again, so a session costs about its own new tokens.
A DFlash2 drafter's inputs are captured while the server speculates (`--set dflash_hessian_dir=<dir>` with `speculator=dflash`; files `L<layer>_s<site>_r<rank>.f32`:
site 0 `q_proj`, 1 `k_proj` / `v_proj` (context and block rows), 2 `o_proj` and 4 `down_proj` per rank, 3 `gate_proj` / `up_proj`; layer 5 site 0 is the `fc`
input), written when the server shuts down.

`hess_compare merge <out> <dir> <dir> ...` sums the capture files every directory has (Hessian sums and row counts), so captures of different kinds of text form one
calibration mixed by their token counts; `tests/tools/hess_merge.py --out DIR --src DIR_A:0.7 --src DIR_B:0.3` mixes them by weights instead. `--route` of
`volta_quantize`, `volta_tune` and `hess_compare` takes several dumps comma-separated (their rows in that order).

### volta_quantize

`volta_quantize --src <original> --template <EXL3 directory> --out <dir> [--hess <capture dir>] [--route <route dump>] [--order volta|stock] [--devices 0,1,2,3]
[--only A,B] [--skip A,B] [--bits B] [--widths FILE] [--only-widths] [--ranks 4] [--route-rows N] [--budget-mb 8192] [--seed N] [--sigma-reg S]
[--no-scaled-hessian] [--affinity-eps E] [--ldl-metric] [--open-trellis] [--tile-gain] [--scale-search full|light|hint] [--workers 1] [--redo] [--plan]
[--no-assemble]` quantizes a checkpoint on the GPUs.

* **What is quantized.** The template is an EXL3 checkpoint of the same model: every tensor it stores as a trellis is quantized from the original at
  the template's width and codebook, everything else is copied as the template has it, and the output keeps the template's shards and file names.
  `--plan` lists what would be done without a GPU. The original's tensors are read as BF16, FP16, FP32 or FP8 E4M3 with block scales
  (`weight_scale_inv`), from fused experts (`experts.gate_up_proj`, `experts.down_proj`), and GLM's fused `qkv_proj` from `q_proj`, `k_proj` and
  `v_proj`; a tensor the original lacks or shapes differently (the vision tower's padded widths) stays as the template has it.
* **Hessians.** `--hess` is a capture directory of `test_ppl --calibrate` (`hessian_dir`), mapped to the tensors by the model's sites (above;
  `--ranks` the capture's rank count); the row-parallel inputs' per-rank files form a block diagonal. A routed expert's down projection takes the
  Hessian of its own activation (the model's SwiGLU, with GLM's limit) over the rows of the route dump (`--route`; `--route-rows` caps the rows per
  layer), computed per layer when its experts are quantized, so no Hessian files are written. `--affinity-eps E` weights the routed experts' Hessians
  by routing as exllamav3 1.6.7's `--moe_affinity_eps` does: a row counts with eps times the expert's mean routing weight plus its routing weight
  where it is routed to the expert, so each expert's gate and up projections get their own input Hessian (from the route dump's rows routed to it) and
  are quantized expert by expert; it needs the dump's routing weights (`<route dump>.w`). A tensor without a Hessian is quantized without error
  feedback.
* **Per tensor.** The Hessian is damped (`--sigma-reg`, default 0.025 of the mean diagonal); the weight gets random output and input sign flips,
  per-channel output and input scales, the 128-point Hadamard rotation on both sides and a searched global scale (a coarse grid, then a fine one
  around its minimum; a projection after the first in a group, or a layer's next expert down projection, starts the fine grid at the previous one's
  scale and runs the coarse grid only when the minimum lands on an edge; `--scale-search hint`, the default, takes that previous one as it is, `light`
  tries three scales around it on a lighter sample first, `full` runs the fine grid); its block LDL factor is that of the Hessian with its own input
  scales folded in (exllamav3 1.6.7's `scaled_hessian`; `--no-scaled-hessian`: one factor of the sign-flipped Hessian per input), its diagonal held
  at 1e-3 of its mean before the Hadamard rotation (a channel the weight barely reads would otherwise fall below fp32 resolution once mixed
  with live ones and fail the factorization); the 16-row blocks
  are quantized from the last to the first with the LDL error feedback; two rounds refit the stored scales (`suh`, `svh`) to the quantized weight in
  the Hessian metric. `--ldl-metric` weights each element's squared error in the Viterbi search by its row's diagonal of the block's LDL metric. The
  reported proxy error is tr(E^T H E) / tr(W^T H W) of what is stored, with the damping taken out of H.
* **Work.** A work item is the projections of one input (every expert's gate and up projections share the block input), side by side in chunks of
  `--budget-mb`, which go through the encoder together block row by block row, or a layer's expert down projections; one thread per device in
  `--devices` (`--workers N`: N per device, each on its own stream, so one's host work and small kernels overlap another's encoder; each holds an
  item's buffers, up to `--budget-mb`) takes items until none are left. Every finished tensor is written to `<out>/.partial`, so a run continues where
  it stopped (`--redo` quantizes again); the shards are written at the end (`--no-assemble` stops before them; a later run without it writes them from
  `.partial`). At the end the devices' seconds are logged by phase: the source loads, the Hessians (read from the capture or computed from the route
  dump), the input factors, the regularization with its scale search, the per-projection factors, LDLQ through the encoder, the back rotation with the
  refit and packing, and the partials written.
* **Tile order and widths.** `--order volta` (the default) writes `"tile_layout": "volta"`; from a stock template, the tensors that stay as the
  template has them are listed in `"tile_layout_stock"`. `--only` (tensors whose names contain any of the substrings) needs the template's order, and
  a Volta template's stock-order exceptions are kept as they are; `--skip` leaves tensors as the template has them. `--bits` gives every quantized
  tensor one width (1 to 8, or 1.5, 2.5, 3.5 with mul1) instead of the template's, for rate ladders on a sample. `--widths FILE` sets widths per
  tensor: lines of `<regex> <bits>` (`#` comments), each tensor taking the first pattern its name matches (the others `--bits` or the template's);
  with `--only-widths` only the matched tensors are quantized and the rest is kept from the template, so an existing checkpoint as the template gets
  another width allocation.
* **Extended trellis.** `--open-trellis` encodes each tile without the tail-biting ring and stores its start state, the bits before its stream, as
  `<base>.tstart` (uint16 per tile); `--tile-gain` stores a gain per tile as `<base>.tgain` (8 bits, the value (896 + c) / 1024, fitted by least
  squares before the error feedback sees the tile). A tensor's companions are its flag. Both options apply to the routed experts only: the engine's
  MoE kernels on CUDA decode them (the coop decode kernels on their tensor-core tile, the persistent prefill kernel at every row tile, both tile
  orders), and the loader reads them with whole expert tensors or a rank's slice of them (GLM's sharded experts: the companions cut by the same
  tile rows or columns as the codes); `tile_gains=false` leaves the gains on disk, so the tiles decode at gain 1 and the arena keeps a byte per
  tile (the starts cannot be left out: a tile's first positions decode from them). Dense layers, the heavy-expert path (`moe_heavy_rows`), CPU and Vulkan experts, ROCm's coop kernels and the
  range readers refuse an extended tensor. `reconstruct_slice_ext` and `reconstruct_had_slice_ext` decode them, and the tools
  read and write them; `hess_compare` scores `.partial` results that have them.

`test_exl3_encode` and `test_exl3_quantize` test the encoder and the quantizer, `test_expert_hessian` the expert Hessians ([TESTS.md](TESTS.md)).

### hess_compare

`hess_compare stats <dir_a> <dir_b>` sets the mean Hessians of two capture directories against each other per site; `cross --src <original> --tensor <name> --a
<file> --b <file> [--bits 3]` quantizes a tensor with each of two Hessians and scores both results under both; `score --src <original> --tensors A,B (--hess-dir
<capture dir> | --route <dump>) --exl3 <dir> ...` scores the tensors of EXL3 checkpoints against the original under one Hessian (the capture's, or a routed expert's
activation Hessian over the route dump); a checkpoint may be a `volta_quantize` `<out>/.partial` directory, whose files carry their tile order and codebook, and
`--affinity-eps E` scores the routed experts under their routing-weighted Hessians instead; `routes --src <original> --route <dump> --layers L,... [--top 32]` lists
each layer's most-routed experts and their projections as a `--only` list; `moe --src <original> --route <dump> --layers L,... [--max-rows 8192] --exl3 <dir> ...`
scores each routed expert of the first directory by its output's error against the original expert's over the dump's rows routed to it, each row weighted by its
routing weight squared (its share of the MoE block's output), per layer and relative to the first directory; `merge` is above.

### volta_tune

`volta_tune --src <original> --exl3 <dir> --route <dump> --out <dir> --layers L,... [--steps 100] [--lr 0.001] [--max-rows 8192] [--devices 0,1,2,3]` tunes the
scale vectors of routed experts (gate, up and down `suh` / `svh`, multiplicatively, with Adam) so that the quantized expert's output matches the original's over the
dump's rows routed to it (weighted as in `hess_compare moe`), keeping the step with the lowest error on every tenth row, which is held out; the trellis cores stay as
they are, and the results go to `<out>/.partial` in `volta_quantize`'s per-tensor form. `--route-q <dump>` switches to block tuning: the dump is of the same text
through the quantized model, so its rows carry the error of every layer before; all of a layer's experts are tuned together so that the quantized routed MoE output on
those rows (the quantized model's routing) matches the original's on the reference rows of `--route` (the reference routing), in batches of `--batch` tokens, every
tenth token held out for the step choice; `--eval-ref` / `--eval-q`, a reference and a quantized dump of held-out text, report the block error there before and
after. It is built on the engine's training primitives (`engine_train`, `inc/engine/train/grad.h`: SwiGLU forward and backward, a linear given as a core between
per-channel scales with its scale and input gradients, a row-weighted squared-error loss, Adam).

### volta_e2e

`volta_e2e --model <EXL3 dir> --teacher <file> --out <dir> [--inputs <file>] [--devices 0,1,2,3] [--tp 1] [--fp16 1] [--norms 1] [--dense 0] [--dense-lr 0.1]
[--router 0] [--route-cand 0] [--hc 0] [--hc-lr 0.1] [--gains 0] [--gains-lr 1] [--reencode L] [--reencode-steps 0,0.003,0.01,0.03,0.1] [--reencode-feedback 0]
[--router-teacher FILES] [--router-kl 0] [--router-kl-layers A-B] [--router-margin 0] [--router-calib 0] [--steps 64] [--batch 4] [--seqs 1] [--shuffle 0]
[--lr 1e-3] [--warmup 2] [--lr-min 0.1] [--hold 4] [--train N] [--eval-every 16] [--adam-bits 32|8] [--opt-host 0|1] [--ckpt-host -1|0|1] [--keep -1|0|1]
[--qsa 1] [--write 0|1] [--timing 0|1] [--check-inputs N] [--check-trace FILE] [--check-routes FILE]` tunes a Flash-Next EXL3 checkpoint end to end by Adam on
KL( teacher || model ) over token sequences.

* **What is tuned.** Every EXL3 tensor's input and output scales (log deltas on the checkpoint's) and the norms' weights (additive deltas: the
  hyper-connection sites', attention's q / k norms, GDN's gated norm, PLE's; `--norms 0` keeps them); codes and the other dense weights frozen.
  `--dense 1` adds the small dense tensors the same way (the hyper-connection sites' inject mixes, GDN's `in_proj_a` / `in_proj_b`, `A_log`, `dt_bias`
  and convolution, the shared expert's gate and PLE's convolution) at `--dense-lr` times the learning rate, since a row of them feeds a dot product
  and Adam's per-entry steps add up over its width; `--hc 1` trains the hyper-connection sites' down and up mixes at `--hc-lr` times the rate;
  `--gains 1` trains the tile gains of the tensors that store them (straight through: FP32 values, the 8-bit codes the forward decodes set from them
  after each step and written back; their gradient is the product's input and output gradient rotated into the tiles' frame, `( X diag( su ) H )^T (
  dY diag( sv ) H )` summed against each tile's decoded values) at `--gains-lr` times the rate.
* **Routers.** `--router 1` trains the routers' weights at `--dense-lr` times the rate (through the selected experts' routing weights; the selection
  itself has no gradient; `--route-cand N` runs each row's next N experts by logit as zero-weight routes in the backward, the output unchanged, and
  gives their logits the gradient of being swapped in). `--router-teacher FILES` (`router_dump` files of the teacher on the teacher files' texts, in
  their order) with `--router-kl W` distills the routing: every MoE layer (or those of `--router-kl-layers A-B`) adds W times KL( the teacher's router
  distribution, its top 16 experts and the mass outside them || the model's softmax over its router logits ) per row to the loss, the gradient into
  the router's input and, with `--router 1`, the router; `--router-margin D` makes that term the top-k boundary's margins instead of the KL (softplus(
  l_w - l_m + D ) for every wrong pick w against every missed teacher pick m, and softplus( D - ( l_c - l_d ) ) for the weakest agreed pick c against
  the strongest expert neither picks d). The logs give the routing term per layer and the share of the teacher's experts the model picks next to the
  output KL. `--router-calib 1` (with `--router 1`) calibrates the routers alone: forward passes only, each router's term's gradient into its own
  weights, each router fitting the teacher's routing on the model's own inputs.
* **Re-encoding.** `--reencode L` moves codes instead of training: one pass over the training sequences sums, for each of layer L's routed experts,
  the gradient G with respect to its projections' tiles R and the Hessian of the rows routed to it; for each `--reencode-steps` size s (a fraction of
  the tiles' norm) every projection is encoded anew toward R - eta D (`exl3_reencode` in `engine/quant/exl3_quantize.h`: the scales and tile gains
  held, each tile encoded at its gain), the held-out KL and the KL on the first 20 training sequences measured (with the first-order change the step
  predicts) and the loaded codes put back. By default each tile is encoded on its own toward D = G (plain metric); `--reencode-feedback 1` runs the
  LDL feedback in the projection's quantization metric M (its input Hessian rotated into the tiles' frame with its input scales folded in) toward D =
  M^-1 G.
* **Schedule.** `--batch` sequences per step (`--seqs N`: up to N of them, and of the held-out ones, through each layer call together, their rows end
  to end, the recurrences, convolutions, positions and attention per sequence: the same gradients with more rows per product, the saved layer inputs
  and the workspace N times larger), the learning rate warming up linearly over `--warmup` steps and then following a cosine down to `--lr-min` of
  `--lr`. The last `--hold` sequences are held out (`--train N` trains on the first N of the others only); their KL is measured before training and
  every `--eval-every` steps.
* **Execution.** The model runs in FP32 training form (`engine/train/qwen4_exp.h`), each layer recomputed from its saved input in the backward (its
  sublayer outputs kept from the forward). With `--tp 1` the devices run it tensor parallel over NCCL, one host thread each: every device holds all
  layers, the per-position work on its block of the sequence's rows, GDN and attention by heads, the routed experts by expert, the head by vocabulary;
  `--tp 0` splits the layers over the devices instead. The routed experts of a layer run in groups of similar route counts, each group's cores decoded
  together and its products one strided batch. The tuned tensors' products run on the tensor cores (`--fp16 1`, the default): the decoded cores in
  FP16 (exact: the trellis decodes to FP16), activations and gradients rounded to FP16 per row at a power-of-two scale, FP32 accumulation; norms,
  recurrences, attention, the scale gradients' sums and Adam stay FP32. `--fp16 0` runs those products in FP32.
* **Memory.** The arena holds exactly the range's EXL3 tensors (codes, FP16 scales, tile starts and gains; a device's share of the routed experts),
  the other dense weights stay FP16 as the checkpoint stores them and each layer call converts its own into the workspace, and the workspace grows to
  what the calls need (a call needing more adds a block; the next call starts with one block of the peak). `--adam-bits 8` keeps Adam's moments as
  8-bit block-wise codes (`engine/train/adam8.h`: blocks of 256 with an absmax each, m and the root of v as small floats of 3 mantissa bits, rounded
  stochastically, the step's m / sqrt v ratio held to Adam's own bound); `--opt-host 1` keeps the optimizer state (reference values, deltas, moments,
  the best values) in pinned host memory and streams it through the device in parts at each step, with the same kernels, so the result is that of the
  device state; the devices then hold the current values, their gradients and one part. The memory by component is logged after the first step, and
  `--timing 1` logs each step's time by phase per rank.
* **Long sequences.** Past the attention's QSA budget (2048 tokens) the training form selects keys as the engine does (the indexer's queries and
  pooled keys with the engine's FP16 roundings, the best 512 pools plus the incomplete tail per row, held fixed like the routing; `--qsa 0`: dense
  attention at every length), the attention runs in blocks of 1024 query rows, `--ckpt-host` keeps the saved layer inputs in pinned host memory
  (copied out on a side stream while a layer runs and back a layer ahead of its backward; -1, the default: past 4 GiB of them) and `--keep` the
  sublayer outputs kept from the forward (-1: within 2 GiB, else recomputed in the backward).
* **Data.** The teacher comes from the reference model on the text (`test_ppl --model qwen4_exp --file TEXT --seq S --tokens N --chunk 512 --topk-out FILE`: the
  top K tokens of every position and the mass outside them); the model's own inputs (its token embeddings and the PLE layer's n-gram rows) are made
  from the teacher file's tokens as the engine makes them (`engine::train::qwen4_exp::Inputs`; with `--tp 1`), or read from `--inputs` files of
  `test_ppl --model qwen4_exp --calibrate` with the debug option `train_inputs` set (`--check-inputs N` compares the two on the first N sequences); `--teacher`
  (and `--inputs`) take comma-separated lists, read as one set of sequences in their order, `--shuffle 1` trains on them in a new order each pass.
* **Output.** The values of the best held-out KL are written as a new checkpoint (the input's files, its shards with the tuned scales, norm weights
  and dense tensors in FP16, the precision the engine computes them in); `--write 0` writes nothing. `test_ppl --kl-teacher FILE` measures any
  checkpoint's KL against such a teacher through the inference path.
* **Checks.** `--check-trace FILE` (a `trace_layers` file of the engine's forward over the first sequence, `test_ppl --calibrate --chunk 512`) feeds
  every layer of the training form the engine's input streams of that layer and compares its output with the engine's, and the first sequence's
  embedding streams with the engine's, then exits; the trace may come from another checkpoint of the model on the same text, giving each layer's own
  deviation from it. With it, `--check-routes FILE` (a `route_dump` of that run) gives per layer, fed that run's input and chained, the share of its
  routed experts this model picks too.
* **GLM-5.3-Flash.** A GLM checkpoint (its `config.json` names `glm5_next`) runs `engine/train/glm5_next.h`'s training form, tensor parallel only: the mHC
  sites with the Sinkhorn mix unrolled in the backward, KDA (`engine/train/glm5_kda.h`) and MLA by heads, the dense MLP and the shared expert on each
  device's rows, the routed experts by expert with sigmoid routing and the clamp after the SiLU (`engine/train/glm5_moe.h`), the head as the streams'
  mean, the final norm and lm_head. It tunes every EXL3 tensor's scales and the norms' weights (the sites' input norms, KDA's output norm, MLA's
  q-lora and kv-lora norms, the final norm), the norms as log deltas too (GLM's are plain weights, from about 0.02 to 1.5); the other kinds above
  are Flash-Next's. The inputs are the teacher tokens' embeddings (no `--inputs`).
  The routed experts' codes (about 26.5 GiB a device at 3.05 bits) stay in pinned host memory and each MoE layer's are copied to one of two device
  stages a layer ahead of its use; `--expert-gib G` keeps whole layers' codes on each device up to G GiB instead, spread over the layers so that the
  staged copies overlap with the resident layers' work. `--check-trace` takes the engine's `trace_layers` file of the first held-out sequence
  (`test_ppl --model glm5_next --seq S --tokens S --chunk 512` with `prefill_window` 1: the last row of each chunk after every layer) and compares the training
  form's streams after every layer, chained from the embeddings, at those rows, with the share of the engine's routed experts (the trace's `.route`
  file) it picks too. `test_train_glm5_layer` and `test_train_glm5_tp` check the layers and the head against central differences and the tensor-parallel
  form against one device ([TESTS.md](TESTS.md)).
