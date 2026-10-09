# Test inventory

`tests/tools/gate.sh fast` runs the first group, `models` the second (`parity` runs both). Tier numbers follow AGENTS.md.

## Fast: no checkpoint needed

| Test | Tier | GPU | Checks |
|---|---|---|---|
| `test_trellis`, `test_gemv_int8` | 2 | yes | trellis and int8 GEMV kernels against FP32 references |
| `test_mhc`, `test_attention`, `test_attention_prefill`, `test_mla`, `test_dsa`, `test_gdn`, `test_norm`, `test_rope`, `test_kv_quant`, `test_prefill_linear` | 2 | yes | module operations against CPU references |
| `test_all_reduce` | 2 | yes (4) | NVLink all-reduce |
| `test_sampling` | 2 | no | top-k / top-p sampler |
| `test_slang_cpu`, `test_slang_kernels` | 2 | partly | Slang kernels on CPU and as generated CUDA |
| `test_ref_linear`, `test_ref_attention`, `test_vulkan_kernels`, `test_vulkan_smoke`, `test_vulkan_ref` | 2 | yes | portable reference kernels on CUDA, CPU and Vulkan |
| `test_ops_linear`, `test_ops_moe`, `test_ops_attention` | 2 | yes | cross-backend op goldens (dumped on CUDA, checked on CUDA, CPU, Vulkan) |
| `test_jinja` | 2 | no | Jinja2 subset against reference Jinja2 renders of both chat templates |
| `test_image` | 2 | no | image decode, smart resize and patch layout against PIL / HF-layout references, GLM canvas sizes against exllamav3's (`tests/tools/gen_image_goldens.py`) |
| `test_output_parser` | 2 | no | reasoning / tool-call parser, chunking independence, template round trip |
| `test_backend_tiers` | 1 | no | capability registry matches the declared module functions |

## Model-backed: checkpoints from `ENGINE_MODEL_ROOT`

| Test | Tier | GPU | Checks |
|---|---|---|---|
| `test_rows`, `test_moe`, `test_nvfp4_linear`, `test_mla_latent`, `test_spec_state`, `test_dflash` | 2 | yes | layer and speculator parity with real weights (`test_mla_latent` also covers the FP8 / int8 / NVFP4 latent caches) |
| `test_tokenizer`, `test_tokenizer_file --cases` | 2 | no | token-exact against HF tokenizers (fixtures in `tests/data`) |
| `test_coherence`, `test_sampling_gpu` | 4 | yes | greedy determinism, seeded sampling, speculative equivalence |
| `test_prefix_restore` (`--spec`, `--glm`, `--flash`, `ENGINE_KV_FORMAT=fp8`) | 2 | yes | a restored snapshot continues exactly like the same prefill without one |
| `test_qwen4_exp` (`--gen N`, `ENGINE_DEBUG_TRACE_LAYERS`, `ENGINE_DEBUG_QWEN4_HALT`, `ENGINE_DEBUG_TEACHER_STREAMS`) | 4 | yes | Flash-Next bring-up driver: traces compared with `tests/tools/ref_qwen4_exp_dump.py` oracle dumps by `qwen4_exp_trace_cmp.py`; greedy decode speed |
| `test_chunk_boundary` (`--cuts a,b`, `ENGINE_DSA=0`, `ENGINE_DEBUG_TRACE_LAYERS=<file>`) | 2 | yes | diagnostic, no pass/fail: GLM logit shift when the prompt is prefilled split at a non-chunk-aligned cut (aligned = 0); the trace file holds the last row after every layer and attention half |
| `test_vision` (`--glm` for GLM's tower) | 2 | yes | vision tower: tensor-core attention against the CUDA-core reference, finite deterministic embeddings (the semantic check is `test_api.sh`) |
| `test_mtp` | 4 | yes | greedy decoding with the MTP head reproduces plain greedy decoding, short and long prompt, and accepts drafts |
| `test_longctx` | 4 | yes | needle retrieval at long context (run by hand) |
| `tests/4_server/test_api.sh` | 4 | yes | OpenAI and Anthropic APIs, tool calls, reasoning, prefix cache, image input (`VOLTA_TEST_MODEL=glm5_next` for GLM, `VOLTA_TEST_SPEC=mtp` for the MTP head) |
| `tests/4_server/test_server.sh` | 4 | yes | server integration, concurrency throughput gate |

CPU and Vulkan forwards (`test_cpu_forward`, `test_vulkan_forward`) compare a whole prompt against CUDA goldens; they run by hand
(slow on CPU).
