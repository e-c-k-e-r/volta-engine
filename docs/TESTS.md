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
| `test_output_parser` | 2 | no | reasoning / tool-call parser, chunking independence, template round trip |
| `test_backend_tiers` | 1 | no | capability registry matches the declared module functions |

## Model-backed: checkpoints from `ENGINE_MODEL_ROOT`

| Test | Tier | GPU | Checks |
|---|---|---|---|
| `test_rows`, `test_moe`, `test_nvfp4_linear`, `test_mla_latent`, `test_spec_state`, `test_dflash` | 2 | yes | layer and speculator parity with real weights |
| `test_tokenizer`, `test_tokenizer_file --cases` | 2 | no | token-exact against HF tokenizers (fixtures in `tests/data`) |
| `test_coherence`, `test_sampling_gpu` | 4 | yes | greedy determinism, seeded sampling, speculative equivalence |
| `test_prefix_restore` (`--spec`, `--glm`, `ENGINE_KV_FORMAT=fp8`) | 2 | yes | a restored snapshot continues exactly like the same prefill without one |
| `test_longctx` | 4 | yes | needle retrieval at long context (run by hand) |
| `tests/4_server/test_api.sh` | 4 | yes | OpenAI and Anthropic APIs, tool calls, reasoning, prefix cache (`VOLTA_TEST_MODEL=glm5_next` for GLM) |
| `tests/4_server/test_server.sh` | 4 | yes | server integration, concurrency throughput gate |

CPU and Vulkan forwards (`test_cpu_forward`, `test_vulkan_forward`) compare a whole prompt against CUDA goldens; they run by hand
(slow on CPU).
