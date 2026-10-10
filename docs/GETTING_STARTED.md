# Getting started

## Build

CUDA 12.x (13+ drops SM_70) and CMake 3.24+ with Ninja. The CUDA build is the primary one:

```
cmake -S . -B build -G Ninja && cmake --build build
```

CPU and Vulkan builds are separate build trees (no CUDA toolchain needed):

```
cmake -S . -B build_cpu -G Ninja -DENGINE_BACKEND=CPU
cmake -S . -B build_vk  -G Ninja -DENGINE_BACKEND=VULKAN
```

A full CUDA build takes a while (the trellis kernels are template heavy); incremental builds are quick.

## Run the server

```
taskset -c 0-9 build/volta_server --model-dir /path/to/Qwen3.8-27B-exl3 \
	--speculative --draft-model-dir /path/to/Qwen3.8-27B-DFlash2-EXL3-5.0bpw --port 8000
```

The model type is read from the checkpoint (`qwen3_5`, `glm5_next` or `qwen4_exp`); anything else is a fatal error. Configuration (launch
file, `--set key=value`, `ENGINE_*` variables, debug switches) is described in [CONFIG.md](CONFIG.md); `configs/` has examples.
The server speaks the OpenAI chat API (`/v1/chat/completions`, `/v1/models`) and the Anthropic API (`/v1/messages`,
`/v1/messages/count_tokens`), so Claude Code can use it with `ANTHROPIC_BASE_URL=http://host:8000`. `/health`, `/metrics`
(JSON) and `/metrics/prometheus` are also served.

Useful options for agent workloads: a large context (`--set max_seq=262144 --set max_slots=2`) and the prefix cache
(`prefix_cache_gib`), which keeps the system prompt and earlier turns resident in pinned host RAM so a long conversation is
not prefilled again on every turn. The resumed tokens show up in the usage (OpenAI `prompt_tokens_details.cached_tokens`, Anthropic `cache_read_input_tokens`, with `input_tokens` excluding them).

Qwen extras:

* **Images**: the checkpoint's vision tower loads after the weights (`max_image_tokens`, default 1024 language-model tokens per
  image, 0 turns it off). Send OpenAI `image_url` parts (base64 `data:` URLs) or Anthropic `image` blocks (base64 sources),
  also inside Anthropic `tool_result` blocks; remote URLs are not fetched. GLM has its own tower (24 blocks, RMSNorm / SwiGLU, 2x2 conv merge); it is loaded last, into what is
  left of the arena. GLM with the DFlash2 drafter leaves about 5 MiB, so images stay off (the log says so and requests answer 400)
  unless memory is freed, for example `--set expert_cache=0.95` keeps 5% of the routed experts in host RAM.
* **GLM at 256K context, 2 slots, drafter and images** (4x 32 GB): the weights alone take 29.2 of the 31 GiB arena, so part of the routed
  experts live in host RAM and the latent cache is stored as FP8:
  `--set max_seq=262144 --set max_slots=2 --set kv_format=fp8 --set expert_cache=0.85 --speculative --draft-model-dir <DFlash2>`
  (arena 28.9 GiB, about 2 GiB spare). `expert_cache=0.88` loads but the vision tower no longer fits (images off); 0.90 and above run
  out of memory. Without the drafter and images, 0.85 with FP16 latent and 1 slot also fits; its 250K needle passes (prefill 208 tok/s,
  decode 14.8 tok/s at the 150 W cap). `kv_format` on GLM applies to the single latent tensor: `k8v4` stores it as 8-bit.
* **Qwen3.8-Flash-Next** (`qwen4_exp`, the 100 GB exl3 checkpoint): runs on all four GPUs only (experts are placed whole, 128 per rank), about 18 GiB of
  the arena per GPU at an 8K window. Plain decode is about 31 tok/s; `--speculative --set speculator=mtp` drafts with the checkpoint's own MTP head and
  roughly doubles it with identical greedy output. Full-attention layers use the model's QSA block selection past 2048 tokens of context; the 256K window
  with 2 slots fits (24.3 of 29.8 GiB per GPU). Images use the checkpoint's own tower. Prefix caching, tools and thinking behave as for Qwen.
* **MTP speculation**: `--speculative --set speculator=mtp` drafts with the checkpoint's own multi-token-prediction head and
  needs no draft model (about 1.7x plain decoding against 2.4x for DFlash2 on the same prompts, and it saves the drafter's VRAM).
* **Two GPUs per instance**: `--tp-size 2` runs one model on two GPUs (about 0.7x the speed of four); start a second server with
  `CUDA_VISIBLE_DEVICES=2,3` on the other pair for twice the aggregate throughput. GLM runs on 1 or 4 GPUs only.

## Tests

`tests/tools/gate.sh` is the regression gate (modes in its header): `static` (builds all three configurations, header and doc
checks), `fast` (tests that need no checkpoint), `models` (tests that load checkpoints from `ENGINE_MODEL_ROOT`, or from
`tests/local.env`), `quick` (build + kernel fingerprint + coherence, for moves and renames), `sass` (order-independent SASS
fingerprint), `bench` and `compare <tag>`. GPU tests run under `taskset -c 0-9`. See [TESTS.md](TESTS.md) for the inventory
and [CAPABILITIES.md](CAPABILITIES.md) for which module functions each backend implements.

Benchmark baselines depend on the GPU power cap: record one per power limit (`results/baseline_<tag>.txt`).
