# Getting started

## Build

CUDA 12.x (13+ drops SM_70) and CMake 3.24+ with Ninja. The CUDA build is the primary one:

```
cmake -S . -B build -G Ninja && cmake --build build
```

### Build time

The EXL3 GEMM kernels are the bulk of a from-scratch CUDA build: each (bit width, codebook) pair is its own generated translation unit, so the
build runs them in parallel. Two options leave out kernels a setup does not use:

* `ENGINE_EXL3_PRUNE` (`AUTO`, `ON`, `OFF`) skips the tile shapes the built architectures never select: shape 1 below SM_90, shape 4 at 7 and 8 bits
  below SM_89. `AUTO`, the default, prunes when SM_70 is the only target (the default target; `native` is resolved to the installed
  GPUs) and builds everything otherwise; `ON` prunes for whatever is built. The selection steps down to a shape that exists
* `ENGINE_EXL3_CODEBOOKS` (default `0;1;2`: 3inst, mcg, mul1; the codebook is in the checkpoint's `config.json`) lists the codebooks to build.
  `-DENGINE_EXL3_CODEBOOKS=2` builds `mul1` only; a checkpoint using a codebook left out stops at its first GEMM with a message naming the option

### Other GPUs (SM_86, SM_89)

The CUDA build targets SM_70 (V100) by default. Ampere (SM_86: RTX 30xx, A10, A40) and Ada (SM_89: RTX 40xx, L4, L40) build from the
same sources; pick the targets with `CMAKE_CUDA_ARCHITECTURES` (several at once give one binary for all of them):

```
cmake -S . -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES="70;86;89"   # or 86, 89, native
```

CUDA 12.x builds all three; CUDA 13 builds 86 / 89 only (configuration fails if 70 is in the list). On SM_86 / SM_89:
* the EXL3 GEMM / GEMV and the cooperative MoE decode kernel take their native `mma.m16n8k16` / `ldmatrix` / `cp.async` paths (the
  upstream exllamav3 ones). Fallbacks if a native path misbehaves on a card: `-DENGINE_EXL3_FORCE_VOLTA_MMA=ON` keeps the Volta
  `m8n8k4` GEMM inner loop everywhere, `-DCMAKE_CUDA_FLAGS=-DMOE_COOP_M8N8K4=1` the Volta MoE tile
* the NVFP4 skinny / MoE kernels and the prefill attention kernels (dense, MLA latent, sparse latent) have `mma.m16n8k16` paths
  over the same tiles and shared-memory staging as their Volta `m8n8k4` forms; the vision tower takes its WMMA attention kernel. No
  `m8n8k4` (which SM_80+ only emulates, at ~80 instructions each) runs there. The fragment maps are checked on the CPU by
  `test_mma_frag`; numerics match the Volta path up to FP32 summation order
* the EXL3 launch choices follow the card: GEMM tile shapes, GEMV configurations and the int8 GEMV's highest bit rate use upstream
  exllamav3's per-architecture tables on SM_86 / SM_89 (the V100 ones are this engine's own measurements); the GEMV kernels for 5 and
  6 bpw are V100-only, as upstream runs those rates through the GEMM
* the SM count is read from the device at startup (cooperative / persistent grids are sized to it). `-DENGINE_SM_COUNT=N` caps it,
  `-DENGINE_MAX_SMEM=bytes` changes the opt-in shared memory the big kernels request (default 98304, allowed by all three)
* startup prints each GPU's architecture and SM count, and warns when the build has no code for a card
* tensor parallelism needs peer access between the cards (NVLink, or PCIe P2P where the driver allows it; GeForce 40xx cards do
  not), so a single consumer card runs `tp` 1

AMD RDNA GPUs (RX 7000 / 9000) build with `-DENGINE_BACKEND=ROCM`: see [ROCM.md](ROCM.md).

CPU and Vulkan builds are separate build trees (no CUDA toolchain needed):

```
cmake -S . -B build_cpu -G Ninja -DENGINE_BACKEND=CPU
cmake -S . -B build_vk  -G Ninja -DENGINE_BACKEND=VULKAN
```

The CPU and Vulkan builds run the portable parts on reference kernels, so they are slow by design:
* the CPU build compiles for the build host (`-march=native`: AVX2, FMA, F16C, about 2x on the reference kernels; `-DENGINE_CPU_NATIVE=OFF` for a portable binary) and runs Qwen and Flash-Next on one rank
  * Flash-Next needs ~66 GiB of host memory, and answers a chat request with the CUDA engine's text (see [TESTS.md](TESTS.md) for timings)
* the Vulkan build runs Qwen on one device, or tensor-parallel over several (`tp` 2 or 4: one Vulkan device per rank, the all-reduce goes through host memory, execution is eager, about 2 s per Qwen token); Flash-Next needs the four devices. Models with a routed MoE need `ENGINE_DEBUG_NO_GRAPHS=1` on one device (the MoE reads its routes back on the host; with `tp` above 1 graphs are off anyway)
* the DFlash2 speculator is CUDA only, and the MTP head has not been run on Vulkan
* MiniMax-H3 is CUDA only

NCCL is optional (CMake looks for it): it enables tensor parallelism for MiniMax-H3, see [video.md](video.md).

A full CUDA build takes a while (the trellis kernels are template heavy); incremental builds are quick.

## Run the server

```
taskset -c 0-9 build/volta_server --model-dir /path/to/Qwen3.8-27B-exl3 \
	--speculative --draft-model-dir /path/to/Qwen3.8-27B-DFlash2-EXL3-5.0bpw --port 8000
```

The model type is read from the checkpoint (`qwen3_5`, `glm5_next`, `qwen4_exp` or `deepseek_v41`; a MiniMax-H3 pipeline directory starts the video service, see [video.md](video.md)); anything else is a fatal error. Configuration (launch
file, `--set key=value`, `ENGINE_*` variables, debug switches) is described in [CONFIG.md](CONFIG.md); `configs/` has examples.
The server speaks the OpenAI chat API (`/v1/chat/completions`, `/v1/models`) and the Anthropic API (`/v1/messages`,
`/v1/messages/count_tokens`), so Claude Code can use it with `ANTHROPIC_BASE_URL=http://host:8000`. `/health`, `/metrics`
(JSON) and `/metrics/prometheus` are also served. `/monitor` is a live page (Monokai or Tomorrow Night theme): throughput, slots with context fill, per-GPU power / clocks / temperature / throttle reasons (NVML, loaded at run time), token-gap and TTFT percentiles, speculation, prefix cache, recent requests, and for MoE models a layer-by-expert routing heat map with the load per rank. It reads `/api/stats` and `/api/experts` (JSON); `expert_stats=false` turns the routing counters off.

Useful options for agent workloads: a large context (`--set max_seq=262144 --set max_slots=2`) and the prefix cache
(`prefix_cache_gib`), which keeps the system prompt and earlier turns resident in pinned host RAM so a long conversation is
not prefilled again on every turn. The resumed tokens show up in the usage (OpenAI `prompt_tokens_details.cached_tokens`, Anthropic `cache_read_input_tokens`, with `input_tokens` excluding them).

Qwen extras:

* **Images**: the checkpoint's vision tower loads after the weights (`max_image_tokens`, default 1024 language-model tokens per
  image, 0 turns it off). Send OpenAI `image_url` parts (base64 `data:` URLs) or Anthropic `image` blocks (base64 sources),
  also inside Anthropic `tool_result` blocks; remote URLs are not fetched. GLM has its own tower (24 blocks, RMSNorm / SwiGLU, 2x2 conv merge); it is loaded last, into what is
  left of the arena. GLM with the DFlash2 drafter leaves about 5 MiB, so images stay off (the log says so and requests answer 400)
  unless memory is freed, for example `--set expert_cache=0.95` keeps 5% of the routed experts in host RAM.
* **GLM at 256K context, 2 slots and the drafter** (4x 32 GB): the latent cache is stored in 4 bits and the two slots share one pool of 262144
  tokens (`kv_pool`, below), and every routed expert fits on the GPUs: `--set max_seq=262144 --set max_slots=2 --set kv_format=nvfp4
  --set kv_pool=262144 --set expert_cache=1.0 --speculative --draft-model-dir <DFlash2>` (images off). Per GPU the arena holds 1.7 GiB of
  buffers and caches (the latent cache 0.8 of them), 28.6 GiB of weights and the 0.6 GiB drafter, leaving 0.08 GiB; `--set mem_log=true` prints this
  as `[mem]` lines at startup. The shared expert and the dense MLP of the first three layers are cut over the ranks (`shared_slice`), which saves 0.6 GiB a GPU, and the attention workspace is sized
  for the 2K keys a row attends to once the indexer selects, not for the whole context. Verified at 200K (a 200K prompt: first token after 248 s;
  two 128K prompts do not fit the pool together, so the second waits for the first) with GPU memory at 32.4 of 32.5 GiB. `configs/glm_256k.json` uses `expert_cache=auto` instead, which leaves a margin
  on a different machine; each point of `expert_cache` costs 0.26 GiB, and the FP8 cache (`kv_format=fp8`, 1.2 GiB more) needs it below 1.
  `--set expert_cache=auto` picks the fraction itself: the largest the arena holds after the other weights, the drafter (its file size) and a 0.25 GiB
  margin; the first MoE layer starts from a conservative estimate and each later layer re-estimates from what was actually loaded, so the last
  layers take up the slack (the log prints the plan and the resident range). GLM only; it uses the expert tier on any rank count.
  Prefill chunks (above 16 rows) copy each layer's cold experts once into device scratch, a layer ahead of the compute (`expert_stage`: auto = on, except for Flash-Next's
  trellis experts, where the persistent kernel's tiles read each cold expert about once and staging measured 15-25% slower at 256-row chunks and level at 512;
  2 x the cold experts of a layer, 126 MiB at 90% resident), instead of reading them over PCIe from every row tile: 4096 tokens in chunks of
  1024 ran at 534 tok/s against 306 at `expert_cache=0.9`, and 721 against 589 at 0.97.
  Decode keeps the fused MoE kernels while at least 75% of the experts are resident (below that the persistent kernel reads the cold experts a
  little faster): at 99% resident a token takes 19.0 ms against 15.5 ms with every expert resident (16.3 ms with the PCIe reads taken out; the
  rest is cold-expert traffic, 2.7 ms for one row and 7.6 ms for an 8-row verify), and 28.5 ms before.
  `--set kv_pool=262144` lets the two slots share one pool of that many tokens instead of owning 256K each (0.95 GiB per GPU less at the 4-bit
  cache, which the expert cache or the drafter can use): a request starts once the pool holds its prompt, otherwise it queues, and one that
  grows past the pool ends with finish reason `length`.
  The 4-bit cache moves the perplexity of GLM by +0.008 NLL against FP16 on 3000 tokens (FP8: +0.001). `kv_format` on GLM applies to the
  single latent tensor: `k8v4` stores it as 8-bit. With images on, the vision tower takes its memory from the same arena.
* **Qwen3.8-Flash-Next** (`qwen4_exp`, the 100 GB exl3 checkpoint): runs on all four GPUs only (experts are placed whole, 128 per rank), about 18 GiB of
  the arena per GPU at an 8K window. Plain decode is about 31 tok/s; `--speculative --set speculator=mtp` drafts with the checkpoint's own MTP head and
  roughly doubles it with identical greedy output. Full-attention layers use the model's QSA block selection past 2048 tokens of context; the 256K window
  with 2 slots fits (24.3 of 29.8 GiB per GPU). Images use the checkpoint's own tower. Prefix caching, tools and thinking behave as for Qwen.
* **DFlash2 with several slots (GLM)**: every slot keeps its own drafter context and tap ring (80 MiB of taps and 10 MiB of drafter K/V per slot on each
  GPU), so concurrent requests all speculate. With two slots a round drafts each slot's block in turn and verifies both in one target forward, which reads the
  weights and the experts the rows share once (`spec_joint`, on by default for DFlash2; `spec_joint=0` runs one slot's round per step, `draft_per_slot=false` goes
  back to one context that the newest request takes over). The speculative verify records each KDA row's inputs and replays the accepted rows
  instead of keeping a 36 MiB state snapshot per row. Two concurrent streams of repetitive text: 149 tok/s aggregate with joint rounds, 137 interleaved,
  122 with one speculating slot and 97 for plain batched decode; on typical agent text (about 3 tokens a round) the controller keeps plain batched decode
  with two streams, which is the faster choice there.
* **MTP speculation**: `--speculative --set speculator=mtp` drafts with the checkpoint's own multi-token-prediction head and
  needs no draft model (about 1.7x plain decoding against 2.4x for DFlash2 on the same prompts, and it saves the drafter's VRAM). GLM-5.3-Flash takes its head from the
  NVFP4 checkpoint (`nvfp4_dense_src`, quantized to NVFP4 at load; example `configs/glm_mtp.json`) and runs as fast as with the DFlash2 drafter.
* **NVFP4 weights** (modelopt NVFP4 checkpoints of the same models; details and numbers in [FLASH_NEXT.md](FLASH_NEXT.md), examples `configs/flash_next_nvfp4.json`,
  `configs/glm_nvfp4_dense.json` for the 12K window and `configs/glm_256k.json` for 256K with two slots; the Qwen NVFP4 checkpoint runs as it is, `configs/qwen_nvfp4.json`): the EXL3 directory stays the model directory and supplies everything not replaced. Flash-Next:
  `nvfp4_experts=<NVFP4 dir>` takes the routed experts from the NVFP4 checkpoint (and is also the BF16 source for dense tensors), `nvfp4_dense=<groups>`
  (`gdn,attn,idx,head,shared_gate,shared_up` recommended; `all` adds `shared_down`, which costs perplexity) quantizes those dense groups at load, and
  `nvfp4_dense_dir=<dir>` loads calibrated tensors made by `tests/tools/nvfp4_ldlq.py` instead of round-to-nearest. GLM-5.3-Flash: `nvfp4_dense_src=<NVFP4 dir>` plus
  `nvfp4_dense=kda,mla,idx,head,dense,shared` (experts stay EXL3: the NVFP4 experts do not fit in VRAM and the offload penalty rules them out). Calibration: run
  `ENGINE_HESSIAN_DIR=<dir> test_ppl --flash|--glm --calibrate --file <text>` (GLM: `ENGINE_HESSIAN_LAYERS=lo-hi`, a third of the layers per pass) and then
  `nvfp4_ldlq.py --model flash|glm`. Speed figures there are indicative (150 W cap, single runs).
* **DeepSeek-V4.1-Flash** (`deepseek_v41`, EXL3 checkpoints of the sudoingX / 0xSero layout, for example `DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw`;
  example `configs/deepseek_v41.json`): the Engram tables are not in those repositories, so `model-00047-of-00048.safetensors` and
  `model-00048-of-00048.safetensors` of `deepseek-ai/DeepSeek-V4.1-Flash-1017` go into the model directory, and
  `python tests/tools/dsv41/engram_hash_tables.py <model dir>` (a venv with `torch`, `transformers` and `sympy`) writes the hashing tables the
  engine reads, `engram_hash.bin`, once per directory. The routed experts that do not fit the arena stay in pinned host memory and are read in
  place (62% resident for the 2.7 bpw checkpoint on four 32 GB GPUs). Its routing is concentrated (on agent text the hottest 10% of a layer's
  experts take half of the routes) and specialized by domain, so which experts are resident decides the speed: with `--set expert_heat_file=<path>`
  the route counts are saved at exit and the next start keeps the hottest experts on the device and balances the heat over the ranks. Counts from five
  recorded Claude Code turns put 95% of the routes of five other turns on resident experts (62% without), and a held-out turn decoded at 59 ms a token
  instead of 196 and prefilled at 188 tok/s instead of 84; a placement made on one kind of text slows down a very different one (a story: 46% resident). While serving, every `expert_adapt_every` steps the experts routed to most often move into device memory in exchange for the least used resident ones of the same bit width (a story from that placement: about 118 -> 75 ms a token over 400 tokens). The prompt format is the checkpoint's `encoding/encoding.py`, built in
  (no chat template): thinking and reasoning effort (low 50, medium 60, high 75, max 100), tools as DSML tool calls. The prefix cache works (a snapshot holds the
  compressed KV and index keys of its positions plus the last 128 window rows of each layer). Not yet: speculation, images.
* **Two GPUs per instance**: `--tp-size 2` runs one model on two GPUs (about 0.7x the speed of four); start a second server with
  `CUDA_VISIBLE_DEVICES=2,3` on the other pair for twice the aggregate throughput. GLM runs on 1, 2 or 4 GPUs (on two, the experts need the expert tier, `expert_cache` below 1, to fit).

## Tests

`tests/tools/gate.sh` is the regression gate (modes in its header): `static` (builds all three configurations, header and doc
checks), `fast` (tests that need no checkpoint), `models` (tests that load checkpoints from `ENGINE_MODEL_ROOT`, or from
`tests/local.env`), `quick` (build + kernel fingerprint + coherence, for moves and renames), `sass` (order-independent SASS
fingerprint), `bench` and `compare <tag>`. GPU tests run under `taskset -c 0-9`. See [TESTS.md](TESTS.md) for the inventory
and [CAPABILITIES.md](CAPABILITIES.md) for which module functions each backend implements.

Benchmark baselines depend on the GPU power cap: record one per power limit (`results/baseline_<tag>.txt`).
