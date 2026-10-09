# Qwen3.8-Flash-Next (`qwen4_exp`) port plan

Checkpoint: `Swift-Qwen3.8-Flash-Next-exl3-4.00bpw` (arch `Qwen4ExpForConditionalGeneration`). Reference: `~/Programs/exllamav3`
(`architecture/qwen4_exp.py`, `modules/qsa_indexer.py`, `modules/ple.py`, `modules/hyperconnections.py::GatedResidual`,
`architecture/qwen4_exp_mtp.py`). The reference runs on SM70 (single process, autosplit over 4 GPUs, loads in ~66 s) and answers
sensibly, so it serves as the parity oracle. It marks the architecture `supports_tp: False`; the multi-GPU split is ours to build.

## Architecture facts

| Item | Value |
|---|---|
| Layers | 48: every 4th is full attention (QSA), the rest GDN linear attention |
| Hidden | 2560, vocab 248320, context 262144, untied head, **no final norm** |
| Full attention | 24 Q heads, 2 KV heads, head_dim 256, partial rotary 0.25, mrope interleaved (11/11/10), q/k RMSNorm (+1), interleaved output gate |
| GDN | 16 K heads x 128, 48 V heads x 128, conv kernel 4, **sigmoid-gated** output RMSNorm (Qwen3.8-27B uses silu) |
| QSA indexer | 4 heads, 1 KV head, head_dim 128, compress ratio 4, token budget 2048 (a DSA-like block selector) |
| MoE | 512 experts, top 10, intermediate 640, plus a shared expert (640) with a sigmoid gate; routed weights trellis-quantized |
| Residual | 4 fp32 streams, **gated residual** (low rank, per-channel) instead of input/post norms; see below |
| PLE | one layer (id 2): hashed 3-gram embedding, 16 heads x 160, trellis-quantized table of 128 shards x 2.5M rows x 61 int16 (39 GB), causal dilated conv, grouped RMSNorms, key/value projections into the streams |
| MTP | 1 hybrid layer (full attention), no dedicated embeddings |
| Vision | tower identical to Qwen3.5's (we already run it) |
| Size | trunk ~63 GB + n-gram table 39 GB; host RAM is 128 GB |

Gated residual (per site, replaces norm + residual add): per-stream grouped RMSNorm (weight zero-init, applied as 1 + w), a low-rank
sigmoid gate over the normed stack picks each stream's share of the elementwise mean that feeds the sublayer, and the sublayer
output is added back to the raw streams with a per-stream scalar `2*sigmoid` gate. No Sinkhorn, no combine matrix (cheaper than
GLM's mHC, same stream layout).

## What exists in the engine

| Need | Reuse |
|---|---|
| GDN recurrence, conv, projections | `modules/gdn` (Qwen3.8-27B); add the sigmoid output gate variant |
| Full attention, partial rotary, M-RoPE, FP16/FP8/K8V4/NVFP4 KV | Qwen path (`attention.cu`, `kv_quant.cuh`) |
| Sparse block selection with pooled keys | GLM DSA (`dsa.cu`): different shapes and scoring, same structure |
| Trellis GEMM, MoE dispatch, expert tier | GLM (`exl3_moe_coop`, `moe.cu`); needs 512 experts, top 10, hidden 2560 |
| Four-stream residual | GLM mHC kernels: different mixing, same layout and tap hooks |
| Vision tower, image cache, MTP speculator | C5 work |
| Prefix snapshots (GDN state + KV rows) | Qwen/GLM snapshot code |

## Work breakdown (milestones, each parity-checked against the reference)

1. **Config and loader**: model type `qwen4_exp`, tensor map, trunk placement over 4 GPUs.
2. **Block parity tools**: reference activation dumps for a fixed prompt (embedding, each site, each block), engine-side trace like `ENGINE_DEBUG_TRACE_LAYERS`.
3. **Gated residual + GDN block** (layers 0-2 minus MoE): kernels and a block parity test.
4. **MoE at 512 experts / top 10**: router (assert is 288 today), dispatch, fused kernels, shared expert + gate; parity.
5. **QSA attention layer**: indexer, block selection, sparse GQA kernel; parity against the reference at several lengths.
6. **PLE layer**: host-resident n-gram table (mmap), trellis row decode, hashing, conv state in snapshots.
7. **Whole model**: TP=4 split (GDN heads, KV heads 2 -> replicate or TP2 groups, experts sharded), logits parity, greedy text, needle.
8. **Serving**: chat template, tool calls (Qwen family format), prefix cache, 256K context and batch 2, images, MTP.
9. **Performance and VRAM**: decode graphs, speculation, `kv_format`, SASS/bench baselines.

## Risks

- **No TP reference**: the sharding of 2 KV heads over 4 ranks and of the GDN state needs a design of our own.
- **PLE table**: 16 random 122-byte row reads per token from a 39 GB mapping; fine from page cache, a cold start reads from disk.
- **Router sensitivity**: like GLM, expect expert-set flips between numerically different paths; compare with route replay, not exact logits.
- **Scale of the port**: comparable to the GLM port (new block structure, new sparse attention, wider MoE).
