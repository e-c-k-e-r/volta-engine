#pragma once

#include <string>

namespace engine::runtime {

// Runtime and model options. The only place that reads the environment is options.cpp: defaults, then the
// legacy EXL3_* overrides, then whatever the launcher (server config, tests) assigns before the first model loads.
// Numeric fields use <0 (or 0 for sizes) for "unset: use the model's default".
struct Options {
	int tp = -1;                  // rank count; <0 keeps the ClusterConfig value
	int arena_gib = -1;           // per-GPU arena override
	int vram_reserve_mib = -1;    // tp=1 arena auto-size: memory left outside the arena
	int max_seq = 0;              // context window (KV pool); 0 = model default
	int max_slots = 0;            // decode slots; 0 = model default
	std::string kv_format;        // "", fp16, fp8, k8v4, nvfp4
	double expert_cache = 1.0;    // GLM resident expert fraction (<1 enables the two-tier cache)
	int expert_adapt_every = 64;  // decode steps between expert residency rebalances (0 = never)
	int expert_swaps = -1;        // max expert swaps per rebalance; <0 = tier default
	bool draft_tp = true;         // false: drafter on rank 0 only
	int lm_shard = -1;            // Qwen lm_head sharding; <0 = on when tp > 1
	bool moe_tp = true;           // GLM sharded routed experts
	bool dsa = true;              // GLM sparse attention indexer
	bool mla_latent = true;       // GLM latent KV path
	bool attn_tc = true;          // Qwen tensor-core decode attention
	bool expert_log = false;      // debug
	bool dsa_diag = false;        // debug
};

// Populated from the environment on first use.
Options& options();

} // namespace engine::runtime
