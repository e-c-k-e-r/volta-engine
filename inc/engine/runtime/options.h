#pragma once

#include <string>

namespace engine::runtime {

// Runtime and model options. The only place that reads the environment is options.cpp: defaults, then the
// legacy EXL3_* environment (compat), then whatever the launcher (config file, CLI, tests) assigns before the first model loads.
// Numeric fields use <0 (or 0 for sizes) for "unset: use the model's default".
struct Options {
	int tp = -1;                  // rank count; <0 keeps the ClusterConfig value
	int arena_gib = -1;           // per-GPU arena override
	int vram_reserve_mib = -1;    // tp=1 arena auto-size: memory left outside the arena
	int max_seq = 0;              // context window (KV pool); 0 = model default
	int max_slots = 0;            // decode slots; 0 = model default
	int prefix_cache_gib = 16;    // pinned host RAM for the prefix cache (sequence snapshots); 0 = off
	std::string kv_format;        // "", fp16, fp8, k8v4, nvfp4
	double expert_cache = 1.0;    // GLM resident expert fraction (<1 enables the two-tier cache)
	int expert_adapt_every = 64;  // decode steps between expert residency rebalances (0 = never)
	int expert_swaps = -1;        // max expert swaps per rebalance; <0 = tier default
	bool draft_tp = true;         // false: drafter on rank 0 only
	int lm_shard = -1;            // Qwen lm_head sharding; <0 = on when tp > 1
	bool moe_tp = true;           // GLM sharded routed experts
	bool dsa = true;              // GLM sparse attention indexer
	bool mla_latent = true;       // GLM latent KV path
	bool spec_ctrl = true;        // adaptive speculation controller (false: always speculate)
	// Debug switches (environment: ENGINE_DEBUG_*)
	bool expert_log = false;      // ENGINE_DEBUG_EXPERT_LOG: per-rebalance expert tier stats
	bool dsa_diag = false;        // ENGINE_DEBUG_DSA: indexer diagnostics
	bool spec_ctrl_log = false;   // ENGINE_DEBUG_SPEC_CTRL: controller decisions
	bool no_graphs = false;       // ENGINE_DEBUG_NO_GRAPHS: eager execution, no CUDA graphs
	std::string dump_requests;    // ENGINE_DEBUG_DUMP_REQUESTS=<dir>: write every API request body to <dir>/req_<id>.json
	int dump_draft_round = -1;    // ENGINE_DEBUG_DUMP_DRAFT=<round>: write /tmp/draft_dump.bin
};

// Populated from the environment on first use.
Options& options();

// Assigns one option by field name (launch config "options" object, --set key=value). False for an unknown key
// or an unparsable value; the caller treats that as a fatal configuration error.
bool set_option( const std::string& key, const std::string& value );

} // namespace engine::runtime
