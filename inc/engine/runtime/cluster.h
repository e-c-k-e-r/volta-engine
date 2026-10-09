#pragma once

#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/core/comm.h"
#include "engine/core/batch.h"
#include "engine/models/model.h"
#include "engine/runtime/rank_pool.h"
#include "engine/runtime/dflash2_runner.h"
#include "engine/runtime/sampling.h"
#include <functional>
#include "engine/backend/types.h"
#include <map>
#include <memory>
#include <mutex>
#include <vector>
#include <string>
#include <vector>

namespace engine::runtime {

struct ClusterConfig {
	std::string model_type;      // "qwen3_5" | "glm5_next"
	std::string model_dir;
	std::string draft_model_dir; // empty => no speculator
	size_t arena_bytes = 0;      // 0 => per-model default
	int tp = ENGINE_NUM_GPUS;    // ranks: 1 (single device, no collective) or ENGINE_NUM_GPUS
};

// TP model instance (1 or ENGINE_NUM_GPUS ranks): per-rank arenas, streams, NVLink mesh, pipelines and a persistent
// pinned worker pool. Shared by the Generator and every test / benchmark driver.
class ModelCluster {
public:
	ModelCluster() = default;
	~ModelCluster() { shutdown(); }

	bool init( const ClusterConfig& cfg );
	void shutdown();

	// Embed + forward on all ranks (Mode A causal block or Mode B batched rows), synchronized.
	void forward( const int* tokens, const pod::BatchRows& rows );
	void forward( const int* tokens, int count, int pos, int slot );

	// Chunked causal prefill of `count` tokens starting at pos 0 into `slot`.
	// Returns the greedy prediction for the token following the prompt.
	// last_cands (optional) receives the final row's top-16 candidates so the caller can sample the first token.
	// chunk 0 = the model's largest row count (128 for Qwen). Chunks above MAX_BATCH rows run eagerly and skip the
	// LM head except for the final row of the final chunk.
	// pos0: position of tokens[0] (a prefill that resumes from a restored prefix starts there)
	int prefill( const int* tokens, int count, int slot, int chunk = 0, Candidates* last_cands = nullptr, int pos0 = 0 );

	// Sequence snapshots (prefix cache): one pinned host buffer per rank holding the slot's recurrent state and the KV
	// rows [0, tokens). Buffers are recycled through a pool, so steady-state saves allocate nothing.
	struct SeqSnapshot {
		int from = 0;                      // the snapshot holds KV rows [from, tokens); a chain links from = the parent's tokens
		int tokens = 0;
		bool with_taps = false;            // carries the drafter tap rows: a restore keeps speculating
		size_t bytes = 0;                  // per rank
		void* host[ENGINE_NUM_GPUS] = {};
	};
	int snapshot_align() const { return models_[0]->seq_align(); }
	size_t snapshot_bytes( int from, int tokens, bool with_taps = false ) const;    // per rank; 0 when the model cannot snapshot
	bool save_snapshot( int slot, int from, int tokens, SeqSnapshot* snap, bool with_taps = true );
	// Writes the snapshot into the slot: its KV rows, and unless kv_only the recurrent state and tap rows (the slot then
	// claims the drafter, which re-reads its window from the restored taps). A chain is restored ancestors first (kv_only),
	// then the node being resumed from.
	bool load_snapshot( int slot, const SeqSnapshot& snap, bool kv_only = false );
	void free_snapshot( SeqSnapshot* snap );

	// Greedy argmax of the last forward's first `count` rows (rank 0).
	void sample( int* out, int count );

	// Top-16 candidates (merged across the vocabulary shards, descending) for each of the last forward's first
	// `count` rows, for temperature / top-p sampling. Synchronizes.
	void sample_topk( int count, Candidates* out );

	// Full-vocabulary FP16 logits of the last forward's first `rows` rows, copied to host
	// ([rows, vocab_size], row-major). Reassembles the TP shards when the LM head is sharded.
	// Diagnostic / test path: synchronizes and issues per-row peer copies.
	void read_logits( int rows, backend::half_t* host_dst );
	int sample_one();
	// Adapts the routed-expert residency (ENGINE_EXPERT_CACHE) to the routing seen since the last call; false when no model tiers
	bool rebalance_experts();

	// Pre-capture decode graphs for row counts 1..max_m (two dummy forwards each on `slot`,
	// which is reset afterwards) so serving never pays eager warm-up or capture mid-stream.
	void warm_graphs( int max_m, int slot = 0 );

	void reset_slot( int slot );
	void checkpoint( int slot );
	void rollback( int slot );
	void set_spec_snapshot( bool on );
	void commit_spec_row( int slot, int row );

	// Speculative decoding (DFlash2). The round itself lives in DFlash2Runner.
	bool has_speculator() const { return runner_ != nullptr; }
	DFlash2Runner& spec() { return *runner_; }
	bool spec_owns( int slot ) const { return runner_ && runner_->owns( slot ); }
	SpecResult speculate( int slot, int anchor, int pos, float depth_threshold = -1.0f,
		const SamplingParams* sp = nullptr, Rng* rng = nullptr ) {
		return runner_->speculate( slot, anchor, pos, depth_threshold, sp, rng );
	}

	// Runs fn( rank ) on every rank's pinned worker thread and waits for all of them.
	void run_ranks( const std::function<void( int )>& fn ) { pool_.run( fn ); }
	engine::models::ModelPipeline& model( int rank = 0 ) { return *models_[rank]; }
	const engine::models::ModelMetadata& metadata() const { return models_[0]->get_metadata(); }
	backend::stream_t stream( int rank ) const { return streams_[rank]; }
	engine::comm::CollectiveRing* ring( int rank ) { return &rings_[rank]; }
	const pod::MemoryArena& arena( int rank ) const { return arenas_[rank]; }
	pod::MemoryArena& arena( int rank ) { return arenas_[rank]; }
	bool is_eos( int token ) const;
	int tp() const { return tp_; }

private:
	bool created_ = false;
	bool initialized_ = false;
	int tp_ = ENGINE_NUM_GPUS;
	std::unique_ptr<engine::models::ModelPipeline> models_[ENGINE_NUM_GPUS];
	pod::MemoryArena arenas_[ENGINE_NUM_GPUS] = {};
	engine::comm::CollectiveRing rings_[ENGINE_NUM_GPUS];
	backend::stream_t streams_[ENGINE_NUM_GPUS] = {};
	void* staging_ptrs_[ENGINE_NUM_GPUS] = {};
	void* flag_ptrs_[ENGINE_NUM_GPUS] = {};
	std::unique_ptr<DFlash2Runner> runner_;
	std::map<size_t, std::vector<void*>> snapshot_free_[ENGINE_NUM_GPUS]; // pinned buffers by rounded size
	std::mutex snapshot_mtx_;
	RankPool pool_;
};

} // namespace engine::runtime
