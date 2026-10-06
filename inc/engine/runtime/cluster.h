#pragma once

#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/core/comm.h"
#include "engine/core/batch.h"
#include "engine/models/model.h"
#include "engine/runtime/rank_pool.h"
#include "engine/spec/dflash.h"
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

namespace engine::runtime {

struct ClusterConfig {
	std::string model_type;      // "qwen38" | "glm53"
	std::string model_dir;
	std::string draft_model_dir; // empty => no speculator
	size_t arena_bytes = 0;      // 0 => per-model default
};

// Outcome of one speculative round on a single slot.
struct SpecResult {
	int proposed = 0;          // draft tokens submitted for verification
	int accepted = 0;          // draft tokens matching the target
	int emitted[pod::MAX_BATCH] = {}; // tokens committed this round (accepted drafts + correction/bonus)
	int emitted_count = 0;
	int next_pos = 0;          // position of the next anchor (== tokens committed to KV)
};

// TP=4 model instance: per-rank arenas, streams, NVLink mesh, pipelines and a persistent
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
	int prefill( const int* tokens, int count, int slot, int chunk = pod::MAX_BATCH );

	// Greedy argmax of the last forward's first `count` rows (rank 0).
	void sample( int* out, int count );
	int sample_one();

	void reset_slot( int slot );
	void checkpoint( int slot );
	void rollback( int slot );

	// One DFlash2 round: anchor forward at pos, K-token draft, batched verify, state repair.
	// context = all committed tokens (prompt + generated, ending with `anchor`).
	SpecResult speculate( int slot, int anchor, int pos, const std::vector<int>& context, float conf_threshold = 0.25f );

	bool has_speculator() const { return has_spec_; }
	engine::models::ModelPipeline& model( int rank = 0 ) { return *models_[rank]; }
	const engine::models::ModelMetadata& metadata() const { return models_[0]->get_metadata(); }
	cudaStream_t stream( int rank ) const { return streams_[rank]; }
	const pod::MemoryArena& arena( int rank ) const { return arenas_[rank]; }
	bool is_eos( int token ) const;

private:
	bool created_ = false;
	bool initialized_ = false;
	bool has_spec_ = false;
	std::unique_ptr<engine::models::ModelPipeline> models_[ENGINE_NUM_GPUS];
	pod::MemoryArena arenas_[ENGINE_NUM_GPUS];
	engine::comm::CollectiveRing rings_[ENGINE_NUM_GPUS];
	cudaStream_t streams_[ENGINE_NUM_GPUS] = {};
	void* staging_ptrs_[ENGINE_NUM_GPUS] = {};
	void* flag_ptrs_[ENGINE_NUM_GPUS] = {};
	engine::spec::DFlashSpeculator speculator_;
	RankPool pool_;
};

} // namespace engine::runtime
