#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/arena.h"
#include "engine/core/comm.h"
#include "engine/core/batch.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <string>
#include <memory>

namespace engine::models {

struct ModelMetadata {
	std::string model_name;
	int hidden_size = 0;
	int vocab_size = 0;
	int num_layers = 0;
	int num_heads = 0;
	int num_kv_heads = 0;
	int head_dim = 0;
	int max_seq_len = 0;
	int tp_size = 0;
	int max_slots = 1;    // independent request slots (KV + recurrent state) resident on device
	int max_rows = 1;     // max rows per forward pass (<= pod::MAX_BATCH)
	int eos_ids[4] = { -1, -1, -1, -1 };
};

// Unified per-rank model contract. One instance per GPU; all ranks are driven in lockstep.
//
// Row semantics (see pod::BatchRows):
//   Mode A (speculative verify / chunked prefill): forward( M, pos, slot ) evaluates the causal
//          candidate block [pos, pos + M) of a single slot.
//   Mode B (concurrent batching): forward( rows ) evaluates one token per independent slot,
//          rows.slot[b] / rows.pos[b] mapped per row.
class ModelPipeline {
public:
	virtual ~ModelPipeline() = default;

	// 1. Lifecycle & weight loading
	virtual bool init( int device_id, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring ) = 0;
	virtual bool load_weights( const char* model_dir, cudaStream_t stream = nullptr ) = 0;

	// 2. Token embedding: token_ids is a host array of `count` ids for rows 0..count-1
	virtual void embed_tokens( const int* token_ids, int count, cudaStream_t stream = nullptr ) = 0;

	// 3. Forward over embedded rows
	virtual void forward( const pod::BatchRows& rows, cudaStream_t stream = nullptr ) = 0;
	void forward( int batch_size, int pos, int slot_id = 0, cudaStream_t stream = nullptr ) {
		forward( pod::BatchRows::causal( batch_size, pos, slot_id ), stream );
	}

	// 4. Greedy sampling of the last forward's first `count` rows into host out_preds (rank 0)
	virtual void sample_tokens( int* out_preds, int count, cudaStream_t stream = nullptr ) = 0;

	// 4b. Vocabulary-parallel sampling: each rank picks the best token inside its own vocabulary
	// shard and reports (token, logit) into caller-owned host arrays. Returns false when the model's
	// logits are not sharded, in which case the caller falls back to sample_tokens on rank 0.
	virtual bool sample_local( int* out_tokens, float* out_vals, int count, cudaStream_t stream = nullptr ) {
		(void) out_tokens;
		(void) out_vals;
		(void) count;
		(void) stream;
		return false;
	}

	// 4c. This rank's local logits shard, for a vocabulary-parallel head. Null when unsharded.
	virtual const half* shard_logits_ptr() const { return nullptr; }
	virtual int shard_logits_cols() const { return 0; }

	// 4d. Materialize the full-vocabulary logits on rank 0 from every rank's shard. Only consumers that
	// need the whole distribution (the DFlash2 drafter) pay for this; sampling uses sample_local.
	// The caller must have synchronized every rank's stream first. Returns logits() when unsharded.
	virtual const pod::TensorView& gather_logits(
		int rows,
		const half* const* peer_shard_ptrs,
		int peer_shard_cols,
		cudaStream_t stream = nullptr
	) {
		(void) rows;
		(void) peer_shard_ptrs;
		(void) peer_shard_cols;
		(void) stream;
		return logits();
	}

	// 5. Speculative state management: snapshot / restore the recurrent state of one slot.
	//    KV cache entries need no snapshot: rejected positions are overwritten by the next forward.
	virtual void checkpoint_state( int slot_id, cudaStream_t stream = nullptr ) = 0;
	virtual void rollback_state( int slot_id, cudaStream_t stream = nullptr ) = 0;
	virtual void reset_state( int slot_id = 0 ) = 0;

	// 6. Output descriptors & tapped states
	virtual pod::TensorView& logits() = 0;
	virtual const pod::TensorView& logits() const = 0;
	virtual const half* tapped_states( int row = 0 ) const = 0;
	virtual const ModelMetadata& get_metadata() const = 0;
};

// Factory helper to instantiate models dynamically
std::unique_ptr<ModelPipeline> create_model_pipeline( const std::string& model_type );

} // namespace engine::models
