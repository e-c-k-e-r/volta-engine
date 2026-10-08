#pragma once
#include <vector>

#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/arena.h"
#include "engine/core/comm.h"
#include "engine/core/batch.h"
#include "engine/backend/types.h"
#include <string>
#include <memory>

namespace engine::models {

// The drafter's query block width, shared by the model scratch, the speculator and the cluster so
// the three cannot disagree about how many rows a draft round has.
static constexpr int kDraftM = 8;
static constexpr int kDraftTopK = 16;   // dflash2.selector_top_k

// Per-rank view of the target that the DFlash2 drafter reads. Filled once at init; every pointer is
// stable for the model's lifetime, so a captured graph may bake them in.
struct DraftHooks {
	const pod::TensorView* embedding = nullptr; // [vocab, hidden] table (replicated on every rank)
	const backend::half_t* taps = nullptr;                 // [taps_cap, 5 * hidden] position-indexed tap ring
	int taps_cap = 0;
	int* taps_owner = nullptr;                  // device scalar: slot whose rows publish taps (-1: none)
	backend::half_t* lm_in = nullptr;                      // [kDraftM, hidden] drafter hidden, written by the runner
	float* cand_vals = nullptr;                 // [kDraftM, kDraftTopK] local top-k logits
	int* cand_idx = nullptr;                    // [kDraftM, kDraftTopK] global token ids
	int shard_offset = 0;                       // first global token id of this rank's slice
	int shard_real_cols = 0;                    // unpadded slice width
	bool valid() const { return embedding != nullptr && taps != nullptr; }
};

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
	// Rank count of the tensor-parallel group; call before init(). A null ring means a group of one.
	void set_tp_size( int tp ) { requested_tp_ = tp; }
	virtual bool init( int device_id, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring ) = 0;
	virtual bool load_weights( const char* model_dir, backend::stream_t stream = nullptr ) = 0;

	// 2. Token embedding: token_ids is a host array of `count` ids for rows 0..count-1
	virtual void embed_tokens( const int* token_ids, int count, backend::stream_t stream = nullptr ) = 0;

	// 3. Forward over embedded rows
	virtual void forward( const pod::BatchRows& rows, backend::stream_t stream = nullptr ) = 0;
	void forward( int batch_size, int pos, int slot_id = 0, backend::stream_t stream = nullptr ) {
		forward( pod::BatchRows::causal( batch_size, pos, slot_id ), stream );
	}

	// 4. Greedy sampling of the last forward's first `count` rows into host out_preds (rank 0)
	virtual void sample_tokens( int* out_preds, int count, backend::stream_t stream = nullptr ) = 0;

	// 4b. Vocabulary-parallel sampling: each rank picks the best token inside its own vocabulary
	// shard and reports (token, logit) into caller-owned host arrays. Returns false when the model's
	// logits are not sharded, in which case the caller falls back to sample_tokens on rank 0.
	virtual bool sample_local( int* out_tokens, float* out_vals, int count, backend::stream_t stream = nullptr ) {
		(void) out_tokens;
		(void) out_vals;
		(void) count;
		(void) stream;
		return false;
	}

	// 4b'. Vocabulary-parallel top-k for sampling: this rank's top-kSampleTopK (16) logits per row over its shard of
	// the last forward's first `count` rows, with global token ids, into caller-owned host arrays
	// [count][16] (descending). False when the logits are not sharded.
	virtual bool topk_local( float* out_vals, int* out_ids, int count, backend::stream_t stream = nullptr ) {
		(void) out_vals;
		(void) out_ids;
		(void) count;
		(void) stream;
		return false;
	}

	// 4c. This rank's local logits shard, for a vocabulary-parallel head. Null when unsharded.
	// Routed-expert route counts since the last reset: layers x experts (this rank's partition), false when not tracked
	virtual bool expert_heat( std::vector<uint32_t>& out, int& layers, int& experts, bool reset = false ) { (void) out; (void) layers; (void) experts; (void) reset; return false; }
	// Adapt the routed-expert residency to the routing seen so far; false when the model keeps every expert resident
	virtual bool rebalance_experts( backend::stream_t stream = nullptr ) { (void) stream; return false; }
	virtual const backend::half_t* shard_logits_ptr() const { return nullptr; }
	virtual int shard_logits_cols() const { return 0; }

	// 4d. DFlash2 drafter attachment. The drafter has no embedding table or output head of its own: it
	// reads this rank's embedding, the tap ring and a head-shard top-k through one descriptor, and
	// asks the model to run the head over caller-provided hidden rows.
	virtual const DraftHooks& draft_hooks() const { static const DraftHooks none; return none; }
	// Runs this rank's LM head shard over hooks.lm_in[rows, hidden] and leaves the local top-k
	// (token ids offset into the global vocabulary, sorted) in hooks.cand_vals / cand_idx.
	virtual void draft_head_topk( int rows, backend::stream_t stream = nullptr ) { (void) rows; (void) stream; }

	// 4e. Materialize the full-vocabulary logits on rank 0 from every rank's shard. Only consumers that
	// need the whole distribution (the DFlash2 drafter) pay for this; sampling uses sample_local.
	// The caller must have synchronized every rank's stream first. Returns logits() when unsharded.
	virtual const pod::TensorView& gather_logits(
		int rows,
		const backend::half_t* const* peer_shard_ptrs,
		int peer_shard_cols,
		backend::stream_t stream = nullptr
	) {
		(void) rows;
		(void) peer_shard_ptrs;
		(void) peer_shard_cols;
		(void) stream;
		return logits();
	}

	// 5. Speculative state management: snapshot / restore the recurrent state of one slot.
	//    KV cache entries need no snapshot: rejected positions are overwritten by the next forward.
	virtual void checkpoint_state( int slot_id, backend::stream_t stream = nullptr ) = 0;
	virtual void rollback_state( int slot_id, backend::stream_t stream = nullptr ) = 0;
	virtual void reset_state( int slot_id = 0 ) = 0;
	// Per-row state slots (replace checkpoint + replay). While on, a forward also publishes the recurrent
	// state after every row; commit_spec_row( slot, t ) makes the state after row t the slot's live state.
	// Stream-ordered, so a flag change applies to the forwards queued after it.
	virtual void set_spec_snapshot( bool on, backend::stream_t stream = nullptr ) { (void) on; (void) stream; ENGINE_ASSERT( false, "model has no speculative state slots" ); }
	virtual void commit_spec_row( int slot_id, int row, backend::stream_t stream = nullptr ) { (void) slot_id; (void) row; (void) stream; ENGINE_ASSERT( false, "model has no speculative state slots" ); }

	// 5b. Sequence snapshots for the prefix cache: everything a slot needs to resume after `tokens` positions (recurrent
	// state and the KV rows [0, tokens)) as bytes in caller-provided host memory. seq_state_bytes() == 0 means the model
	// (or its KV format) cannot snapshot. Stream-ordered: the caller synchronizes the stream before using the buffer.
	// Snapshots also carry the drafter tap ring rows of the saved prefix when on (a speculative cluster), so a restored slot keeps speculating.
	virtual void set_snapshot_taps( bool on ) { (void) on; }
	// A snapshot covers positions [from, tokens): the recurrent state and tap rows at `tokens` plus only the KV rows of that
	// range, so a chain of snapshots (from = the parent's tokens) stores each KV row once. load with kv_only skips the state.
	// with_taps: also store the drafter tap rows (only meaningful after set_snapshot_taps); a snapshot without them resumes a
	// slot that cannot speculate.
	virtual size_t seq_state_bytes( int from, int tokens, bool with_taps ) const { (void) from; (void) tokens; (void) with_taps; return 0; }
	virtual bool save_seq_state( int slot_id, int from, int tokens, bool with_taps, void* host_dst, backend::stream_t stream = nullptr ) { (void) slot_id; (void) from; (void) tokens; (void) with_taps; (void) host_dst; (void) stream; return false; }
	virtual bool load_seq_state( int slot_id, int from, int tokens, bool with_taps, const void* host_src, bool kv_only, backend::stream_t stream = nullptr ) { (void) slot_id; (void) from; (void) tokens; (void) with_taps; (void) host_src; (void) kv_only; (void) stream; return false; }

	// 6. Output descriptors & tapped states
	virtual pod::TensorView& logits() = 0;
	virtual const pod::TensorView& logits() const = 0;
	virtual const backend::half_t* tapped_states( int row = 0 ) const = 0;
	virtual const ModelMetadata& get_metadata() const = 0;

protected:
	int requested_tp_ = ENGINE_NUM_GPUS;
};

// Factory helper to instantiate models dynamically
// Registered architectures use the checkpoint config.json "model_type": qwen3_5, glm5_next. Anything else is unsupported.
bool is_known_model_type( const std::string& model_type );
std::unique_ptr<ModelPipeline> create_model_pipeline( const std::string& model_type );

} // namespace engine::models
