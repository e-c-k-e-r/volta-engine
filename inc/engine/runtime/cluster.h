#pragma once

#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/core/comm.h"
#include "engine/core/batch.h"
#include "engine/models/model.h"
#include "engine/models/vision_tower.h"
#include "engine/runtime/rank_pool.h"
#include "engine/runtime/dflash2_runner.h"
#include "engine/runtime/speculator.h"
#include "engine/runtime/sampling.h"
#include "engine/runtime/multimodal.h"
#include <functional>
#include "engine/backend/types.h"
#include <map>
#include <memory>
#include <mutex>
#include <thread>
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
	bool speculate = false;      // build a speculator (the MTP head, or DFlash2 when draft_model_dir is set)
	bool spec_rows = false;      // the verify snapshots without a speculator (a test that runs and commits speculative rows itself)
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
	// The same with the ids in device 0's memory, written by work already queued on rank 0's stream (a drafter's tokens): every rank's stream
	// waits for that work on the device, so the forward is queued before the host has the ids. With preds, the greedy argmax of every row
	// follows in the same queue (as sample). False (nothing run) when the model embeds host ids only or a slot has images
	bool forward_device( const int* d_tokens, const pod::BatchRows& rows, int* preds );

	// Chunked causal prefill of `count` tokens starting at pos 0 into `slot`.
	// Returns the greedy prediction for the token following the prompt.
	// last_cands (optional) receives the final row's top-16 candidates so the caller can sample the first token.
	// chunk 0 = the model's largest row count (128 for Qwen). Chunks above MAX_BATCH rows run eagerly and skip the
	// LM head except for the final row of the final chunk.
	// pos0: position of tokens[0] (a prefill that resumes from a restored prefix starts there)
	int prefill( const int* tokens, int count, int slot, int chunk = 0, Candidates* last_cands = nullptr, int pos0 = 0 );
	// A window of consecutive prefill chunks of one slot (each above MAX_BATCH rows, at most window_chunks() of them) run layer by layer, as prefill
	// runs long prompts. GLM: when every chunk asks for LOGITS_ALL (a scoring pass), every row of the window has logits after it (device_logits /
	// read_logits of the window's total rows; glm_logit_rows must hold them)
	void forward_window( const int* tokens, const pod::BatchRows* chunks, int n );
	int window_chunks() const { return models_[0]->window_chunks(); }
	// Where a forward of the slot's positions [from, end) may stop when the model's images are bidirectional (a forward holds whole images):
	// an end inside an image moves back to the image's start, or past its end when the image starts at `from`; otherwise `end`
	int span_safe_end( int slot, int from, int end ) const;

	// Sequence snapshots (prefix cache): one pinned host buffer per rank holding the slot's recurrent state and the KV
	// rows [0, tokens). Buffers are recycled through a pool, so steady-state saves allocate nothing.
	struct SeqSnapshot {
		int from = 0;                      // the snapshot holds KV rows [from, tokens); a chain links from = the parent's tokens
		int tokens = 0;
		bool with_taps = false;            // carries the drafter tap rows: a restore keeps speculating
		bool rows_only = false;            // only the KV rows (rows_snapshot dropped the state): loads kv_only, as a chain ancestor
		size_t bytes = 0;                  // per rank
		void* host[ENGINE_NUM_GPUS] = {};
	};
	int snapshot_align() const { return models_[0]->seq_align(); }
	size_t snapshot_bytes( int from, int tokens, bool with_taps = false ) const;    // per rank; 0 when the model cannot snapshot
	bool save_snapshot( int slot, int from, int tokens, SeqSnapshot* snap, bool with_taps = true );
	// Host buffers of `bytes` per rank for a snapshot that is filled from elsewhere (the prefix cache's disk tier); false when pinning fails
	bool alloc_snapshot( int from, int tokens, bool with_taps, size_t bytes, SeqSnapshot* snap );
	// Pins the host buffers of snapshots that a long prompt is about to store (cudaHostAlloc costs ~0.75 s per GiB, which would stall every
	// decoding slot when done inside a checkpoint) on a helper thread, so the store finds them in the free list.
	struct SnapshotPlan { int from, tokens; bool with_taps; };
	void prewarm_snapshots( const std::vector<SnapshotPlan>& plan );
	// Writes the snapshot into the slot: its KV rows, and unless kv_only the recurrent state and tap rows (the slot then
	// claims the drafter, which re-reads its window from the restored taps). A chain is restored ancestors first (kv_only),
	// then the node being resumed from.
	bool load_snapshot( int slot, const SeqSnapshot& snap, bool kv_only = false );
	void free_snapshot( SeqSnapshot* snap );
	// A copy of a snapshot's KV rows alone (rows_only: the recurrent state and tap rows dropped), what a chain ancestor needs, at a
	// fraction of the bytes when the state is large. False when the model cannot, the copy would not be smaller, or host memory runs out
	bool rows_snapshot( const SeqSnapshot& snap, SeqSnapshot* out );

	// Greedy argmax of the last forward's first `count` rows (rank 0).
	void sample( int* out, int count );
	// the largest logit over the ranks' shard winners (ties keep the lower id)
	void merge_shard_argmax( const int ( *toks )[pod::MAX_BATCH], const float ( *vals )[pod::MAX_BATCH], int count, int* out ) const;

	// Top-16 candidates (merged across the vocabulary shards, descending) for each of the last forward's first
	// `count` rows, for temperature / top-p sampling. Synchronizes.
	void sample_topk( int count, Candidates* out );
	// sample_topk also fills Candidates::lse (a log-sum-exp over the vocabulary per row) while a request wants log-probabilities
	void set_logprobs( bool on ) { logprobs_ = on; }

	// Full-vocabulary FP16 logits of the last forward's first `rows` rows, copied to host
	// ([rows, vocab_size], row-major). Reassembles the TP shards when the LM head is sharded.
	// Diagnostic / test path: synchronizes and issues per-row peer copies.
	void read_logits( int rows, backend::half_t* host_dst );
	// The same reassembled logits left on device 0 ([rows, vocab_size] FP16), ready in order on the returned stream (no copy, no sync); row0 > 0
	// (models that gathers_offset_rows) gathers the forward's rows from row0 on
	const backend::half_t* device_logits( int rows, backend::stream_t* stream, int row0 = 0 );
	int sample_one();
	// Adapts the routed-expert residency (expert_cache) to the routing seen since the last call; false when no model tiers
	bool rebalance_experts();
	// route_resident_bias: gives every rank's router the bias for the experts resident on any rank (after a rebalance and at start)
	void update_resident_bias();
	// Routed-expert route counts since the previous call, per rank: layers x experts of that rank's partition; false when the model keeps none
	bool expert_heat( std::vector<std::vector<uint32_t>>& per_rank, int& layers, int& experts );

	// Pre-capture decode graphs for row counts 1..max_m (two dummy forwards each on `slot`,
	// which is reset afterwards) so serving never pays eager warm-up or capture mid-stream.
	void warm_graphs( int max_m, int slot = 0 );

	void reset_slot( int slot );
	void set_hessian_capture( bool on ) { for ( int d = 0; d < tp_; ++d ) models_[d]->set_hessian_capture( on ); }
	// Runtime steering (option steer), on every rank; see ModelPipeline::steer_*. Stream-ordered with the forwards.
	bool steer_enabled() const { return models_[0] && models_[0]->steer_enabled(); }
	void steer_set( int slot, const std::vector<engine::modules::SteerEntry>& entries );
	void steer_control( int slot, int scope, int prompt_len, int think_end );
	void steer_penalty( int slot, const std::vector<int>& ids, float penalty );
	void steer_range( int* first_layer, int* num_layers ) const { models_[0]->steer_range( first_layer, num_layers ); }
	int steer_route_experts() const { return models_[0]->steer_route_experts(); }
	void steer_capture( const std::vector<int>& layers, bool on );
	bool steer_read( std::vector<float>& out );   // rank 0's capture (the residual is replicated)
	bool probe_read( int slot, int first_pos, int count, float* out );   // rank 0's probe scores, out [kProbes][count]
	// Shared token pool (model option kv_pool; always roomy without it): reserve room for positions [0, tokens) of a slot on every rank, or
	// take nothing and return false; release_tokens hands the slot's pages back. Host bookkeeping only.
	bool reserve_tokens( int slot, int tokens );
	void release_tokens( int slot );
	int pool_tokens() const;   // 0 = every slot owns max_seq tokens

	// Multimodal input. A slot with a layout rotates its image tokens (and the text after them) with M-RoPE positions and takes the
	// image embeddings in place of the placeholder tokens' embeddings; set it before prefilling, reset_slot clears it.
	void set_slot_mm( int slot, MmLayout layout );
	void apply_mrope( pod::BatchRows& rows ) const;
	const engine::loader::VisionParams* vision_params() const { return host_vision_ ? &host_vision_->params() : models_[0]->vision_params(); }
	// True when the vision_cpu tower on the host encodes the images: any thread may call encode_image then, unlike with the on-GPU tower
	bool host_vision() const { return host_vision_ != nullptr; }
	bool encode_image( const engine::loader::PreparedImage& image, std::vector<backend::half_t>* out );
	void checkpoint( int slot );
	void rollback( int slot );
	void set_spec_snapshot( bool on );
	void commit_spec_row( int slot, int row );
	void commit_spec_rows( int slot, int first_row, int last_row );
	bool spec_commit_always() const;

	// Speculative decoding (DFlash2). The round itself lives in DFlash2Runner.
	bool has_speculator() const { return runner_ != nullptr; }
	// Drafter taps of positions [first_pos, first_pos + count) of a slot, from rank 0 (every rank holds the same values), count * 5 * hidden halves
	bool read_taps( int slot, int first_pos, int count, backend::half_t* host_dst );
	Speculator& speculator() { return *runner_; }
	const Speculator& speculator() const { return *runner_; }
	DFlash2Runner& spec() { ENGINE_ASSERT( dflash_ != nullptr, "ModelCluster::spec: the speculator is not DFlash2" ); return *dflash_; }
	bool spec_owns( int slot ) const { return runner_ && runner_->owns( slot ); }
	SpecResult speculate( int slot, int anchor, int pos, float depth_threshold = -1.0f,
		const SamplingParams* sp = nullptr, Rng* rng = nullptr ) {
		return runner_->speculate( slot, anchor, pos, depth_threshold, sp, rng );
	}

	// DSpark drafter round (see ModelPipeline::dspark_forward): the block's draft tokens and confidence logits, from rank 0
	void dspark_draft( int slot, int pos, int from, int win_start, int anchor, int* drafts, float* conf );

	// MTP head steps on all ranks (see ModelPipeline::mtp_step); positions pos0 .. pos0 + count - 1 of `slot`, tokens[j] follows position pos0 + j.
	void mtp_step( int slot, int pos0, int count, const int* tokens, int hidden_mode, int hist_row0, bool logits_last );
	void mtp_save_hist( int src_row, int rows, int dst_row );
	// Joint rounds (see ModelPipeline::mtp_step_rows): rows of several slots in one head step, logits for rows out_rows[0 .. n_out) only
	// The MTP draft chain on the device: catch-up over pend_n pending rows from pend_pos (the last takes catch_tokens' anchor), then steps - 1 head
	// steps at pos ..; the drafted tokens into out. False when the model or the mtp_chain option says no (step the head from the host then).
	bool mtp_chain( int slot, int pend_pos, int pend_n, const int* catch_tokens, int hist_row0, int pos, int steps, int* out );
	bool mtp_joint() const;
	void mtp_step_rows( const pod::BatchRows& rows, const int* tokens, int hidden_mode, int hist_row0, const int* out_rows, int n_out );
	// Sparse drafter head (sparse_head option): the vocabulary ids in `ids`, most important first, become the draft head's subset; each rank
	// takes the ids of its shard up to its capacity (the rest are dropped). Returns the ids kept (sorted), empty when the model has none
	std::vector<int> sparse_head_set( const std::vector<int>& ids );
	void sparse_head_use( bool on );
	int sparse_head_cap() const { return models_[0] ? models_[0]->sparse_head_cap() : 0; }
	void mtp_gather_hist( const int* src_rows, int count );

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
	std::unique_ptr<engine::models::VisionTower> host_vision_;   // vision_cpu: the image encoder on the host CPU
	bool created_ = false;
	bool initialized_ = false;
	bool resident_bias_logged_ = false;   // route_resident_bias: the first update logs the table
	int tp_ = ENGINE_NUM_GPUS;
	std::unique_ptr<engine::models::ModelPipeline> models_[ENGINE_NUM_GPUS];
	pod::MemoryArena arenas_[ENGINE_NUM_GPUS] = {};
	engine::comm::CollectiveRing rings_[ENGINE_NUM_GPUS];
	backend::stream_t streams_[ENGINE_NUM_GPUS] = {};
	void* staging_ptrs_[ENGINE_NUM_GPUS] = {};
	void* flag_ptrs_[ENGINE_NUM_GPUS] = {};
	std::unique_ptr<Speculator> runner_;
	bool logprobs_ = false;
	DFlash2Runner* dflash_ = nullptr;   // runner_ when it is DFlash2
	std::vector<MmLayout> slot_mm_;   // per slot
	int mm_slots_ = 0;                // slots with a non-empty layout
	std::map<size_t, std::vector<void*>> snapshot_free_[ENGINE_NUM_GPUS]; // pinned buffers by rounded size
	size_t snapshot_free_bytes_[ENGINE_NUM_GPUS] = {};                    // what those lists hold, at most snapshot_free_cap()
	std::mutex snapshot_mtx_;
	std::thread snapshot_prewarm_;
	std::thread snapshot_release_;   // unpins the buffers the free lists do not keep, off the caller's thread
	std::mutex snapshot_release_mtx_;
	size_t snapshot_free_cap() const;
	void release_pinned( std::vector<void*> buffers );
	RankPool pool_;
};

} // namespace engine::runtime
