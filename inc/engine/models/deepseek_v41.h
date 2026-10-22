#pragma once

#include "engine/config.h"
#include "engine/models/model.h"
#include "engine/core/tensor.h"
#include "engine/core/arena.h"
#include "engine/core/comm.h"
#include "engine/core/graph_cache.h"
#include "engine/modules/linear.h"
#include "engine/modules/moe.h"
#include "engine/models/deepseek_v41_vision.h"
#include "engine/backend/types.h"
#include <cstdint>
#include <map>
#include <memory>
#include <thread>
#include <vector>

namespace engine::loader { class SafeTensorsRepo; }

namespace engine::models {

// DeepSeek-V4.1-Flash (model_type deepseek_v41; reference: DeepSeek's inference/model.py, tests/tools/dsv41). 40 layers of sliding-window
// + compressed sparse attention and routed MoE (384 experts, 6 per token) on four hyper-connection streams, with Engram n-gram lookups
// at two layers. The EXL3 checkpoints quantize the attention, shared-expert, routed-expert and head linears per expert / tensor; the rest
// (norms, routers, compressors, indexers, Engram tables) are the official tensors.
struct DeepseekV41Config {
	int num_layers = 40;
	int hidden_size = 5120;
	int num_heads = 64;
	int q_lora_rank = 1280;
	int o_groups = 8;
	int o_lora_rank = 1024;
	int num_experts = 384;
	int top_k = 6;
	int intermediate_size = 2304;
	int vocab_size = 129280;
	float route_scale = 1.5f;
	float swiglu_limit = 10.0f;
	float norm_eps = 1e-20f;
	float hc_eps = 1e-6f;
	int hc_sinkhorn_iters = 20;
	float rope_theta = 10000.0f;
	float compress_rope_theta = 160000.0f;
	int rope_original_len = 65536;
	float rope_factor = 16.0f;
	float beta_fast = 32.0f;
	float beta_slow = 1.0f;
	std::vector<int> compress_ratios;     // per layer: 0 window only, r > 0 compressed attention at ratio r
	std::vector<int> kv_sources;          // layers that compress KV (and own the index keys)
	std::vector<int> index_sources;       // layers that run an indexer
	int candidate_source = 20;
	int candidate_blocks = 2048;
	int candidate_block = 8;
	std::vector<int> engram_layers;
	int engram_heads = 8;
	int engram_head_dim = 256;
	int engram_ngram = 4;
	int max_seq = 16384;
	int max_slots = 1;
	int tp_size = 4;
	int eos_id = 1;
};

// One range of routed experts of a layer that share a bit width: engine ids [first, first + count), this rank's
struct DeepseekV41ExpertGroup {
	int bits = 0;
	int first = 0;
	int count = 0;
	int spans = 0;                   // expert_shard: the width of this rank's slice of these experts, in 128-wide spans
	std::unique_ptr<pod::MoELayerWeights> mw;
	std::vector<char> resident;      // [count] 1 when the expert's codes are in device memory, 0 in pinned host memory
	std::vector<uint64_t> score;     // [count] decayed route counts (the swap decisions)
};

struct DeepseekV41LayerWeights {
	int ratio = 0;            // compress ratio
	int kv_src = -1;          // index into the compressed caches (the source layer this one reads), -1 without compression
	int idx_src = -1;         // the index-source layer whose selection this one uses
	bool is_kv_source = false, is_index_source = false;
	int engram = -1;          // engram slot (0, 1) or -1
	pod::TensorView attn_norm, ffn_norm, q_norm, kv_norm, sink;
	pod::LinearWeight wq_a, wq_b, wkv, wo_b;
	std::vector<pod::LinearWeight> wo_a;          // this rank's groups
	// compressor / index keys (kv sources), indexer (index sources)
	pod::LinearWeight c_wkv, c_wgate, idx_wk, idx_wq_b, idx_wproj;
	pod::TensorView c_norm, idx_knorm;
	// routing (checkpoint expert ids -> engine ids through perm) and experts
	pod::TensorView router_w_t, router_bias, router_bias_vl;   // bias_vl: the router bias of image tokens
	int* perm = nullptr;
	std::vector<DeepseekV41ExpertGroup> groups;
	pod::LinearWeight sh_gate, sh_up, sh_down;     // rank 0
	// engram
	pod::LinearWeight e_wkv;
	pod::TensorView e_w;                           // q_weight * k_weight [4, hidden] FP32
	// hyper-connections
	pod::TensorView hc_attn_fn, hc_attn_base, hc_attn_scale, hc_ffn_fn, hc_ffn_base, hc_ffn_scale;
};

class DeepseekV41Pipeline : public ModelPipeline {
public:
	DeepseekV41Pipeline();
	~DeepseekV41Pipeline() override;

	using ModelPipeline::forward;

	bool init( int device_id, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring ) override;
	bool load_weights( const char* model_dir, backend::stream_t stream = nullptr ) override;
	void embed_tokens( const int* token_ids, int count, backend::stream_t stream = nullptr ) override;
	void forward( const pod::BatchRows& rows, backend::stream_t stream = nullptr ) override;
	// Prefill chunks layer by layer (prefill_window): the routed experts of a layer run once over the rows of all the window's chunks, so the
	// cold experts' codes cross the PCIe link once per window instead of once per 512-row chunk
	int window_chunks() const override { return window_; }
	void forward_window( const int* tokens, const pod::BatchRows* chunks, int n, backend::stream_t stream = nullptr ) override;
	// The Engram rows of the next window are requested from the table files while this one runs (the lookups' pages are the NVMe reads of a prefill)
	void set_prefetch_hint( const int* tokens, int count, int slot, int pos0 ) override { hint_tokens_ = tokens; hint_count_ = count; hint_slot_ = slot; hint_pos0_ = pos0; }
	void sample_tokens( int* out_preds, int count, backend::stream_t stream = nullptr ) override;
	bool sample_local( int* out_tokens, float* out_vals, int count, backend::stream_t stream = nullptr ) override;
	bool topk_local( float* out_vals, int* out_ids, int count, backend::stream_t stream = nullptr ) override;
	const backend::half_t* shard_logits_ptr() const override { return logits_.ptr<const backend::half_t>(); }
	int shard_logits_cols() const override { return lm_shard_cols_; }
	const pod::TensorView& gather_logits( int rows, const backend::half_t* const* peer_shard_ptrs, int peer_shard_cols, backend::stream_t stream = nullptr ) override;

	// Every cache is indexed by position, so a rewind needs no copy
	void checkpoint_state( int slot_id, backend::stream_t stream = nullptr ) override { (void) slot_id; (void) stream; }
	void rollback_state( int slot_id, backend::stream_t stream = nullptr ) override { (void) slot_id; (void) stream; }
	void reset_state( int slot_id = 0 ) override { (void) slot_id; }
	// Prefix-cache snapshots: positions [from, tokens) store their compressed KV and index-key entries and their Engram ids; the state at `tokens`
	// is the last WINDOW rows of every layer's window ring. Positions are even, so no 2:1 compression group is ever half filled at a boundary
	int seq_align() const override { return 2; }
	bool free_rollback() const override { return true; }
	// Images: the tower sits on rank 0 (its arena share comes out of that rank's resident experts); the cluster overwrites the embedded rows of an
	// image's token span with encode_image's rows (inject_embeddings), which then route with bias_vl and stay out of the Engram n-grams
	bool load_vision( const char* model_dir, backend::stream_t stream = nullptr ) override;
	const engine::loader::VisionParams* vision_params() const override { return vision_.loaded() ? &vision_.params() : nullptr; }
	bool encode_image( const engine::loader::PreparedImage& image, std::vector<backend::half_t>* out, backend::stream_t stream = nullptr ) override;
	bool inject_embeddings( const backend::half_t* host_rows, int dst_row, int rows, backend::stream_t stream = nullptr ) override;
	// DSpark block drafter (speculator=dspark, draft_model_dir = the directory quantize_dspark.py wrote): three blocks of window attention and a
	// 128-expert MoE on the same hyper-connection streams, entered through main_proj from the trunk's stream means at layers 37 / 38 / 39. The
	// main-KV rows of the window live in three extra layers of the window ring; see ModelPipeline::dspark_forward
	void set_reserve_after_weights( size_t bytes ) override { reserve_after_ = bytes; }
	bool load_dspark( const char* dir, backend::stream_t stream = nullptr ) override;
	bool has_dspark() const override { return ds_.loaded; }
	int dspark_block() const override { return ds_.block; }
	void dspark_forward( int slot, int pos, int from, int win_start, int anchor, backend::stream_t stream = nullptr ) override;
	void dspark_chain( const backend::half_t* const* peer_shards, int peer_cols, int anchor, int* drafts, float* conf, backend::stream_t stream = nullptr ) override;
	size_t seq_state_bytes( int from, int tokens, bool with_taps ) const override;
	bool save_seq_state( int slot_id, int from, int tokens, bool with_taps, void* host_dst, backend::stream_t stream = nullptr ) override;
	bool load_seq_state( int slot_id, int from, int tokens, bool with_taps, const void* host_src, bool kv_only, backend::stream_t stream = nullptr ) override;

	pod::TensorView& logits() override { return logits_; }
	const pod::TensorView& logits() const override { return logits_; }
	const backend::half_t* tapped_states( int row = 0 ) const override { (void) row; return nullptr; }
	const ModelMetadata& get_metadata() const override { return metadata_; }
	// route counts of this rank's experts per layer, in engine order; the totals (checkpoint ids) go to expert_heat_file at exit
	bool expert_heat( std::vector<uint32_t>& out, int& layers, int& experts, bool reset = false ) override;
	// Folds the routes since the last pass into per-expert scores and exchanges the hottest cold experts with the coldest resident ones of the same
	// bit width (same code size: the bytes trade places, no allocation moves)
	bool rebalance_experts( backend::stream_t stream = nullptr ) override;

private:
	static int lm_shard_spans( int vocab, int tp, int rank ) { return vocab / 128 / tp + ( rank < vocab / 128 % tp ? 1 : 0 ); }
	static int lm_shard_offset( int vocab, int tp, int rank ) { return 128 * ( rank * ( vocab / 128 / tp ) + MIN( rank, vocab / 128 % tp ) ); }
	static constexpr int kTopkTile = 4096;
	static constexpr float kOutScale = 8.0f;   // sublayer outputs are stored divided by this (the BOS row's massive activation exceeds FP16)

	bool read_config( const char* model_dir );
	bool load_engram( const char* model_dir );
	// moves the snapshot rows of [from, tokens) of a slot between the caches and host memory (layout: see seq_state_bytes)
	void copy_seq_rows( int slot_id, int from, int tokens, char* host, bool to_host, backend::stream_t stream );
	// phases: 1 = hashes and the page requests, 2 = gather and dequantize (and upload); a window requests the pages of all its chunks first
	void engram_prepare( const pod::BatchRows& rows, backend::stream_t stream, int chunk = 0, bool upload = true, int phases = 3 );
	void set_rows( int m );
	void forward_body( int m, backend::stream_t stream );
	void attention( int layer, DeepseekV41LayerWeights& lw, int m, backend::stream_t stream );
	void ffn( int layer, DeepseekV41LayerWeights& lw, int m, backend::stream_t stream );
	void ffn_front( DeepseekV41LayerWeights& lw, int m, backend::stream_t stream, int experts = 0, int top_k = 0 );     // mHC collapse, norm, router (the drafter's has fewer experts)
	void dspark_ingest( int slot, int from, int to, backend::stream_t stream );
	void engram_hash( const int64_t* toks, int64_t* out ) const;                       // the table rows of one position: out [layers][(ngram - 1) * heads]
	void engram_stage_start( const pod::BatchRows* chunks, int n );                    // rank 0: reads the window's table rows (and the next window's, from the hint) into shared staging
	void copy_taps( int slot_id, int tokens, char* host, bool to_host, backend::stream_t stream );
	void embed_raw( const int* token_ids, int count, backend::stream_t stream );
	void ffn_experts( DeepseekV41LayerWeights& lw, int rows, backend::stream_t stream ); // the routed experts over norm_x_ rows
	void ffn_back( DeepseekV41LayerWeights& lw, int m, backend::stream_t stream );      // shared expert, all-reduce
	void forward_head( int m, backend::stream_t stream );
	void trace( int tag, int m, backend::stream_t stream );
	void trace_tensor( int tag, const pod::TensorView& t, int m, int cols, backend::stream_t stream );   // debug: <trace_layers>.t

	DeepseekV41Config config_;
	ModelMetadata metadata_;
	int rank_ = 0;
	int lh_ = 16;               // attention heads of this rank
	int gl_ = 2;                // wo_a groups of this rank
	pod::MemoryArena* arena_ = nullptr;
	pod::MemoryArena host_arena_{};   // pinned cold experts
	engine::comm::CollectiveRing* ring_ = nullptr;
	engine::core::GraphCache graphs_;
	bool weights_loaded_ = false;
	std::unique_ptr<engine::loader::SafeTensorsRepo> repo_;   // kept open: the Engram tables are read through its mappings
	int lm_mode_ = pod::BatchRows::LOGITS_ALL;
	int last_rows_ = 0;
	int key_hint_ = 0;

	std::vector<DeepseekV41LayerWeights> layers_;
	pod::TensorView embed_table_;          // host BF16, rank 0
	pod::TensorView final_norm_;
	pod::LinearWeight lm_head_;
	float* rope_plain_ = nullptr;          // [max_seq][32][2]: layers without compression
	float* rope_comp_ = nullptr;           // compressing layers (YaRN)

	// caches
	backend::half_t* win_ring_ = nullptr;  size_t win_slot_stride_ = 0;
	std::vector<backend::half_t*> comp_kv_;  std::vector<size_t> comp_slot_stride_;
	std::vector<backend::half_t*> idx_k_;    std::vector<size_t> idx_slot_stride_;
	std::vector<float*> cring_kv_, cring_sc_; size_t cring_slot_stride_ = 0;
	std::vector<int> src_cap_;             // compressed entries per slot of each source

	// per-forward buffers [MAX_ROWS, ...]
	pod::BatchRows* d_rows_ = nullptr;
	int* d_token_ids_ = nullptr;
	pod::TensorView embed_rows_, streams_, norm_x_, collapsed_, attn_out_, moe_out_, moe_tmp_, sh_out_;
	pod::TensorView mhc_inv_rms_, mhc_mix_, mhc_post_, mhc_comb_;
	float* pre_[2] = {};
	int pre_cur_ = 0;
	pod::TensorView qr_, qr_n_, q_, kv_, kv_n_, ckv_, csc_, latent_, kraw_, qi_, wi_, o_g_, woa_out_, woa_cat_;
	pod::TensorView sh_g_, sh_u_, sh_i_;
	float* idx_scores_ = nullptr;
	uint8_t* idx_cand_ = nullptr;
	int* idx_sel_ = nullptr;
	int* idx_nsel_ = nullptr;
	int* d_topk_idx_ = nullptr;
	backend::half_t* d_topk_w_ = nullptr;
	float* route_scratch_ = nullptr;
	float* attend_ws_ = nullptr;   // split-KV partials of the decode attention
	// Window prefill: what a chunk keeps from layer to layer (its residual streams, lagged pre-mix, mHC post / comb of the FFN in flight, the
	// indexer's selection and candidate blocks) lives in a per-chunk set; the window's normed FFN input, router results and routed output are
	// one block of rows
	struct WinChunk {
		void* streams = nullptr;
		float* pre[2] = {};
		int pre_cur = 0;
		float* post = nullptr;
		float* comb = nullptr;
		int* sel = nullptr;
		int* nsel = nullptr;
		uint8_t* cand = nullptr;
	};
	// DSpark drafter
	struct DSparkState {
		bool enabled = false;                    // buffers allocated (speculator=dspark)
		bool loaded = false;
		int block = 5, noise = 128799, tap0 = 37, taps = 3, experts = 128, top_k = 3, blocks = 3, markov_rank = 256;
		std::vector<DeepseekV41LayerWeights> layers;
		pod::LinearWeight main_proj;
		pod::TensorView main_norm, final_norm;
		std::unique_ptr<engine::loader::SafeTensorsRepo> repo;
		pod::TensorView markov_embed;            // rank 0: device BF16 [vocab, markov_rank]
		pod::LinearWeight markov_head;           // rank 0: [markov_rank -> vocab]
		std::vector<float> conf_w;               // rank 0: confidence head over [hidden | markov_rank]
		backend::half_t* tap_ring = nullptr;     // [slot][WINDOW][taps][hidden]
		size_t tap_stride = 0;
		backend::half_t* blk_ring = nullptr;     // scratch window ring for the block's own keys
		pod::BatchRows* d_ing_rows = nullptr;
		pod::BatchRows* d_blk_rows = nullptr;
		pod::TensorView taps_in, mx_pre, mx, dkv, dkvn;   // [WINDOW rows, ...]
		int* d_chain = nullptr;                  // rank 0: [8] token ids of the chain
		pod::TensorView memb, bias;              // rank 0: [8, markov_rank], [1, vocab]
	} ds_;
	static constexpr float kTapScale = 1.0f / 16.0f;   // taps are stored scaled (the stream means pass FP16 at the BOS row); main_norm removes the scale
	const int* hint_tokens_ = nullptr;     // set_prefetch_hint
	int hint_count_ = 0, hint_slot_ = 0, hint_pos0_ = 0;
	size_t reserve_after_ = 0;
	int tap_floor_ = 0;
	int win_layers_ = 0;
	int window_ = 1;
	std::vector<WinChunk> win_;
	pod::TensorView win_norm_, win_moe_out_, win_moe_tmp_;
	int* win_topk_idx_ = nullptr;
	backend::half_t* win_topk_w_ = nullptr;
	pod::MoEScratch moe_scratch_;
	int max_group_ = 1;
	DeepseekV41VisionTower vision_;
	size_t vision_reserve_ = 0;            // arena bytes the tower will need (rank 0), kept out of the resident share
	std::vector<char> img_rows_;           // rows of the current forward that hold image tokens
	uint8_t* d_img_flag_ = nullptr;        // the same on the device, read by the router
	bool img_flag_dirty_ = false;          // the device flags hold ones that have not been cleared yet
	void sync_image_flags( backend::stream_t stream );
	bool expert_tp_ = false;               // every rank holds a slice of every expert (expert_shard) instead of whole experts
	// expert_shard: the intermediate width is 18 spans of 128; rank r of tp takes 18 / tp of them plus one of the 18 % tp extra, and which ranks get
	// the extra rotates with the expert so every rank carries the same share of the bytes and of the work
	void expert_slice( int expert, int rank, int* begin, int* spans ) const;
	std::vector<std::vector<int>> mine_;   // [layer] this rank's checkpoint expert ids in engine order
	std::vector<uint64_t> heat_total_;     // [layer][checkpoint expert] routes counted on this rank
	std::vector<uint32_t> pending_;        // [layer][checkpoint expert] routes drained by expert_heat that the residency has not seen yet
	std::map<int, uint8_t*> swap_staging_; // pinned, one expert projection per (bits * 32 + spans); rotates through the swaps
	uint64_t swaps_ = 0, window_routes_ = 0, window_hits_ = 0;
	int rebalances_ = 0;

	// engram: host hashing and row gathers, two device row buffers
	struct EngramHost {
		int vocab = 0, layers = 0, ngram = 0, heads = 0, pad = 0, dim = 256, block = 32;
		std::vector<int> token_map;
		std::vector<int64_t> mult, primes, offsets;
		const uint8_t* table[2] = {};
		const uint8_t* scale[2] = {};
		std::vector<std::vector<int>> hist;       // [slot][pos] compressed ids
		std::vector<int64_t> idx[16];              // the table rows of each chunk of the current forward [rows][layers][(ngram - 1) * heads]
		int rows_c[16] = {};
		std::thread advise[16];                    // a window: rank 0 requests the pages of a chunk in the background
		size_t page = 4096;
		backend::half_t* host_rows = nullptr;     // pinned [2][MAX_ROWS][cols]
		float e4m3[256];
		int fd[2] = { -1, -1 };                    // the table shards: rank 0 reads the rows of a window through them (deep pread queues, not page faults)
		size_t toff[2] = {};                       // absolute file offsets of the two tables
		int chunk_off[16] = {};                    // first row of each chunk of the current window in the window's staging
		int win_slot = -1, win_from = 0;           // the window the chunks belong to
		std::thread fetcher;                       // rank 0: reads the rows of this window and of the next one
		double t_ms[4] = {};                       // host time in the Engram lookups of this forward: hashes + page requests, waiting for the page-request thread, gather + dequantize, upload (expert_log)
	} eg_;
	std::vector<int> pending_tokens_;
	pod::TensorView e_rows_[2], e_kv_;

	// logits / sampling
	pod::TensorView logits_, full_logits_;
	int logit_rows_ = 16;
	int lm_shard_cols_ = 0, lm_shard_real_cols_ = 0, lm_shard_offset_ = 0;
	int* d_argmax_idx_ = nullptr;
	float* d_argmax_val_ = nullptr;
	int* h_argmax_idx_ = nullptr;
	float* samp_part_v_ = nullptr;
	int* samp_part_i_ = nullptr;
	int* samp_counters_ = nullptr;
	float* d_samp_vals_ = nullptr;
	int* d_samp_idx_ = nullptr;
};

} // namespace engine::models
