#include "engine/spec/dflash.h"
#include "engine/runtime/options.h"
#include "engine/core/log.h"
#include "engine/modules/norm.h"
#include "engine/modules/swiglu.h"
#include "engine/modules/embedding.h"
#include "engine/modules/linear.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <chrono>
#include <cstring>

namespace engine::spec {

// ---------------------------------------------------------------------------
// DFlashSpeculator Implementation
// ---------------------------------------------------------------------------

DFlashSpeculator::DFlashSpeculator()
	: device_{ 0 }
	, arena_{ nullptr }
	, initialized_{ false }
	, weights_loaded_{ false }
	, rank_{ 0 }
	, tp_size_{ 1 }
	, d_draft_tokens_{ nullptr }
	, d_conf_{ nullptr }
	, d_input_ids_{ nullptr }
	, d_topk_values_{ nullptr }
	, d_topk_indices_{ nullptr }
	, d_topk_part_v_{ nullptr }
	, d_topk_part_ix_{ nullptr }
	, d_topk_counters_{ nullptr }
	, d_gate_{ nullptr }
{
}

DFlashSpeculator::~DFlashSpeculator() {
	destroy();
}

bool DFlashSpeculator::init(
	const pod::DFlashConfig& config,
	int device,
	pod::MemoryArena& arena,
	cudaStream_t stream,
	int rank,
	int tp_size
) {
	config_ = config;
	device_ = device;
	arena_ = &arena;
	rank_ = rank;
	tp_size_ = tp_size;
	(void) stream;

	CUDA_CHECK( cudaSetDevice( device_ ) );

	int M = config_.block_size; // 8
	int K = M - 1;              // 7
	int top_k = config_.selector_top_k;
	const int sel_rank = config_.selector_rank;
	const int H = config_.hidden_size;
	const int TP = tp_size_;
	const int q_heads_local = config_.num_q_heads / TP;
	const int kv_heads_local = config_.num_kv_heads / TP;
	const int inter_local = config_.intermediate_size / TP;
	const int groups = config_.conv_groups();
	const int proj_out = config_.conv_proj_out();
	ENGINE_ASSERT( config_.num_q_heads % TP == 0 && config_.num_kv_heads % TP == 0
		&& config_.intermediate_size % TP == 0 && H % TP == 0,
		"DFlashSpeculator::init: TP size must divide the head and intermediate counts" );

	// 1. Allocate device execution buffers in static arena
	d_draft_tokens_ = static_cast<int*>( engine::core::arena_alloc( arena, M * sizeof( int ) ) );
	// The walk seeds itself from the anchor token and then overwrites d_draft_tokens_, so the anchor
	// needs its own slot; aliasing them fed the walk its own uninitialized output.
	d_anchor_ = static_cast<int*>( engine::core::arena_alloc( arena, sizeof( int ) ) );
	d_conf_ = static_cast<float*>( engine::core::arena_alloc( arena, M * sizeof( float ) ) );
	d_input_ids_ = static_cast<int*>( engine::core::arena_alloc( arena, M * sizeof( int ) ) );
	CUDA_CHECK( cudaMemsetAsync( d_input_ids_, 0, M * sizeof( int ), 0 ) );

	// 2. Scratch buffers for selector top-k & walk
	int splits = ( config_.vocab_size + kDflashTopkTile - 1 ) / kDflashTopkTile;
	// block_size rows, not K. The LM head's fast path only covers m in {1, 2, 4, 8}, so the
	// per-row candidate top-k runs on the full block and the walk drops the last row. Sizing these on K
	// overflowed by one row of values, one of indices and one of counters.
	int num_rows = M;

	d_topk_values_ = static_cast<float*>( engine::core::arena_alloc( arena, num_rows * top_k * sizeof( float ) ) );
	d_topk_indices_ = static_cast<int*>( engine::core::arena_alloc( arena, num_rows * top_k * sizeof( int ) ) );
	d_topk_part_v_ = static_cast<float*>( engine::core::arena_alloc( arena, num_rows * splits * top_k * sizeof( float ) ) );
	d_topk_part_ix_ = static_cast<int*>( engine::core::arena_alloc( arena, num_rows * splits * top_k * sizeof( int ) ) );
	d_topk_counters_ = static_cast<int*>( engine::core::arena_alloc( arena, num_rows * sizeof( int ) ) );
	d_gate_ = static_cast<half*>( engine::core::arena_alloc( arena, num_rows * sel_rank * sizeof( half ) ) );

	int h_shape[2] = { M, H };
	draft_hidden_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );

	// Merge staging for the distributed per-row candidate top-k. Rank d's slice starts at d * rows * k
	// and every rank writes the same width, so the merge can walk them with one stride.
	{
		const int mr = M;
		// cand_ranks_, not tp_size_: the drafter runs unsharded on rank 0 while the target's head is
		// sharded, and sizing this from tp_size_ let rank 1's slice land on top of the index buffer.
		const int mel = cand_ranks_ * mr * top_k;
		int mv[1] = { mel };
		merge_vals_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, mv, 1 );
		merge_ix_ = engine::core::arena_alloc_tensor( arena, pod::DType::INT32, mv, 1 );
	}

	int proj_shape[2] = { 1, H };
	draft_projected_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, proj_shape, 2 );

	// 3. Backbone activation scratch, sized for block_size rows and this rank's shard widths. The
	// residual stream is always full width and replicated; only q/k/v and the MLP intermediate are
	// per-rank. Coeff holds the per-token per-group conv deltas for both sides of each sandwich.
	int res_shape[2] = { M, H };
	bb_residual_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, res_shape, 2 );
	bb_normed_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, res_shape, 2 );
	bb_attn_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, res_shape, 2 );
	bb_hidden_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, res_shape, 2 );

	int q_shape[3] = { M, q_heads_local, config_.head_dim };
	bb_q_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_shape, 3 );
	int kv_shape[3] = { M, kv_heads_local, config_.head_dim };
	bb_k_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_shape, 3 );
	bb_v_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_shape, 3 );

	int ffn_shape[2] = { M, inter_local };
	bb_ffn_gate_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ffn_shape, 2 );
	bb_ffn_up_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ffn_shape, 2 );
	bb_ffn_mlp_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ffn_shape, 2 );

	int dyn_shape[3] = { M, config_.conv_kernel_size, groups };
	draft_dyn_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, dyn_shape, 3 );
	// One coefficient tensor per conv side: [rows, taps, groups], split out of the projection's
	// 2*taps*groups output so the conv's (dyn[t, k, group]) indexing is direct.
	int coef_shape[3] = { M, config_.conv_kernel_size, groups };
	bb_coef_a0_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, coef_shape, 3 );
	bb_coef_a1_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, coef_shape, 3 );
	bb_coef_m0_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, coef_shape, 3 );
	bb_coef_m1_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, coef_shape, 3 );

	int dynproj_shape[2] = { M, proj_out };
	bb_dyn_a_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, dynproj_shape, 2 );
	bb_dyn_m_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, dynproj_shape, 2 );

	// q, k and v all stage through this buffer, and q is the widest: qh_local * head_dim columns
	// against kvh_local * head_dim for k and v. Sizing it on k/v alone overflows by the GQA ratio
	// (4x at the real config) and corrupts whatever the arena placed next.
	//
	// Rows: the context projection reuses this same buffer with up to ctx_chunk_ rows, which is 256
	// in production against block_size 8. Sizing on the block size alone overflows on any context
	// longer than the block.
	// 4. Drafter KV cache for the projected target context: a ring of sliding_window slots plus a
	// reserved tail of block_size slots for the current block's own transient K/V.
	ring_len_ = MIN( config_.sliding_window, 2048 );
	own_slot_ = ring_len_;

	// Chunk width for the context projection. 256 rows is fine in principle, but the Trellis fp16
	// path has a gemv fast path only for m in {1,2,4,8} and falls back to the cooperative GEMM
	// otherwise -- and that fallback is the one code path the block side never exercises. Keep the
	// chunk at 8 until the cooperative GEMM has its own parity gate.
	//
	// This has to be settled before any buffer that is sized from it. It used to be assigned further
	// down, after bb_kv_stage_ was already allocated from it, so that staging buffer was sized
	// MAX(M, 0) = block_size rows while the context loop wrote up to ctx_chunk_ rows into it -- 32 rows
	// of overrun at the parity geometry, 256 in production. The arena is a bump allocator with no
	// guard, so it silently overwrote the buffers allocated after it (the FP32 residual carrier, the
	// pre-attn/pre-MLP conv staging and the K/V cache itself) rather than faulting.
	ctx_chunk_ = MIN( ring_len_, 256 );

	const int kv_stage_cols = MAX( q_heads_local * config_.head_dim, kv_heads_local * config_.head_dim );
	int kvflat_shape[2] = { MAX( M, ctx_chunk_ ), kv_stage_cols };
	bb_kv_stage_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kvflat_shape, 2 );
	bb_stage_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, res_shape, 2 );
	// Pre-attention and pre-MLP convs read a normalized hidden, so FP16 staging is safe there.
	bb_stage_h_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, res_shape, 2 );

	// FP32 carrier for the two conv outputs that can exceed FP16's range: the post-attention conv over
	// o_proj and the post-MLP conv over down_proj. Both feed a norm directly, so nothing downstream
	// needs them narrowed again.
	bb_stage_f32_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, res_shape, 2 );
	bb_o_proj_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, res_shape, 2 );

	const int kv_total = ring_len_ + M;
	int kv_cache_shape[4] = { kv_total, config_.num_hidden_layers, kv_heads_local, config_.head_dim };
	ctxs_.assign( static_cast<size_t>( num_contexts_ ), Ctx{} );
	for ( Ctx& c : ctxs_ ) {
		c.k = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_cache_shape, 4 );
		c.v = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_cache_shape, 4 );
	}
	ctx_cur_ = 0;
	drafter_k_ = ctxs_[0].k;
	drafter_v_ = ctxs_[0].v;
	kv_len_ = 0;
	kv_pos0_ = 0;

	int ctx_tap_shape[2] = { ctx_chunk_, 5 * H };
	bb_ctx_fc_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ctx_tap_shape, 2 );
	int ctx_shape[2] = { ctx_chunk_, H };
	bb_ctx_normed_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ctx_shape, 2 );
	int ctx_res_shape[2] = { ctx_chunk_, H };
	bb_ctx_resid_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, ctx_res_shape, 2 );

	// Default dummy codebooks (for standalone benchmarks or tests without weights on disk)
	// Only the output rank (0) runs the selector walk, so the other ranks of a TP drafter skip the 254 MiB.
	if ( rank_ == 0 ) {
		int cb_shape[2] = { config_.vocab_size, sel_rank };
		pred_codebook_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, cb_shape, 2 );
		succ_codebook_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, cb_shape, 2 );
	}

	if ( rank_ == 0 ) {
		const size_t kv_bytes = 2 * static_cast<size_t>( kv_total ) * config_.num_hidden_layers * kv_heads_local * config_.head_dim * sizeof( half );
		const size_t book_bytes = 2 * static_cast<size_t>( config_.vocab_size ) * sel_rank * sizeof( half );
		if ( engine::runtime::options().mem_log ) LOG_INFO( "[mem] drafter rank 0: KV ring %.0f MiB, selector codebooks %.0f MiB (rank 0 only)\n", kv_bytes / 1048576.0, book_bytes / 1048576.0 );
	}
	initialized_ = true;
	return true;
}


// One-hot rows c0..c0+R of the K x K identity: X[r][c0 + r] = 1
__global__ void identity_chunk_kernel( half* x, int K, int c0, int R ) {
	const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= (size_t) R * K ) return;
	const int r = (int) ( i / K ), c = (int) ( i % K );
	x[i] = __float2half( c == c0 + r ? 1.0f : 0.0f );
}

// dst[n][c0 + r] = src[r][n]: the chunk of W^T rows (Y = X W^T, X one-hot) back into the dense [N, K] layout
__global__ void transpose_chunk_kernel( const half* src, half* dst, int N, int K, int c0, int R ) {
	const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= (size_t) R * N ) return;
	const int r = (int) ( i / N ), n = (int) ( i % N );
	dst[(size_t) n * K + c0 + r] = src[i];
}

// Dense FP16 [N, K] weight from an EXL3 tensor, by pushing the identity through the same Linear path the model uses (so
// Hadamards, scales and the codebook come out exactly as at run time).
static size_t g_dense_proj_bytes = 0;   // FP16 projections dequantized at load (the [mem] startup line reports them)
static pod::TensorView dequant_dense( const pod::LinearWeight& lw, pod::MemoryArena& arena, cudaStream_t stream ) {
	ENGINE_ASSERT( lw.is_valid() && lw.type == pod::QuantType::TRELLIS, "dflash2: projection is not an EXL3 tensor" );
	const int K = lw.w1.shape[0]; // suh
	const int N = lw.w2.shape[0]; // svh
	const int shape[2] = { N, K };
	pod::TensorView dense = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, shape, 2 );
	ENGINE_ASSERT( dense.data != nullptr, "dflash2: arena exhausted dequantizing a projection" );
	g_dense_proj_bytes += static_cast<size_t>( N ) * K * sizeof( half );
	constexpr int R = 256; // above the fused-GEMM row limit, so it takes the same dequantize + cuBLAS path as prefill
	half *x = nullptr, *y = nullptr;
	CUDA_CHECK( cudaMalloc( &x, (size_t) R * K * sizeof( half ) ) );
	CUDA_CHECK( cudaMalloc( &y, (size_t) R * N * sizeof( half ) ) );
	pod::TensorView xt = pod::make_tensor_2d( x, pod::DType::FP16, R, K );
	pod::TensorView yt = pod::make_tensor_2d( y, pod::DType::FP16, R, N );
	for ( int c0 = 0; c0 < K; c0 += R ) {
		const int rows = MIN( R, K - c0 );
		xt.shape[0] = rows;
		yt.shape[0] = rows;
		identity_chunk_kernel<<<(unsigned) ( ( (size_t) rows * K + 255 ) / 256 ), 256, 0, stream>>>( x, K, c0, rows );
		engine::modules::Linear::forward( xt, lw, yt, stream );
		transpose_chunk_kernel<<<(unsigned) ( ( (size_t) rows * N + 255 ) / 256 ), 256, 0, stream>>>( y, dense.ptr<half>(), N, K, c0, rows );
	}
	CUDA_CHECK( cudaStreamSynchronize( stream ) );
	CUDA_CHECK( cudaFree( x ) );
	CUDA_CHECK( cudaFree( y ) );
	return dense;
}

// A projection the DFlash2 code runs as a plain FP16 GEMV ([N, K] weight): read as such from an unquantized checkpoint, or
// dequantized from the EXL3 tensors of a quantized one. These few projections are small and latency-bound, which is why
// they stay dense.
static pod::TensorView load_dense_proj(
	engine::loader::SafeTensorsRepo& repo,
	const char* base,
	pod::MemoryArena& arena,
	cudaStream_t stream
) {
	char key[256];
	std::snprintf( key, sizeof( key ), "%s.weight", base );
	if ( repo.find( key ) != nullptr ) return repo.load( key, arena, stream, pod::DType::FP16 );
	return dequant_dense( repo.load_linear( base, arena, 5, stream ), arena, stream );
}

// Loads one projection as a TP shard: column-parallel (slice_dim 1, splits out_features) for
// q/k/v/gate/up, row-parallel (slice_dim 0, splits in_features) for o_proj/down_proj. The full
// out_features value is the shape[0]*16 of the trellis tensor.
static pod::LinearWeight load_shard(
	engine::loader::SafeTensorsRepo& repo,
	const char* key,
	pod::MemoryArena& arena,
	int slice_dim,
	int rank,
	int tp_size,
	cudaStream_t stream
) {
	if ( tp_size <= 1 ) return repo.load_linear( key, arena, 5, stream );
	return repo.load_linear_slice( key, arena, 5, slice_dim, rank, tp_size, stream );
}

bool DFlashSpeculator::load_weights( const char* model_dir, cudaStream_t stream ) {
	if ( !model_dir || !initialized_ ) return false;

	engine::loader::SafeTensorsRepo repo;
	if ( !repo.open( model_dir ) ) {
		return false;
	}

	CUDA_CHECK( cudaSetDevice( device_ ) );

	const int R = rank_;
	const int TP = tp_size_;
	const int H = config_.hidden_size;

	// Validate the geometry against the checkpoint before allocating anything.
	//
	// DFlashConfig's defaults happen to describe the Qwen3.8-27B-DFlash2 drafter, and ModelCluster
	// only sets hidden_size, vocab_size and block_size -- every other field falls through to those
	// defaults. A drafter for any other target would therefore load and run with Qwen geometry
	// instead of failing. Checking the shapes that actually determine kernel indexing (heads,
	// head_dim, intermediate, conv grouping, codebook rank) turns that into a loud failure.
	{
		const auto check = [&]( const char* key, int d0, int d1 ) {
			const pod::SafeTensorMeta* m = repo.find( key );
			ENGINE_ASSERT( m != nullptr, "dflash2 checkpoint is missing a required tensor" );
			ENGINE_ASSERT( m->ndim == 2 && m->shape[0] == d0 && m->shape[1] == d1,
				"dflash2 checkpoint geometry does not match DFlashConfig -- DFlashConfig is under-populated" );
		};
		const auto check1d = [&]( const char* key, int d0 ) {
			const pod::SafeTensorMeta* m = repo.find( key );
			ENGINE_ASSERT( m != nullptr, "dflash2 checkpoint is missing a required tensor" );
			ENGINE_ASSERT( m->ndim >= 1 && m->shape[0] == d0,
				"dflash2 checkpoint geometry does not match DFlashConfig -- DFlashConfig is under-populated" );
		};
		check1d( "layers.0.input_layernorm.weight", H );
		check1d( "layers.0.post_attention_layernorm.weight", H );
		check1d( "norm.weight", H );
		if ( repo.find( "candidate_selector.hidden_projection.weight" ) != nullptr ) {
			check( "candidate_selector.hidden_projection.weight", config_.selector_rank, H );
		}
		else {
			const pod::SafeTensorMeta* sv = repo.find( "candidate_selector.hidden_projection.svh" );
			ENGINE_ASSERT( sv != nullptr && sv->shape[0] == config_.selector_rank, "dflash2 selector projection geometry does not match DFlashConfig" );
		}
		ENGINE_ASSERT( repo.find( "candidate_selector.predecessor_codebook" ) != nullptr
			&& repo.find( "candidate_selector.successor_codebook" ) != nullptr,
			"dflash2 checkpoint is missing the selector codebooks" );
		// The conv kernel_projection emits 2 * taps * groups columns; groups comes from the config's
		// conv_group_size, so a mismatch there means every coefficient is misindexed.
		ENGINE_ASSERT( config_.hidden_size % config_.conv_group_size == 0,
			"dflash2 conv_group_size does not divide hidden_size" );
		ENGINE_ASSERT( config_.num_q_heads % config_.num_kv_heads == 0,
			"dflash2 num_q_heads is not a multiple of num_kv_heads" );
	}
	const int L = config_.num_hidden_layers;

	// Context projection: 5 tapped target states (5*H) -> H. Column-parallel on the output with an
	// all-gather back to full width, so the 82 MB weight is a quarter per rank instead of replicated.
	fc_ = load_shard( repo, "fc", *arena_, 1, R, TP, stream );
	if ( fc_.type == pod::QuantType::TRELLIS && fc_.bits > 6 ) {
		// The 8-bit EXL3 fc has no fused-GEMM instantiation (only 2..6 bits are built); run it as the dense FP16 shard
		// an unquantized checkpoint would give.
		const pod::TensorView dense = dequant_dense( fc_, *arena_, stream );
		fc_ = pod::LinearWeight::make_fp16( pod::make_tensor_2d( dense.data, pod::DType::FP16, dense.shape[1], dense.shape[0] ) );
	}
	fc_norm_ = repo.load( "hidden_norm.weight", *arena_, stream, pod::DType::FP16 );
	final_norm_ = repo.load( "norm.weight", *arena_, stream, pod::DType::FP16 );

	// Candidate selector: read by the walk on the output rank (0) only.
	if ( R == 0 ) {
		pred_codebook_ = repo.load( "candidate_selector.predecessor_codebook", *arena_, stream, pod::DType::FP16 );
		succ_codebook_ = repo.load( "candidate_selector.successor_codebook", *arena_, stream, pod::DType::FP16 );
		// The selector's hidden projection is a plain FP16 [rank, hidden] matrix, not a trellis tensor, so
		// it cannot go through load_linear. Stored as a raw weight for a small dedicated GEMV.
		hidden_proj_w_ = load_dense_proj( repo, "candidate_selector.hidden_projection", *arena_, stream );
	}

	// Backbone. The residual stream stays full width and replicated; only heads and the MLP
	// intermediate are split, so this is 10 all-reduces per draft round and no more.
	char key[256];
	for ( int i = 0; i < L; ++i ) {
		DFlashLayer& ly = layers_[i];

		std::snprintf( key, sizeof( key ), "layers.%d.input_layernorm.weight", i );
		ly.input_norm = repo.load( key, *arena_, stream, pod::DType::FP16 );
		std::snprintf( key, sizeof( key ), "layers.%d.post_attention_layernorm.weight", i );
		ly.post_attn_norm = repo.load( key, *arena_, stream, pod::DType::FP16 );

		std::snprintf( key, sizeof( key ), "layers.%d.self_attn.q_proj", i );
		ly.q_proj = load_shard( repo, key, *arena_, 1, R, TP, stream );
		std::snprintf( key, sizeof( key ), "layers.%d.self_attn.k_proj", i );
		ly.k_proj = load_shard( repo, key, *arena_, 1, R, TP, stream );
		std::snprintf( key, sizeof( key ), "layers.%d.self_attn.v_proj", i );
		ly.v_proj = load_shard( repo, key, *arena_, 1, R, TP, stream );
		std::snprintf( key, sizeof( key ), "layers.%d.self_attn.o_proj", i );
		ly.o_proj = load_shard( repo, key, *arena_, 0, R, TP, stream );

		std::snprintf( key, sizeof( key ), "layers.%d.self_attn.q_norm.weight", i );
		ly.q_norm = repo.load( key, *arena_, stream, pod::DType::FP16 );
		std::snprintf( key, sizeof( key ), "layers.%d.self_attn.k_norm.weight", i );
		ly.k_norm = repo.load( key, *arena_, stream, pod::DType::FP16 );

		std::snprintf( key, sizeof( key ), "layers.%d.mlp.gate_proj", i );
		ly.gate_proj = load_shard( repo, key, *arena_, 1, R, TP, stream );
		std::snprintf( key, sizeof( key ), "layers.%d.mlp.up_proj", i );
		ly.up_proj = load_shard( repo, key, *arena_, 1, R, TP, stream );
		std::snprintf( key, sizeof( key ), "layers.%d.mlp.down_proj", i );
		ly.down_proj = load_shard( repo, key, *arena_, 0, R, TP, stream );

		// Sandwiched grouped dynamic convs: static per-tap per-channel base plus a plain FP16
		// projection emitting per-token per-group coefficients for both sides of the sandwich.
		std::snprintf( key, sizeof( key ), "layers.%d.attention_conv.base_kernel", i );
		ly.attn_conv_base = repo.load( key, *arena_, stream, pod::DType::FP16 );
		std::snprintf( key, sizeof( key ), "layers.%d.mlp_conv.base_kernel", i );
		ly.mlp_conv_base = repo.load( key, *arena_, stream, pod::DType::FP16 );
		std::snprintf( key, sizeof( key ), "layers.%d.attention_conv.kernel_projection", i );
		ly.attn_conv_proj = load_dense_proj( repo, key, *arena_, stream );
		std::snprintf( key, sizeof( key ), "layers.%d.mlp_conv.kernel_projection", i );
		ly.mlp_conv_proj = load_dense_proj( repo, key, *arena_, stream );
	}

	weights_loaded_ = true;
	LOG_INFO( "[dflash] backbone loaded: %d layers, hidden %d, tp %d (rank %d), conv groups %d\n",
		L, H, TP, R, config_.conv_groups() );
	if ( rank_ == 0 && engine::runtime::options().mem_log ) LOG_INFO( "[mem] drafter rank 0: weights include %.0f MiB of dense FP16 projections (dequantized at load)\n", g_dense_proj_bytes / 1048576.0 );
	return true;
}

void DFlashSpeculator::destroy() {
	initialized_ = false;
	weights_loaded_ = false;
}

} // namespace engine::spec
