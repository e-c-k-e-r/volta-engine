#include "engine/models/glm5_next.h"
#include "engine/core/log.h"
#include "engine/runtime/options.h"
#include "engine/modules/hessian.h"
#include <memory>
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#include "engine/loader/safetensors.h"
#include "engine/modules/gdn.h"
#include "engine/modules/dsa.h"
#include "engine/modules/embedding.h"
#include "engine/spec/dflash.h"
#include "engine/spec/tap_ring.h"
#if ENGINE_BACKEND_CUDA
#include "trellis_context.h"
#endif
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <array>
#include <algorithm>

namespace engine::models {

// Resident fraction of the expert tier from which decode keeps the fused (exl3_moe_coop) kernels; below it the persistent kernel reads the cold experts a little faster
constexpr double kFusedTierMinFraction = 0.75;

namespace {

// Host-resident embedding table shared by all ranks (see load_weights)
pod::TensorView g_embed_table;

} // namespace

namespace impl {

pod::TensorView load_router_weights_glm(
	engine::loader::SafeTensorsRepo& repo,
	const char* name,
	pod::MemoryArena& arena,
	backend::stream_t stream
) {
	int shard_idx = -1;
	const auto* meta = repo.find( name, &shard_idx );
	if ( !meta || shard_idx < 0 ) return pod::TensorView{};

	int num_experts = meta->shape[0]; // 288
	int hidden_size = meta->shape[1]; // 4096
	const backend::half_t* src = static_cast<const backend::half_t*>( engine::loader::safetensors_get_data_cpu( *repo.get_shard( shard_idx ), meta ) );
	if ( !src ) return pod::TensorView{};
	// The NVFP4 checkpoint stores the gate in BF16 (a BF16 gate read as FP16 routes every token wrongly)
	std::vector<backend::half_t> converted;
	if ( meta->dtype == pod::DType::BF16 ) {
		converted.resize( static_cast<size_t>( num_experts ) * hidden_size );
		const uint16_t* bf = reinterpret_cast<const uint16_t*>( src );
		for ( size_t i = 0; i < converted.size(); ++i ) {
			const uint32_t u = static_cast<uint32_t>( bf[i] ) << 16;
			float f;
			std::memcpy( &f, &u, 4 );
			converted[i] = backend::from_float_half( f );
		}
		src = converted.data();
	}

	int shape[2] = { hidden_size, num_experts };
	pod::TensorView tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, shape, 2 );
	if ( tv.data != nullptr ) {
		std::vector<backend::half_t> h_transposed( static_cast<size_t>( hidden_size ) * num_experts );
		for ( int e = 0; e < num_experts; ++e ) {
			for ( int d = 0; d < hidden_size; ++d ) {
				h_transposed[d * num_experts + e] = src[e * hidden_size + d];
			}
		}
		backend::memcpy_sync( tv.data, h_transposed.data(), h_transposed.size() * sizeof( backend::half_t ), backend::Copy::H2D );
	}
	return tv;
}

} // namespace impl

Glm5NextPipeline::Glm5NextPipeline() = default;

Glm5NextPipeline::~Glm5NextPipeline() {
	destroy();
}

bool Glm5NextPipeline::init(
	const Glm5NextConfig& config,
	int rank,
	pod::MemoryArena& arena,
	engine::comm::CollectiveRing* ring
) {
	config_ = config;
	config_.tp_size = ( ring != nullptr ) ? requested_tp_ : 1;
	ENGINE_ASSERT( config_.tp_size == 1 || config_.tp_size == 2 || config_.tp_size == ENGINE_NUM_GPUS, "GLM runs on 1, 2 or 4 ranks" );
	// 64 attention heads (MLA and KDA alike) split over the ranks: 16 per rank on four, 32 on two, all 64 on one
	lh_ = 64 / config_.tp_size;
	config_.kda_heads = lh_;
	config_.kda_channels = lh_ * 3 * config_.kda_head_dim;
	rank_ = rank;
#if ENGINE_BACKEND_CUDA
	backend::set_device( rank_ );
	side_stream_ = backend::stream_create();
#endif
	arena_ = &arena;
	ring_ = ring;
	weights_loaded_ = false;

	backend::set_device( rank_ );

	const int R = pod::MAX_ROWS;
	const int S = config_.max_slots;
	size_t mem_mark = arena.offset_bytes;
	const auto mem = [&]( const char* what ) {
		if ( rank_ == 0 ) LOG_INFO( "[mem] rank 0: %s %.2f GiB\n", what, ( arena.offset_bytes - mem_mark ) / 1073741824.0 );
		mem_mark = arena.offset_bytes;
	};
	// The MTP head (speculator=mtp) is one more MLA layer with its own latent cache and indexer state
	mtp_on_ = runtime::options().speculator == "mtp";
	mla_slots_ = config_.num_mla_layers + ( mtp_on_ ? 1 : 0 );
	// Routed experts: sharded over the ranks (every rank holds a quarter of every expert's intermediate width, moe_tp,
	// default) or expert-parallel (72 whole experts per rank, forced by the two-tier expert cache). The sharded layout balances
	// the load: each active expert costs every rank the same, where expert-parallel waits for the rank that drew the most.
	moe_tp_ = engine::modules::ExpertTier::resident_fraction() >= 1.0 && config_.tp_size == ENGINE_NUM_GPUS;
	const int E_local = moe_tp_ ? config_.num_experts : config_.num_experts / config_.tp_size;

	// 1. Router outputs + routed-MoE dispatch scratch
	state_.d_topk_indices = static_cast<int*>( engine::core::arena_alloc( arena, static_cast<size_t>( R ) * config_.top_k * sizeof( int ) ) );
	state_.d_topk_weights = static_cast<backend::half_t*>( engine::core::arena_alloc( arena, static_cast<size_t>( R ) * config_.top_k * sizeof( backend::half_t ) ) );
	state_.moe = engine::modules::MoE::alloc_scratch( arena, E_local, config_.top_k, config_.hidden_size, config_.intermediate_size, R, mtp_on_ );
	// The fused decode kernels read cold experts from host memory a little slower than the persistent kernel (6.4 vs 6.8 tok/s at a
	// 50% tier), but with most experts resident they are far faster (the persistent kernel costs ~11 ms a token more at 99% resident): only a
	// small resident set keeps the persistent path. expert_cache=auto decides when the fraction is known
#if !defined(USE_ROCM)
	if ( const double f = engine::modules::ExpertTier::resident_fraction(); f > 0.0 && f < kFusedTierMinFraction ) state_.moe.coop_ctr = nullptr;
#endif   // the ROCm build has no persistent kernel (its tensor-core path is emulated, 70 ms a layer): the fused launch serves the tier too

	mem( "router outputs and MoE scratch" );
	// KDA recurrent state + conv history per slot: [S, 45, ...] FP32, plus one-slot checkpoints
	// Only the linear-attention layers carry state: every fourth layer (3, 7, ...) is MLA
	state_.layer_kda_floats = static_cast<size_t>( config_.kda_heads ) * config_.kda_head_dim * config_.kda_head_dim;
	state_.layer_conv_floats = static_cast<size_t>( config_.kda_channels ) * 3;
	const int n_lin = config_.num_layers - config_.num_mla_layers;
	ENGINE_ASSERT( config_.num_layers == 45 && n_lin == 34, "Glm5NextPipeline: linear-layer indexing assumes 45 layers with every fourth MLA" );
	const size_t slot_kda = state_.layer_kda_floats * n_lin;
	const size_t slot_conv = state_.layer_conv_floats * n_lin;
	state_.slot_kda_floats = slot_kda;
	state_.slot_conv_floats = slot_conv;
	state_.kda_states = static_cast<float*>( engine::core::arena_alloc( arena, S * slot_kda * sizeof( float ) ) );
	state_.conv_history = static_cast<float*>( engine::core::arena_alloc( arena, S * slot_conv * sizeof( float ) ) );
	state_.kda_checkpoint = static_cast<float*>( engine::core::arena_alloc( arena, slot_kda * sizeof( float ) ) );
	state_.conv_checkpoint = static_cast<float*>( engine::core::arena_alloc( arena, slot_conv * sizeof( float ) ) );
	backend::memset_sync( state_.kda_states, 0, S * slot_kda * sizeof( float ) );
	backend::memset_sync( state_.conv_history, 0, S * slot_conv * sizeof( float ) );
	// Speculative verify (DFlash2). Journal mode: the recurrent state is left alone by the verify forward and each row's k, v, decay and beta are
	// recorded (16 KB a layer and row); the commit replays the accepted rows. It replaces a 36 MiB state snapshot per row and holds the blocks of two slots
	// (a joint round). The MTP speculator and the backends without the kernels keep the per-row snapshots
	use_journal_ = !mtp_on_ && engine::modules::GDN::kda_journal_supported();
	state_.snap_rows_cap = use_journal_ ? 2 * kDraftM : kDraftM;
	if ( use_journal_ ) {
		state_.journal_row_stride = engine::modules::GDN::kda_journal_row_bytes( config_.kda_heads );
		state_.journal_layer_stride = static_cast<size_t>( state_.snap_rows_cap ) * state_.journal_row_stride;
		state_.kda_journal = static_cast<unsigned char*>( engine::core::arena_alloc( arena, static_cast<size_t>( n_lin ) * state_.journal_layer_stride, 128 ) );
		ENGINE_ASSERT( state_.kda_journal != nullptr, "Glm5NextPipeline::init: arena exhausted allocating the KDA journal" );
	}
	else {
		state_.kda_snap = static_cast<float*>( engine::core::arena_alloc( arena, kDraftM * slot_kda * sizeof( float ) ) );
	}
	state_.conv_snap = static_cast<float*>( engine::core::arena_alloc( arena, static_cast<size_t>( state_.snap_rows_cap ) * slot_conv * sizeof( float ) ) );
	state_.d_snap_on = static_cast<int*>( engine::core::arena_alloc( arena, sizeof( int ) ) );
	backend::memset_sync( state_.d_snap_on, 0, sizeof( int ) );

	mem( "KDA states, checkpoint and per-row snapshots" );
	// 2. Activations: [R, ...] row buffers, viewed as [m, ...] per forward
	int rows_shape[2] = { R, config_.hidden_size };
	embed_rows_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, rows_shape, 2 );
	d_token_ids_ = static_cast<int*>( engine::core::arena_alloc( arena, R * sizeof( int ) ) );
	d_rows_ = static_cast<pod::BatchRows*>( engine::core::arena_alloc( arena, sizeof( pod::BatchRows ) ) );
	int h_shape[2] = { R, config_.hidden_size };
	hidden_states_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	norm_hidden_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );

	int streams_shape[2] = { 4 * R, config_.hidden_size };
	hidden_streams_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, streams_shape, 2 );
	mhc_collapsed_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );

	int rms_shape[1] = { R };
	mhc_inv_rms_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, rms_shape, 1 );

	int mix_shape[1] = { 24 * R };
	mhc_mix_logits_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, mix_shape, 1 );

	int hc_shape[1] = { 4 * R };
	mhc_pre_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, hc_shape, 1 );
	mhc_post_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, hc_shape, 1 );

	int comb_shape[2] = { 4 * R, 4 };
	mhc_comb_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, comb_shape, 2 );

	mem( "activations" );
	// 3. MLA KV cache pool [S, 11, max_seq, 16 * 256] FP16 & execution buffers
	attn_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	{
		latent_slot_stride_ = static_cast<size_t>( mla_slots_ ) * config_.max_seq * 512;
		const int pool_opt = runtime::options().kv_pool;
		// Latent and indexer arrays: one per slot, or one pool of pages for all of them
		size_t cache_slots = S;
		if ( pool_opt > 0 ) {
			ENGINE_ASSERT( runtime::options().dsa, "Glm5NextPipeline::init: kv_pool needs the DSA indexer pools (dsa) in the same paged layout" );
			const int page = 1 << kKvPageShift;
			kv_pool_tokens_ = ( pool_opt + page - 1 ) / page * page;
			kv_pages_per_slot_ = ( config_.max_seq + page - 1 ) / page;
			kv_table_.assign( static_cast<size_t>( S ) * kv_pages_per_slot_, 0 );
			kv_slot_pages_.assign( S, 0 );
			for ( int p = kv_pool_tokens_ / page - 1; p >= 0; --p ) kv_free_.push_back( p );
			d_kv_table_ = static_cast<int*>( engine::core::arena_alloc( arena, kv_table_.size() * sizeof( int ), 128 ) );
			ENGINE_ASSERT( d_kv_table_ != nullptr, "Glm5NextPipeline::init: arena exhausted allocating the page table" );
			backend::memset_sync( d_kv_table_, 0, kv_table_.size() * sizeof( int ) );
			latent_slot_stride_ = static_cast<size_t>( mla_slots_ ) * kv_pool_tokens_ * 512;
			cache_slots = 1;
			if ( rank_ == 0 ) LOG_INFO( "[kv] shared token pool: %d tokens in pages of %d for %d slots of up to %d tokens\n", kv_pool_tokens_, page, S, config_.max_seq );
		}
		// kv_format applies to the latent as one tensor: k8v4 stores it as 8-bit (K precision), the others as named
		const std::string& fmt = engine::runtime::options().kv_format;
		if ( fmt == "fp8" ) latent_q_ = pod::KVQ::E5M2;
		else if ( fmt == "k8v4" ) latent_q_ = pod::KVQ::I8;
		else if ( fmt == "nvfp4" ) latent_q_ = pod::KVQ::FP4;
		else latent_q_ = pod::KVQ::F16;
		const size_t data_bytes = cache_slots * latent_data_bytes( latent_slot_stride_ );
		const size_t scale_bytes = cache_slots * latent_scale_bytes( latent_slot_stride_ );
		latent_cache_ = engine::core::arena_alloc( arena, data_bytes, 128 );
		ENGINE_ASSERT( latent_cache_ != nullptr, "Glm5NextPipeline::init: arena exhausted allocating the latent cache" );
		backend::memset_sync( latent_cache_, 0, data_bytes );
		if ( scale_bytes ) {
			latent_scales_ = engine::core::arena_alloc( arena, scale_bytes, 128 );
			ENGINE_ASSERT( latent_scales_ != nullptr, "Glm5NextPipeline::init: arena exhausted allocating the latent scales" );
			backend::memset_sync( latent_scales_, 0, scale_bytes );
		}
		mem( "latent cache" );
		// With the indexer on, a row attends to at most DSA::KPAD (~2.1K) keys: dense below the selection threshold, a selected list above it. The
		// split-KV partials then need chunks for ~4K keys, not for the whole context
		ws_seq_ = runtime::options().dsa ? std::min( config_.max_seq, 4096 ) : config_.max_seq;
		latent_ws_ = static_cast<float*>( engine::core::arena_alloc( arena, engine::modules::MLALatent::workspace_floats( ws_seq_, lh_ ) * sizeof( float ), 128 ) );
		ENGINE_ASSERT( latent_ws_ != nullptr, "Glm5NextPipeline::init: arena exhausted allocating the latent workspace" );
		mem( "attention split-KV workspace" );
		int lat_shape[2] = { R, lh_ * 512 };
		q_lat_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, lat_shape, 2 );
		o_lat_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, lat_shape, 2 );

		dsa_ = runtime::options().dsa;
		if ( dsa_ ) {
			const int NP = config_.max_seq / engine::modules::DSA::POOL;
			dsa_plane_stride_ = static_cast<size_t>( mla_slots_ ) * engine::modules::DSA::PLANE_CAP * engine::modules::DSA::PLANE;
			dsa_pool_stride_ = static_cast<size_t>( mla_slots_ ) * ( kv_pool_tokens_ > 0 ? kv_pool_tokens_ / engine::modules::DSA::POOL : NP ) * engine::modules::DSA::DIM;
			auto halves = [&]( size_t n ) { return static_cast<backend::half_t*>( engine::core::arena_alloc( arena, n * sizeof( backend::half_t ), 128 ) ); };
			dsa_plane_ = halves( static_cast<size_t>( S ) * dsa_plane_stride_ );
			dsa_pool_ = halves( cache_slots * dsa_pool_stride_ );
			dsa_proj_ = halves( static_cast<size_t>( R ) * engine::modules::DSA::PROJ );
			dsa_w_ = halves( static_cast<size_t>( R ) * engine::modules::DSA::HEADS );
			dsa_q_ = halves( static_cast<size_t>( R ) * engine::modules::DSA::HEADS * engine::modules::DSA::DIM );
			dsa_scores_ = halves( static_cast<size_t>( R ) * NP );
			dsa_idx_ = static_cast<int*>( engine::core::arena_alloc( arena, static_cast<size_t>( R ) * engine::modules::DSA::KPAD * sizeof( int ), 128 ) );
			dsa_nk_ = static_cast<int*>( engine::core::arena_alloc( arena, static_cast<size_t>( R ) * sizeof( int ), 128 ) );
			ENGINE_ASSERT( dsa_plane_ && dsa_pool_ && dsa_proj_ && dsa_w_ && dsa_q_ && dsa_scores_ && dsa_idx_ && dsa_nk_, "Glm5NextPipeline::init: arena exhausted allocating the DSA state" );
			backend::memset_sync( dsa_plane_, 0, static_cast<size_t>( S ) * dsa_plane_stride_ * sizeof( backend::half_t ) );
			backend::memset_sync( dsa_pool_, 0, cache_slots * dsa_pool_stride_ * sizeof( backend::half_t ) );
			mem( "DSA indexer planes, pooled keys and scratch" );
		}
	}

	int q_lora_shape[2] = { R, 1536 };
	q_lora_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_lora_shape, 2 );
	q_lora_norm_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_lora_shape, 2 );

	int q_shape[2] = { R, lh_ * 256 };
	q_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_shape, 2 );

	int kv_lora_shape[2] = { R, 512 };
	kv_lora_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_lora_shape, 2 );
	kv_lora_norm_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_lora_shape, 2 );

	int kv_head_shape[2] = { R, lh_ * 256 };
	k_head_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_head_shape, 2 );
	v_head_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_head_shape, 2 );
	mla_heads_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_head_shape, 2 );

	// KDA linear attention buffers
	int qkv_shape[2] = { R, config_.kda_channels };
	qkv_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, qkv_shape, 2 );

	int fa_shape[2] = { R, 128 };
	f_a_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, fa_shape, 2 );
	g_a_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, fa_shape, 2 );

	int beta_shape[2] = { R, lh_ };
	beta_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, beta_shape, 2 );

	int gdec_shape[2] = { R, lh_ * 128 };
	g_decay_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, gdec_shape, 2 );

	int z_shape[2] = { R, lh_ * 128 };
	z_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, z_shape, 2 );
	attn_head_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, z_shape, 2 );

	mem( "latent cache, attention workspaces" );
	// 4. Dense / shared-expert FFN buffers
	int moe_inter_shape[2] = { R, config_.dense_intermediate_size };
	moe_gate_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, moe_inter_shape, 2 );
	moe_up_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, moe_inter_shape, 2 );
	moe_inter_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, moe_inter_shape, 2 );
	moe_expert_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	local_moe_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	ffn_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );

	mem( "FFN buffers" );
	// 5. Logits & argmax: the LM head is vocabulary-parallel, every rank owns vocab_size / tp_size columns padded
	// up to a multiple of 256 (so the dp4a int8 GEMV engages); lm_shard_real_cols_ excludes the pad
	ENGINE_ASSERT( config_.vocab_size % 128 == 0, "Glm5NextPipeline: vocab_size must be a multiple of 128" );
	lm_shard_real_cols_ = 128 * lm_shard_spans( config_.vocab_size, config_.tp_size, rank_ );
	lm_shard_offset_ = lm_shard_offset( config_.vocab_size, config_.tp_size, rank_ );
	{
		const int widest = 128 * lm_shard_spans( config_.vocab_size, config_.tp_size, 0 );
		lm_shard_cols_ = widest + ( 256 - widest % 256 ) % 256;   // uniform across ranks so logits gather with one stride
	}
	{
		const int B = pod::MAX_BATCH;
		logit_rows_ = std::max( B, runtime::options().glm_logit_rows );
		int logit_shape[2] = { logit_rows_, lm_shard_cols_ };
		logits_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, logit_shape, 2 );
		d_argmax_idx_ = static_cast<int*>( engine::core::arena_alloc( arena, B * sizeof( int ) ) );
		d_argmax_val_ = static_cast<float*>( engine::core::arena_alloc( arena, B * sizeof( float ) ) );
		h_argmax_idx_ = static_cast<int*>( engine::core::host_alloc_pinned( B * sizeof( int ) ) );
		std::memset( h_argmax_idx_, 0, B * sizeof( int ) );
		const int splits = ( lm_shard_cols_ + kTopkTile - 1 ) / kTopkTile;
		const size_t per = static_cast<size_t>( B ) * splits * 16;
		samp_part_v_ = static_cast<float*>( engine::core::arena_alloc( arena, per * sizeof( float ) ) );
		samp_part_i_ = static_cast<int*>( engine::core::arena_alloc( arena, per * sizeof( int ) ) );
		samp_counters_ = static_cast<int*>( engine::core::arena_alloc( arena, B * sizeof( int ) ) );
		d_samp_vals_ = static_cast<float*>( engine::core::arena_alloc( arena, static_cast<size_t>( B ) * 16 * sizeof( float ) ) );
		d_samp_idx_ = static_cast<int*>( engine::core::arena_alloc( arena, static_cast<size_t>( B ) * 16 * sizeof( int ) ) );
		backend::memset_sync( samp_counters_, 0, B * sizeof( int ) );
		if ( rank_ == 0 ) {
			// Only whole-distribution consumers (the drafter's tests) need the gathered logits
			int full_shape[2] = { logit_rows_, config_.vocab_size };
			full_logits_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, full_shape, 2 );
		}
	}

	mem( "logits" );
	// 6. DFlash2 attachment: a position-indexed tap ring per slot (every slot's rows publish to its own, so each can feed the drafter), head scratch,
	// candidate top-k scratch
	{
		draft_hooks_.taps_cap = MAX( MIN( config_.max_seq, 2048 ), 256 );
		int wtap_shape[2] = { S * draft_hooks_.taps_cap, 5 * config_.hidden_size };
		draft_hooks_.taps_slot_stride = static_cast<size_t>( draft_hooks_.taps_cap ) * 5 * config_.hidden_size;
		draft_taps_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, wtap_shape, 2 );
		backend::memset_sync( draft_taps_.data, 0, static_cast<size_t>( wtap_shape[0] ) * wtap_shape[1] * sizeof( backend::half_t ) );
		d_taps_owner_ = static_cast<int*>( engine::core::arena_alloc( arena, sizeof( int ) ) );
		backend::memset_sync( d_taps_owner_, 0xFF, sizeof( int ) );
		draft_hooks_.taps = draft_taps_.ptr<const backend::half_t>();
		draft_hooks_.taps_owner = d_taps_owner_;
		// embedding: set when the table is loaded (host-resident, shared by every rank)
		int tap_shape[2] = { R, config_.hidden_size };
		tap_tmp_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, tap_shape, 2 );

		const int kM = kDraftM;
		int ds[2] = { kM, config_.hidden_size };
		draft_hooks_.lm_in = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ds, 2 ).ptr<backend::half_t>();
		int dl[2] = { kM, lm_shard_cols_ };
		draft_lm_logits_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, dl, 2 );
		const int splits = ( lm_shard_cols_ + kTopkTile - 1 ) / kTopkTile;
		const size_t per = static_cast<size_t>( kM ) * splits * kDraftTopK;
		draft_cand_part_v_ = static_cast<float*>( engine::core::arena_alloc( arena, per * sizeof( float ) ) );
		draft_cand_part_i_ = static_cast<int*>( engine::core::arena_alloc( arena, per * sizeof( int ) ) );
		draft_cand_counters_ = static_cast<int*>( engine::core::arena_alloc( arena, kM * sizeof( int ) ) );
		draft_hooks_.cand_vals = static_cast<float*>( engine::core::arena_alloc( arena, static_cast<size_t>( kM ) * kDraftTopK * sizeof( float ) ) );
		draft_hooks_.cand_idx = static_cast<int*>( engine::core::arena_alloc( arena, static_cast<size_t>( kM ) * kDraftTopK * sizeof( int ) ) );
		backend::memset_sync( draft_cand_counters_, 0, kM * sizeof( int ) );
		draft_hooks_.shard_offset = lm_shard_offset_;
		draft_hooks_.shard_real_cols = lm_shard_real_cols_;
		draft_hooks_.embedding = &g_embed_table;
	}

	if ( mtp_on_ ) {
		const int H = config_.hidden_size;
		mtp_emb_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
		mtp_xe_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
		mtp_xh_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
		mtp_stage_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
		mtp_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
		int in_shape[2] = { R, 2 * H };
		mtp_in_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, in_shape, 2 );
		int hist_shape[2] = { S * 2 * ( kDraftM + 1 ), H };   // a region per slot (the runner offsets by slot)
		mtp_hist_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, hist_shape, 2 );
		int chain_shape[2] = { 1, H };
		mtp_chain_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, chain_shape, 2 );
	}

#if ENGINE_BACKEND_CUDA
	engine::modules::trellis_get_workspace( rank_, 16 * 1024 * 1024 );
	engine::modules::trellis_get_locks( rank_ );
#endif

	metadata_.model_name = "glm5_next";
	metadata_.hidden_size = config_.hidden_size;
	metadata_.vocab_size = config_.vocab_size;
	metadata_.num_layers = config_.num_layers;
	metadata_.num_heads = lh_;
	metadata_.num_kv_heads = lh_;
	metadata_.head_dim = 256;
	metadata_.max_seq_len = config_.max_seq;
	metadata_.tp_size = config_.tp_size;
	metadata_.max_slots = S;
	metadata_.max_rows = R;
	metadata_.eos_ids[0] = 154820;
	metadata_.eos_ids[1] = 154827;
	metadata_.eos_ids[2] = 154829;

	mem( "DFlash2 taps, MTP and the rest" );
	return true;
}

bool Glm5NextPipeline::init( int device_id, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring ) {
	Glm5NextConfig config;
	if ( runtime::options().max_seq > 0 ) config.max_seq = runtime::options().max_seq;  // context window (KV pool size)
	if ( runtime::options().max_slots > 0 ) config.max_slots = runtime::options().max_slots;
	return init( config, device_id, arena, ring );
}

bool Glm5NextPipeline::rebalance_experts( backend::stream_t stream ) {
	if ( !tier_.active() ) return false;
	backend::set_device( rank_ );
	tier_.reset_window();
	int moved = 0;
	for ( int l = 0; l < tier_layers_; ++l ) moved += tier_.rebalance( l, stream );
	if ( ++tier_.rebalances % 8 == 0 ) tier_.save_heat();
	if ( runtime::options().expert_log ) {
		LOG_INFO( "[experts rank %d] window hit rate %.3f over %llu routes, %d swaps (%llu total)\n", rank_, tier_.hit_rate(),
			static_cast<unsigned long long>( tier_.window_routes ), moved, static_cast<unsigned long long>( tier_.swaps ) );
	}
	return true;
}

bool Glm5NextPipeline::expert_heat( std::vector<uint32_t>& out, int& layers, int& experts, bool reset ) {
	backend::set_device( rank_ );
	backend::device_sync();
	layers = 0;
	experts = moe_tp_ ? config_.num_experts : config_.num_experts / config_.tp_size;
	out.clear();
	for ( auto& lw : layers_ ) {
		if ( !lw.is_moe || lw.moe_weights.heat == nullptr ) continue;
		const size_t at = out.size();
		out.resize( at + experts );
		backend::memcpy_sync( out.data() + at, lw.moe_weights.heat, static_cast<size_t>( experts ) * sizeof( uint32_t ), backend::Copy::D2H );
		if ( reset ) backend::memset_sync( lw.moe_weights.heat, 0, static_cast<size_t>( experts ) * sizeof( uint32_t ) );
		++layers;
	}
	return layers > 0;
}

pod::KVRef Glm5NextPipeline::latent_ref() const {
	pod::KVRef r;
	r.k = latent_cache_;
	r.v = latent_cache_;
	r.ks = latent_scales_;
	r.vs = latent_scales_;
	r.kq = latent_q_;
	r.vq = latent_q_;
	r.pages = kv_pages();
	return r;
}

pod::KVPages Glm5NextPipeline::kv_pages() const {
	pod::KVPages pg;
	if ( kv_pool_tokens_ == 0 ) return pg;
	pg.table = d_kv_table_;
	pg.shift = kKvPageShift;
	pg.per_slot = kv_pages_per_slot_;
	pg.pool_tokens = static_cast<size_t>( kv_pool_tokens_ );
	return pg;
}

bool Glm5NextPipeline::reserve_tokens( int slot_id, int tokens ) {
	ENGINE_ASSERT( slot_id >= 0 && slot_id < config_.max_slots, "Glm5NextPipeline::reserve_tokens: slot out of range" );
	if ( kv_pool_tokens_ == 0 ) return true;
	const int need = ( std::min( tokens, config_.max_seq ) + ( 1 << kKvPageShift ) - 1 ) >> kKvPageShift;
	int& have = kv_slot_pages_[slot_id];
	if ( need <= have ) return true;
	if ( static_cast<int>( kv_free_.size() ) < need - have ) return false;
	for ( ; have < need; ++have ) {
		kv_table_[static_cast<size_t>( slot_id ) * kv_pages_per_slot_ + have] = kv_free_.back();
		kv_free_.pop_back();
	}
	kv_dirty_ = true;
	return true;
}

void Glm5NextPipeline::release_tokens( int slot_id ) {
	ENGINE_ASSERT( slot_id >= 0 && slot_id < config_.max_slots, "Glm5NextPipeline::release_tokens: slot out of range" );
	if ( kv_pool_tokens_ == 0 ) return;
	// Highest page first, so a slot that starts again takes the pool's lowest pages in order
	for ( int i = kv_slot_pages_[slot_id] - 1; i >= 0; --i ) {
		int& e = kv_table_[static_cast<size_t>( slot_id ) * kv_pages_per_slot_ + i];
		kv_free_.push_back( e );
		e = 0;
	}
	kv_slot_pages_[slot_id] = 0;
	kv_dirty_ = true;
}

int Glm5NextPipeline::auto_resident( const engine::loader::SafeTensorsRepo& repo, int layer, size_t layers_begin, size_t experts_cost, int layers_done, int experts, size_t slot_bytes, int moe_layers ) const {
	constexpr double kGiB = 1073741824.0;
	auto prefix = []( int l ) { return "model.language_model.layers." + std::to_string( l ) + "."; };
	// What the layers loaded so far took in the arena (less their experts) against their checkpoint bytes, applied to the checkpoint bytes still to
	// come: this layer's shared expert and the layers after it. The first estimate rests on the dense layers and one attention block; every
	// later layer refines it
	size_t counted = repo.bytes_of( prefix( layer ), { ".mlp.experts.", ".mlp.shared_experts." } ), total = 0;
	for ( int l = 0; l < layer; ++l ) counted += repo.bytes_of( prefix( l ), { ".mlp.experts." } );
	for ( int l = 0; l < config_.num_layers; ++l ) total += repo.bytes_of( prefix( l ), { ".mlp.experts." } );
	const size_t used = arena_->offset_bytes - layers_begin - experts_cost;
	const double per_ckpt_byte = counted ? static_cast<double>( used ) / static_cast<double>( counted ) : 1.0;
	// The shared experts are cut over the ranks (shared_slice): only 1/tp of their checkpoint bytes land here
	size_t shared_left = 0;
	if ( hess_.empty() && runtime::options().shared_slice && config_.tp_size > 1 ) {
		for ( int l = layer; l < config_.num_layers; ++l ) shared_left += repo.bytes_of( prefix( l ) + "mlp.shared_experts." );
	}
	const double shared_saved = static_cast<double>( shared_left ) * ( 1.0 - 1.0 / config_.tp_size );
	const double rest_ckpt = std::max( 0.0, static_cast<double>( total > counted ? total - counted : 0 ) - shared_saved );
	const size_t rest = static_cast<size_t>( per_ckpt_byte * rest_ckpt );
	// The device-side tensors every expert keeps (signs, scales), resident or not
	const size_t expert_all = repo.bytes_of( prefix( layer ) + "mlp.experts.0." );
	const int layers_left = moe_layers - layers_done;
	const size_t extras = ( expert_all > slot_bytes ? expert_all - slot_bytes : 0 ) * static_cast<size_t>( experts ) * static_cast<size_t>( layers_left );
	// The speculator (rank 0 holds the selector codebooks, the others about half of the file), the MTP head, and a margin for what the estimate misses
	const size_t speculator = rank_ == 0 ? reserve_after_weights_ : reserve_after_weights_ / 2;
	const size_t head = mtp_on_ ? ( 768ULL << 20 ) : 0, margin = 256ULL << 20;
	const size_t free_now = arena_->total_bytes - arena_->offset_bytes;
	const size_t other = rest + extras + speculator + head + margin;
	ENGINE_ASSERT( free_now > other, "Glm5NextPipeline: expert_cache=auto: the arena cannot hold the other weights; set max_seq / kv_format smaller or give expert_cache" );
	// The remaining layers share what is left evenly; the estimate only improves, so the last layers take up the slack
	const size_t per_resident = static_cast<size_t>( layers_left ) * slot_bytes;
	const int floor_res = std::max( 1, experts / 10 );
	auto fit = [&]( size_t reserved ) { return std::max( floor_res, std::min( experts, static_cast<int>( free_now > other + reserved ? ( free_now - other - reserved ) / per_resident : 0 ) ) ); };
	// Prefill staging keeps two layers' cold experts in device scratch (sized by the first layer's count): leave room for it
	const bool stage = runtime::options().expert_stage && !runtime::options().expert_stream;
	int resident = fit( 0 );
	if ( stage ) {
		const int base = tier_.resident > 0 ? tier_.resident : resident;
		resident = fit( 2 * static_cast<size_t>( experts - base ) * slot_bytes );
		if ( tier_.resident == 0 ) resident = fit( 2 * static_cast<size_t>( experts - resident ) * slot_bytes );
	}
	if ( rank_ == 0 && layers_done == 0 ) {
		LOG_INFO( "[experts] expert_cache auto: %d of %d experts per layer resident to start (%.1f%%), later layers take what the refined estimate leaves. Arena free %.2f GiB; still to load: other weights %.2f, expert scales %.2f, speculator %.2f%s; margin %.2f\n",
			resident, experts, 100.0 * resident / experts, free_now / kGiB, rest / kGiB, extras / kGiB, speculator / kGiB, mtp_on_ ? " + MTP head 0.75" : "", margin / kGiB );
	}
	return resident;
}

void Glm5NextPipeline::kv_prepare( const pod::BatchRows& rows, backend::stream_t stream ) {
	if ( kv_pool_tokens_ == 0 ) return;
	for ( int r = 0; r < rows.count; ++r ) {
		ENGINE_ASSERT( reserve_tokens( rows.slot[r], rows.pos[r] + 1 ), "Glm5NextPipeline: the shared token pool (option kv_pool) is full" );
	}
	if ( !kv_dirty_ ) return;
	backend::memcpy_async( d_kv_table_, kv_table_.data(), kv_table_.size() * sizeof( int ), backend::Copy::H2D, stream );
	kv_dirty_ = false;
}

template <typename F>
void Glm5NextPipeline::for_kv_ranges( int slot_id, int from, int tokens, F&& fn ) const {
	const int page = 1 << kKvPageShift;
	for ( int t = from; t < tokens; ) {
		const int n = std::min( tokens - t, page - ( t & ( page - 1 ) ) );
		fn( t, n, static_cast<size_t>( kv_table_[static_cast<size_t>( slot_id ) * kv_pages_per_slot_ + ( t >> kKvPageShift )] ) * page + ( t & ( page - 1 ) ) );
		t += n;
	}
}

void Glm5NextPipeline::reset_state( int slot_id ) {
	ENGINE_ASSERT( slot_id >= 0 && slot_id < config_.max_slots, "Glm5NextPipeline::reset_state: slot out of range" );
	backend::set_device( rank_ );
	// A pooled slot keeps its pages (the request that owns them ends with release_tokens); every position is written before it is read
	if ( kv_pool_tokens_ == 0 ) {
		backend::memset_sync( static_cast<char*>( latent_cache_ ) + static_cast<size_t>( slot_id ) * latent_data_bytes( latent_slot_stride_ ), 0, latent_data_bytes( latent_slot_stride_ ) );
		if ( latent_scales_ ) backend::memset_sync( static_cast<char*>( latent_scales_ ) + static_cast<size_t>( slot_id ) * latent_scale_bytes( latent_slot_stride_ ), 0, latent_scale_bytes( latent_slot_stride_ ) );
	}
	const size_t slot_kda = state_.slot_kda_floats;
	const size_t slot_conv = state_.slot_conv_floats;
	backend::memset_sync( state_.kda_states + slot_id * slot_kda, 0, slot_kda * sizeof( float ) );
	backend::memset_sync( state_.conv_history + slot_id * slot_conv, 0, slot_conv * sizeof( float ) );
}

void Glm5NextPipeline::checkpoint_state( int slot_id, backend::stream_t stream ) {
	ENGINE_ASSERT( slot_id >= 0 && slot_id < config_.max_slots, "Glm5NextPipeline::checkpoint_state: slot out of range" );
	backend::set_device( rank_ );
	const size_t slot_kda = state_.slot_kda_floats;
	const size_t slot_conv = state_.slot_conv_floats;
	backend::memcpy_async( state_.kda_checkpoint, state_.kda_states + slot_id * slot_kda, slot_kda * sizeof( float ), backend::Copy::D2D, stream );
	backend::memcpy_async( state_.conv_checkpoint, state_.conv_history + slot_id * slot_conv, slot_conv * sizeof( float ), backend::Copy::D2D, stream );
}

void Glm5NextPipeline::rollback_state( int slot_id, backend::stream_t stream ) {
	ENGINE_ASSERT( slot_id >= 0 && slot_id < config_.max_slots, "Glm5NextPipeline::rollback_state: slot out of range" );
	backend::set_device( rank_ );
	const size_t slot_kda = state_.slot_kda_floats;
	const size_t slot_conv = state_.slot_conv_floats;
	backend::memcpy_async( state_.kda_states + slot_id * slot_kda, state_.kda_checkpoint, slot_kda * sizeof( float ), backend::Copy::D2D, stream );
	backend::memcpy_async( state_.conv_history + slot_id * slot_conv, state_.conv_checkpoint, slot_conv * sizeof( float ), backend::Copy::D2D, stream );
}

bool Glm5NextPipeline::load_weights( const char* model_dir, backend::stream_t stream ) {
	size_t bytes_before = arena_->offset_bytes;
	engine::loader::SafeTensorsRepo repo;
	if ( !repo.open( model_dir ) ) {
		LOG_ERROR( "[loader] FAILED to open model directory %s\n", model_dir );
		return false;
	}

	int missing_count = 0;

	// Dense groups quantized to NVFP4 at load from the BF16 tensors of a modelopt checkpoint (ENGINE_NVFP4_DENSE_SRC, ENGINE_NVFP4_DENSE); the
	// routed experts stay EXL3. The dense MLP layers are NVFP4 in that checkpoint already and load as they are (group `dense`)
	std::unique_ptr<engine::loader::SafeTensorsRepo> nv_repo;
	if ( const std::string& nv_dir = runtime::options().nvfp4_dense_src; !nv_dir.empty() && ( !runtime::options().nvfp4_dense.empty() || mtp_on_ ) ) {
		nv_repo = std::make_unique<engine::loader::SafeTensorsRepo>();
		if ( !nv_repo->open( nv_dir.c_str() ) ) {
			LOG_ERROR( "[loader] FAILED to open the NVFP4 directory %s\n", nv_dir.c_str() );
			return false;
		}
	}
	const std::string& nv_groups = runtime::options().nvfp4_dense;
	auto nv_listed = [&]( const std::string& g ) { return ( "," + nv_groups + "," ).find( "," + g + "," ) != std::string::npos; };
	auto nv_on = [&]( const char* group ) {
		if ( !nv_repo ) return false;
		const std::string g = group;
		return nv_groups == "all" || nv_listed( g ) || ( g.rfind( "shared_", 0 ) == 0 && nv_listed( "shared" ) );
	};
	int nv_count = 0, nv_calibrated = 0;
	auto nv_check = [&]( pod::LinearWeight lw, const char* name ) {
		if ( !lw.is_valid() ) { LOG_ERROR( "[RANK %d] MISSING (NVFP4): %s\n", rank_, name ); ++missing_count; }
		++nv_count;
		return lw;
	};
	// Calibrated tensors (tests/tools/nvfp4_ldlq.py --model glm) when the directory has them, otherwise round-to-nearest from the BF16 weight
	std::unique_ptr<engine::loader::SafeTensorsRepo> ov_repo;
	if ( const std::string& ov_dir = runtime::options().nvfp4_dense_dir; !ov_dir.empty() && nv_repo ) {
		ov_repo = std::make_unique<engine::loader::SafeTensorsRepo>();
		if ( !ov_repo->open( ov_dir.c_str() ) ) {
			LOG_ERROR( "[loader] FAILED to open the calibrated NVFP4 directory %s\n", ov_dir.c_str() );
			return false;
		}
	}
	auto has_ov = [&]( const char* name ) { return ov_repo && ov_repo->find( ( std::string( name ) + ".weight_scale_2" ).c_str() ); };
	auto nv_slice = [&]( const char* name, bool split_n, int rank, int parts ) {
		if ( has_ov( name ) ) { ++nv_calibrated; return nv_check( ov_repo->load_nvfp4_slice( name, *arena_, split_n, rank, parts, stream ), name ); }
		return nv_check( nv_repo->quantize_nvfp4_slice( name, *arena_, split_n, rank, parts, 0, stream ), name );
	};

	if ( !runtime::options().hessian_dir.empty() ) {
		// Calibration accumulators (sites: see hessian()); replicated inputs live on rank 0, the o_proj inputs on every rank
		hess_.assign( static_cast<size_t>( config_.num_layers ) * 6, HessianSite{} );
		auto site = [&]( int layer, int idx, int k ) {
			HessianSite& hs = hess_[static_cast<size_t>( layer ) * 6 + idx];
			hs.k = k;
			hs.h = static_cast<float*>( engine::core::arena_alloc( *arena_, static_cast<size_t>( k ) * k * sizeof( float ), 128 ) );
			backend::memset_async( hs.h, 0, static_cast<size_t>( k ) * k * sizeof( float ), stream );
		};
		int hl_lo = 0, hl_hi = config_.num_layers - 1;
		if ( const std::string& hl = runtime::options().hessian_layers; !hl.empty() ) std::sscanf( hl.c_str(), "%d-%d", &hl_lo, &hl_hi );
		for ( int L = 0; L < config_.num_layers; ++L ) {
			if ( L < hl_lo || L > hl_hi ) continue;
			const bool mla = ( L % 4 == 3 ) && L < 44;
			if ( rank_ == 0 ) {
				site( L, 0, config_.hidden_size );
				if ( L >= 3 ) { site( L, 1, config_.hidden_size ); site( L, 3, 2048 ); }
				if ( mla ) site( L, 4, 1536 );
			}
			site( L, 2, mla ? 16384 / config_.tp_size : 8192 / config_.tp_size );
		}
		if ( rank_ == 0 && hl_lo == 0 ) site( 0, 5, config_.hidden_size );
		backend::stream_sync( stream );
	}

	// 1. Embedding table: 1.27 GB of BF16 kept in portable pinned host memory instead of rank 0's arena (which has
	// no room for the drafter beside it). Lookups touch a few rows per forward and read over the link; every rank
	// shares the one table (the drafter looks up its anchor token on all four).
	if ( rank_ == 0 ) {
		const char* ename = "model.language_model.embed_tokens.weight";
		int shard_idx = -1;
		const auto* meta = repo.find( ename, &shard_idx );
		if ( meta == nullptr || shard_idx < 0 ) { LOG_ERROR( "[RANK %d] MISSING: embed_tokens\n", rank_ ); ++missing_count; }
		else {
			const size_t bytes = static_cast<size_t>( config_.vocab_size ) * config_.hidden_size * sizeof( uint16_t );
			const void* src = engine::loader::safetensors_get_data_cpu( *repo.get_shard( shard_idx ), meta );
			if ( g_embed_table.data == nullptr ) {
				void* host = backend::host_alloc_pinned( bytes, backend::Pin::MappedPortable );
				ENGINE_ASSERT( host != nullptr, "Glm5NextPipeline: pinned embedding table allocation failed" );
				g_embed_table = pod::make_tensor_2d( host, pod::DType::BF16, config_.vocab_size, config_.hidden_size );
			}
			if ( src == nullptr ) { ++missing_count; }
			else { std::memcpy( g_embed_table.data, src, bytes ); }
		}
	}
	embed_tokens_ = g_embed_table;
	// Final norm + this rank's vocabulary shard of the LM head
	final_norm_ = repo.load( "model.language_model.norm.weight", *arena_, stream, pod::DType::FP16 );
	if ( !final_norm_.data ) { LOG_ERROR( "[RANK %d] MISSING: norm\n", rank_ ); ++missing_count; }
	if ( nv_on( "head" ) ) {
		const int begin = lm_shard_offset_, count = lm_shard_real_cols_;
		if ( has_ov( "lm_head" ) ) { ++nv_calibrated; lm_head_ = nv_check( ov_repo->load_nvfp4( "lm_head", *arena_, &begin, &count, 1, 0, config_.hidden_size, stream, lm_shard_cols_ ), "lm_head" ); }
		else lm_head_ = nv_check( nv_repo->quantize_nvfp4( "lm_head", *arena_, &begin, &count, 1, 0, config_.hidden_size, lm_shard_cols_, stream ), "lm_head" );
	}
	else lm_head_ = repo.load_linear_cols_padded( "lm_head", *arena_, 6, lm_shard_offset_ / 16, lm_shard_real_cols_ / 16, lm_shard_cols_ - lm_shard_real_cols_, stream );
	if ( !lm_head_.is_valid() ) { LOG_ERROR( "[RANK %d] MISSING: lm_head\n", rank_ ); ++missing_count; }

	// 2. Per-layer weights across 45 layers
	char name_buf[256];
	int experts_per_rank = moe_tp_ ? config_.num_experts : config_.num_experts / config_.tp_size; // 288 sharded, 72 expert-parallel
	int start_expert = moe_tp_ ? 0 : rank_ * experts_per_rank;

	static constexpr int GLM_QKV_BLOCKS[3] = { 512, 512, 512 };
	const int GLM_QKV_RANK_BLOCKS[3] = { 512 / config_.tp_size, 512 / config_.tp_size, 512 / config_.tp_size };
	static constexpr int GLM_CONV_CHANNELS[3] = { 8192, 8192, 8192 };
	const int GLM_CONV_RANK_CHANNELS[3] = { 8192 / config_.tp_size, 8192 / config_.tp_size, 8192 / config_.tp_size };

	const size_t layers_begin = arena_->offset_bytes;
	size_t wb_attn = 0, wb_router = 0, wb_shared = 0, wb_dense = 0;   // [mem] weights breakdown (rank 0 log)
	size_t experts_cost = 0;   // arena bytes the experts took so far (codes in device slots, signs and scales)
	int auto_base = 0;
	for ( int L = 0; L < config_.num_layers; ++L ) {
		auto& lw = layers_[L];
		lw.is_moe = ( L >= 3 );
		const size_t wb_layer0 = arena_->offset_bytes;
		size_t wb_shared_start = 0;

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.input_layernorm.weight", L );
		lw.input_norm = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.post_attention_layernorm.weight", L );
		lw.post_attn_norm = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.qkv_proj", L );
		{
			char trellis_name[300];
			std::snprintf( trellis_name, sizeof( trellis_name ), "%s.trellis", name_buf );
			if ( nv_on( "kda" ) && repo.find( trellis_name ) ) {
				// q / k / v are three [8192, 4096] BF16 tensors in the NVFP4 checkpoint; every rank takes 2048 rows of each, stacked into one weight
				char qn[3][300];
				const char* parts[3] = { "q_proj", "k_proj", "v_proj" };
				const char* qp[3];
				int begin[3], count[3];
				for ( int i = 0; i < 3; ++i ) {
					std::snprintf( qn[i], sizeof( qn[i] ), "model.language_model.layers.%d.self_attn.%s", L, parts[i] );
					qp[i] = qn[i];
					count[i] = 8192 / config_.tp_size;
					begin[i] = rank_ * count[i];
				}
				if ( has_ov( qp[0] ) ) { ++nv_calibrated; lw.qkv = nv_check( ov_repo->load_nvfp4_concat( qp, begin, count, 3, 0, config_.hidden_size, 0, *arena_, stream ), name_buf ); }
				else lw.qkv = nv_check( nv_repo->quantize_nvfp4_concat( qp, begin, count, 3, 0, config_.hidden_size, 0, *arena_, stream ), name_buf );
			}
			else lw.qkv = repo.load_grouped_trellis_slice( name_buf, *arena_, GLM_QKV_BLOCKS, GLM_QKV_RANK_BLOCKS, 3, rank_, stream );
		}
		lw.is_linear_attn = lw.qkv.is_valid();

		if ( lw.is_linear_attn ) {
			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.conv1d.weight", L );
			lw.conv1d_weights = repo.load_grouped_slice( name_buf, *arena_, GLM_CONV_CHANNELS, GLM_CONV_RANK_CHANNELS, 3, rank_, 0, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.A_log", L );
			lw.A_log = repo.load_slice( name_buf, *arena_, 0, rank_, config_.tp_size, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.dt_bias", L );
			lw.dt_bias = repo.load_slice( name_buf, *arena_, 0, rank_, config_.tp_size, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.b_proj.weight", L );
			lw.b_proj = repo.load_slice( name_buf, *arena_, 0, rank_, config_.tp_size, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.f_a_proj.weight", L );
			lw.f_a_proj = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.f_b_proj.weight", L );
			lw.f_b_proj = repo.load_slice( name_buf, *arena_, 0, rank_, config_.tp_size, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.g_a_proj.weight", L );
			lw.g_a_proj = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.g_b_proj.weight", L );
			lw.g_b_proj = repo.load_slice( name_buf, *arena_, 0, rank_, config_.tp_size, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.o_norm.weight", L );
			lw.o_norm = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.o_proj", L );
			lw.o_proj = nv_on( "kda" ) ? nv_slice( name_buf, false, rank_, config_.tp_size )
				: repo.load_linear_slice( name_buf, *arena_, 5, 0, rank_, config_.tp_size, stream );
		}

		bool is_mla = ( ( L % 4 == 3 ) && L < 44 );
		lw.is_mla = is_mla;
		if ( lw.is_mla ) {
			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.q_a_proj", L );
			lw.q_a = nv_on( "mla" ) ? nv_slice( name_buf, true, 0, 1 ) : repo.load_linear( name_buf, *arena_, 5, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.q_a_layernorm.weight", L );
			lw.q_a_norm = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.q_b_proj", L );
			lw.q_b = nv_on( "mla" ) ? nv_slice( name_buf, true, rank_, config_.tp_size )
				: repo.load_linear_slice( name_buf, *arena_, 5, 1, rank_, config_.tp_size, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.kv_a_proj_with_mqa", L );
			lw.kv_a = nv_on( "mla" ) ? nv_slice( name_buf, true, 0, 1 ) : repo.load_linear( name_buf, *arena_, 5, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.kv_a_layernorm.weight", L );
			lw.kv_a_norm = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.kv_b_proj.weight", L );
			lw.kv_b_proj = repo.load_slice( name_buf, *arena_, 0, rank_, config_.tp_size, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.o_proj", L );
			lw.o_proj = nv_on( "mla" ) ? nv_slice( name_buf, false, rank_, config_.tp_size )
				: repo.load_linear_slice( name_buf, *arena_, 5, 0, rank_, config_.tp_size, stream );

			if ( dsa_ ) {
				char pre[160];
				std::snprintf( pre, sizeof( pre ), "model.language_model.layers.%d.self_attn.indexer", L );
				std::snprintf( name_buf, sizeof( name_buf ), "%s.wq_b", pre );
				lw.idx_wq_b = nv_on( "idx" ) ? nv_slice( name_buf, true, 0, 1 ) : repo.load_linear( name_buf, *arena_, 5, stream );
				const int dshape[2] = { engine::modules::DSA::PROJ, 4096 };
				lw.idx_dense = engine::core::arena_alloc_tensor( *arena_, pod::DType::FP16, dshape, 2 );
				const size_t keep = arena_->offset_bytes;
				const char* parts[3] = { "wk.weight", "weights_proj.weight", "index_kpool_compress_gate" };
				const int part_rows[3] = { engine::modules::DSA::DIM, engine::modules::DSA::HEADS, engine::modules::DSA::DIM };
				int row0 = 0;
				for ( int i = 0; i < 3; ++i ) {
					std::snprintf( name_buf, sizeof( name_buf ), "%s.%s", pre, parts[i] );
					const pod::TensorView part = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );
					ENGINE_ASSERT( part.data != nullptr && part.shape[0] == part_rows[i] && part.shape[1] == 4096, "Glm5NextPipeline: indexer projection shape" );
					backend::memcpy_async( lw.idx_dense.ptr<backend::half_t>() + static_cast<size_t>( row0 ) * 4096, part.data, static_cast<size_t>( part_rows[i] ) * 4096 * sizeof( backend::half_t ), backend::Copy::D2D, stream );
					row0 += part_rows[i];
				}
				backend::stream_sync( stream );
				arena_->offset_bytes = keep; // the staging copies are dead
				std::snprintf( name_buf, sizeof( name_buf ), "%s.k_norm.weight", pre );
				lw.idx_ln_w = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );
				std::snprintf( name_buf, sizeof( name_buf ), "%s.k_norm.bias", pre );
				lw.idx_ln_b = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );
				std::snprintf( name_buf, sizeof( name_buf ), "%s.index_kpool_compress_ape", pre );
				lw.idx_ape = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );
			}
		}

		const size_t wb_attn_end = arena_->offset_bytes;
		wb_attn += wb_attn_end - wb_layer0;
		if ( lw.is_moe ) {
			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.gate.weight", L );
			lw.router_weights = impl::load_router_weights_glm( repo, name_buf, *arena_, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.gate.e_score_correction_bias", L );
			lw.e_score_bias = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

			wb_router += arena_->offset_bytes - wb_attn_end;
			// 72 local experts for this rank
			auto& mw = lw.moe_weights;
			mw.type = pod::QuantType::TRELLIS;
			mw.num_local_experts = experts_per_rank;
			// Tiered experts (ENGINE_EXPERT_CACHE < 1): the codes of resident experts go to device slots, the rest to pinned host memory
			double tier_fraction = engine::modules::ExpertTier::resident_fraction();
			const bool resident_fraction_auto = tier_fraction <= 0.0;
			size_t experts_cost_from = arena_->offset_bytes;
			int tier_layer = -1;
			if ( tier_fraction < 1.0 ) {
				if ( tier_layers_ == 0 ) {
					size_t bytes[3];
					const char* kinds[3] = { "gate_proj", "up_proj", "down_proj" };
					for ( int p = 0; p < 3; ++p ) {
						std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.experts.%d.%s", L, start_expert, kinds[p] );
						bytes[p] = repo.trellis_bytes( name_buf );
						if ( bytes[p] == 0 && runtime::options().expert_synth >= 2 ) bytes[p] = repo.synth_trellis_bytes( name_buf );
						ENGINE_ASSERT( bytes[p] > 0, "Glm5NextPipeline: expert codes not found for the tier" );
					}
					int moe_layers = 0;
					for ( int ll = 0; ll < config_.num_layers; ++ll ) moe_layers += ( ll >= 3 ) ? 1 : 0;
					if ( tier_fraction <= 0.0 ) {
						auto_base = auto_resident( repo, L, layers_begin, experts_cost, 0, experts_per_rank, bytes[0] + bytes[1] + bytes[2], moe_layers );
						tier_fraction = ( auto_base + 0.25 ) / experts_per_rank;
#if !defined(USE_ROCM)
						if ( tier_fraction < kFusedTierMinFraction ) state_.moe.coop_ctr = nullptr;
#endif
					}
					tier_.start( experts_per_rank, tier_fraction, bytes, rank_, config_.tp_size, moe_layers, runtime::options().expert_stream, host_arena_ );
				}
				experts_cost_from = arena_->offset_bytes;
				int layer_resident = tier_.resident;
				if ( resident_fraction_auto && tier_layers_ > 0 ) {
					layer_resident = std::max( tier_.resident, auto_resident( repo, L, layers_begin, experts_cost, tier_layers_, experts_per_rank, tier_.slot_bytes, config_.num_layers - 3 ) );
				}
				tier_.add_layer( &mw, *arena_, host_arena_, tier_.initial_set( tier_layers_, layer_resident ) );
				tier_layer = tier_layers_++;
			}
			for ( int e = 0; e < experts_per_rank; ++e ) {
				int gid = start_expert + e;
				pod::MemoryArena ca[3];
				if ( tier_layer >= 0 ) for ( int p = 0; p < 3; ++p ) tier_.code_arena( tier_layer, e, p, ca[p] );
				pod::MemoryArena* cp[3] = { tier_layer >= 0 ? &ca[0] : nullptr, tier_layer >= 0 ? &ca[1] : nullptr, tier_layer >= 0 ? &ca[2] : nullptr };
				if ( runtime::options().expert_synth >= 2 ) {
					// Timing only: experts that are not in the checkpoint. The resident ones are generated, a cold one is a copy of a resident one
					ENGINE_ASSERT( tier_layer >= 0 && !moe_tp_, "Glm5NextPipeline: ENGINE_DEBUG_EXPERT_SYNTH=2 needs expert_cache < 1 on one rank" );
					if ( e >= tier_.resident ) {
						mw.gate[e] = mw.gate[e % tier_.resident];
						mw.up[e] = mw.up[e % tier_.resident];
						mw.down[e] = mw.down[e % tier_.resident];
						continue;
					}
					static const char* const kinds[3] = { "gate_proj", "up_proj", "down_proj" };
					pod::LinearWeight* dst[3] = { &mw.gate[e], &mw.up[e], &mw.down[e] };
					for ( int p = 0; p < 3; ++p ) {
						std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.experts.%d.%s", L, gid, kinds[p] );
						*dst[p] = repo.synth_linear( name_buf, *arena_, stream, cp[p] );
						ENGINE_ASSERT( dst[p]->w0.data != nullptr, "Glm5NextPipeline: no entry for a generated expert in quantization_config.json" );
					}
					continue;
				}

				if ( tier_layer >= 0 && runtime::options().expert_stream ) {
					static const char* const kinds[3] = { "gate_proj", "up_proj", "down_proj" };
					for ( int p = 0; p < 3; ++p ) {
						std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.experts.%d.%s.trellis", L, gid, kinds[p] );
						int fd = -1; size_t off = 0, nb = 0;
						ENGINE_ASSERT( repo.file_ref( name_buf, &fd, &off, &nb ), "Glm5NextPipeline: expert_stream: tensor not found" );
						tier_.set_source( tier_layer, e, p, fd, off, nb );
					}
				}
				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.experts.%d.gate_proj", L, gid );
				mw.gate[e] = moe_tp_ ? repo.load_linear_slice( name_buf, *arena_, 4, 1, rank_, config_.tp_size, stream ) : repo.load_linear( name_buf, *arena_, 4, stream, cp[0] );

				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.experts.%d.up_proj", L, gid );
				mw.up[e] = moe_tp_ ? repo.load_linear_slice( name_buf, *arena_, 4, 1, rank_, config_.tp_size, stream ) : repo.load_linear( name_buf, *arena_, 4, stream, cp[1] );

				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.experts.%d.down_proj", L, gid );
				mw.down[e] = moe_tp_ ? repo.load_linear_slice( name_buf, *arena_, 4, 0, rank_, config_.tp_size, stream ) : repo.load_linear( name_buf, *arena_, 4, stream, cp[2] );
			}
			engine::modules::MoE::init_device_ptrs( mw, *arena_, stream );
			mw.heat = static_cast<uint32_t*>( engine::core::arena_alloc( *arena_, static_cast<size_t>( experts_per_rank ) * sizeof( uint32_t ), 128 ) );
			backend::memset_async( mw.heat, 0, static_cast<size_t>( experts_per_rank ) * sizeof( uint32_t ), stream );
			experts_cost += arena_->offset_bytes - experts_cost_from;

			// Shared experts
			wb_shared_start = arena_->offset_bytes;
			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.shared_experts.gate_proj", L );
			// Sliced over the ranks like the tensor-parallel experts (gate / up by output width, down by input width): 1/tp of the weights on each
			// rank instead of the whole shared expert on every rank for rank 0 to run. Hessian calibration keeps it whole on rank 0
			const bool sh_slice = hess_.empty() && runtime::options().shared_slice;
			const int sh_parts = sh_slice ? config_.tp_size : 1, sh_rank = sh_slice ? rank_ : 0;
			lw.shared_sliced = sh_parts > 1;
			lw.shared_expert.gate = nv_on( "shared_gate" ) ? nv_slice( name_buf, true, sh_rank, sh_parts ) : repo.load_linear_slice( name_buf, *arena_, 4, 1, sh_rank, sh_parts, stream );
			lw.has_shared_expert = lw.shared_expert.gate.is_valid();

			if ( lw.has_shared_expert ) {
				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.shared_experts.up_proj", L );
				lw.shared_expert.up = nv_on( "shared_up" ) ? nv_slice( name_buf, true, sh_rank, sh_parts ) : repo.load_linear_slice( name_buf, *arena_, 4, 1, sh_rank, sh_parts, stream );

				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.shared_experts.down_proj", L );
				lw.shared_expert.down = nv_on( "shared_down" ) ? nv_slice( name_buf, false, sh_rank, sh_parts ) : repo.load_linear_slice( name_buf, *arena_, 4, 0, sh_rank, sh_parts, stream );
			}
		}
		else {
			// Dense layer
			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.gate_proj", L );
			// Cut over the ranks like the shared expert (gate / up by output width, down by input width), a partial sum per rank that one all-reduce
			// completes: 1/tp of the weights on each rank instead of a copy on every rank computing the same product
			const bool dn_slice = hess_.empty() && runtime::options().shared_slice && config_.tp_size > 1 && ring_ != nullptr;
			const int dn_parts = dn_slice ? config_.tp_size : 1, dn_rank = dn_slice ? rank_ : 0;
			lw.dense_sliced = dn_slice;
			lw.dense_mlp.gate = nv_on( "dense" ) ? nv_check( nv_repo->load_nvfp4_slice( name_buf, *arena_, true, dn_rank, dn_parts, stream ), name_buf ) : repo.load_linear_slice( name_buf, *arena_, 4, 1, dn_rank, dn_parts, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.up_proj", L );
			lw.dense_mlp.up = nv_on( "dense" ) ? nv_check( nv_repo->load_nvfp4_slice( name_buf, *arena_, true, dn_rank, dn_parts, stream ), name_buf ) : repo.load_linear_slice( name_buf, *arena_, 4, 1, dn_rank, dn_parts, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.down_proj", L );
			lw.dense_mlp.down = nv_on( "dense" ) ? nv_check( nv_repo->load_nvfp4_slice( name_buf, *arena_, false, dn_rank, dn_parts, stream ), name_buf ) : repo.load_linear_slice( name_buf, *arena_, 4, 0, dn_rank, dn_parts, stream );
		}

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.hc_attn_base", L );
		lw.hc_attn_base = repo.load( name_buf, *arena_, stream, pod::DType::FP32 );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.hc_attn_scale", L );
		lw.hc_attn_scale = repo.load( name_buf, *arena_, stream, pod::DType::FP32 );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.hc_attn_fn", L );
		lw.hc_attn_fn = repo.load( name_buf, *arena_, stream, pod::DType::FP32 );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.hc_ffn_base", L );
		lw.hc_ffn_base = repo.load( name_buf, *arena_, stream, pod::DType::FP32 );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.hc_ffn_scale", L );
		lw.hc_ffn_scale = repo.load( name_buf, *arena_, stream, pod::DType::FP32 );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.hc_ffn_fn", L );
		lw.hc_ffn_fn = repo.load( name_buf, *arena_, stream, pod::DType::FP32 );

		if ( !lw.input_norm.data ) { LOG_ERROR( "[RANK %d L%d] MISSING: input_norm\n", rank_, L ); ++missing_count; }
		if ( !lw.post_attn_norm.data ) { LOG_ERROR( "[RANK %d L%d] MISSING: post_attn_norm\n", rank_, L ); ++missing_count; }
		if ( lw.is_linear_attn ) {
			if ( !lw.qkv.is_valid() ) { LOG_ERROR( "[RANK %d L%d] MISSING: qkv\n", rank_, L ); ++missing_count; }
			if ( !lw.conv1d_weights.data ) { LOG_ERROR( "[RANK %d L%d] MISSING: conv1d_weights\n", rank_, L ); ++missing_count; }
			if ( !lw.A_log.data ) { LOG_ERROR( "[RANK %d L%d] MISSING: A_log\n", rank_, L ); ++missing_count; }
			if ( !lw.dt_bias.data ) { LOG_ERROR( "[RANK %d L%d] MISSING: dt_bias\n", rank_, L ); ++missing_count; }
			if ( !lw.b_proj.data ) { LOG_ERROR( "[RANK %d L%d] MISSING: b_proj\n", rank_, L ); ++missing_count; }
			if ( !lw.f_a_proj.data ) { LOG_ERROR( "[RANK %d L%d] MISSING: f_a_proj\n", rank_, L ); ++missing_count; }
			if ( !lw.f_b_proj.data ) { LOG_ERROR( "[RANK %d L%d] MISSING: f_b_proj\n", rank_, L ); ++missing_count; }
			if ( !lw.g_a_proj.data ) { LOG_ERROR( "[RANK %d L%d] MISSING: g_a_proj\n", rank_, L ); ++missing_count; }
			if ( !lw.g_b_proj.data ) { LOG_ERROR( "[RANK %d L%d] MISSING: g_b_proj\n", rank_, L ); ++missing_count; }
			if ( !lw.o_norm.data ) { LOG_ERROR( "[RANK %d L%d] MISSING: o_norm\n", rank_, L ); ++missing_count; }
			if ( !lw.o_proj.is_valid() ) { LOG_ERROR( "[RANK %d L%d] MISSING: o_proj (linear)\n", rank_, L ); ++missing_count; }
		}
		if ( lw.is_mla ) {
			if ( !lw.q_a.is_valid() ) { LOG_ERROR( "[RANK %d L%d] MISSING: q_a\n", rank_, L ); ++missing_count; }
			if ( !lw.q_a_norm.data ) { LOG_ERROR( "[RANK %d L%d] MISSING: q_a_norm\n", rank_, L ); ++missing_count; }
			if ( !lw.q_b.is_valid() ) { LOG_ERROR( "[RANK %d L%d] MISSING: q_b\n", rank_, L ); ++missing_count; }
			if ( !lw.kv_a.is_valid() ) { LOG_ERROR( "[RANK %d L%d] MISSING: kv_a\n", rank_, L ); ++missing_count; }
			if ( !lw.kv_a_norm.data ) { LOG_ERROR( "[RANK %d L%d] MISSING: kv_a_norm\n", rank_, L ); ++missing_count; }
			if ( !lw.kv_b_proj.data ) { LOG_ERROR( "[RANK %d L%d] MISSING: kv_b_proj\n", rank_, L ); ++missing_count; }
			if ( !lw.o_proj.is_valid() ) { LOG_ERROR( "[RANK %d L%d] MISSING: o_proj (mla)\n", rank_, L ); ++missing_count; }
		}
		( lw.is_moe ? wb_shared : wb_dense ) += arena_->offset_bytes - ( lw.is_moe ? wb_shared_start : wb_attn_end );
		if ( lw.is_moe ) {
			if ( !lw.router_weights.data ) { LOG_ERROR( "[RANK %d L%d] MISSING: router_weights\n", rank_, L ); ++missing_count; }
			if ( !lw.e_score_bias.data ) { LOG_ERROR( "[RANK %d L%d] MISSING: e_score_bias\n", rank_, L ); ++missing_count; }
			int expert_miss = 0;
			for ( int e = 0; e < experts_per_rank; ++e ) {
				const auto& mw = lw.moe_weights;
				if ( !mw.gate[e].is_valid() || !mw.up[e].is_valid() || !mw.down[e].is_valid() ) {
					++expert_miss;
				}
			}
			if ( expert_miss > 0 ) {
				LOG_ERROR( "[RANK %d L%d] MISSING: %d/%d local experts invalid!\n", rank_, L, expert_miss, experts_per_rank );
				missing_count += expert_miss;
			}
			if ( !lw.shared_expert.gate.is_valid() || !lw.shared_expert.up.is_valid() || !lw.shared_expert.down.is_valid() ) {
				LOG_ERROR( "[RANK %d L%d] MISSING: shared_expert\n", rank_, L );
				++missing_count;
			}
		}
		else {
			if ( !lw.dense_mlp.gate.is_valid() || !lw.dense_mlp.up.is_valid() || !lw.dense_mlp.down.is_valid() ) {
				LOG_ERROR( "[RANK %d L%d] MISSING: dense_mlp\n", rank_, L );
				++missing_count;
			}
		}
		if ( !lw.hc_attn_base.data ) { LOG_ERROR( "[RANK %d L%d] MISSING: hc_attn_base\n", rank_, L ); ++missing_count; }
		if ( !lw.hc_attn_scale.data ) { LOG_ERROR( "[RANK %d L%d] MISSING: hc_attn_scale\n", rank_, L ); ++missing_count; }
		if ( !lw.hc_attn_fn.data ) { LOG_ERROR( "[RANK %d L%d] MISSING: hc_attn_fn\n", rank_, L ); ++missing_count; }
		if ( !lw.hc_ffn_base.data ) { LOG_ERROR( "[RANK %d L%d] MISSING: hc_ffn_base\n", rank_, L ); ++missing_count; }
		if ( !lw.hc_ffn_scale.data ) { LOG_ERROR( "[RANK %d L%d] MISSING: hc_ffn_scale\n", rank_, L ); ++missing_count; }
		if ( !lw.hc_ffn_fn.data ) { LOG_ERROR( "[RANK %d L%d] MISSING: hc_ffn_fn\n", rank_, L ); ++missing_count; }
	}

	// 3. MTP head (speculator=mtp): layer 45 of the NVFP4 checkpoint, a plain MLA + MoE layer without hyper-connections. Every dense tensor is
	// quantized to NVFP4 at load from the BF16 weights; the 288 routed experts too, as this rank's slice of each (gate / up rows, down columns)
	if ( mtp_on_ ) {
		if ( !nv_repo ) {
			LOG_ERROR( "[RANK %d] speculator=mtp needs nvfp4_dense_src (the NVFP4 checkpoint that holds the head)\n", rank_ );
			return false;
		}
		const int L = 45;
		auto& lw = mtp_layer_;
		char pre[160];
		std::snprintf( pre, sizeof( pre ), "model.language_model.layers.%d", L );
		auto nm = [&]( const char* tail ) { std::snprintf( name_buf, sizeof( name_buf ), "%s.%s", pre, tail ); return name_buf; };
		auto bf = [&]( const char* tail ) {
			const pod::TensorView t = nv_repo->load( nm( tail ), *arena_, stream, pod::DType::FP16 );
			if ( !t.data ) { LOG_ERROR( "[RANK %d] MISSING (MTP): %s\n", rank_, name_buf ); ++missing_count; }
			return t;
		};
		mtp_eh_ = nv_check( nv_repo->quantize_nvfp4_slice( nm( "eh_proj" ), *arena_, true, 0, 1, 0, stream ), name_buf );
		mtp_enorm_ = bf( "enorm.weight" );
		mtp_hnorm_ = bf( "hnorm.weight" );
		mtp_norm_ = bf( "shared_head.norm.weight" );
		lw.is_moe = true;
		lw.is_mla = true;
		lw.input_norm = bf( "input_layernorm.weight" );
		lw.post_attn_norm = bf( "post_attention_layernorm.weight" );
		lw.q_a = nv_check( nv_repo->quantize_nvfp4_slice( nm( "self_attn.q_a_proj" ), *arena_, true, 0, 1, 0, stream ), name_buf );
		lw.q_a_norm = bf( "self_attn.q_a_layernorm.weight" );
		lw.q_b = nv_check( nv_repo->quantize_nvfp4_slice( nm( "self_attn.q_b_proj" ), *arena_, true, rank_, config_.tp_size, 0, stream ), name_buf );
		lw.kv_a = nv_check( nv_repo->quantize_nvfp4_slice( nm( "self_attn.kv_a_proj_with_mqa" ), *arena_, true, 0, 1, 0, stream ), name_buf );
		lw.kv_a_norm = bf( "self_attn.kv_a_layernorm.weight" );
		lw.kv_b_proj = nv_repo->load_slice( nm( "self_attn.kv_b_proj.weight" ), *arena_, 0, rank_, config_.tp_size, stream, pod::DType::FP16 );
		if ( !lw.kv_b_proj.data ) { LOG_ERROR( "[RANK %d] MISSING (MTP): %s\n", rank_, name_buf ); ++missing_count; }
		lw.o_proj = nv_check( nv_repo->quantize_nvfp4_slice( nm( "self_attn.o_proj" ), *arena_, false, rank_, config_.tp_size, 0, stream ), name_buf );
		if ( dsa_ ) {
			lw.idx_wq_b = nv_check( nv_repo->quantize_nvfp4_slice( nm( "self_attn.indexer.wq_b" ), *arena_, true, 0, 1, 0, stream ), name_buf );
			const int dshape[2] = { engine::modules::DSA::PROJ, 4096 };
			lw.idx_dense = engine::core::arena_alloc_tensor( *arena_, pod::DType::FP16, dshape, 2 );
			const size_t keep = arena_->offset_bytes;
			const char* parts[3] = { "self_attn.indexer.wk.weight", "self_attn.indexer.weights_proj.weight", "self_attn.indexer.index_kpool_compress_gate" };
			const int part_rows[3] = { engine::modules::DSA::DIM, engine::modules::DSA::HEADS, engine::modules::DSA::DIM };
			int row0 = 0;
			for ( int i = 0; i < 3; ++i ) {
				const pod::TensorView part = nv_repo->load( nm( parts[i] ), *arena_, stream, pod::DType::FP16 );
				ENGINE_ASSERT( part.data != nullptr && part.shape[0] == part_rows[i] && part.shape[1] == 4096, "Glm5NextPipeline: MTP indexer projection shape" );
				backend::memcpy_async( lw.idx_dense.ptr<backend::half_t>() + static_cast<size_t>( row0 ) * 4096, part.data, static_cast<size_t>( part_rows[i] ) * 4096 * sizeof( backend::half_t ), backend::Copy::D2D, stream );
				row0 += part_rows[i];
			}
			backend::stream_sync( stream );
			arena_->offset_bytes = keep;
			lw.idx_ln_w = bf( "self_attn.indexer.k_norm.weight" );
			lw.idx_ln_b = bf( "self_attn.indexer.k_norm.bias" );
			lw.idx_ape = bf( "self_attn.indexer.index_kpool_compress_ape" );
		}
		lw.router_weights = impl::load_router_weights_glm( *nv_repo, nm( "mlp.gate.weight" ), *arena_, stream );
		if ( !lw.router_weights.data ) { LOG_ERROR( "[RANK %d] MISSING (MTP): %s\n", rank_, name_buf ); ++missing_count; }
		lw.e_score_bias = bf( "mlp.gate.e_score_correction_bias" );
		auto& mw = lw.moe_weights;
		mw.type = pod::QuantType::NVFP4;
		// The head's experts follow the layer layout of the trunk: every expert's intermediate width sliced over the ranks (moe_tp_), or
		// this rank's whole experts (expert-parallel, the tier)
		const int head_experts = moe_tp_ ? config_.num_experts : experts_per_rank;
		const int head_first = moe_tp_ ? 0 : start_expert;
		const int head_rank = moe_tp_ ? rank_ : 0, head_tp = moe_tp_ ? config_.tp_size : 1;
		mw.num_local_experts = head_experts;
		mw.top_k = config_.top_k;
		for ( int e = 0; e < head_experts; ++e ) {
			const int gid = head_first + e;
			std::snprintf( name_buf, sizeof( name_buf ), "%s.mlp.experts.%d.gate_proj", pre, gid );
			mw.gate[e] = nv_check( nv_repo->quantize_nvfp4_slice( name_buf, *arena_, true, head_rank, head_tp, 0, stream ), name_buf );
			std::snprintf( name_buf, sizeof( name_buf ), "%s.mlp.experts.%d.up_proj", pre, gid );
			mw.up[e] = nv_check( nv_repo->quantize_nvfp4_slice( name_buf, *arena_, true, head_rank, head_tp, 0, stream ), name_buf );
			std::snprintf( name_buf, sizeof( name_buf ), "%s.mlp.experts.%d.down_proj", pre, gid );
			mw.down[e] = nv_check( nv_repo->quantize_nvfp4_slice( name_buf, *arena_, false, head_rank, head_tp, 0, stream ), name_buf );
		}
		engine::modules::MoE::init_device_ptrs( mw, *arena_, stream );
		mw.heat = static_cast<uint32_t*>( engine::core::arena_alloc( *arena_, static_cast<size_t>( head_experts ) * sizeof( uint32_t ), 128 ) );
		backend::memset_async( mw.heat, 0, static_cast<size_t>( head_experts ) * sizeof( uint32_t ), stream );
		lw.shared_expert.gate = nv_check( nv_repo->quantize_nvfp4_slice( nm( "mlp.shared_experts.gate_proj" ), *arena_, true, 0, 1, 0, stream ), name_buf );
		lw.shared_expert.up = nv_check( nv_repo->quantize_nvfp4_slice( nm( "mlp.shared_experts.up_proj" ), *arena_, true, 0, 1, 0, stream ), name_buf );
		lw.shared_expert.down = nv_check( nv_repo->quantize_nvfp4_slice( nm( "mlp.shared_experts.down_proj" ), *arena_, true, 0, 1, 0, stream ), name_buf );
		lw.has_shared_expert = true;
		if ( rank_ == 0 ) LOG_INFO( "[RANK 0] MTP head from the NVFP4 checkpoint: layer %d, %d experts quantized to NVFP4 (this rank's %s)\n", L, head_experts, moe_tp_ ? "slice" : "whole experts" );
	}

	tier_.stage_init( *arena_ );
	size_t bytes_after = arena_->offset_bytes;
	if ( rank_ <= 1 ) {
		constexpr double kGiB = 1073741824.0;
		LOG_INFO( "[mem] rank %d weights: before the layers (lm_head, final norm) %.3f GiB, attention + norms + hyper-connections %.3f, routers %.3f, shared experts %.3f, dense MLPs %.3f, experts %.3f (codes, signs, scales), MTP head and the rest %.3f\n",
			rank_, ( layers_begin - bytes_before ) / kGiB, wb_attn / kGiB, wb_router / kGiB, wb_shared / kGiB, wb_dense / kGiB, experts_cost / kGiB,
			( bytes_after - layers_begin - wb_attn - wb_router - wb_shared - wb_dense - experts_cost ) / kGiB );
	}
	if ( rank_ == 0 && !tier_.layers.empty() ) {
		int lo = tier_.experts, hi = 0;
		double sum = 0.0;
		for ( const auto& l : tier_.layers ) {
			const int r = static_cast<int>( l.owner.size() );
			lo = std::min( lo, r );
			hi = std::max( hi, r );
			sum += r;
		}
		LOG_INFO( "[experts] resident per layer: %d to %d of %d (mean %.1f, %.1f%%)\n", lo, hi, tier_.experts, sum / tier_.layers.size(), 100.0 * sum / tier_.layers.size() / tier_.experts );
	}
	if ( nv_count > 0 ) LOG_INFO( "[RANK %d] NVFP4 dense tensors: %d, %d calibrated (groups %s)\n", rank_, nv_count, nv_calibrated, nv_groups.c_str() );
	if ( missing_count > 0 || rank_ == 0 ) {
		LOG_ERROR( "[RANK %d] load_weights complete: missing_count=%d | arena: before=%.3f GiB, after=%.3f GiB, loaded=%.3f GiB\n",
			rank_, missing_count,
			bytes_before / ( 1024.0 * 1024.0 * 1024.0 ),
			bytes_after / ( 1024.0 * 1024.0 * 1024.0 ),
			( bytes_after - bytes_before ) / ( 1024.0 * 1024.0 * 1024.0 ) );
	}

	if ( stream ) {
		backend::stream_sync( stream );
	}
	repo.close();
	weights_loaded_ = true;
	return true;
}

bool Glm5NextPipeline::load_vision( const char* model_dir, backend::stream_t stream ) {
	if ( rank_ != 0 || runtime::options().max_image_tokens <= 0 ) return false;
	backend::set_device( rank_ );
	return vision_.load( model_dir, *arena_, runtime::options().max_image_tokens * 4, stream );
}

bool Glm5NextPipeline::encode_image( const engine::loader::PreparedImage& image, std::vector<backend::half_t>* out, backend::stream_t stream ) {
	if ( rank_ != 0 || !vision_.loaded() ) return false;
	backend::set_device( rank_ );
	return vision_.encode( image, out, stream );
}

// Overwrites embedded rows (replicated on every rank after the embedding all-reduce) before the hyper-connection streams start
bool Glm5NextPipeline::inject_embeddings( const backend::half_t* host_rows, int dst_row, int rows, backend::stream_t stream ) {
	ENGINE_ASSERT( dst_row >= 0 && rows >= 1 && dst_row + rows <= pod::MAX_ROWS, "Glm5NextPipeline::inject_embeddings: rows out of range" );
	backend::set_device( rank_ );
	const size_t row_bytes = static_cast<size_t>( config_.hidden_size ) * sizeof( backend::half_t );
	backend::memcpy_async( embed_rows_.ptr<char>() + static_cast<size_t>( dst_row ) * row_bytes, host_rows, static_cast<size_t>( rows ) * row_bytes, backend::Copy::H2D, stream );
	return true;
}

void Glm5NextPipeline::embed_tokens( const int* token_ids, int count, backend::stream_t stream ) {
	ENGINE_ASSERT( count >= 1 && count <= pod::MAX_ROWS, "Glm5NextPipeline::embed_tokens: count out of range" );
	backend::set_device( rank_ );
	set_rows( count );
	// Rank 0 hosts the vocabulary table; other ranks contribute zeros to the all-reduce broadcast
	if ( rank_ == 0 ) {
		backend::memcpy_async( d_token_ids_, token_ids, count * sizeof( int ), backend::Copy::H2D, stream );
		engine::modules::Embedding::forward( embed_tokens_, d_token_ids_, count, embed_rows_, stream );
	}
	else {
		backend::memset_async( embed_rows_.data, 0, embed_rows_.size_bytes(), stream );
	}
	if ( ring_ ) {
		ring_->all_reduce( embed_rows_, stream );
	}
}

void Glm5NextPipeline::set_rows( int m ) {
	hidden_states_.shape[0] = m;
	norm_hidden_.shape[0] = m;
	mhc_collapsed_.shape[0] = m;
	attn_out_.shape[0] = m;
	q_lora_buf_.shape[0] = m;
	q_lora_norm_buf_.shape[0] = m;
	q_buf_.shape[0] = m;
	kv_lora_buf_.shape[0] = m;
	kv_lora_norm_buf_.shape[0] = m;
	k_head_buf_.shape[0] = m;
	v_head_buf_.shape[0] = m;
	mla_heads_out_.shape[0] = m;
	qkv_buf_.shape[0] = m;
	f_a_buf_.shape[0] = m;
	g_a_buf_.shape[0] = m;
	beta_buf_.shape[0] = m;
	g_decay_buf_.shape[0] = m;
	z_buf_.shape[0] = m;
	attn_head_out_.shape[0] = m;
	moe_gate_buf_.shape[0] = m;
	moe_up_buf_.shape[0] = m;
	moe_inter_buf_.shape[0] = m;
	moe_expert_out_.shape[0] = m;
	local_moe_out_.shape[0] = m;
	ffn_out_.shape[0] = m;
	embed_rows_.shape[0] = m;
}

void Glm5NextPipeline::sample_tokens( int* out_preds, int count, backend::stream_t stream ) {
	(void) out_preds;
	(void) count;
	(void) stream;
	ENGINE_ASSERT( false, "Glm5NextPipeline::sample_tokens: the vocabulary is sharded, use sample_local" );
}

bool Glm5NextPipeline::sample_local( int* out_tokens, float* out_vals, int count, backend::stream_t stream ) {
	ENGINE_ASSERT( count >= 1 && count <= last_rows_, "Glm5NextPipeline::sample_local: count exceeds last forward rows" );
	backend::set_device( rank_ );
	logits_.shape[0] = count;
	// Scan only this rank's real columns: the pad block decodes to 0.0 and would beat a real negative logit.
	// index_offset turns the shard-local winner into a global token id.
	engine::modules::Elementwise::batched_argmax_range(
		logits_, d_argmax_idx_, d_argmax_val_, lm_shard_real_cols_, count, lm_shard_offset_, stream );
	backend::memcpy_async( h_argmax_idx_, d_argmax_idx_, count * sizeof( int ), backend::Copy::D2H, stream );
	backend::stream_sync( stream );
	std::memcpy( out_tokens, h_argmax_idx_, count * sizeof( int ) );
	if ( out_vals != nullptr ) {
		float h_vals[pod::MAX_BATCH];
		backend::memcpy_async( h_vals, d_argmax_val_, count * sizeof( float ), backend::Copy::D2H, stream );
		backend::stream_sync( stream );
		std::memcpy( out_vals, h_vals, count * sizeof( float ) );
	}
	return true;
}

bool Glm5NextPipeline::topk_local( float* out_vals, int* out_ids, int count, backend::stream_t stream ) {
	ENGINE_ASSERT( count >= 1 && count <= last_rows_, "Glm5NextPipeline::topk_local: count exceeds last forward rows" );
	backend::set_device( rank_ );
	// Real columns only (pad columns decode to 0.0), global ids via the shard offset
	engine::spec::dflash2_topk_shard(
		logits_.ptr<const backend::half_t>(), count, lm_shard_cols_, lm_shard_real_cols_, 16,
		d_samp_vals_, d_samp_idx_, lm_shard_offset_, samp_part_v_, samp_part_i_, samp_counters_, stream );
	backend::memcpy_async( out_vals, d_samp_vals_, static_cast<size_t>( count ) * 16 * sizeof( float ), backend::Copy::D2H, stream );
	backend::memcpy_async( out_ids, d_samp_idx_, static_cast<size_t>( count ) * 16 * sizeof( int ), backend::Copy::D2H, stream );
	backend::stream_sync( stream );
	return true;
}

const pod::TensorView& Glm5NextPipeline::gather_logits(
	int rows,
	const backend::half_t* const* peer_shard_ptrs,
	int peer_shard_cols,
	backend::stream_t stream
) {
	ENGINE_ASSERT( rank_ == 0, "Glm5NextPipeline::gather_logits: only rank 0 materializes the full vocabulary" );
	ENGINE_ASSERT( rows >= 1 && rows <= last_rows_, "Glm5NextPipeline::gather_logits: rows exceeds last forward rows" );
	ENGINE_ASSERT( peer_shard_cols == lm_shard_cols_, "Glm5NextPipeline::gather_logits: shard width mismatch" );
	full_logits_.shape[0] = rows;
	// Row r of the gathered buffer is [shard 0 | shard 1 | ...]: source rows are lm_shard_cols_ apart (pad included),
	// destination rows lm_shard_real_cols_ * tp apart
	for ( int d = 0; d < config_.tp_size; ++d ) {
		const size_t shard_elems = static_cast<size_t>( 128 * lm_shard_spans( config_.vocab_size, config_.tp_size, d ) );
		const size_t shard_off = static_cast<size_t>( lm_shard_offset( config_.vocab_size, config_.tp_size, d ) );
		const backend::half_t* src_base = ( d == rank_ ) ? logits_.ptr<const backend::half_t>() : peer_shard_ptrs[d];
		ENGINE_ASSERT( src_base != nullptr, "Glm5NextPipeline::gather_logits: missing peer shard pointer" );
		for ( int r = 0; r < rows; ++r ) {
			const backend::half_t* src = src_base + static_cast<size_t>( r ) * lm_shard_cols_;
			backend::half_t* dst = full_logits_.ptr<backend::half_t>() + static_cast<size_t>( r ) * config_.vocab_size + shard_off;
			backend::peer_copy_async( dst, rank_, src, d, shard_elems * sizeof( backend::half_t ), stream );
		}
	}
	return full_logits_;
}

// Runs this rank's head shard over the drafter's hidden rows and leaves the local top-k in the hooks' cand buffers.
void Glm5NextPipeline::draft_head_topk( int rows, backend::stream_t stream ) {
	ENGINE_ASSERT( weights_loaded_, "Glm5NextPipeline::draft_head_topk: weights not loaded" );
	ENGINE_ASSERT( rows >= 1 && rows <= kDraftM, "Glm5NextPipeline::draft_head_topk: rows out of range" );
	backend::set_device( rank_ );
	// All kDraftM rows, whatever `rows` is: the head's fast path only covers m in {1, 2, 4, 8}
	pod::TensorView in = pod::make_tensor_2d( draft_hooks_.lm_in, pod::DType::FP16, kDraftM, config_.hidden_size );
	engine::modules::Linear::forward( in, lm_head_, draft_lm_logits_, stream );
	engine::spec::dflash2_topk_shard(
		draft_lm_logits_.ptr<const backend::half_t>(), kDraftM, lm_shard_cols_, lm_shard_real_cols_, kDraftTopK,
		draft_hooks_.cand_vals, draft_hooks_.cand_idx, draft_hooks_.shard_offset,
		draft_cand_part_v_, draft_cand_part_i_, draft_cand_counters_, stream );
}

void Glm5NextPipeline::set_spec_snapshot( bool on, backend::stream_t stream ) {
	backend::set_device( rank_ );
	const int v = on ? 1 : 0;
	backend::memcpy_async( state_.d_snap_on, &v, sizeof( int ), backend::Copy::H2D, stream );
}

void Glm5NextPipeline::commit_spec_rows( int slot_id, int first_row, int last_row, backend::stream_t stream ) {
	ENGINE_ASSERT( slot_id >= 0 && slot_id < config_.max_slots, "Glm5NextPipeline::commit_spec_rows: slot out of range" );
	ENGINE_ASSERT( first_row >= 0 && last_row >= first_row && last_row < state_.snap_rows_cap, "Glm5NextPipeline::commit_spec_rows: rows out of range" );
	if ( !use_journal_ ) {
		ENGINE_ASSERT( first_row == 0, "Glm5NextPipeline::commit_spec_rows: per-row snapshots cover one block" );
		commit_spec_row( slot_id, last_row, stream );
		return;
	}
	backend::set_device( rank_ );
	// The slot's state still holds its value before the block: apply the accepted rows from the journal, and take the conv history after the last one
	engine::modules::GDN::kda_replay( state_.kda_states, state_.slot_kda_floats, state_.layer_kda_floats, config_.num_layers - config_.num_mla_layers, slot_id,
		state_.kda_journal, state_.journal_layer_stride, state_.journal_row_stride, first_row, last_row - first_row + 1, config_.kda_heads, stream );
	backend::memcpy_async( state_.conv_history + static_cast<size_t>( slot_id ) * state_.slot_conv_floats, state_.conv_snap + static_cast<size_t>( last_row ) * state_.slot_conv_floats, state_.slot_conv_floats * sizeof( float ), backend::Copy::D2D, stream );
}

void Glm5NextPipeline::commit_spec_row( int slot_id, int row, backend::stream_t stream ) {
	ENGINE_ASSERT( slot_id >= 0 && slot_id < config_.max_slots, "Glm5NextPipeline::commit_spec_row: slot out of range" );
	ENGINE_ASSERT( row >= 0 && row < kDraftM, "Glm5NextPipeline::commit_spec_row: row out of range" );
	if ( use_journal_ ) {
		commit_spec_rows( slot_id, 0, row, stream );
		return;
	}
	backend::set_device( rank_ );
	backend::memcpy_async( state_.kda_states + static_cast<size_t>( slot_id ) * state_.slot_kda_floats, state_.kda_snap + static_cast<size_t>( row ) * state_.slot_kda_floats, state_.slot_kda_floats * sizeof( float ), backend::Copy::D2D, stream );
	backend::memcpy_async( state_.conv_history + static_cast<size_t>( slot_id ) * state_.slot_conv_floats, state_.conv_snap + static_cast<size_t>( row ) * state_.slot_conv_floats, state_.slot_conv_floats * sizeof( float ), backend::Copy::D2D, stream );
}

void Glm5NextPipeline::forward( const pod::BatchRows& rows, backend::stream_t stream ) {
	ENGINE_ASSERT( weights_loaded_, "Glm5NextPipeline::forward: weights not loaded" );
	ENGINE_ASSERT( rows.count >= 1 && rows.count <= pod::MAX_ROWS, "Glm5NextPipeline::forward: row count out of range" );
	backend::set_device( rank_ );
	for ( int r = 0; r < rows.count; ++r ) {
		ENGINE_ASSERT( rows.slot[r] >= 0 && rows.slot[r] < config_.max_slots, "Glm5NextPipeline::forward: slot out of range" );
		ENGINE_ASSERT( rows.pos[r] >= 0 && rows.pos[r] < config_.max_seq, "Glm5NextPipeline::forward: position out of range" );
	}
	const int m = rows.count;
	key_hint_ = rows.pos[m - 1] + 1;
	// Logits selection applies to eager prefill chunks only (decode graphs are keyed on m alone)
	lm_mode_ = ( m > pod::MAX_BATCH ) ? rows.logits_mode : pod::BatchRows::LOGITS_ALL;
	last_rows_ = ( lm_mode_ == pod::BatchRows::LOGITS_NONE ) ? 0 : ( lm_mode_ == pod::BatchRows::LOGITS_LAST ? 1 : m );
	ENGINE_ASSERT( last_rows_ <= logit_rows_, "Glm5NextPipeline::forward: more logit rows than the buffer holds (ENGINE_DEBUG_GLM_LOGIT_ROWS)" );
	set_rows( m );
	kv_prepare( rows, stream );
	// Pageable source: the copy is staged before return, so `rows` may be a temporary
	backend::memcpy_async( d_rows_, &rows, rows.upload_bytes(), backend::Copy::H2D, stream );
	graphs_.run( m, stream, [&]( backend::stream_t s ) { forward_body( m, s ); } );
}

void Glm5NextPipeline::forward_body( int m, backend::stream_t stream ) {
	// Slot-0 base pointers; slot and position of each row are resolved on device from d_rows_
	const int H = config_.hidden_size;

	// mHC: expand every row's embedding into its 4 residual streams
	if ( static const double eps = runtime::options().perturb_embed; eps != 0.0 && m > pod::MAX_BATCH ) {
		// Debug: rounding-scale noise on the embeddings, a function of (position, dim) only
		std::vector<backend::half_t> e( static_cast<size_t>( m ) * H );
		backend::stream_sync( stream );
		backend::memcpy_async( e.data(), embed_rows_.ptr<const backend::half_t>(), e.size() * sizeof( backend::half_t ), backend::Copy::D2H, stream );
		backend::stream_sync( stream );
		for ( int r = 0; r < m; ++r ) {
			for ( int d = 0; d < H; ++d ) {
				uint32_t h = static_cast<uint32_t>( key_hint_ - m + r ) * 2654435761u ^ static_cast<uint32_t>( d ) * 40503u;
				h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
				const float u = ( static_cast<float>( h & 0xffff ) / 32767.5f ) - 1.0f;
				backend::half_t& v = e[static_cast<size_t>( r ) * H + d];
				v = backend::from_float_half( backend::to_float( v ) * ( 1.0f + static_cast<float>( eps ) * u ) );
			}
		}
		backend::memcpy_async( embed_rows_.ptr<backend::half_t>(), e.data(), e.size() * sizeof( backend::half_t ), backend::Copy::H2D, stream );
		backend::stream_sync( stream );
	}
	engine::modules::HyperConnection::init_streams( embed_rows_, hidden_streams_, H, stream, m );



	for ( int layer = 0; layer < config_.num_layers; ++layer ) layer_body( layer, layers_[layer], m, stream );

	if ( layers_[0].hc_attn_fn.data != nullptr ) {
		// HyperHead unweighted mean across 4 streams
		engine::modules::HyperHead::mean( hidden_streams_, hidden_states_, H, stream, m );
	}

	// Final norm + this rank's vocabulary shard of the LM head
	if ( !hess_.empty() && m > pod::MAX_BATCH && hess_[5].h != nullptr ) {
		// Calibration passes skip the head (its logits buffer holds MAX_BATCH rows); the head's input statistics need the normed rows only
		engine::modules::RMSNorm::forward( hidden_states_, final_norm_, norm_hidden_, 1e-5f, 0.0f, stream );
		hessian( 0, 5, norm_hidden_.data, config_.hidden_size, m, stream );
	}
	// The MTP head reads the final-norm hidden of every prefill row, whether or not the chunk needs logits
	const bool norm_all = mtp_on_ && m > pod::MAX_BATCH;
	if ( norm_all ) engine::modules::RMSNorm::forward( hidden_states_, final_norm_, norm_hidden_, 1e-5f, 0.0f, stream );
	if ( lm_mode_ == pod::BatchRows::LOGITS_NONE ) return;
	if ( lm_mode_ == pod::BatchRows::LOGITS_LAST ) {
		// Prefill: only the final row's logits are read
		pod::TensorView h1 = pod::make_tensor_2d( hidden_states_.ptr<backend::half_t>() + static_cast<size_t>( m - 1 ) * H, pod::DType::FP16, 1, H );
		pod::TensorView n1 = pod::make_tensor_2d( norm_hidden_.ptr<backend::half_t>() + ( norm_all ? static_cast<size_t>( m - 1 ) * H : 0 ), pod::DType::FP16, 1, H );
		if ( norm_all ) {   // already normed above (n1 is the last row of norm_hidden_, which the head step still needs whole)
			pod::TensorView l1 = pod::make_tensor_2d( logits_.ptr<backend::half_t>(), pod::DType::FP16, 1, lm_shard_cols_ );
			engine::modules::Linear::forward( n1, lm_head_, l1, stream );
			return;
		}
		pod::TensorView l1 = pod::make_tensor_2d( logits_.ptr<backend::half_t>(), pod::DType::FP16, 1, lm_shard_cols_ );
		engine::modules::RMSNorm::forward( h1, final_norm_, n1, 1e-5f, 0.0f, stream );
		engine::modules::Linear::forward( n1, lm_head_, l1, stream );
		return;
	}
	logits_.shape[0] = m;
	engine::modules::RMSNorm::forward( hidden_states_, final_norm_, norm_hidden_, 1e-5f, 0.0f, stream );
	engine::modules::Linear::forward( norm_hidden_, lm_head_, logits_, stream );
}

void Glm5NextPipeline::layer_body( int layer, Glm5NextLayerWeights& lw, int m, backend::stream_t stream ) {
	const pod::SlotRef kv_ref{ d_rows_, 0, latent_slot_stride_ };
	const size_t slot_kda_floats = state_.slot_kda_floats;
	const size_t slot_conv_floats = state_.slot_conv_floats;
	const int snap_rows = MIN( m, state_.snap_rows_cap );
	const int H = config_.hidden_size;
	// Routed experts: this rank's 72 whole experts, or a quarter of all 288
	int experts_per_rank = moe_tp_ ? config_.num_experts : config_.num_experts / config_.tp_size;
	int start_expert = moe_tp_ ? 0 : rank_ * experts_per_rank;
	int end_expert = start_expert + experts_per_rank;
#if ENGINE_BACKEND_CUDA
	// Decode-sized rows: the all-reduce of a sublayer output and the mHC post-step that consumes it run as one kernel
	static const bool hc_fuse = runtime::options().hc_fuse && runtime::options().trace_layers.empty() && runtime::options().hessian_dir.empty();
	const bool fuse_reduce_update = hc_fuse && m <= pod::MAX_BATCH;
#else
	const bool fuse_reduce_update = false;
#endif
	const bool fuse_ffn_reduce = fuse_reduce_update && ring_ != nullptr && lw.hc_ffn_fn.data != nullptr && ( lw.is_moe || lw.dense_sliced );

	// Debug (trace_layers): header (tag, rows, first position) then the last row's four streams; tag = layer after the layer, 100 + layer after attention
	static const std::string& trace_path = runtime::options().trace_layers;
	auto trace_row = [&]( int tag ) {
		if ( trace_path.empty() || rank_ != 0 || m <= pod::MAX_BATCH ) return;
		std::vector<float> row( static_cast<size_t>( 4 ) * H );
		backend::stream_sync( stream );
		backend::memcpy_async( row.data(), hidden_streams_.ptr<const float>() + static_cast<size_t>( m - 1 ) * 4 * H, row.size() * sizeof( float ), backend::Copy::D2H, stream );
		backend::stream_sync( stream );
		if ( FILE* f = std::fopen( trace_path.c_str(), "ab" ) ) {
			const int hdr[3] = { tag, m, key_hint_ - m };
			std::fwrite( hdr, sizeof( int ), 3, f );
			std::fwrite( row.data(), sizeof( float ), row.size(), f );
			std::fclose( f );
		}
	};


		if ( lw.hc_attn_fn.data != nullptr ) {
			// mHC pre-step: attention
			// Collapse and the input RMSNorm (norm_hidden = rms_norm(collapsed)) in one launch when the shape allows
			if ( !engine::modules::HyperConnection::collapse_norm(
				hidden_streams_, lw.hc_attn_fn, lw.hc_attn_base, lw.hc_attn_scale, lw.input_norm, norm_hidden_,
				mhc_post_, mhc_comb_, mhc_inv_rms_, mhc_mix_logits_, mhc_pre_,
				H, 20, 1e-6f, 1e-5f, 0.0f, stream, m
			) ) {
				engine::modules::HyperConnection::collapse(
					hidden_streams_, lw.hc_attn_fn, lw.hc_attn_base, lw.hc_attn_scale,
					mhc_collapsed_, mhc_post_, mhc_comb_, mhc_inv_rms_, mhc_mix_logits_, mhc_pre_,
					H, 20, 1e-6f, stream, m
				);
				engine::modules::RMSNorm::forward(
					mhc_collapsed_, lw.input_norm, norm_hidden_, 1e-5f, 0.0f, stream
				);
			}
		}
		else {
			// 1. Input RMSNorm: norm_hidden = rms_norm(hidden_states)
			engine::modules::RMSNorm::forward(
				hidden_states_, lw.input_norm, norm_hidden_, 1e-5f, 0.0f, stream
			);
		}

		// 2. Attention
		hessian( layer, 0, norm_hidden_.data, H, m, stream );
		if ( lw.is_linear_attn && lw.qkv.is_valid() ) {
			// Linear layers are every layer but the MLA ones (3, 7, ...): ordinal = layer - (layer + 1) / 4
			const size_t lin = static_cast<size_t>( layer - ( layer + 1 ) / 4 );
			float* layer_kda = state_.kda_states + lin * state_.layer_kda_floats;
			float* layer_conv = state_.conv_history + lin * state_.layer_conv_floats;
			float* kda_snap = state_.kda_snap != nullptr ? state_.kda_snap + lin * state_.layer_kda_floats : nullptr;
			float* conv_snap = state_.conv_snap + lin * state_.layer_conv_floats;

			engine::modules::Linear::forward(
				norm_hidden_, lw.qkv, qkv_buf_, stream
			);

			engine::modules::GDN::causal_conv1d_silu_rows(
				qkv_buf_, layer_conv, lw.conv1d_weights, config_.kda_channels, m, d_rows_, slot_conv_floats,
				conv_snap, slot_conv_floats, snap_rows, state_.d_snap_on, stream
			);

			engine::modules::GDN::l2_norm_heads(
				qkv_buf_, lh_, 128, stream, true, m, config_.kda_channels
			);

			engine::modules::GDN::project_kda_gates(
				norm_hidden_, lw.b_proj, lw.f_a_proj, lw.f_b_proj, lw.g_a_proj, lw.g_b_proj,
				lw.A_log, lw.dt_bias, beta_buf_, g_decay_buf_, z_buf_, f_a_buf_, g_a_buf_,
				lh_, 128, H, stream, m
			);

			if ( use_journal_ ) {
				ENGINE_ASSERT( lh_ == config_.kda_heads, "Glm5NextPipeline: the KDA journal is sized for the configured head count" );
				engine::modules::GDN::recurrent_step_kda_rows_journal(
					layer_kda, qkv_buf_, g_decay_buf_, beta_buf_, z_buf_, lw.o_norm, attn_head_out_,
					lh_, 128, 1e-5f, m, d_rows_, slot_kda_floats, state_.kda_journal + lin * state_.journal_layer_stride, state_.journal_row_stride, snap_rows,
					state_.d_snap_on, stream
				);
			}
			else {
				engine::modules::GDN::recurrent_step_kda_rows(
					layer_kda, qkv_buf_, g_decay_buf_, beta_buf_, z_buf_, lw.o_norm, attn_head_out_,
					lh_, 128, 1e-5f, m, d_rows_, slot_kda_floats, kda_snap, slot_kda_floats, snap_rows, state_.d_snap_on, stream
				);
			}

			hessian( layer, 2, attn_head_out_.data, 8192 / config_.tp_size, m, stream );
			if ( lw.o_proj.is_valid() ) {
				engine::modules::Linear::forward(
					attn_head_out_, lw.o_proj, attn_out_, stream
				);
			}

			if ( ring_ && lw.hc_attn_fn.data != nullptr && fuse_reduce_update ) {
				// the all-reduce and the mHC post-step (attention update) in one kernel
				ring_->all_reduce_hc_apply( attn_out_, hidden_streams_.ptr<float>(), mhc_post_.ptr<const float>(), mhc_comb_.ptr<const float>(), stream );
			}
			else {
				if ( ring_ ) {
					ring_->all_reduce( attn_out_, stream );
				}

				if ( lw.hc_attn_fn.data != nullptr ) {
					// mHC post-step: attention update
					engine::modules::HyperConnection::update(
						hidden_streams_, attn_out_, mhc_post_, mhc_comb_, H, stream, m
					);
				}
				else {
					// Residual connection: hidden_states += attn_out
					engine::modules::Elementwise::add_residual( hidden_states_, attn_out_, stream );
				}
			}
		}
		else if ( lw.is_mla && lw.q_a.is_valid() ) {
			int mla_idx = layer / 4;

			// Project Q: q_a_proj -> q_a_norm -> q_b_proj
			engine::modules::Linear::forward(
				norm_hidden_, lw.q_a, q_lora_buf_, stream
			);

			engine::modules::RMSNorm::forward(
				q_lora_buf_, lw.q_a_norm, q_lora_norm_buf_, 1e-5f, 0.0f, stream
			);

			hessian( layer, 4, q_lora_norm_buf_.data, 1536, m, stream );
			engine::modules::Linear::forward(
				q_lora_norm_buf_, lw.q_b, q_buf_, stream
			);

			// Project KV: kv_a_proj_with_mqa -> kv_a_norm -> kv_b_proj
			engine::modules::Linear::forward(
				norm_hidden_, lw.kv_a, kv_lora_buf_, stream
			);

			engine::modules::RMSNorm::forward(
				kv_lora_buf_, lw.kv_a_norm, kv_lora_norm_buf_, 1e-5f, 0.0f, stream
			);

			{
				// Absorbed: cache the normalized latent, attend in latent space, expand the value side per head
				pod::TensorView q_lat = q_lat_, o_lat = o_lat_;
				q_lat.shape[0] = m;
				o_lat.shape[0] = m;
				engine::modules::MLALatent::store( latent_ref(), kv_lora_norm_buf_, mla_idx, config_.max_seq, m, kv_ref, stream );
				engine::modules::MLALatent::absorb_q( q_buf_, lw.kv_b_proj, q_lat, lh_, m, stream );
				const int* sel_idx = nullptr;
				const int* sel_n = nullptr;
				if ( dsa_ && lw.idx_dense.data != nullptr ) {
					using engine::modules::DSA;
					// Indexer keys and gates of this chunk go into the plane, finished pools into the pool plane
					if ( m <= pod::MAX_BATCH ) {
						DSA::project_small( lw.idx_dense.ptr<backend::half_t>(), norm_hidden_.ptr<backend::half_t>(), dsa_proj_, m, DSA::PROJ, 4096, stream );
					}
					else {
						DSA::project_large( lw.idx_dense.ptr<backend::half_t>(), norm_hidden_.ptr<backend::half_t>(), dsa_proj_, m, DSA::PROJ, 4096, stream );
					}
					DSA::prep( dsa_proj_, lw.idx_ln_w.ptr<backend::half_t>(), lw.idx_ln_b.ptr<backend::half_t>(), dsa_plane_, dsa_w_, mla_idx, config_.max_seq, m,
						d_rows_, dsa_plane_stride_, stream );
					DSA::pool_update( dsa_plane_, dsa_pool_, lw.idx_ape.ptr<backend::half_t>(), mla_idx, config_.max_seq, m, d_rows_,
						dsa_plane_stride_, dsa_pool_stride_, stream, kv_pages() );
					// Selection matters once a row sees more than topk tokens; decode graphs always run it (rows below the
					// threshold exit early on the device), eager chunks skip it while every row is still dense
					if ( m <= pod::MAX_BATCH || key_hint_ > DSA::TOPK + DSA::POOL - 1 ) {
						pod::TensorView qi = pod::make_tensor_2d( dsa_q_, pod::DType::FP16, m, DSA::HEADS * DSA::DIM );
						engine::modules::Linear::forward( q_lora_norm_buf_, lw.idx_wq_b, qi, stream );
						DSA::score( dsa_q_, dsa_w_, dsa_pool_, dsa_scores_, config_.max_seq / DSA::POOL, mla_idx, config_.max_seq, m,
							d_rows_, dsa_pool_stride_, stream, kv_pages() );
						DSA::select( dsa_scores_, config_.max_seq / DSA::POOL, dsa_idx_, dsa_nk_, m, d_rows_, stream );
						sel_idx = dsa_idx_;
						sel_n = dsa_nk_;
					}
				}
				engine::modules::MLALatent::attend( latent_ref(), q_lat, o_lat, mla_idx, config_.max_seq, m, lh_, kv_ref, latent_ws_, key_hint_, stream, sel_idx, sel_n, ws_seq_ );
				static const bool dsa_diag = runtime::options().dsa_diag;
				if ( dsa_diag && sel_idx != nullptr && m > pod::MAX_BATCH && rank_ == 0 && lh_ == 16 ) {
					// Recall of the dense attention mass by the selection, last row of the chunk (eager prefill only)
					const int n = key_hint_;
					ENGINE_ASSERT( latent_q_ == pod::KVQ::F16 && kv_pool_tokens_ == 0, "dsa_diag: FP16 latent cache without kv_pool only" );
					std::vector<float> sc( static_cast<size_t>( 16 ) * n );
					float* d_sc = static_cast<float*>( backend::device_malloc( sc.size() * sizeof( float ) ) );
					engine::modules::MLALatent::diag_scores( q_lat.ptr<backend::half_t>() + static_cast<size_t>( m - 1 ) * 16 * 512,
						static_cast<const backend::half_t*>( latent_cache_ ) + static_cast<size_t>( mla_idx ) * config_.max_seq * 512, n, 16, d_sc, stream );
					std::vector<int> sel( engine::modules::DSA::KPAD );
					int nk = 0;
					backend::memcpy_async( &nk, dsa_nk_ + ( m - 1 ), sizeof( int ), backend::Copy::D2H, stream );
					backend::memcpy_async( sel.data(), dsa_idx_ + static_cast<size_t>( m - 1 ) * engine::modules::DSA::KPAD, sel.size() * sizeof( int ), backend::Copy::D2H, stream );
					backend::memcpy_async( sc.data(), d_sc, sc.size() * sizeof( float ), backend::Copy::D2H, stream );
					backend::stream_sync( stream );
					backend::device_free( d_sc );
					if ( nk > 0 ) {
						std::vector<char> in_sel( n, 0 );
						for ( int j = 0; j < nk; ++j ) if ( sel[j] >= 0 && sel[j] < n ) in_sel[sel[j]] = 1;
						double recall = 0.0;
						for ( int h = 0; h < 16; ++h ) {
							const float* s = sc.data() + static_cast<size_t>( h ) * n;
							float mx = -1e30f;
							for ( int t = 0; t < n; ++t ) mx = std::max( mx, s[t] );
							double tot = 0.0, got = 0.0;
							for ( int t = 0; t < n; ++t ) {
								const double e = std::exp( static_cast<double>( s[t] - mx ) );
								tot += e;
								if ( in_sel[t] ) got += e;
							}
							recall += got / tot;
						}
						LOG_INFO( "[dsa diag] layer %d: %d of %d tokens selected, dense-attention mass recall %.4f (mean over heads)\n", mla_idx, nk, n, recall / 16.0 );
					}
				}
				engine::modules::MLALatent::expand_out( o_lat, lw.kv_b_proj, mla_heads_out_, lh_, m, stream );
			}

			// Project output: o_proj -> ring_->all_reduce across the ranks
			hessian( layer, 2, mla_heads_out_.data, 16384 / config_.tp_size, m, stream );
			if ( lw.o_proj.is_valid() ) {
				engine::modules::Linear::forward(
					mla_heads_out_, lw.o_proj, attn_out_, stream
				);
			}

			if ( ring_ && lw.hc_attn_fn.data != nullptr && fuse_reduce_update ) {
				// the all-reduce and the mHC post-step (attention update) in one kernel
				ring_->all_reduce_hc_apply( attn_out_, hidden_streams_.ptr<float>(), mhc_post_.ptr<const float>(), mhc_comb_.ptr<const float>(), stream );
			}
			else {
				if ( ring_ ) {
					ring_->all_reduce( attn_out_, stream );
				}

				if ( lw.hc_attn_fn.data != nullptr ) {
					// mHC post-step: attention update
					engine::modules::HyperConnection::update(
						hidden_streams_, attn_out_, mhc_post_, mhc_comb_, H, stream, m
					);
				}
				else {
					// Residual connection: hidden_states += attn_out
					engine::modules::Elementwise::add_residual( hidden_states_, attn_out_, stream );
				}
			}
		}

		trace_row( 100 + layer );
		if ( lw.hc_ffn_fn.data != nullptr ) {
			// mHC pre-step: FFN
			// Collapse and the post-attention RMSNorm in one launch when the shape allows
			if ( !engine::modules::HyperConnection::collapse_norm(
				hidden_streams_, lw.hc_ffn_fn, lw.hc_ffn_base, lw.hc_ffn_scale, lw.post_attn_norm, norm_hidden_,
				mhc_post_, mhc_comb_, mhc_inv_rms_, mhc_mix_logits_, mhc_pre_,
				H, 20, 1e-6f, 1e-5f, 0.0f, stream, m
			) ) {
				engine::modules::HyperConnection::collapse(
					hidden_streams_, lw.hc_ffn_fn, lw.hc_ffn_base, lw.hc_ffn_scale,
					mhc_collapsed_, mhc_post_, mhc_comb_, mhc_inv_rms_, mhc_mix_logits_, mhc_pre_,
					H, 20, 1e-6f, stream, m
				);
				engine::modules::RMSNorm::forward(
					mhc_collapsed_, lw.post_attn_norm, norm_hidden_, 1e-5f, 0.0f, stream
				);
			}
		}
		else {
			// 3. Post-attention RMSNorm: norm_hidden = rms_norm(hidden_states)
			engine::modules::RMSNorm::forward(
				hidden_states_, lw.post_attn_norm, norm_hidden_, 1e-5f, 0.0f, stream
			);
		}

		// 4. MoE FFN
		hessian( layer, 1, norm_hidden_.data, H, m, stream );
		if ( lw.is_moe ) {
			if ( tier_.active() && engine::modules::ExpertTier::stream_mode() ) tier_.stream_fill( layer - 3, stream );   // tier layers are the MoE layers from 3 on
			// Replicated Top-K Router: 288 experts, top-8 selection evaluated on all ranks
			engine::modules::MoERouter::route_topk(
				norm_hidden_, lw.router_weights, lw.e_score_bias,
				state_.d_topk_indices, state_.d_topk_weights,
				H, config_.num_experts, config_.top_k, 2.5f, stream, m
			);
			static const std::string& dump_path = runtime::options().route_dump;
			if ( !dump_path.empty() && rank_ == 0 && m > pod::MAX_BATCH ) {
				// Debug (route_dump): {layer, rows, first position, K}, the rows' expert ids (int32), the router's input rows (FP16)
				const int K = config_.top_k;
				std::vector<int> ids( static_cast<size_t>( m ) * K );
				std::vector<backend::half_t> xr( static_cast<size_t>( m ) * H );
				backend::stream_sync( stream );
				backend::memcpy_async( ids.data(), state_.d_topk_indices, ids.size() * sizeof( int ), backend::Copy::D2H, stream );
				backend::memcpy_async( xr.data(), norm_hidden_.data, xr.size() * sizeof( backend::half_t ), backend::Copy::D2H, stream );
				backend::stream_sync( stream );
				if ( FILE* f = std::fopen( dump_path.c_str(), "ab" ) ) {
					const int hdr[4] = { layer, m, key_hint_ - m, K };
					std::fwrite( hdr, sizeof( int ), 4, f );
					std::fwrite( ids.data(), sizeof( int ), ids.size(), f );
					std::fwrite( xr.data(), sizeof( backend::half_t ), xr.size(), f );
					std::fclose( f );
				}
			}
			static const std::string& replay_path = runtime::options().route_replay;
			if ( !replay_path.empty() && m > pod::MAX_BATCH ) {
				// Debug: overwrite the routing of rows that have a record (key layer * 2^20 + position)
				static std::unordered_map<int64_t, std::pair<std::array<int, 16>, std::array<backend::half_t, 16>>> table;
				static bool loaded = false;
				const int K = config_.top_k;
				if ( !loaded ) {
					loaded = true;
					if ( FILE* f = std::fopen( replay_path.c_str(), "rb" ) ) {
						int hdr[3];
						while ( std::fread( hdr, sizeof( int ), 3, f ) == 3 ) {
							std::vector<int> ids( static_cast<size_t>( hdr[1] ) * K );
							std::vector<backend::half_t> wts( ids.size() );
							if ( std::fread( ids.data(), sizeof( int ), ids.size(), f ) != ids.size() || std::fread( wts.data(), sizeof( backend::half_t ), wts.size(), f ) != wts.size() ) break;
							for ( int r = 0; r < hdr[1]; ++r ) {
								auto& e = table[( static_cast<int64_t>( hdr[0] ) << 20 ) + hdr[2] + r];
								for ( int k = 0; k < K; ++k ) { e.first[k] = ids[static_cast<size_t>( r ) * K + k]; e.second[k] = wts[static_cast<size_t>( r ) * K + k]; }
							}
						}
						std::fclose( f );
					}
					LOG_INFO( "route replay: %zu records\n", table.size() );
				}
				std::vector<int> ids( static_cast<size_t>( m ) * K );
				std::vector<backend::half_t> wts( ids.size() );
				backend::stream_sync( stream );
				backend::memcpy_async( ids.data(), state_.d_topk_indices, ids.size() * sizeof( int ), backend::Copy::D2H, stream );
				backend::memcpy_async( wts.data(), state_.d_topk_weights, wts.size() * sizeof( backend::half_t ), backend::Copy::D2H, stream );
				backend::stream_sync( stream );
				for ( int r = 0; r < m; ++r ) {
					const auto it = table.find( ( static_cast<int64_t>( layer ) << 20 ) + key_hint_ - m + r );
					if ( it == table.end() ) continue;
					for ( int k = 0; k < K; ++k ) { ids[static_cast<size_t>( r ) * K + k] = it->second.first[k]; wts[static_cast<size_t>( r ) * K + k] = it->second.second[k]; }
				}
				backend::memcpy_async( state_.d_topk_indices, ids.data(), ids.size() * sizeof( int ), backend::Copy::H2D, stream );
				backend::memcpy_async( state_.d_topk_weights, wts.data(), wts.size() * sizeof( backend::half_t ), backend::Copy::H2D, stream );
				backend::stream_sync( stream );
			}
			if ( !trace_path.empty() && rank_ == 0 && m > pod::MAX_BATCH ) {
				// Debug: every row's selected experts and weights (header: layer, rows, first position)
				const size_t n = static_cast<size_t>( m ) * config_.top_k;
				std::vector<int> ids( n );
				std::vector<backend::half_t> wts( n );
				backend::stream_sync( stream );
				backend::memcpy_async( ids.data(), state_.d_topk_indices, n * sizeof( int ), backend::Copy::D2H, stream );
				backend::memcpy_async( wts.data(), state_.d_topk_weights, n * sizeof( backend::half_t ), backend::Copy::D2H, stream );
				backend::stream_sync( stream );
				if ( FILE* f = std::fopen( ( trace_path + ".route" ).c_str(), "ab" ) ) {
					const int hdr[3] = { layer, m, key_hint_ - m };
					std::fwrite( hdr, sizeof( int ), 3, f );
					std::fwrite( ids.data(), sizeof( int ), n, f );
					std::fwrite( wts.data(), sizeof( backend::half_t ), n, f );
					std::fclose( f );
				}
			}

				// Shared expert: every rank runs its slice (the partial outputs meet in the all-reduce below), or rank 0 the whole of it. Decode rows on
				// CUDA run it on a side stream beside the fused MoE kernels (they use no trellis workspace and the chain has its own buffers: the
				// shared GEMVs fill 16 to 128 blocks, the experts the rest of the machine); the accumulate below waits for both
			const bool shared_here = lw.has_shared_expert && ( lw.shared_sliced || rank_ == 0 );
			const bool overlap = shared_here && side_stream_ != nullptr && m <= pod::MAX_BATCH && engine::modules::MoE::fused_decode( lw.moe_weights, state_.moe, m );
			backend::stream_t sh_stream = overlap ? side_stream_ : stream;
			if ( overlap ) backend::stream_fork( stream, sh_stream );
			if ( shared_here ) {
				pod::TensorView sh_g = moe_gate_buf_, sh_u = moe_up_buf_, sh_i = moe_inter_buf_;
				const int sh_n = lw.shared_sliced ? config_.intermediate_size / config_.tp_size : config_.intermediate_size;
				sh_g.shape[1] = sh_u.shape[1] = sh_i.shape[1] = sh_n;
				engine::modules::Linear::forward(
					norm_hidden_, lw.shared_expert.gate, sh_g, sh_stream
				);
				engine::modules::Linear::forward(
					norm_hidden_, lw.shared_expert.up, sh_u, sh_stream
				);
				engine::modules::SwiGLU::forward( sh_g, sh_u, sh_i, sh_stream, 10.0f );
				hessian( layer, 3, sh_i.data, 2048, m, sh_stream );
				engine::modules::Linear::forward(
					sh_i, lw.shared_expert.down, moe_expert_out_, sh_stream
				);
			}

				// Routed experts in this rank's partition: device-resident dispatch, no host sync. A prefill chunk reads the layer's cold experts from device
				// scratch, copied a layer ahead
			const bool staged = m > pod::MAX_BATCH && tier_.prefill_staging() && tier_.stage_begin( layer - 3, stream );
			if ( staged && layer + 1 < config_.num_layers ) tier_.stage_prefetch( layer - 2, stream );
			engine::modules::MoE::forward(
				norm_hidden_, lw.moe_weights,
				state_.d_topk_indices, state_.d_topk_weights,
				local_moe_out_, start_expert, end_expert,
				state_.moe, 10.0f, stream
			);
			if ( staged ) tier_.stage_end( layer - 3, stream );
			if ( overlap ) backend::stream_join( stream, sh_stream );

			if ( shared_here ) {
				engine::modules::MoERouter::accumulate(
					local_moe_out_.ptr<backend::half_t>(),
					moe_expert_out_.ptr<const backend::half_t>(),
					backend::from_float_half( 1.0f ),
					m * H,
					stream
				);
			}

				// Sum local expert contributions across NVLink mesh
			if ( ring_ && !fuse_ffn_reduce ) {
				ring_->all_reduce( local_moe_out_, stream );
			}
		}
		else {
			// Dense layer
			if ( lw.dense_mlp.gate.is_valid() ) {
				pod::TensorView dn_g = moe_gate_buf_, dn_u = moe_up_buf_, dn_i = moe_inter_buf_;
				if ( lw.dense_sliced ) dn_g.shape[1] = dn_u.shape[1] = dn_i.shape[1] = config_.dense_intermediate_size / config_.tp_size;
				engine::modules::Linear::forward(
					norm_hidden_, lw.dense_mlp.gate, dn_g, stream
				);
				engine::modules::Linear::forward(
					norm_hidden_, lw.dense_mlp.up, dn_u, stream
				);
				engine::modules::SwiGLU::forward( dn_g, dn_u, dn_i, stream, 10.0f );
				engine::modules::Linear::forward(
					dn_i, lw.dense_mlp.down, ffn_out_, stream
				);
				if ( lw.dense_sliced && !fuse_ffn_reduce ) ring_->all_reduce( ffn_out_, stream );
			}
		}

		const pod::TensorView& ffn_res = lw.is_moe ? local_moe_out_ : ffn_out_;

		if ( lw.hc_ffn_fn.data != nullptr ) {
			// mHC post-step: FFN update (with the all-reduce of the output, when that is fused)
			if ( fuse_ffn_reduce ) {
				ring_->all_reduce_hc_apply( const_cast<pod::TensorView&>( ffn_res ), hidden_streams_.ptr<float>(), mhc_post_.ptr<const float>(), mhc_comb_.ptr<const float>(), stream );
			}
			else {
				engine::modules::HyperConnection::update(
					hidden_streams_, ffn_res, mhc_post_, mhc_comb_, H, stream, m
				);
			}
		}
		else {
			if ( lw.is_moe ) {
				engine::modules::Elementwise::add_residual( hidden_states_, local_moe_out_, stream );
			}
			else {
				engine::modules::Elementwise::add_residual( hidden_states_, ffn_out_, stream );
			}
		}

		// Tapped layer states for the DFlash2 drafter: position-indexed ring, owner/slot/position read on device.
		// The tapped hidden state of an mHC layer is the mean of its four residual streams (as HyperHead).
		for ( int tap = 0; tap < 5; ++tap ) {
			if ( layer != config_.tap_layers[tap] ) continue;
			engine::spec::dflash_reduce_streams( hidden_streams_.ptr<const float>(), tap_tmp_.ptr<backend::half_t>(), m, H, 0, stream );
			engine::spec::dflash_publish_taps_slots(
				draft_taps_.ptr<backend::half_t>(), tap_tmp_.ptr<const backend::half_t>(), d_rows_,
				m, tap, H, draft_hooks_.taps_cap, draft_hooks_.taps_slot_stride, stream );
		}

		trace_row( layer );
}

void Glm5NextPipeline::hessian( int layer, int site, const void* x, int k, int m, backend::stream_t stream ) {
	if ( hess_.empty() || m <= pod::MAX_BATCH ) return;
	HessianSite& hs = hess_[static_cast<size_t>( layer ) * 6 + site];
	if ( hs.h == nullptr ) return;
	ENGINE_ASSERT( hs.k == k, "Glm5NextPipeline::hessian: site width changed" );
	engine::modules::Hessian::accumulate( hs.h, static_cast<const backend::half_t*>( x ), m, k, stream );
	hs.rows += m;
}

// One file per site: int32 magic, int32 k, int64 rows, then k * k fp32 (x^T x summed over the rows)
void Glm5NextPipeline::hessian_dump() {
	if ( hess_.empty() ) return;
	const std::string& dir = runtime::options().hessian_dir;
	backend::stream_sync( nullptr );
	std::vector<float> host;
	for ( size_t i = 0; i < hess_.size(); ++i ) {
		const HessianSite& hs = hess_[i];
		if ( hs.h == nullptr || hs.rows == 0 ) continue;
		host.resize( static_cast<size_t>( hs.k ) * hs.k );
		backend::memcpy_sync( host.data(), hs.h, host.size() * sizeof( float ), backend::Copy::D2H );
		char path[512];
		std::snprintf( path, sizeof( path ), "%s/L%02d_s%d_r%d.f32", dir.c_str(), static_cast<int>( i / 6 ), static_cast<int>( i % 6 ), rank_ );
		if ( FILE* f = std::fopen( path, "wb" ) ) {
			const int32_t hdr[2] = { 0x48455353, hs.k };
			std::fwrite( hdr, sizeof( int32_t ), 2, f );
			std::fwrite( &hs.rows, sizeof( int64_t ), 1, f );
			std::fwrite( host.data(), sizeof( float ), host.size(), f );
			std::fclose( f );
		}
		else LOG_ERROR( "[RANK %d] hessian: cannot write %s\n", rank_, path );
	}
	LOG_INFO( "[RANK %d] hessians written to %s\n", rank_, dir.c_str() );
}

// One MTP forward (DeepSeek-style head): row j takes the target's final-norm hidden at its position and the embedding of the token that
// follows, x = eh_proj( [enorm(embedding) | hnorm(hidden)] ), runs the head's MLA + MoE layer on x and predicts the token after that.
// The head's latent cache and indexer state are the last slot of the sequence (layer 45).
void Glm5NextPipeline::mtp_step( const pod::BatchRows& rows, const int* tokens, int hidden_mode, int hist_row0, bool logits_last, backend::stream_t stream ) {
	ENGINE_ASSERT( mtp_on_ && weights_loaded_, "Glm5NextPipeline::mtp_step: no MTP head" );
	backend::set_device( rank_ );
	const int m = rows.count;
	const int H = config_.hidden_size;
	const size_t row_bytes = static_cast<size_t>( H ) * sizeof( backend::half_t );
	ENGINE_ASSERT( m >= 1 && m <= pod::MAX_ROWS, "Glm5NextPipeline::mtp_step: rows out of range" );
	for ( int r = 0; r < m; ++r ) {
		ENGINE_ASSERT( rows.slot[r] >= 0 && rows.slot[r] < config_.max_slots && rows.pos[r] >= 0 && rows.pos[r] < config_.max_seq, "Glm5NextPipeline::mtp_step: slot or position out of range" );
	}
	key_hint_ = rows.pos[m - 1] + 1;
	lm_mode_ = pod::BatchRows::LOGITS_ALL;
	set_rows( m );
	kv_prepare( rows, stream );
	backend::memcpy_async( d_rows_, &rows, rows.upload_bytes(), backend::Copy::H2D, stream );

	// 1. Normed embedding of the following tokens
	embed_tokens( tokens, m, stream );
	pod::TensorView emb = pod::make_tensor_2d( embed_rows_.data, pod::DType::FP16, m, H );
	pod::TensorView xe = pod::make_tensor_2d( mtp_xe_.data, pod::DType::FP16, m, H );
	pod::TensorView xh = pod::make_tensor_2d( mtp_xh_.data, pod::DType::FP16, m, H );
	engine::modules::RMSNorm::forward( emb, mtp_enorm_, xe, 1e-5f, 0.0f, stream );

	// 2. Normed hidden rows
	const backend::half_t* src = nullptr;
	switch ( hidden_mode ) {
		case 0: src = norm_hidden_.ptr<const backend::half_t>(); break;   // the last forward's final-norm rows
		case 1: {   // saved row 0, then the last forward's rows 0 .. m - 2 (a prefill chunk continuing a previous one)
			backend::half_t* st = mtp_stage_.ptr<backend::half_t>();
			backend::memcpy_async( st, mtp_hist_.ptr<char>() + static_cast<size_t>( hist_row0 ) * row_bytes, row_bytes, backend::Copy::D2D, stream );
			if ( m > 1 ) backend::memcpy_async( st + H, norm_hidden_.data, static_cast<size_t>( m - 1 ) * row_bytes, backend::Copy::D2D, stream );
			src = st;
		} break;
		case 2: src = mtp_hist_.ptr<const backend::half_t>() + static_cast<size_t>( hist_row0 ) * H; break;   // saved rows 0 .. m - 1
		default: src = mtp_chain_.ptr<const backend::half_t>(); break;   // the previous MTP step's last output row
	}
	pod::TensorView src_view = pod::make_tensor_2d( const_cast<backend::half_t*>( src ), pod::DType::FP16, m, H );
	engine::modules::RMSNorm::forward( src_view, mtp_hnorm_, xh, 1e-5f, 0.0f, stream );

	// 3. eh_proj over [embedding | hidden] (replicated on every rank), into the head layer's input
	backend::memcpy2d_async( mtp_in_.ptr<backend::half_t>(), 2 * row_bytes, xe.data, row_bytes, row_bytes, m, backend::Copy::D2D, stream );
	backend::memcpy2d_async( mtp_in_.ptr<backend::half_t>() + H, 2 * row_bytes, xh.data, row_bytes, row_bytes, m, backend::Copy::D2D, stream );
	pod::TensorView in2 = pod::make_tensor_2d( mtp_in_.data, pod::DType::FP16, m, 2 * H );
	pod::TensorView x0 = pod::make_tensor_2d( hidden_states_.data, pod::DType::FP16, m, H );
	engine::modules::Linear::forward( in2, mtp_eh_, x0, stream );

	// 4. The head's layer (MLA + MoE on the single residual stream)
	layer_body( config_.num_layers, mtp_layer_, m, stream );

	// 5. Final norm; the last row stays for the next chained step and feeds the shared LM head
	pod::TensorView out = pod::make_tensor_2d( mtp_out_.data, pod::DType::FP16, m, H );
	engine::modules::RMSNorm::forward( x0, mtp_norm_, out, 1e-5f, 0.0f, stream );
	backend::memcpy_async( mtp_chain_.data, mtp_out_.ptr<backend::half_t>() + static_cast<size_t>( m - 1 ) * H, row_bytes, backend::Copy::D2D, stream );
	if ( logits_last ) {
		pod::TensorView n1 = pod::make_tensor_2d( mtp_chain_.data, pod::DType::FP16, 1, H );
		pod::TensorView l1 = pod::make_tensor_2d( logits_.ptr<backend::half_t>(), pod::DType::FP16, 1, lm_shard_cols_ );
		engine::modules::Linear::forward( n1, lm_head_, l1, stream );
		logits_.shape[0] = 1;
		last_rows_ = 1;
	}
}

// Keeps rows [src_row, src_row + rows) of the last forward's final-norm hidden for later MTP steps (row dst_row onwards)
void Glm5NextPipeline::mtp_save_hist( int src_row, int rows, int dst_row, backend::stream_t stream ) {
	ENGINE_ASSERT( mtp_on_ && rows >= 1 && dst_row + rows <= config_.max_slots * 2 * ( kDraftM + 1 ), "Glm5NextPipeline::mtp_save_hist: rows out of range" );
	backend::set_device( rank_ );
	const size_t row_bytes = static_cast<size_t>( config_.hidden_size ) * sizeof( backend::half_t );
	backend::memcpy_async( mtp_hist_.ptr<char>() + static_cast<size_t>( dst_row ) * row_bytes, norm_hidden_.ptr<const char>() + static_cast<size_t>( src_row ) * row_bytes,
		static_cast<size_t>( rows ) * row_bytes, backend::Copy::D2D, stream );
}

void Glm5NextPipeline::destroy() {
	if ( arena_ != nullptr ) {
		backend::set_device( rank_ );
		hessian_dump();
		graphs_.reset();
		if ( side_stream_ != nullptr ) backend::stream_destroy( side_stream_ );
		side_stream_ = nullptr;
	}
	engine::core::arena_destroy( host_arena_ );
	if ( h_argmax_idx_ != nullptr ) {
		engine::core::host_free_pinned( h_argmax_idx_ );
		h_argmax_idx_ = nullptr;
	}
	if ( rank_ == 0 && g_embed_table.data != nullptr ) {
		backend::host_free_pinned( g_embed_table.data );
		g_embed_table.data = nullptr;
	}
}

// -- Sequence snapshots (prefix cache), absorbed-MLA + DSA configuration only. Host layout: [KDA states][conv history]
// [DSA planes][tap rows][per MLA layer: latent rows, then pooled index keys for positions [from, tokens)]. The plane is a
// small ring of recent per-token index rows (the whole slot is stored; it is part of the state, not of the segment).
size_t Glm5NextPipeline::tap_snapshot_bytes( int tokens ) const {
	if ( !snap_taps_ ) return 0;
	return static_cast<size_t>( std::min( tokens, draft_hooks_.taps_cap ) ) * 5 * config_.hidden_size * sizeof( backend::half_t );
}

size_t Glm5NextPipeline::seq_state_bytes( int from, int tokens, bool with_taps ) const {
	if ( !dsa_ || from < 0 || tokens < from || tokens > config_.max_seq ) return 0;
	const int pool = engine::modules::DSA::POOL;
	const size_t recurrent = ( state_.slot_kda_floats + state_.slot_conv_floats ) * sizeof( float );
	const size_t planes = dsa_plane_stride_ * sizeof( backend::half_t );
	const size_t lat_elems = static_cast<size_t>( tokens - from ) * 512;
	const size_t latent = static_cast<size_t>( mla_slots_ ) * ( latent_data_bytes( lat_elems ) + latent_scale_bytes( lat_elems ) );
	const size_t pool_rows = static_cast<size_t>( ( tokens + pool - 1 ) / pool - from / pool );
	const size_t pools = static_cast<size_t>( mla_slots_ ) * pool_rows * engine::modules::DSA::DIM * sizeof( backend::half_t );
	return recurrent + planes + ( with_taps ? tap_snapshot_bytes( tokens ) : 0 ) + latent + pools + 256;
}

void Glm5NextPipeline::copy_seq_rows( int slot_id, int from, int tokens, char* host, bool to_host, backend::stream_t stream ) {
	const int pool = engine::modules::DSA::POOL;
	const int NP = config_.max_seq / pool;
	const backend::Copy dir = to_host ? backend::Copy::D2H : backend::Copy::H2D;
	auto move = [&]( void* dev, char* at, size_t bytes ) {
		if ( bytes == 0 ) return;
		if ( to_host ) backend::memcpy_async( at, dev, bytes, dir, stream );
		else backend::memcpy_async( dev, at, bytes, dir, stream );
	};
	const size_t lat_elems = static_cast<size_t>( tokens - from ) * 512;
	const size_t lat_bytes = latent_data_bytes( lat_elems ), lat_scale_bytes = latent_scale_bytes( lat_elems );
	const int pool_from = from / pool, pool_to = ( tokens + pool - 1 ) / pool;
	const size_t pool_bytes = static_cast<size_t>( pool_to - pool_from ) * engine::modules::DSA::DIM * sizeof( backend::half_t );
	const size_t D = engine::modules::DSA::DIM;
	for ( int layer = 0; layer < mla_slots_; ++layer ) {
		// Host layout of one layer: latent rows, their scales, pooled keys (the same in both cache layouts)
		char* h_lat = host;
		char* h_scale = h_lat + lat_bytes;
		char* h_pool = h_scale + lat_scale_bytes;
		host = h_pool + pool_bytes;
		if ( kv_pool_tokens_ == 0 ) {
			const size_t lat_off = static_cast<size_t>( slot_id ) * latent_slot_stride_ + ( static_cast<size_t>( layer ) * config_.max_seq + from ) * 512;
			move( static_cast<char*>( latent_cache_ ) + latent_data_bytes( lat_off ), h_lat, lat_bytes );
			if ( lat_scale_bytes ) move( static_cast<char*>( latent_scales_ ) + latent_scale_bytes( lat_off ), h_scale, lat_scale_bytes );
			move( dsa_pool_ + static_cast<size_t>( slot_id ) * dsa_pool_stride_ + ( static_cast<size_t>( layer ) * NP + pool_from ) * D, h_pool, pool_bytes );
			continue;
		}
		// Pooled: one range per page. The pooled key of tokens [4p, 4p + 4) sits in the page of token 4p.
		for_kv_ranges( slot_id, from, tokens, [&]( int t, int n, size_t phys ) {
			const size_t lat_off = ( static_cast<size_t>( layer ) * kv_pool_tokens_ + phys ) * 512;
			const size_t rel = static_cast<size_t>( t - from ) * 512;
			move( static_cast<char*>( latent_cache_ ) + latent_data_bytes( lat_off ), h_lat + latent_data_bytes( rel ), latent_data_bytes( static_cast<size_t>( n ) * 512 ) );
			if ( lat_scale_bytes ) move( static_cast<char*>( latent_scales_ ) + latent_scale_bytes( lat_off ), h_scale + latent_scale_bytes( rel ), latent_scale_bytes( static_cast<size_t>( n ) * 512 ) );
			const int p0 = t / pool, p1 = ( t + n + pool - 1 ) / pool;   // pools touching [t, t + n)
			if ( p1 > p0 ) {
				const size_t pool_off = ( static_cast<size_t>( layer ) * ( kv_pool_tokens_ / pool ) + ( phys - static_cast<size_t>( t - p0 * pool ) ) / pool ) * D;
				move( dsa_pool_ + pool_off, h_pool + static_cast<size_t>( p0 - pool_from ) * D * sizeof( backend::half_t ), static_cast<size_t>( p1 - p0 ) * D * sizeof( backend::half_t ) );
			}
		} );
	}
}

bool Glm5NextPipeline::save_seq_state( int slot_id, int from, int tokens, bool with_taps, void* host_dst, backend::stream_t stream ) {
	if ( seq_state_bytes( from, tokens, with_taps ) == 0 || slot_id < 0 || slot_id >= config_.max_slots ) return false;
	backend::set_device( rank_ );
	char* dst = static_cast<char*>( host_dst );
	const size_t kda = state_.slot_kda_floats * sizeof( float ), conv = state_.slot_conv_floats * sizeof( float ), planes = dsa_plane_stride_ * sizeof( backend::half_t );
	backend::memcpy_async( dst, state_.kda_states + static_cast<size_t>( slot_id ) * state_.slot_kda_floats, kda, backend::Copy::D2H, stream );
	backend::memcpy_async( dst + kda, state_.conv_history + static_cast<size_t>( slot_id ) * state_.slot_conv_floats, conv, backend::Copy::D2H, stream );
	backend::memcpy_async( dst + kda + conv, dsa_plane_ + static_cast<size_t>( slot_id ) * dsa_plane_stride_, planes, backend::Copy::D2H, stream );
	dst += kda + conv + planes;
	if ( const size_t tap_bytes = with_taps ? tap_snapshot_bytes( tokens ) : 0 ) {
		backend::memcpy_async( dst, draft_taps_.ptr<backend::half_t>() + static_cast<size_t>( slot_id ) * draft_hooks_.taps_slot_stride, tap_bytes, backend::Copy::D2H, stream );
		dst += tap_bytes;
	}
	copy_seq_rows( slot_id, from, tokens, dst, true, stream );
	return true;
}

bool Glm5NextPipeline::load_seq_state( int slot_id, int from, int tokens, bool with_taps, const void* host_src, bool kv_only, backend::stream_t stream ) {
	if ( seq_state_bytes( from, tokens, with_taps ) == 0 || slot_id < 0 || slot_id >= config_.max_slots ) return false;
	if ( !reserve_tokens( slot_id, tokens ) ) return false;
	backend::set_device( rank_ );
	const char* src = static_cast<const char*>( host_src );
	const size_t kda = state_.slot_kda_floats * sizeof( float ), conv = state_.slot_conv_floats * sizeof( float ), planes = dsa_plane_stride_ * sizeof( backend::half_t );
	const size_t tap_bytes = with_taps ? tap_snapshot_bytes( tokens ) : 0;
	if ( !kv_only ) {
		backend::memcpy_async( state_.kda_states + static_cast<size_t>( slot_id ) * state_.slot_kda_floats, src, kda, backend::Copy::H2D, stream );
		backend::memcpy_async( state_.conv_history + static_cast<size_t>( slot_id ) * state_.slot_conv_floats, src + kda, conv, backend::Copy::H2D, stream );
		backend::memcpy_async( dsa_plane_ + static_cast<size_t>( slot_id ) * dsa_plane_stride_, src + kda + conv, planes, backend::Copy::H2D, stream );
		if ( tap_bytes ) backend::memcpy_async( draft_taps_.ptr<backend::half_t>() + static_cast<size_t>( slot_id ) * draft_hooks_.taps_slot_stride, src + kda + conv + planes, tap_bytes, backend::Copy::H2D, stream );
	}
	copy_seq_rows( slot_id, from, tokens, const_cast<char*>( src ) + kda + conv + planes + tap_bytes, false, stream );
	return true;
}

} // namespace engine::models
