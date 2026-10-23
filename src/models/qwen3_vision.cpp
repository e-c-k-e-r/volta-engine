#include "engine/models/qwen3_vision.h"
#include "engine/modules/vision.h"
#include "engine/loader/safetensors.h"
#include "engine/backend/numeric.h"
#include "engine/backend/runtime.h"
#include "engine/core/log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>

namespace engine::models {

bool Qwen3VisionTower::load( const char* model_dir, pod::MemoryArena& arena, int max_patches, backend::stream_t stream, bool allow_dense ) {
	const auto cfg_json = read_json( std::string( model_dir ) + "/config.json" );
	if ( cfg_json.is_discarded() || !cfg_json.contains( "vision_config" ) ) return false;
	const auto& vc = cfg_json["vision_config"];
	cfg_.depth = vc.value( "depth", cfg_.depth );
	cfg_.hidden = vc.value( "hidden_size", cfg_.hidden );
	cfg_.heads = vc.value( "num_heads", cfg_.heads );
	cfg_.intermediate = vc.value( "intermediate_size", cfg_.intermediate );
	cfg_.out_hidden = vc.value( "out_hidden_size", cfg_.out_hidden );
	cfg_.patch_size = vc.value( "patch_size", cfg_.patch_size );
	cfg_.merge_size = vc.value( "spatial_merge_size", cfg_.merge_size );
	cfg_.temporal_patch_size = vc.value( "temporal_patch_size", cfg_.temporal_patch_size );
	cfg_.num_position_embeddings = vc.value( "num_position_embeddings", cfg_.num_position_embeddings );
	cfg_.deepstack_blocks.clear();
	for ( const auto& d : vc.value( "deepstack_visual_indexes", nlohmann::json::array() ) ) { cfg_.deepstack_blocks.push_back( d.get<int>() ); }
	params_.patch_size = cfg_.patch_size;
	params_.merge_size = cfg_.merge_size;
	params_.temporal_patch_size = cfg_.temporal_patch_size;
	params_.max_pixels = max_patches * cfg_.patch_size * cfg_.patch_size;

	engine::loader::SafeTensorsRepo repo;
	if ( !repo.open( model_dir ) ) return false;
	const bool trellis = repo.find( "model.visual.merger.linear_fc1.trellis" ) && repo.find( "model.visual.blocks.0.attn.q_proj.trellis" );
	// an unquantized tower: dense `.weight` tensors, with either separate q / k / v projections or the fused `attn.qkv` of the original release
	fused_qkv_ = false;
	const bool dense = !trellis && allow_dense && repo.find( "model.visual.merger.linear_fc1.weight" ) &&
		( repo.find( "model.visual.blocks.0.attn.q_proj.weight" ) || ( fused_qkv_ = repo.find( "model.visual.blocks.0.attn.qkv.weight" ) != nullptr ) );
	if ( !trellis && !dense ) {
		LOG_WARN( "[vision] checkpoint has no quantized vision tower; image input disabled\n" );
		repo.close();
		return false;
	}
	const size_t weight_bytes = dense ? repo.bytes_with_prefix( "model.visual." ) : vision_weight_bytes( repo, cfg_.depth, cfg_.hidden );
	if ( !vision_fits( arena, weight_bytes + static_cast<size_t>( max_patches ) * 48 * 1024 + ( 16u << 20 ), "the Qwen vision tower" ) ) {
		repo.close();
		return false;
	}
	max_patches_ = max_patches;

	auto fp16 = [&]( const std::string& name ) { return repo.load( name.c_str(), arena, stream, pod::DType::FP16 ); };
	auto lin = [&]( const std::string& name ) { return repo.load_linear( name.c_str(), arena, 6, stream ); };
	// The patch projection is a 5-D conv kernel [hidden, 3, T, P, P]; the repo's views stop at 4 dims, so copy the bytes as [hidden, patch_dim].
	{
		const auto* meta = repo.find( "model.visual.patch_embed.proj.weight" );
		const void* raw = repo.cpu_data( "model.visual.patch_embed.proj.weight" );
		const size_t bytes = static_cast<size_t>( cfg_.hidden ) * cfg_.patch_dim() * sizeof( backend::half_t );
		ENGINE_ASSERT( meta && raw && ( ( meta->dtype == pod::DType::FP16 && meta->size_bytes == bytes ) || ( meta->dtype == pod::DType::BF16 && meta->size_bytes == bytes ) ), "vision: patch embedding must be FP16 or BF16 [hidden, 3, T, P, P]" );
		void* dev = engine::core::arena_alloc( arena, bytes );
		if ( meta->dtype == pod::DType::BF16 ) { // converted on the host (the repo's views stop at 4 dimensions)
			std::vector<backend::half_t> conv( bytes / sizeof( backend::half_t ) );
			const uint16_t* src = static_cast<const uint16_t*>( raw );
			for ( size_t i = 0; i < conv.size(); ++i ) {
				const uint32_t bits = static_cast<uint32_t>( src[i] ) << 16;
				float f;
				std::memcpy( &f, &bits, 4 );
				conv[i] = backend::from_float_half( f );
			}
			backend::memcpy_async( dev, conv.data(), bytes, backend::Copy::H2D, stream );
			backend::stream_sync( stream ); // the host buffer goes out of scope
		} else {
			backend::memcpy_async( dev, raw, bytes, backend::Copy::H2D, stream );
		}
		patch_w_ = pod::make_tensor_2d( dev, pod::DType::FP16, cfg_.hidden, cfg_.patch_dim() );
	}
	patch_b_ = fp16( "model.visual.patch_embed.proj.bias" );

	// Position table (interpolated on the device per image)
	{
		const auto* meta = repo.find( "model.visual.pos_embed.weight" );
		const void* raw = repo.cpu_data( "model.visual.pos_embed.weight" );
		ENGINE_ASSERT( meta && raw && ( meta->dtype == pod::DType::FP16 || meta->dtype == pod::DType::BF16 ), "vision: pos_embed must be FP16 or BF16" );
		const size_t count = static_cast<size_t>( cfg_.num_position_embeddings ) * cfg_.hidden;
		if ( meta->dtype == pod::DType::FP16 ) {
			d_pos_table_ = take<backend::half_t>( arena, count );
			backend::memcpy_async( d_pos_table_, raw, count * sizeof( backend::half_t ), backend::Copy::H2D, stream );
		}
		else {
			// BF16 table (Flash-Next): the repo converts to FP16 on load
			d_pos_table_ = repo.load( "model.visual.pos_embed.weight", arena, stream, pod::DType::FP16 ).ptr<backend::half_t>();
		}
	}

	blocks_.resize( cfg_.depth );
	char pre[96];
	for ( int i = 0; i < cfg_.depth; ++i ) {
		std::snprintf( pre, sizeof( pre ), "model.visual.blocks.%d.", i );
		const std::string p( pre );
		Block& b = blocks_[i];
		b.norm1_w = fp16( p + "norm1.weight" );
		b.norm1_b = fp16( p + "norm1.bias" );
		b.norm2_w = fp16( p + "norm2.weight" );
		b.norm2_b = fp16( p + "norm2.bias" );
		if ( fused_qkv_ ) { // the fused [3 hidden, hidden] projection and its bias, cut into thirds
			pod::LinearWeight* part[3] = { &b.q, &b.k, &b.v };
			pod::TensorView* bias[3] = { &b.q_b, &b.k_b, &b.v_b };
			for ( int t = 0; t < 3; ++t ) {
				pod::TensorView w = repo.load_slice( ( p + "attn.qkv.weight" ).c_str(), arena, 0, t, 3, stream, pod::DType::FP16 );
				*part[t] = pod::LinearWeight::make_fp16( pod::make_tensor_2d( w.data, pod::DType::FP16, w.shape[1], w.shape[0] ) ); // [n, k] in memory, tagged [k, n]
				*bias[t] = repo.load_slice( ( p + "attn.qkv.bias" ).c_str(), arena, 0, t, 3, stream, pod::DType::FP16 );
			}
		} else {
			b.q = lin( p + "attn.q_proj" );
			b.k = lin( p + "attn.k_proj" );
			b.v = lin( p + "attn.v_proj" );
			b.q_b = fp16( p + "attn.q_proj.bias" );
			b.k_b = fp16( p + "attn.k_proj.bias" );
			b.v_b = fp16( p + "attn.v_proj.bias" );
		}
		b.proj = lin( p + "attn.proj" );
		b.proj_b = fp16( p + "attn.proj.bias" );
		b.fc1 = lin( p + "mlp.linear_fc1" );
		b.fc2 = lin( p + "mlp.linear_fc2" );
		b.fc1_b = fp16( p + "mlp.linear_fc1.bias" );
		b.fc2_b = fp16( p + "mlp.linear_fc2.bias" );
	}
	merge_norm_w_ = fp16( "model.visual.merger.norm.weight" );
	merge_norm_b_ = fp16( "model.visual.merger.norm.bias" );
	merge_fc1_ = lin( "model.visual.merger.linear_fc1" );
	merge_fc2_ = lin( "model.visual.merger.linear_fc2" );
	merge_fc1_b_ = fp16( "model.visual.merger.linear_fc1.bias" );
	merge_fc2_b_ = fp16( "model.visual.merger.linear_fc2.bias" );
	deep_.resize( cfg_.deepstack_blocks.size() );
	for ( size_t i = 0; i < deep_.size(); ++i ) {
		const std::string p = "model.visual.deepstack_merger_list." + std::to_string( i ) + ".";
		deep_[i].norm_w = fp16( p + "norm.weight" );
		deep_[i].norm_b = fp16( p + "norm.bias" );
		deep_[i].fc1 = lin( p + "linear_fc1" );
		deep_[i].fc2 = lin( p + "linear_fc2" );
		deep_[i].fc1_b = fp16( p + "linear_fc1.bias" );
		deep_[i].fc2_b = fp16( p + "linear_fc2.bias" );
	}
	repo.close();

	const size_t M = static_cast<size_t>( max_patches );
	const size_t H = cfg_.hidden;
	const size_t mlp_w = static_cast<size_t>( blocks_[0].fc1_b.shape[0] );
	d_patches_ = take<backend::half_t>( arena, M * cfg_.patch_dim() );
	d_pos_ = take<backend::half_t>( arena, M * H );
	d_cos_ = take<float>( arena, M * cfg_.head_dim() );
	d_sin_ = take<float>( arena, M * cfg_.head_dim() );
	d_hidden_ = take<float>( arena, M * H );
	d_ln_ = take<backend::half_t>( arena, M * H );
	d_q_ = take<backend::half_t>( arena, M * H );
	d_k_ = take<backend::half_t>( arena, M * H );
	d_v_ = take<backend::half_t>( arena, M * H );
	d_attn_ = take<backend::half_t>( arena, M * H );
	d_tmp_ = take<backend::half_t>( arena, M * H );
	d_mlp_ = take<backend::half_t>( arena, M * mlp_w );
	const size_t merged_w = H * cfg_.merge_size * cfg_.merge_size;
	d_merged_ = take<backend::half_t>( arena, M / ( cfg_.merge_size * cfg_.merge_size ) * merged_w );
	d_merged_fc1_ = take<backend::half_t>( arena, M / ( cfg_.merge_size * cfg_.merge_size ) * merged_w );
	d_out_ = take<backend::half_t>( arena, M / ( cfg_.merge_size * cfg_.merge_size ) * cfg_.out_hidden );
	if ( stream ) backend::stream_sync( stream );
	loaded_ = true;
	LOG_INFO( "[vision] tower loaded: %d blocks, up to %d patches (%d tokens)\n", cfg_.depth, max_patches_, max_patches_ / ( cfg_.merge_size * cfg_.merge_size ) );
	return true;
}

void vision_rotary_tables( int gh, int gw, int m, int hd, std::vector<float>* cos_t, std::vector<float>* sin_t ) {
	const int quarter = hd / 4;
	std::vector<float> inv( quarter );
	for ( int i = 0; i < quarter; ++i ) inv[i] = 1.0f / std::pow( 10000.0f, static_cast<float>( 2 * i ) / ( hd / 2 ) );
	cos_t->assign( static_cast<size_t>( gh ) * gw * hd, 0.0f );
	sin_t->assign( cos_t->size(), 0.0f );
	// Row and column angles depend on one grid coordinate each: tabulate them once
	std::vector<float> rc( static_cast<size_t>( gh ) * quarter ), rs( rc.size() ), cc( static_cast<size_t>( gw ) * quarter ), cs_( cc.size() );
	for ( int y = 0; y < gh; ++y ) {
		for ( int i = 0; i < quarter; ++i ) {
			rc[y * quarter + i] = std::cos( y * inv[i] );
			rs[y * quarter + i] = std::sin( y * inv[i] );
		}
	}
	for ( int x = 0; x < gw; ++x ) {
		for ( int i = 0; i < quarter; ++i ) {
			cc[x * quarter + i] = std::cos( x * inv[i] );
			cs_[x * quarter + i] = std::sin( x * inv[i] );
		}
	}
	size_t row = 0;
	for ( int by = 0; by < gh / m; ++by ) {
		for ( int bx = 0; bx < gw / m; ++bx ) {
			for ( int my = 0; my < m; ++my ) {
				for ( int mx = 0; mx < m; ++mx, ++row ) {
					const int y = by * m + my, x = bx * m + mx;
					float* cs = &( *cos_t )[row * hd];
					float* sn = &( *sin_t )[row * hd];
					// emb = [row freqs | col freqs], repeated twice (rotate_half layout)
					for ( int i = 0; i < quarter; ++i ) {
						cs[i] = cs[i + hd / 2] = rc[y * quarter + i];
						sn[i] = sn[i + hd / 2] = rs[y * quarter + i];
						cs[quarter + i] = cs[quarter + i + hd / 2] = cc[x * quarter + i];
						sn[quarter + i] = sn[quarter + i + hd / 2] = cs_[x * quarter + i];
					}
				}
			}
		}
	}
}

void Qwen3VisionTower::rotary_tables( int gh, int gw, std::vector<float>* cos_t, std::vector<float>* sin_t ) const {
	vision_rotary_tables( gh, gw, cfg_.merge_size, cfg_.head_dim(), cos_t, sin_t );
}

bool Qwen3VisionTower::encode( const engine::loader::PreparedImage& img, std::vector<backend::half_t>* out, backend::stream_t stream ) {
	return encode( img, out, nullptr, stream );
}

bool Qwen3VisionTower::encode( const engine::loader::PreparedImage& img, std::vector<backend::half_t>* out, std::vector<std::vector<backend::half_t>>* deepstack, backend::stream_t stream ) {
	if ( !loaded_ ) return false;
	if ( deepstack ) deepstack->assign( cfg_.deepstack_blocks.size(), std::vector<backend::half_t>() );
	const int m = img.num_patches();
	if ( m < 1 || m > max_patches_ ) {
		LOG_WARN( "[vision] image has %d patches, limit is %d\n", m, max_patches_ );
		return false;
	}
	const int H = cfg_.hidden, hd = cfg_.head_dim(), mrg = cfg_.merge_size * cfg_.merge_size;
	const int mlp_w = blocks_[0].fc1_b.shape[0];

	// Patches go up as FP32 through the MLP scratch (unused until the first block) and are narrowed on the device
	std::vector<float> cs, sn;
	rotary_tables( img.grid_h, img.grid_w, &cs, &sn );
	const size_t patch_floats = static_cast<size_t>( m ) * cfg_.patch_dim();
	ENGINE_ASSERT( patch_floats * sizeof( float ) <= static_cast<size_t>( max_patches_ ) * mlp_w * sizeof( backend::half_t ), "vision: patch staging does not fit" );
	backend::memcpy_async( d_mlp_, img.patches.data(), patch_floats * sizeof( float ), backend::Copy::H2D, stream );
	backend::memcpy_async( d_cos_, cs.data(), cs.size() * sizeof( float ), backend::Copy::H2D, stream );
	backend::memcpy_async( d_sin_, sn.data(), sn.size() * sizeof( float ), backend::Copy::H2D, stream );

	using engine::modules::Vision;
	using engine::modules::Linear;
	Vision::f32_to_f16( reinterpret_cast<const float*>( d_mlp_ ), d_patches_, patch_floats, stream );
	Vision::pos_interp( d_pos_table_, d_pos_, img.grid_h, img.grid_w, cfg_.merge_size, static_cast<int>( std::lround( std::sqrt( static_cast<double>( cfg_.num_position_embeddings ) ) ) ), H, stream );
	Vision::patch_embed( d_patches_, patch_w_.ptr<const backend::half_t>(), patch_b_.ptr<const backend::half_t>(), d_pos_, d_hidden_, m, H, cfg_.patch_dim(), stream );
	const float scale = 1.0f / std::sqrt( static_cast<float>( hd ) );
	auto in_h = [&]( void* p, int rows ) { return view2d( p, rows, H ); };
	const int mrg_tokens = m / mrg, mrg_w = H * mrg;
	for ( size_t bi = 0; bi < blocks_.size(); ++bi ) {
		const Block& b = blocks_[bi];
		Vision::layer_norm( d_hidden_, b.norm1_w.ptr<const backend::half_t>(), b.norm1_b.ptr<const backend::half_t>(), d_ln_, m, H, cfg_.eps, stream );
		pod::TensorView ln = in_h( d_ln_, m ), qv = in_h( d_q_, m ), kv = in_h( d_k_, m ), vv = in_h( d_v_, m );
		Linear::forward( ln, b.q, qv, stream );
		Linear::forward( ln, b.k, kv, stream );
		Linear::forward( ln, b.v, vv, stream );
		Vision::qkv_bias_rope( d_q_, d_k_, d_v_, b.q_b.ptr<const backend::half_t>(), b.k_b.ptr<const backend::half_t>(), b.v_b.ptr<const backend::half_t>(),
			d_cos_, d_sin_, m, cfg_.heads, hd, stream );
		Vision::attention( d_q_, d_k_, d_v_, d_attn_, m, cfg_.heads, hd, scale, stream );
		pod::TensorView at = in_h( d_attn_, m ), tmp = in_h( d_tmp_, m );
		Linear::forward( at, b.proj, tmp, stream );
		Vision::bias_residual( d_hidden_, d_tmp_, b.proj_b.ptr<const backend::half_t>(), m, H, stream );

		Vision::layer_norm( d_hidden_, b.norm2_w.ptr<const backend::half_t>(), b.norm2_b.ptr<const backend::half_t>(), d_ln_, m, H, cfg_.eps, stream );
		pod::TensorView mlp = view2d( d_mlp_, m, mlp_w );
		Linear::forward( ln, b.fc1, mlp, stream );
		Vision::bias_act( d_mlp_, b.fc1_b.ptr<const backend::half_t>(), m, mlp_w, Vision::ACT_GELU_TANH, stream );
		Linear::forward( mlp, b.fc2, tmp, stream );
		Vision::bias_residual( d_hidden_, d_tmp_, b.fc2_b.ptr<const backend::half_t>(), m, H, stream );
		for ( size_t d = 0; d < cfg_.deepstack_blocks.size(); ++d ) {
			if ( cfg_.deepstack_blocks[d] != static_cast<int>( bi ) ) continue;
			// DeepStack tap: the merger with its norm over the merged width ("post shuffle")
			const DeepMerger& g = deep_[d];
			Vision::layer_norm( d_hidden_, g.norm_w.ptr<const backend::half_t>(), g.norm_b.ptr<const backend::half_t>(), d_merged_, mrg_tokens, mrg_w, cfg_.eps, stream );
			pod::TensorView dm = view2d( d_merged_, mrg_tokens, mrg_w ), df = view2d( d_merged_fc1_, mrg_tokens, mrg_w ), dout = view2d( d_out_, mrg_tokens, cfg_.out_hidden );
			Linear::forward( dm, g.fc1, df, stream );
			Vision::bias_act( d_merged_fc1_, g.fc1_b.ptr<const backend::half_t>(), mrg_tokens, mrg_w, Vision::ACT_GELU_ERF, stream );
			Linear::forward( df, g.fc2, dout, stream );
			Vision::bias_act( d_out_, g.fc2_b.ptr<const backend::half_t>(), mrg_tokens, cfg_.out_hidden, Vision::ACT_NONE, stream );
			if ( deepstack ) {
				( *deepstack )[d].resize( static_cast<size_t>( mrg_tokens ) * cfg_.out_hidden );
				backend::memcpy_async( ( *deepstack )[d].data(), d_out_, ( *deepstack )[d].size() * sizeof( backend::half_t ), backend::Copy::D2H, stream );
				backend::stream_sync( stream ); // d_out_ is reused by the next tap and the final merger
			}
		}
	}
	Vision::layer_norm_merge( d_hidden_, merge_norm_w_.ptr<const backend::half_t>(), merge_norm_b_.ptr<const backend::half_t>(), d_merged_, m, H, mrg, cfg_.eps, stream );
	const int tokens = m / mrg, mw = H * mrg;
	pod::TensorView mg = view2d( d_merged_, tokens, mw ), f1 = view2d( d_merged_fc1_, tokens, mw ), o = view2d( d_out_, tokens, cfg_.out_hidden );
	Linear::forward( mg, merge_fc1_, f1, stream );
	Vision::bias_act( d_merged_fc1_, merge_fc1_b_.ptr<const backend::half_t>(), tokens, mw, Vision::ACT_GELU_ERF, stream );
	Linear::forward( f1, merge_fc2_, o, stream );
	Vision::bias_act( d_out_, merge_fc2_b_.ptr<const backend::half_t>(), tokens, cfg_.out_hidden, Vision::ACT_NONE, stream );

	out->resize( static_cast<size_t>( tokens ) * cfg_.out_hidden );
	backend::memcpy_async( out->data(), d_out_, out->size() * sizeof( backend::half_t ), backend::Copy::D2H, stream );
	backend::stream_sync( stream );   // also covers the staging vectors above
	return true;
}

} // namespace engine::models
