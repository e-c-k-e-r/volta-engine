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

namespace {

pod::TensorView view2d( void* p, int rows, int cols ) {
	return pod::make_tensor_2d( p, pod::DType::FP16, rows, cols );
}

template <typename T>
T* take( pod::MemoryArena& arena, size_t count ) {
	return static_cast<T*>( engine::core::arena_alloc( arena, count * sizeof( T ) ) );
}

} // namespace

bool Qwen3VisionTower::load( const char* model_dir, pod::MemoryArena& arena, int max_patches, backend::stream_t stream ) {
	std::ifstream cf( std::string( model_dir ) + "/config.json" );
	if ( !cf ) return false;
	const auto cfg_json = nlohmann::json::parse( std::string( std::istreambuf_iterator<char>( cf ), std::istreambuf_iterator<char>() ), nullptr, false );
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
	if ( !vc.value( "deepstack_visual_indexes", nlohmann::json::array() ).empty() ) {
		LOG_WARN( "[vision] deepstack vision features are not supported\n" );
		return false;
	}
	params_.patch_size = cfg_.patch_size;
	params_.merge_size = cfg_.merge_size;
	params_.temporal_patch_size = cfg_.temporal_patch_size;
	params_.max_pixels = max_patches * cfg_.patch_size * cfg_.patch_size;

	engine::loader::SafeTensorsRepo repo;
	if ( !repo.open( model_dir ) ) return false;
	if ( !repo.find( "model.visual.merger.linear_fc1.trellis" ) || !repo.find( "model.visual.blocks.0.attn.q_proj.trellis" ) ) {
		LOG_WARN( "[vision] checkpoint has no quantized vision tower; image input disabled\n" );
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
		ENGINE_ASSERT( meta && raw && meta->dtype == pod::DType::FP16 && meta->size_bytes == bytes, "vision: patch embedding must be FP16 [hidden, 3, T, P, P]" );
		void* dev = engine::core::arena_alloc( arena, bytes );
		backend::memcpy_async( dev, raw, bytes, backend::Copy::H2D, stream );
		patch_w_ = pod::make_tensor_2d( dev, pod::DType::FP16, cfg_.hidden, cfg_.patch_dim() );
	}
	patch_b_ = fp16( "model.visual.patch_embed.proj.bias" );

	// Position table on the host (interpolated per image)
	{
		const auto* meta = repo.find( "model.visual.pos_embed.weight" );
		const void* raw = repo.cpu_data( "model.visual.pos_embed.weight" );
		ENGINE_ASSERT( meta && raw && meta->dtype == pod::DType::FP16, "vision: pos_embed must be FP16" );
		pos_table_.resize( static_cast<size_t>( cfg_.num_position_embeddings ) * cfg_.hidden );
		const auto* h = static_cast<const backend::half_t*>( raw );
		for ( size_t i = 0; i < pos_table_.size(); ++i ) pos_table_[i] = backend::to_float( h[i] );
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
		b.q = lin( p + "attn.q_proj" );
		b.k = lin( p + "attn.k_proj" );
		b.v = lin( p + "attn.v_proj" );
		b.proj = lin( p + "attn.proj" );
		b.q_b = fp16( p + "attn.q_proj.bias" );
		b.k_b = fp16( p + "attn.k_proj.bias" );
		b.v_b = fp16( p + "attn.v_proj.bias" );
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

std::vector<float> Qwen3VisionTower::position_embeddings( int gh, int gw ) const {
	const int side = static_cast<int>( std::lround( std::sqrt( static_cast<double>( cfg_.num_position_embeddings ) ) ) );
	const int D = cfg_.hidden, m = cfg_.merge_size;
	auto lin_idx = []( int i, int n, int side_ ) { return ( n == 1 ) ? 0.0f : static_cast<float>( i ) * ( side_ - 1 ) / ( n - 1 ); };
	std::vector<float> out( static_cast<size_t>( gh ) * gw * D );
	size_t row = 0;
	for ( int by = 0; by < gh / m; ++by ) {
		for ( int bx = 0; bx < gw / m; ++bx ) {
			for ( int my = 0; my < m; ++my ) {
				for ( int mx = 0; mx < m; ++mx, ++row ) {
					const float hy = lin_idx( by * m + my, gh, side ), wx = lin_idx( bx * m + mx, gw, side );
					const int hf = static_cast<int>( hy ), wf = static_cast<int>( wx );
					const int hc = std::min( hf + 1, side - 1 ), wc = std::min( wf + 1, side - 1 );
					const float dh = hy - hf, dw = wx - wf;
					const float* t00 = &pos_table_[( static_cast<size_t>( hf ) * side + wf ) * D];
					const float* t01 = &pos_table_[( static_cast<size_t>( hf ) * side + wc ) * D];
					const float* t10 = &pos_table_[( static_cast<size_t>( hc ) * side + wf ) * D];
					const float* t11 = &pos_table_[( static_cast<size_t>( hc ) * side + wc ) * D];
					float* o = &out[row * D];
					for ( int d = 0; d < D; ++d ) {
						o[d] = ( 1 - dh ) * ( 1 - dw ) * t00[d] + ( 1 - dh ) * dw * t01[d] + dh * ( 1 - dw ) * t10[d] + dh * dw * t11[d];
					}
				}
			}
		}
	}
	return out;
}

void Qwen3VisionTower::rotary_tables( int gh, int gw, std::vector<float>* cos_t, std::vector<float>* sin_t ) const {
	const int hd = cfg_.head_dim(), quarter = hd / 4, m = cfg_.merge_size;
	std::vector<float> inv( quarter );
	for ( int i = 0; i < quarter; ++i ) inv[i] = 1.0f / std::pow( 10000.0f, static_cast<float>( 2 * i ) / ( hd / 2 ) );
	cos_t->assign( static_cast<size_t>( gh ) * gw * hd, 0.0f );
	sin_t->assign( cos_t->size(), 0.0f );
	size_t row = 0;
	for ( int by = 0; by < gh / m; ++by ) {
		for ( int bx = 0; bx < gw / m; ++bx ) {
			for ( int my = 0; my < m; ++my ) {
				for ( int mx = 0; mx < m; ++mx, ++row ) {
					const float r = static_cast<float>( by * m + my ), c = static_cast<float>( bx * m + mx );
					float* cs = &( *cos_t )[row * hd];
					float* sn = &( *sin_t )[row * hd];
					// emb = [row freqs | col freqs], repeated twice (rotate_half layout)
					for ( int i = 0; i < quarter; ++i ) {
						const float fr = r * inv[i], fc = c * inv[i];
						cs[i] = cs[i + hd / 2] = std::cos( fr );
						sn[i] = sn[i + hd / 2] = std::sin( fr );
						cs[quarter + i] = cs[quarter + i + hd / 2] = std::cos( fc );
						sn[quarter + i] = sn[quarter + i + hd / 2] = std::sin( fc );
					}
				}
			}
		}
	}
}

bool Qwen3VisionTower::encode( const engine::loader::PreparedImage& img, std::vector<backend::half_t>* out, backend::stream_t stream ) {
	if ( !loaded_ ) return false;
	const int m = img.num_patches();
	if ( m < 1 || m > max_patches_ ) {
		LOG_WARN( "[vision] image has %d patches, limit is %d\n", m, max_patches_ );
		return false;
	}
	const int H = cfg_.hidden, hd = cfg_.head_dim(), mrg = cfg_.merge_size * cfg_.merge_size;
	const int mlp_w = blocks_[0].fc1_b.shape[0];

	std::vector<backend::half_t> patches( static_cast<size_t>( m ) * cfg_.patch_dim() );
	for ( size_t i = 0; i < patches.size(); ++i ) patches[i] = backend::from_float_half( img.patches[i] );
	const std::vector<float> pos_f = position_embeddings( img.grid_h, img.grid_w );
	std::vector<backend::half_t> pos( pos_f.size() );
	for ( size_t i = 0; i < pos.size(); ++i ) pos[i] = backend::from_float_half( pos_f[i] );
	std::vector<float> cs, sn;
	rotary_tables( img.grid_h, img.grid_w, &cs, &sn );
	backend::memcpy_async( d_patches_, patches.data(), patches.size() * sizeof( backend::half_t ), backend::Copy::H2D, stream );
	backend::memcpy_async( d_pos_, pos.data(), pos.size() * sizeof( backend::half_t ), backend::Copy::H2D, stream );
	backend::memcpy_async( d_cos_, cs.data(), cs.size() * sizeof( float ), backend::Copy::H2D, stream );
	backend::memcpy_async( d_sin_, sn.data(), sn.size() * sizeof( float ), backend::Copy::H2D, stream );
	// The staging vectors die at return; the copies must be done before then.
	backend::stream_sync( stream );

	using engine::modules::Vision;
	using engine::modules::Linear;
	Vision::patch_embed( d_patches_, patch_w_.ptr<const backend::half_t>(), patch_b_.ptr<const backend::half_t>(), d_pos_, d_hidden_, m, H, cfg_.patch_dim(), stream );
	const float scale = 1.0f / std::sqrt( static_cast<float>( hd ) );
	auto in_h = [&]( void* p, int rows ) { return view2d( p, rows, H ); };
	for ( const Block& b : blocks_ ) {
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
	backend::stream_sync( stream );
	return true;
}

} // namespace engine::models
