#include "engine/models/glm5_vision.h"
#include "engine/modules/vision.h"
#include "engine/modules/swiglu.h"
#include "engine/loader/safetensors.h"
#include "engine/backend/numeric.h"
#include "engine/backend/runtime.h"
#include "engine/core/log.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <iterator>

namespace engine::models {

bool Glm5VisionTower::load( const char* model_dir, pod::MemoryArena& arena, int max_patches, backend::stream_t stream ) {
	const nlohmann::json cfg_json = read_json( std::string( model_dir ) + "/config.json" );
	if ( cfg_json.is_discarded() || !cfg_json.contains( "vision_config" ) ) return false;
	const auto& vc = cfg_json["vision_config"];
	cfg_.depth = vc.value( "depth", cfg_.depth );
	cfg_.hidden = vc.value( "hidden_size", cfg_.hidden );
	cfg_.heads = vc.value( "num_heads", cfg_.heads );
	cfg_.intermediate = vc.value( "intermediate_size", cfg_.intermediate );
	cfg_.out_hidden = vc.value( "out_hidden_size", cfg_.out_hidden );
	cfg_.merger_intermediate = vc.value( "projection_intermediate_size", cfg_.merger_intermediate );
	cfg_.patch_size = vc.value( "patch_size", cfg_.patch_size );
	cfg_.merge_size = vc.value( "spatial_merge_size", cfg_.merge_size );
	cfg_.temporal_patch_size = vc.value( "temporal_patch_size", cfg_.temporal_patch_size );
	cfg_.eps = vc.value( "rms_norm_eps", cfg_.eps );
	cfg_.swiglu_limit = vc.value( "swiglu_limit", cfg_.swiglu_limit );
	if ( cfg_.head_dim() != 64 || cfg_.merge_size != 2 ) {
		LOG_WARN( "[vision] unsupported GLM vision geometry (head_dim %d, merge %d)\n", cfg_.head_dim(), cfg_.merge_size );
		return false;
	}

	params_.patch_size = cfg_.patch_size;
	params_.merge_size = cfg_.merge_size;
	params_.temporal_patch_size = cfg_.temporal_patch_size;
	params_.canvas = true;
	params_.max_pixels = max_patches * cfg_.patch_size * cfg_.patch_size;
	const nlohmann::json pc = read_json( std::string( model_dir ) + "/processor_config.json" );
	if ( !pc.is_discarded() && pc.contains( "image_processor" ) ) {
		const auto& ip = pc["image_processor"];
		if ( ip.contains( "image_mean" ) && ip["image_mean"].size() == 3 && ip.contains( "image_std" ) && ip["image_std"].size() == 3 ) {
			for ( int c = 0; c < 3; ++c ) {
				params_.mean[c] = ip["image_mean"][c].get<float>();
				params_.stddev[c] = ip["image_std"][c].get<float>();
			}
		}
		params_.min_pixels = ip.value( "min_image_tokens", 16 ) * cfg_.merge_size * cfg_.merge_size * cfg_.patch_size * cfg_.patch_size;
	}

	engine::loader::SafeTensorsRepo repo;
	if ( !repo.open( model_dir ) ) return false;
	if ( !repo.find( "model.visual.merger.proj.trellis" ) || !repo.find( "model.visual.blocks.0.attn.q_proj.trellis" ) ) {
		LOG_WARN( "[vision] checkpoint has no quantized vision tower; image input disabled\n" );
		repo.close();
		return false;
	}
	if ( !vision_fits( arena, vision_weight_bytes( repo, cfg_.depth, cfg_.hidden ) + static_cast<size_t>( max_patches ) * 72 * 1024 + ( 16u << 20 ), "the GLM vision tower" ) ) {
		repo.close();
		return false;
	}
	max_patches_ = max_patches;

	auto fp16 = [&]( const std::string& name ) { return repo.load( name.c_str(), arena, stream, pod::DType::FP16 ); };
	auto lin = [&]( const std::string& name ) { return repo.load_linear( name.c_str(), arena, 6, stream ); };
	{
		// 5-D conv kernel [hidden, 3, T, P, P]: the repo's views stop at 4 dims, so copy the bytes as [hidden, patch_dim]
		const auto* meta = repo.find( "model.visual.patch_embed.proj.weight" );
		const void* raw = repo.cpu_data( "model.visual.patch_embed.proj.weight" );
		const size_t bytes = static_cast<size_t>( cfg_.hidden ) * cfg_.patch_dim() * sizeof( backend::half_t );
		ENGINE_ASSERT( meta && raw && meta->dtype == pod::DType::FP16 && meta->size_bytes == bytes, "vision: patch embedding must be FP16 [hidden, 3, T, P, P]" );
		void* dev = engine::core::arena_alloc( arena, bytes );
		backend::memcpy_async( dev, raw, bytes, backend::Copy::H2D, stream );
		patch_w_ = pod::make_tensor_2d( dev, pod::DType::FP16, cfg_.hidden, cfg_.patch_dim() );
	}
	patch_b_ = fp16( "model.visual.patch_embed.proj.bias" );
	post_norm_w_ = fp16( "model.visual.post_layernorm.weight" );
	down_w_ = fp16( "model.visual.downsample.weight" );   // [out, hidden, 2, 2] == [out, hidden * 4], channel-major
	down_b_ = fp16( "model.visual.downsample.bias" );
	ENGINE_ASSERT( down_w_.numel() == static_cast<size_t>( cfg_.out_hidden ) * cfg_.hidden * 4, "vision: downsample weight shape" );

	blocks_.resize( cfg_.depth );
	for ( int i = 0; i < cfg_.depth; ++i ) {
		const std::string p = "model.visual.blocks." + std::to_string( i ) + ".";
		ENGINE_ASSERT( repo.find( ( p + "attn.q_proj.trellis" ).c_str() ) != nullptr, "vision: block without quantized q_proj" );
		Block& b = blocks_[i];
		b.norm1_w = fp16( p + "norm1.weight" );
		b.norm2_w = fp16( p + "norm2.weight" );
		b.q_norm = fp16( p + "attn.q_norm.weight" );
		b.k_norm = fp16( p + "attn.k_norm.weight" );
		b.q = lin( p + "attn.q_proj" );
		b.k = lin( p + "attn.k_proj" );
		b.v = lin( p + "attn.v_proj" );
		b.proj = lin( p + "attn.proj" );
		b.q_b = fp16( p + "attn.q_proj.bias" );
		b.k_b = fp16( p + "attn.k_proj.bias" );
		b.v_b = fp16( p + "attn.v_proj.bias" );
		b.proj_b = fp16( p + "attn.proj.bias" );
		b.gate = lin( p + "mlp.gate_proj" );
		b.up = lin( p + "mlp.up_proj" );
		b.down = lin( p + "mlp.down_proj" );
		b.gate_b = fp16( p + "mlp.gate_proj.bias" );
		b.up_b = fp16( p + "mlp.up_proj.bias" );
		b.down_b = fp16( p + "mlp.down_proj.bias" );
	}
	merge_proj_ = lin( "model.visual.merger.proj" );
	merge_norm_w_ = fp16( "model.visual.merger.post_projection_norm.weight" );
	merge_norm_b_ = fp16( "model.visual.merger.post_projection_norm.bias" );
	merge_gate_ = lin( "model.visual.merger.gate_proj" );
	merge_up_ = lin( "model.visual.merger.up_proj" );
	merge_down_ = lin( "model.visual.merger.down_proj" );
	repo.close();

	const size_t M = static_cast<size_t>( max_patches ), H = cfg_.hidden, T = M / 4, O = cfg_.out_hidden;
	d_patches_ = take<backend::half_t>( arena, M * cfg_.patch_dim() );
	d_cos_ = take<float>( arena, M * cfg_.head_dim() );
	d_sin_ = take<float>( arena, M * cfg_.head_dim() );
	d_hidden_ = take<float>( arena, M * H );
	d_ln_ = take<backend::half_t>( arena, M * H );
	d_q_ = take<backend::half_t>( arena, M * H );
	d_k_ = take<backend::half_t>( arena, M * H );
	d_v_ = take<backend::half_t>( arena, M * H );
	d_attn_ = take<backend::half_t>( arena, M * H );
	d_tmp_ = take<backend::half_t>( arena, M * H );
	d_gate_ = take<backend::half_t>( arena, M * cfg_.intermediate );
	d_up_ = take<backend::half_t>( arena, M * cfg_.intermediate );
	d_mlp_ = take<backend::half_t>( arena, M * cfg_.intermediate );
	d_gathered_ = take<backend::half_t>( arena, T * H * 4 );
	d_merged_ = take<backend::half_t>( arena, T * O );
	d_mgate_ = take<backend::half_t>( arena, T * cfg_.merger_intermediate );
	d_mup_ = take<backend::half_t>( arena, T * cfg_.merger_intermediate );
	d_mmlp_ = take<backend::half_t>( arena, T * cfg_.merger_intermediate );
	d_out_ = take<backend::half_t>( arena, T * O );
	if ( stream ) backend::stream_sync( stream );
	loaded_ = true;
	LOG_INFO( "[vision] GLM tower loaded: %d blocks, up to %d patches (%d tokens)\n", cfg_.depth, max_patches_, max_patches_ / 4 );
	return true;
}

bool Glm5VisionTower::encode( const engine::loader::PreparedImage& img, std::vector<backend::half_t>* out, backend::stream_t stream ) {
	if ( !loaded_ ) return false;
	const int m = img.num_patches();
	if ( m < 4 || m > max_patches_ ) {
		LOG_WARN( "[vision] image has %d patches, limit is %d\n", m, max_patches_ );
		return false;
	}
	const int H = cfg_.hidden, hd = cfg_.head_dim(), I = cfg_.intermediate, O = cfg_.out_hidden, MI = cfg_.merger_intermediate;
	const int tokens = m / 4;

	// Patches go up as FP32 through the MLP scratch (unused until the first block) and are narrowed on the device
	std::vector<float> cs, sn;
	vision_rotary_tables( img.grid_h, img.grid_w, cfg_.merge_size, hd, &cs, &sn );
	const size_t patch_floats = static_cast<size_t>( m ) * cfg_.patch_dim();
	ENGINE_ASSERT( patch_floats * sizeof( float ) <= static_cast<size_t>( max_patches_ ) * I * sizeof( backend::half_t ), "vision: patch staging does not fit" );
	backend::memcpy_async( d_mlp_, img.patches.data(), patch_floats * sizeof( float ), backend::Copy::H2D, stream );
	backend::memcpy_async( d_cos_, cs.data(), cs.size() * sizeof( float ), backend::Copy::H2D, stream );
	backend::memcpy_async( d_sin_, sn.data(), sn.size() * sizeof( float ), backend::Copy::H2D, stream );
	backend::stream_sync( stream );   // the staging vectors die at return

	using engine::modules::Vision;
	using engine::modules::Linear;
	using H16 = const backend::half_t*;
	Vision::f32_to_f16( reinterpret_cast<const float*>( d_mlp_ ), d_patches_, patch_floats, stream );
	Vision::patch_embed( d_patches_, patch_w_.ptr<const backend::half_t>(), patch_b_.ptr<const backend::half_t>(), nullptr, d_hidden_, m, H, cfg_.patch_dim(), stream );
	const float scale = 1.0f / std::sqrt( static_cast<float>( hd ) );
	for ( const Block& b : blocks_ ) {
		Vision::rms_norm( d_hidden_, b.norm1_w.ptr<const backend::half_t>(), d_ln_, m, H, cfg_.eps, stream );
		pod::TensorView ln = view2d( d_ln_, m, H ), qv = view2d( d_q_, m, H ), kv = view2d( d_k_, m, H ), vv = view2d( d_v_, m, H );
		Linear::forward( ln, b.q, qv, stream );
		Linear::forward( ln, b.k, kv, stream );
		Linear::forward( ln, b.v, vv, stream );
		Vision::qkv_bias_norm_rope( d_q_, d_k_, d_v_, b.q_b.ptr<const backend::half_t>(), b.k_b.ptr<const backend::half_t>(), b.v_b.ptr<const backend::half_t>(),
			b.q_norm.ptr<const backend::half_t>(), b.k_norm.ptr<const backend::half_t>(), d_cos_, d_sin_, m, cfg_.heads, hd, cfg_.eps, stream );
		Vision::attention( d_q_, d_k_, d_v_, d_attn_, m, cfg_.heads, hd, scale, stream );
		pod::TensorView at = view2d( d_attn_, m, H ), tmp = view2d( d_tmp_, m, H );
		Linear::forward( at, b.proj, tmp, stream );
		Vision::bias_residual( d_hidden_, d_tmp_, b.proj_b.ptr<const backend::half_t>(), m, H, stream );

		Vision::rms_norm( d_hidden_, b.norm2_w.ptr<const backend::half_t>(), d_ln_, m, H, cfg_.eps, stream );
		pod::TensorView gate = view2d( d_gate_, m, I ), up = view2d( d_up_, m, I ), mlp = view2d( d_mlp_, m, I );
		Linear::forward( ln, b.gate, gate, stream );
		Linear::forward( ln, b.up, up, stream );
		Vision::bias_act( d_gate_, b.gate_b.ptr<const backend::half_t>(), m, I, Vision::ACT_NONE, stream );
		Vision::bias_act( d_up_, b.up_b.ptr<const backend::half_t>(), m, I, Vision::ACT_NONE, stream );
		engine::modules::SwiGLU::forward( gate, up, mlp, stream, cfg_.swiglu_limit );
		Linear::forward( mlp, b.down, tmp, stream );
		Vision::bias_residual( d_hidden_, d_tmp_, b.down_b.ptr<const backend::half_t>(), m, H, stream );
	}
	Vision::rms_norm( d_hidden_, post_norm_w_.ptr<const backend::half_t>(), d_ln_, m, H, cfg_.eps, stream );

	// 2x2 patch merge as a stride-2 convolution (channel-major gather + GEMM), then the SwiGLU projector
	Vision::merge_gather( d_ln_, d_gathered_, tokens, H, 4, stream );
	Vision::dense_gemm( d_gathered_, down_w_.ptr<const backend::half_t>(), d_merged_, tokens, O, H * 4, stream );
	Vision::bias_act( d_merged_, down_b_.ptr<const backend::half_t>(), tokens, O, Vision::ACT_NONE, stream );
	pod::TensorView mg = view2d( d_merged_, tokens, O ), t1 = view2d( d_out_, tokens, O );
	Linear::forward( mg, merge_proj_, t1, stream );
	Vision::layer_norm_half( d_out_, merge_norm_w_.ptr<const backend::half_t>(), merge_norm_b_.ptr<const backend::half_t>(), d_merged_, tokens, O, cfg_.eps, stream );
	Vision::bias_act( d_merged_, nullptr, tokens, O, Vision::ACT_GELU_ERF, stream );
	pod::TensorView mgate = view2d( d_mgate_, tokens, MI ), mup = view2d( d_mup_, tokens, MI ), mmlp = view2d( d_mmlp_, tokens, MI ), o = view2d( d_out_, tokens, O );
	Linear::forward( mg, merge_gate_, mgate, stream );
	Linear::forward( mg, merge_up_, mup, stream );
	engine::modules::SwiGLU::forward( mgate, mup, mmlp, stream, cfg_.swiglu_limit );
	Linear::forward( mmlp, merge_down_, o, stream );

	out->resize( static_cast<size_t>( tokens ) * O );
	backend::memcpy_async( out->data(), d_out_, out->size() * sizeof( backend::half_t ), backend::Copy::D2H, stream );
	backend::stream_sync( stream );
	return true;
}

} // namespace engine::models
