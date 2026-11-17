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
#include "host_vision_ops.h"
#include "engine/kernels/cpu/cpu_util.h"
#include <chrono>
#include <mutex>
#include <memory>
#include <string>
#include <vector>

namespace engine::models {

bool glm5_vision_config( const char* model_dir, int max_patches, Glm5VisionConfig* out_cfg, engine::loader::VisionParams* out_params ) {
	Glm5VisionConfig cfg;
	engine::loader::VisionParams params;
	const nlohmann::json cfg_json = read_json( std::string( model_dir ) + "/config.json" );
	if ( cfg_json.is_discarded() || !cfg_json.contains( "vision_config" ) ) return false;
	const auto& vc = cfg_json["vision_config"];
	cfg.depth = vc.value( "depth", cfg.depth );
	cfg.hidden = vc.value( "hidden_size", cfg.hidden );
	cfg.heads = vc.value( "num_heads", cfg.heads );
	cfg.intermediate = vc.value( "intermediate_size", cfg.intermediate );
	cfg.out_hidden = vc.value( "out_hidden_size", cfg.out_hidden );
	cfg.merger_intermediate = vc.value( "projection_intermediate_size", cfg.merger_intermediate );
	cfg.patch_size = vc.value( "patch_size", cfg.patch_size );
	cfg.merge_size = vc.value( "spatial_merge_size", cfg.merge_size );
	cfg.temporal_patch_size = vc.value( "temporal_patch_size", cfg.temporal_patch_size );
	cfg.eps = vc.value( "rms_norm_eps", cfg.eps );
	cfg.swiglu_limit = vc.value( "swiglu_limit", cfg.swiglu_limit );
	if ( cfg.head_dim() != 64 || cfg.merge_size != 2 ) {
		LOG_WARN( "[vision] unsupported GLM vision geometry (head_dim %d, merge %d)\n", cfg.head_dim(), cfg.merge_size );
		return false;
	}

	params.patch_size = cfg.patch_size;
	params.merge_size = cfg.merge_size;
	params.temporal_patch_size = cfg.temporal_patch_size;
	params.canvas = true;
	params.max_pixels = max_patches * cfg.patch_size * cfg.patch_size;
	const nlohmann::json pc = read_json( std::string( model_dir ) + "/processor_config.json" );
	if ( !pc.is_discarded() && pc.contains( "image_processor" ) ) {
		const auto& ip = pc["image_processor"];
		if ( ip.contains( "image_mean" ) && ip["image_mean"].size() == 3 && ip.contains( "image_std" ) && ip["image_std"].size() == 3 ) {
			for ( int c = 0; c < 3; ++c ) {
				params.mean[c] = ip["image_mean"][c].get<float>();
				params.stddev[c] = ip["image_std"][c].get<float>();
			}
		}
		params.min_pixels = ip.value( "min_image_tokens", 16 ) * cfg.merge_size * cfg.merge_size * cfg.patch_size * cfg.patch_size;
	}
	*out_cfg = cfg;
	*out_params = params;
	return true;
}

bool Glm5VisionTower::load( const char* model_dir, pod::MemoryArena& arena, int max_patches, backend::stream_t stream ) {
	if ( !glm5_vision_config( model_dir, max_patches, &cfg_, &params_ ) ) return false;

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


// ---- the same tower on the host CPU (vision_cpu, host_vision.h): the arithmetic of encode() above over decoded panels or dense weights ----


namespace {

namespace cpu = engine::kernels::cpu;
using namespace hv;

class HostGlm5Tower : public VisionTower {
public:
	bool load( const char* model_dir, const char* mmproj_dir, int max_tokens, const std::vector<int>& cpus, std::string* error );
	bool loaded() const override { return loaded_; }
	const engine::loader::VisionParams& params() const override { return params_; }
	bool encode( const engine::loader::PreparedImage& img, std::vector<backend::half_t>* out, backend::stream_t stream ) override;

private:
	struct Block {
		std::vector<float> norm1_w, norm2_w, q_norm, k_norm, q_b, k_b, v_b, proj_b, gate_b, up_b, down_b;
		Lin q, k, v, proj, gate, up, down;
	};

	bool loaded_ = false;
	Glm5VisionConfig cfg_;
	engine::loader::VisionParams params_;
	int max_patches_ = 0;
	std::unique_ptr<Par> par_;
	std::mutex mtx_;
	Scratch scratch_;
	std::vector<Block> blocks_;
	Dense patch_w_, down_w_;
	std::vector<float> patch_b_, post_norm_w_, down_b_, merge_norm_w_, merge_norm_b_;
	Lin merge_proj_, merge_gate_, merge_up_, merge_down_;
	// per-encode buffers, grown as needed
	std::vector<float> hidden_;
	std::vector<uint16_t> ln_, q_, k_, v_, attn_, tmp_, gate_, up_, mlp_, gathered_, merged_, out_, mgate_, mup_, mmlp_;
};

bool HostGlm5Tower::load( const char* model_dir, const char* mmproj_dir, int max_tokens, const std::vector<int>& cpus, std::string* error ) {
	const auto t0 = std::chrono::steady_clock::now();
	max_patches_ = max_tokens * 4;
	if ( !glm5_vision_config( model_dir, max_patches_, &cfg_, &params_ ) ) {
		*error = "the checkpoint has no usable GLM vision_config";
		return false;
	}
	par_ = std::make_unique<Par>( cpus );
	Par::Pin pin( *par_ );
	engine::loader::SafeTensorsRepo repo;
	const bool from_mmproj = mmproj_dir != nullptr && mmproj_dir[0] != '\0';
	if ( !repo.open( from_mmproj ? mmproj_dir : model_dir, from_mmproj ) ) {
		*error = std::string( "cannot open the " ) + ( from_mmproj ? "vision projector in " : "checkpoint " ) + ( from_mmproj ? mmproj_dir : model_dir );
		return false;
	}
	std::string err;
	const int H = cfg_.hidden;
	auto vec = [&]( const std::string& name ) { return load_vec( repo, name, &err ); };

	// Everything that reads the repo first; the decode of the projections then runs on the pool
	std::vector<LinSource> sources;
	std::vector<Lin*> targets;
	auto want = [&]( const std::string& name, Lin* dst ) {
		LinSource s;
		if ( !find_lin( repo, name, &s, &err ) ) return false;
		sources.push_back( s );
		targets.push_back( dst );
		return true;
	};

	{
		// [hidden, 3, T, P, P] in the checkpoint; a vision projector GGUF has one [hidden, 3, P, P] kernel per temporal frame (.t0, .t1)
		const int P2 = cfg_.patch_size * cfg_.patch_size, T = cfg_.temporal_patch_size;
		std::vector<uint16_t> w;
		if ( repo.find( "model.visual.patch_embed.proj.weight" ) ) w = load_f16( repo, "model.visual.patch_embed.proj.weight", &err );
		else if ( T == 2 && repo.find( "model.visual.patch_embed.proj.weight.t0" ) && repo.find( "model.visual.patch_embed.proj.weight.t1" ) ) {
			const std::vector<uint16_t> t0 = load_f16( repo, "model.visual.patch_embed.proj.weight.t0", &err ), t1 = load_f16( repo, "model.visual.patch_embed.proj.weight.t1", &err );
			if ( t0.size() == static_cast<size_t>( H ) * 3 * P2 && t1.size() == t0.size() ) {
				w.resize( static_cast<size_t>( H ) * cfg_.patch_dim() );
				for ( int o = 0; o < H; ++o )
					for ( int c = 0; c < 3; ++c )
						for ( int q = 0; q < P2; ++q ) {
							w[static_cast<size_t>( o ) * cfg_.patch_dim() + ( c * T + 0 ) * P2 + q] = t0[( static_cast<size_t>( o ) * 3 + c ) * P2 + q];
							w[static_cast<size_t>( o ) * cfg_.patch_dim() + ( c * T + 1 ) * P2 + q] = t1[( static_cast<size_t>( o ) * 3 + c ) * P2 + q];
						}
			}
		}
		if ( w.size() != static_cast<size_t>( H ) * cfg_.patch_dim() || H % 16 != 0 ) {
			*error = "vision: patch embedding shape " + err;
			return false;
		}
		patch_w_ = pack_nk( w.data(), H, cfg_.patch_dim() );
	}
	patch_b_ = vec( "model.visual.patch_embed.proj.bias" );
	post_norm_w_ = vec( "model.visual.post_layernorm.weight" );
	{
		std::vector<uint16_t> w = load_f16( repo, "model.visual.downsample.weight", &err );
		if ( w.size() != static_cast<size_t>( cfg_.out_hidden ) * H * 4 || cfg_.out_hidden % 16 != 0 ) {
			*error = "vision: downsample shape " + err;
			return false;
		}
		down_w_ = pack_nk( w.data(), cfg_.out_hidden, H * 4 );   // [out, hidden, 2, 2] == [out, hidden * 4], channel-major
	}
	down_b_ = vec( "model.visual.downsample.bias" );

	blocks_.resize( cfg_.depth );
	for ( int i = 0; i < cfg_.depth; ++i ) {
		const std::string p = "model.visual.blocks." + std::to_string( i ) + ".";
		Block& b = blocks_[i];
		b.norm1_w = vec( p + "norm1.weight" );
		b.norm2_w = vec( p + "norm2.weight" );
		b.q_norm = vec( p + "attn.q_norm.weight" );
		b.k_norm = vec( p + "attn.k_norm.weight" );
		// the biases of q / k / v: separate in the EXL3 checkpoint, fused in a vision projector GGUF
		if ( repo.find( ( p + "attn.q_proj.bias" ).c_str() ) ) {
			b.q_b = vec( p + "attn.q_proj.bias" );
			b.k_b = vec( p + "attn.k_proj.bias" );
			b.v_b = vec( p + "attn.v_proj.bias" );
		}
		else {
			const std::vector<float> qkv = vec( p + "attn.qkv.bias" );
			if ( qkv.size() == static_cast<size_t>( 3 ) * H ) {
				b.q_b.assign( qkv.begin(), qkv.begin() + H );
				b.k_b.assign( qkv.begin() + H, qkv.begin() + 2 * H );
				b.v_b.assign( qkv.begin() + 2 * H, qkv.end() );
			}
			else err += " (no q / k / v bias)";
		}
		b.proj_b = vec( p + "attn.proj.bias" );
		b.gate_b = vec( p + "mlp.gate_proj.bias" );
		b.up_b = vec( p + "mlp.up_proj.bias" );
		b.down_b = vec( p + "mlp.down_proj.bias" );
		if ( !want( p + "attn.q_proj", &b.q ) || !want( p + "attn.k_proj", &b.k ) || !want( p + "attn.v_proj", &b.v ) || !want( p + "attn.proj", &b.proj ) ||
		     !want( p + "mlp.gate_proj", &b.gate ) || !want( p + "mlp.up_proj", &b.up ) || !want( p + "mlp.down_proj", &b.down ) ) {
			*error = err;
			return false;
		}
	}
	merge_norm_w_ = vec( "model.visual.merger.post_projection_norm.weight" );
	merge_norm_b_ = vec( "model.visual.merger.post_projection_norm.bias" );
	if ( !want( "model.visual.merger.proj", &merge_proj_ ) || !want( "model.visual.merger.gate_proj", &merge_gate_ ) || !want( "model.visual.merger.up_proj", &merge_up_ ) ||
	     !want( "model.visual.merger.down_proj", &merge_down_ ) ) {
		*error = err;
		return false;
	}
	if ( !err.empty() ) {
		*error = err;
		return false;
	}
	decode_all( *par_, sources, targets );
	repo.close();

	size_t bytes = ( patch_w_.w.size() + down_w_.w.size() ) * 2;
	for ( const Lin* l : targets ) bytes += l->d.w.size() * 2;
	loaded_ = true;
	LOG_INFO( "[vision] GLM tower on the host CPU: %d blocks, up to %d patches (%d tokens), %.2f GiB of decoded weights, %d threads, loaded in %.1f s\n", cfg_.depth,
		max_patches_, max_tokens, bytes / 1073741824.0, par_->threads(), std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count() );
	return true;
}

bool HostGlm5Tower::encode( const engine::loader::PreparedImage& img, std::vector<backend::half_t>* out, backend::stream_t ) {
	if ( !loaded_ ) return false;
	const int m = img.num_patches();
	if ( m < 4 || m > max_patches_ ) {
		LOG_WARN( "[vision] image has %d patches, limit is %d\n", m, max_patches_ );
		return false;
	}
	std::lock_guard<std::mutex> lock( mtx_ );
	Par& par = *par_;
	Par::Pin pin( par );
	const auto t0 = std::chrono::steady_clock::now();
	const int H = cfg_.hidden, hd = cfg_.head_dim(), I = cfg_.intermediate, O = cfg_.out_hidden, MI = cfg_.merger_intermediate, P = cfg_.patch_dim();
	const int tokens = m / 4;
	auto fit = []( auto& v, size_t n ) { if ( v.size() < n ) v.resize( n ); };
	const size_t mh = static_cast<size_t>( m ) * H, mi = static_cast<size_t>( m ) * I;
	fit( hidden_, mh );
	for ( auto* v : { &ln_, &q_, &k_, &v_, &attn_, &tmp_ } ) fit( *v, mh );
	for ( auto* v : { &gate_, &up_, &mlp_ } ) fit( *v, mi );

	std::vector<float> cs, sn;
	vision_rotary_tables( img.grid_h, img.grid_w, cfg_.merge_size, hd, &cs, &sn );

	// Patch embedding: the patches narrowed to FP16, times the convolution kernel, plus its bias; the residual stream stays FP32
	fit( scratch_.xin, static_cast<size_t>( m ) * P );
	fit( scratch_.raw, mh );
	par.run( m, 16, [&]( int64_t r0, int64_t r1 ) {
		thread_local std::vector<uint16_t> h;
		h.resize( P );
		for ( int64_t r = r0; r < r1; ++r ) {
			cpu::float_to_half( img.patches.data() + static_cast<size_t>( r ) * P, h.data(), P );
			cpu::half_to_float( h.data(), scratch_.xin.data() + static_cast<size_t>( r ) * P, P );
		}
	} );
	gemm( par, scratch_.xin.data(), m, patch_w_, scratch_.raw.data() );
	par.run( m, 16, [&]( int64_t r0, int64_t r1 ) {
		for ( int64_t r = r0; r < r1; ++r )
			for ( int c = 0; c < H; ++c ) hidden_[static_cast<size_t>( r ) * H + c] = scratch_.raw[static_cast<size_t>( r ) * H + c] + patch_b_[c];
	} );

	const float scale = 1.0f / std::sqrt( static_cast<float>( hd ) );
	for ( const Block& b : blocks_ ) {
		rms_norm( par, hidden_.data(), b.norm1_w.data(), ln_.data(), m, H, cfg_.eps );
		forward( par, scratch_, b.q, ln_.data(), m, q_.data() );
		forward( par, scratch_, b.k, ln_.data(), m, k_.data() );
		forward( par, scratch_, b.v, ln_.data(), m, v_.data() );
		// bias, per-head q / k RMS norm and the 2-D rotary
		par.run( m, 4, [&]( int64_t r0, int64_t r1 ) {
			float t[64];
			for ( int64_t row = r0; row < r1; ++row ) {
				const float* c = cs.data() + static_cast<size_t>( row ) * 64;
				const float* s = sn.data() + static_cast<size_t>( row ) * 64;
				for ( int head = 0; head < cfg_.heads; ++head ) {
					const size_t base = static_cast<size_t>( row ) * H + head * 64;
					for ( int which = 0; which < 3; ++which ) {
						uint16_t* p = ( which == 0 ? q_.data() : which == 1 ? k_.data() : v_.data() ) + base;
						const float* bias = ( which == 0 ? b.q_b.data() : which == 1 ? b.k_b.data() : b.v_b.data() ) + head * 64;
						cpu::half_to_float( p, t, 64 );
						float ss = 0.0f;
						for ( int d = 0; d < 64; ++d ) {
							t[d] += bias[d];
							ss += t[d] * t[d];
						}
						if ( which < 2 ) {
							const float* gain = which == 0 ? b.q_norm.data() : b.k_norm.data();
							const float inv = 1.0f / std::sqrt( ss / 64.0f + cfg_.eps );
							for ( int d = 0; d < 32; ++d ) {
								const float na = t[d] * inv * gain[d], nb = t[d + 32] * inv * gain[d + 32];
								t[d] = na * c[d] - nb * s[d];
								t[d + 32] = nb * c[d + 32] + na * s[d + 32];
							}
						}
						cpu::float_to_half( t, p, 64 );
					}
				}
			}
		} );
		attention( par, scratch_, q_.data(), k_.data(), v_.data(), attn_.data(), m, cfg_.heads, scale );
		forward( par, scratch_, b.proj, attn_.data(), m, tmp_.data() );
		bias_residual( par, hidden_.data(), tmp_.data(), b.proj_b.data(), m, H );

		rms_norm( par, hidden_.data(), b.norm2_w.data(), ln_.data(), m, H, cfg_.eps );
		forward( par, scratch_, b.gate, ln_.data(), m, gate_.data() );
		forward( par, scratch_, b.up, ln_.data(), m, up_.data() );
		add_bias( par, gate_.data(), b.gate_b.data(), m, I );
		add_bias( par, up_.data(), b.up_b.data(), m, I );
		swiglu( par, gate_.data(), up_.data(), mlp_.data(), static_cast<int64_t>( m ) * I, cfg_.swiglu_limit );
		forward( par, scratch_, b.down, mlp_.data(), m, tmp_.data() );
		bias_residual( par, hidden_.data(), tmp_.data(), b.down_b.data(), m, H );
	}
	rms_norm( par, hidden_.data(), post_norm_w_.data(), ln_.data(), m, H, cfg_.eps );

	// 2x2 patch merge as a stride-2 convolution (channel-major gather + product), then the SwiGLU projector
	const size_t tg = static_cast<size_t>( tokens ) * H * 4, to = static_cast<size_t>( tokens ) * O, tm = static_cast<size_t>( tokens ) * MI;
	fit( gathered_, tg );
	fit( merged_, to );
	fit( out_, to );
	for ( auto* v : { &mgate_, &mup_, &mmlp_ } ) fit( *v, tm );
	par.run( tokens, 8, [&]( int64_t t0r, int64_t t1r ) {
		for ( int64_t t = t0r; t < t1r; ++t )
			for ( int idx = 0; idx < H * 4; ++idx ) {
				const int c = idx / 4, j = idx % 4;
				gathered_[static_cast<size_t>( t ) * H * 4 + idx] = ln_[( static_cast<size_t>( t ) * 4 + j ) * H + c];
			}
	} );
	fit( scratch_.xin, tg );
	fit( scratch_.raw, std::max( to, tm ) );
	par.run( tokens, 8, [&]( int64_t r0, int64_t r1 ) {
		for ( int64_t r = r0; r < r1; ++r ) cpu::half_to_float( gathered_.data() + static_cast<size_t>( r ) * H * 4, scratch_.xin.data() + static_cast<size_t>( r ) * H * 4, H * 4 );
	} );
	gemm( par, scratch_.xin.data(), tokens, down_w_, scratch_.raw.data() );
	par.run( tokens, 8, [&]( int64_t r0, int64_t r1 ) {
		for ( int64_t r = r0; r < r1; ++r ) cpu::float_to_half( scratch_.raw.data() + static_cast<size_t>( r ) * O, merged_.data() + static_cast<size_t>( r ) * O, O );
	} );
	add_bias( par, merged_.data(), down_b_.data(), tokens, O );
	forward( par, scratch_, merge_proj_, merged_.data(), tokens, out_.data() );
	layer_norm( par, out_.data(), merge_norm_w_.data(), merge_norm_b_.data(), merged_.data(), tokens, O, cfg_.eps );
	gelu_erf( par, merged_.data(), nullptr, tokens, O );
	forward( par, scratch_, merge_gate_, merged_.data(), tokens, mgate_.data() );
	forward( par, scratch_, merge_up_, merged_.data(), tokens, mup_.data() );
	swiglu( par, mgate_.data(), mup_.data(), mmlp_.data(), static_cast<int64_t>( tokens ) * MI, cfg_.swiglu_limit );
	forward( par, scratch_, merge_down_, mmlp_.data(), tokens, out_.data() );

	out->resize( to );
	std::memcpy( out->data(), out_.data(), to * sizeof( uint16_t ) );
	LOG_INFO( "[vision] host encode: %d patches in %.1f s\n", m, std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count() );
	return true;
}

} // namespace

std::unique_ptr<VisionTower> make_host_glm5_tower( const char* model_dir, const char* mmproj_dir, int max_tokens, const std::vector<int>& cpus, std::string* error ) {
	auto t = std::make_unique<HostGlm5Tower>();
	if ( !t->load( model_dir, mmproj_dir, max_tokens, cpus, error ) ) return nullptr;
	return t;
}

} // namespace engine::models
