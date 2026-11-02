#include "engine/models/deepseek_v41_vision.h"
#include "engine/modules/vision.h"
#include "engine/modules/swiglu.h"
#include "engine/modules/linear.h"
#include "engine/loader/safetensors.h"
#include "engine/backend/numeric.h"
#include "engine/backend/runtime.h"
#include "engine/core/log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>

namespace engine::models {

namespace {

// The tower's projections: an EXL3 group (name.trellis ...) where the checkpoint has one, else rows [row0, row0 + n) of the dense tensor `dense`
bool has_group( engine::loader::SafeTensorsRepo& repo, const std::string& name ) { return repo.find( ( name + ".trellis" ).c_str() ) != nullptr; }

// Arena bytes of the tower's weights as load() takes them
size_t tower_bytes( engine::loader::SafeTensorsRepo& repo, int depth ) {
	auto pre = [&]( const std::string& s ) { return repo.bytes_with_prefix( s.c_str() ); };
	auto group_or = [&]( const std::string& name, const std::string& dense ) { return has_group( repo, name ) ? pre( name + "." ) : pre( dense + "." ); };
	size_t b = pre( "vision.patch_embed." ) + pre( "vision.norm." );
	for ( int i = 0; i < depth; ++i ) {
		const std::string p = "vision.blocks." + std::to_string( i ) + ".";
		b += pre( p + "norm1." ) + pre( p + "norm2." );
		b += has_group( repo, p + "attn.q_proj" ) ? pre( p + "attn.q_proj." ) + pre( p + "attn.k_proj." ) + pre( p + "attn.v_proj." )
			: pre( p + "attn.wqkv." );
		b += pre( p + "attn.wo." );
		b += has_group( repo, p + "mlp.gate_proj" ) ? pre( p + "mlp.gate_proj." ) + pre( p + "mlp.up_proj." ) : pre( p + "mlp.w1." );
		b += pre( p + "mlp.w2." );
	}
	return b + group_or( "aligner.w1", "aligner.w1" ) + group_or( "aligner.w2", "aligner.w2" );
}

// Output width of the aligner (the delimiter embeddings have it too)
int aligner_out( engine::loader::SafeTensorsRepo& repo ) {
	if ( const auto* m = repo.find( "aligner.w2.weight" ) ) return static_cast<int>( m->shape[0] );
	if ( const auto* m = repo.find( "aligner.w2.trellis" ) ) return static_cast<int>( m->shape[1] ) * 16;
	return 0;
}

} // namespace

size_t DeepseekV41VisionTower::arena_bytes( const char* model_dir, int max_image_tokens ) {
	const nlohmann::json cfg_json = read_json( std::string( model_dir ) + "/config.json" );
	if ( max_image_tokens <= 0 || cfg_json.is_discarded() || !cfg_json.contains( "vision_config" ) ) return 0;
	const auto& vc = cfg_json["vision_config"];
	engine::loader::SafeTensorsRepo repo;
	if ( !repo.open( model_dir ) ) return 0;
	size_t bytes = 0;
	if ( repo.find( "vision.patch_embed.proj.weight" ) && aligner_out( repo ) > 0 ) {
		const int tokens = std::min( max_image_tokens, vc.value( "max_image_tokens", max_image_tokens ) );
		const int ds = vc.value( "downsample_ratio", 3 ), patch = vc.value( "patch_size", 14 );
		const size_t M = static_cast<size_t>( tokens ) * ds * ds, H = vc.value( "hidden_size", 1024 ), I = vc.value( "intermediate_size", 2816 ), T = tokens, O = aligner_out( repo );
		const size_t scratch = M * ( 3 * patch * patch * 2 + 64 * 8 + H * 4 + H * 2 * 6 + I * 2 * 3 ) + T * ( H * 9 * 2 + O * 2 * 2 ) + ( 16u << 20 );
		bytes = tower_bytes( repo, vc.value( "num_hidden_layers", 24 ) ) + 3 * O * 2 + scratch;
	}
	repo.close();
	return bytes;
}

bool DeepseekV41VisionTower::load( const char* model_dir, pod::MemoryArena& arena, int max_image_tokens, backend::stream_t stream ) {
	const nlohmann::json cfg_json = read_json( std::string( model_dir ) + "/config.json" );
	if ( cfg_json.is_discarded() || !cfg_json.contains( "vision_config" ) ) return false;
	const auto& vc = cfg_json["vision_config"];
	cfg_.depth = vc.value( "num_hidden_layers", cfg_.depth );
	cfg_.hidden = vc.value( "hidden_size", cfg_.hidden );
	cfg_.heads = vc.value( "num_attention_heads", cfg_.heads );
	cfg_.intermediate = vc.value( "intermediate_size", cfg_.intermediate );
	cfg_.patch_size = vc.value( "patch_size", cfg_.patch_size );
	cfg_.downsample = vc.value( "downsample_ratio", cfg_.downsample );
	cfg_.rope_theta = vc.value( "rope_theta", cfg_.rope_theta );
	if ( cfg_.head_dim() != 64 ) {
		LOG_WARN( "[vision] unsupported DeepSeek vision geometry (head_dim %d)\n", cfg_.head_dim() );
		return false;
	}
	max_tokens_ = std::min( max_image_tokens, vc.value( "max_image_tokens", max_image_tokens ) );
	max_patches_ = max_tokens_ * cfg_.downsample * cfg_.downsample;

	params_.deepseek = true;
	params_.patch_size = cfg_.patch_size;
	params_.downsample = cfg_.downsample;
	params_.max_tokens = max_tokens_;
	params_.min_pixels = vc.value( "min_pixels", 544 * 544 );

	engine::loader::SafeTensorsRepo repo;
	if ( !repo.open( model_dir ) ) return false;
	if ( !repo.find( "vision.patch_embed.proj.weight" ) || aligner_out( repo ) == 0 ) {
		LOG_WARN( "[vision] checkpoint has no vision tower; image input disabled\n" );
		repo.close();
		return false;
	}
	cfg_.out_hidden = aligner_out( repo );
	const size_t M = static_cast<size_t>( max_patches_ ), H = cfg_.hidden, T = static_cast<size_t>( max_tokens_ ), O = cfg_.out_hidden, I = cfg_.intermediate;
	const size_t scratch = M * ( cfg_.patch_dim() * 2 + 64 * 8 + H * 4 + H * 2 * 6 + I * 2 * 3 ) + T * ( H * 9 * 2 + O * 2 * 2 ) + ( 16u << 20 );
	const size_t weights = tower_bytes( repo, cfg_.depth ) + 3 * O * 2;
	if ( !vision_fits( arena, weights + scratch, "the DeepSeek vision tower" ) ) {
		repo.close();
		return false;
	}

	auto fp16 = [&]( const std::string& name ) {
		pod::TensorView t = repo.load( name.c_str(), arena, stream, pod::DType::FP16 );
		ENGINE_ASSERT( t.data != nullptr, "vision: missing tensor" );
		return t;
	};
	// A projection: the EXL3 group `name`, else rows [row0, row0 + n) of the dense [*, k] tensor `dense` as an FP16 weight
	auto proj = [&]( const std::string& name, const std::string& dense, int row0, int n, int k ) -> pod::LinearWeight {
		if ( has_group( repo, name ) ) {
			pod::LinearWeight w = repo.load_linear( name.c_str(), arena, 4, stream );
			ENGINE_ASSERT( w.is_valid(), "vision: missing projection" );
			return w;
		}
		pod::TensorView t = fp16( dense + ".weight" );
		return pod::LinearWeight::make_fp16( pod::make_tensor_2d( t.ptr<backend::half_t>() + static_cast<size_t>( row0 ) * k, pod::DType::FP16, k, n ) );   // Linear takes k from shape[0]
	};
	patch_w_ = fp16( "vision.patch_embed.proj.weight" );   // [hidden, 3 * 14 * 14]
	patch_b_ = fp16( "vision.patch_embed.proj.bias" );
	norm_w_ = fp16( "vision.norm.weight" );
	blocks_.resize( cfg_.depth );
	for ( int i = 0; i < cfg_.depth; ++i ) {
		const std::string p = "vision.blocks." + std::to_string( i ) + ".";
		Block& b = blocks_[i];
		b.norm1_w = fp16( p + "norm1.weight" );
		b.norm2_w = fp16( p + "norm2.weight" );
		const int Hb = cfg_.hidden, Ib = cfg_.intermediate;
		if ( has_group( repo, p + "attn.q_proj" ) ) {
			b.q = proj( p + "attn.q_proj", "", 0, Hb, Hb );
			b.k = proj( p + "attn.k_proj", "", 0, Hb, Hb );
			b.v = proj( p + "attn.v_proj", "", 0, Hb, Hb );
			b.q_b = fp16( p + "attn.q_proj.bias" ).ptr<const backend::half_t>();
			b.k_b = fp16( p + "attn.k_proj.bias" ).ptr<const backend::half_t>();
			b.v_b = fp16( p + "attn.v_proj.bias" ).ptr<const backend::half_t>();
		}
		else {
			// the fused [3 * hidden, hidden] wqkv: q rows, k rows, v rows
			b.q = proj( "", p + "attn.wqkv", 0, Hb, Hb );
			b.k = proj( "", p + "attn.wqkv", Hb, Hb, Hb );
			b.v = proj( "", p + "attn.wqkv", 2 * Hb, Hb, Hb );
			const backend::half_t* qkv_b = fp16( p + "attn.wqkv.bias" ).ptr<const backend::half_t>();
			b.q_b = qkv_b;
			b.k_b = qkv_b + Hb;
			b.v_b = qkv_b + 2 * Hb;
		}
		b.wo = proj( p + "attn.wo", p + "attn.wo", 0, Hb, Hb );
		b.wo_b = fp16( p + "attn.wo.bias" );
		// [2 * intermediate, hidden] w1: gate rows, then up rows
		b.gate = proj( p + "mlp.gate_proj", p + "mlp.w1", 0, Ib, Hb );
		b.up = proj( p + "mlp.up_proj", p + "mlp.w1", Ib, Ib, Hb );
		b.down = proj( p + "mlp.w2", p + "mlp.w2", 0, Hb, Ib );
	}
	const int R = cfg_.downsample;
	al_w1_ = proj( "aligner.w1", "aligner.w1", 0, static_cast<int>( O ), static_cast<int>( H ) * R * R );
	al_b1_ = fp16( "aligner.w1.bias" );
	al_w2_ = proj( "aligner.w2", "aligner.w2", 0, static_cast<int>( O ), static_cast<int>( O ) );
	al_b2_ = fp16( "aligner.w2.bias" );
	{
		pod::TensorView z = pod::make_tensor_2d( engine::core::arena_alloc( arena, H * sizeof( backend::half_t ) ), pod::DType::FP16, 1, static_cast<int>( H ) );
		backend::memset_async( z.data, 0, H * sizeof( backend::half_t ), stream );
		zero_bias_ = z;
	}
	for ( auto [name, dst] : { std::pair<const char*, std::vector<backend::half_t>*>{ "image_start", &start_ }, { "image_end", &end_ }, { "image_newline", &newline_ } } ) {
		pod::TensorView t = fp16( name );
		dst->resize( O );
		if ( stream ) backend::stream_sync( stream );
		backend::memcpy_sync( dst->data(), t.data, O * sizeof( backend::half_t ), backend::Copy::D2H );
	}
	repo.close();

	d_patches_ = take<backend::half_t>( arena, M * cfg_.patch_dim() );
	d_cos_ = take<float>( arena, M * 64 );
	d_sin_ = take<float>( arena, M * 64 );
	d_hidden_ = take<float>( arena, M * H );
	d_ln_ = take<backend::half_t>( arena, M * H );
	d_q_ = take<backend::half_t>( arena, M * H );
	d_k_ = take<backend::half_t>( arena, M * H );
	d_v_ = take<backend::half_t>( arena, M * H );
	d_attn_ = take<backend::half_t>( arena, M * H );
	d_tmp_ = take<backend::half_t>( arena, M * H );
	d_gate_ = take<backend::half_t>( arena, M * I );
	d_up_ = take<backend::half_t>( arena, M * I );
	d_mlp_ = take<backend::half_t>( arena, M * I );
	d_gathered_ = take<backend::half_t>( arena, T * H * 9 );
	d_al1_ = take<backend::half_t>( arena, T * O );
	d_out_ = take<backend::half_t>( arena, T * O );
	if ( stream ) backend::stream_sync( stream );
	loaded_ = true;
	LOG_INFO( "[vision] DeepSeek tower loaded: %d blocks, up to %d patches (%d tokens), %.2f GiB with scratch\n", cfg_.depth, max_patches_, max_tokens_, ( weights + scratch ) / 1073741824.0 );
	return true;
}

bool DeepseekV41VisionTower::encode( const engine::loader::PreparedImage& img, std::vector<backend::half_t>* out, backend::stream_t stream ) {
	if ( !loaded_ ) return false;
	const int m = img.num_patches();
	const int R = cfg_.downsample, tokens = img.llm_h * img.llm_w;
	if ( m < 1 || m > max_patches_ || img.llm_h < 1 || img.llm_w < 1 || img.span_tokens() > max_tokens_ ) {
		LOG_WARN( "[vision] image has %d patches and %d tokens, limits are %d and %d\n", m, img.span_tokens(), max_patches_, max_tokens_ );
		return false;
	}
	const int H = cfg_.hidden, I = cfg_.intermediate, O = cfg_.out_hidden, hd = cfg_.head_dim(), P = cfg_.patch_dim();

	// 2-D rotary of the raster grid: the first 16 of the 32 frequencies turn with the row, the last 16 with the column; both halves of the 64-wide head
	// use the same 32 (rotate_half pairs channel i with i + 32)
	std::vector<float> cs( static_cast<size_t>( m ) * hd ), sn( static_cast<size_t>( m ) * hd );
	{
		const int quarter = hd / 4;   // 16 frequencies per axis
		for ( int y = 0; y < img.grid_h; ++y ) {
			for ( int x = 0; x < img.grid_w; ++x ) {
				const size_t row = static_cast<size_t>( y ) * img.grid_w + x;
				for ( int j = 0; j < 2 * quarter; ++j ) {
					const int f = j % quarter;
					const double inv = 1.0 / std::pow( static_cast<double>( cfg_.rope_theta ), 2.0 * f / ( hd / 2 ) );
					const double a = ( j < quarter ? y : x ) * inv;
					cs[row * hd + j] = cs[row * hd + j + hd / 2] = static_cast<float>( std::cos( a ) );
					sn[row * hd + j] = sn[row * hd + j + hd / 2] = static_cast<float>( std::sin( a ) );
				}
			}
		}
	}
	const size_t patch_floats = static_cast<size_t>( m ) * P;
	ENGINE_ASSERT( patch_floats * sizeof( float ) <= static_cast<size_t>( max_patches_ ) * I * sizeof( backend::half_t ), "vision: patch staging does not fit" );
	backend::memcpy_async( d_mlp_, img.patches.data(), patch_floats * sizeof( float ), backend::Copy::H2D, stream );
	backend::memcpy_async( d_cos_, cs.data(), cs.size() * sizeof( float ), backend::Copy::H2D, stream );
	backend::memcpy_async( d_sin_, sn.data(), sn.size() * sizeof( float ), backend::Copy::H2D, stream );
	backend::stream_sync( stream );   // the staging vectors die at return

	using engine::modules::Vision;
	// out [rows, n] = x [rows, k] times a projection
	auto lin = [&]( const backend::half_t* x, const pod::LinearWeight& w, backend::half_t* out, int rows, int k, int n ) {
		const pod::TensorView xv = pod::make_tensor_2d( const_cast<backend::half_t*>( x ), pod::DType::FP16, rows, k );
		pod::TensorView ov = pod::make_tensor_2d( out, pod::DType::FP16, rows, n );
		engine::modules::Linear::forward( xv, w, ov, stream );
	};
	Vision::f32_to_f16( reinterpret_cast<const float*>( d_mlp_ ), d_patches_, patch_floats, stream );
	Vision::patch_embed( d_patches_, patch_w_.ptr<const backend::half_t>(), patch_b_.ptr<const backend::half_t>(), nullptr, d_hidden_, m, H, P, stream );
	const float scale = 1.0f / std::sqrt( static_cast<float>( hd ) );
	for ( const Block& b : blocks_ ) {
		Vision::rms_norm( d_hidden_, b.norm1_w.ptr<const backend::half_t>(), d_ln_, m, H, cfg_.eps, stream );
		lin( d_ln_, b.q, d_q_, m, H, H );
		lin( d_ln_, b.k, d_k_, m, H, H );
		lin( d_ln_, b.v, d_v_, m, H, H );
		Vision::qkv_bias_rope( d_q_, d_k_, d_v_, b.q_b, b.k_b, b.v_b, d_cos_, d_sin_, m, cfg_.heads, hd, stream );
		Vision::attention( d_q_, d_k_, d_v_, d_attn_, m, cfg_.heads, hd, scale, stream );
		lin( d_attn_, b.wo, d_tmp_, m, H, H );
		Vision::bias_residual( d_hidden_, d_tmp_, b.wo_b.ptr<const backend::half_t>(), m, H, stream );

		Vision::rms_norm( d_hidden_, b.norm2_w.ptr<const backend::half_t>(), d_ln_, m, H, cfg_.eps, stream );
		lin( d_ln_, b.gate, d_gate_, m, H, I );
		lin( d_ln_, b.up, d_up_, m, H, I );
		pod::TensorView gate = view2d( d_gate_, m, I ), up = view2d( d_up_, m, I ), mlp = view2d( d_mlp_, m, I );
		engine::modules::SwiGLU::forward( gate, up, mlp, stream, 0.0f );
		lin( d_mlp_, b.down, d_tmp_, m, I, H );
		Vision::bias_residual( d_hidden_, d_tmp_, zero_bias_.ptr<const backend::half_t>(), m, H, stream );
	}
	Vision::rms_norm( d_hidden_, norm_w_.ptr<const backend::half_t>(), d_ln_, m, H, cfg_.eps, stream );

	// The aligner's R x R unfold (F.unfold on the zero-padded grid, kernel and stride R): token (ty, tx) takes channel c of patch (ty * R + ky, tx * R + kx)
	// at column c * R * R + ky * R + kx
	std::vector<backend::half_t> feat( static_cast<size_t>( m ) * H ), cat( static_cast<size_t>( tokens ) * H * R * R, backend::from_float_half( 0.0f ) );
	backend::stream_sync( stream );
	backend::memcpy_sync( feat.data(), d_ln_, feat.size() * sizeof( backend::half_t ), backend::Copy::D2H );
	for ( int ty = 0; ty < img.llm_h; ++ty ) {
		for ( int tx = 0; tx < img.llm_w; ++tx ) {
			backend::half_t* dst = &cat[( static_cast<size_t>( ty ) * img.llm_w + tx ) * H * R * R];
			for ( int ky = 0; ky < R; ++ky ) {
				for ( int kx = 0; kx < R; ++kx ) {
					const int y = ty * R + ky, x = tx * R + kx;
					if ( y >= img.grid_h || x >= img.grid_w ) continue;
					const backend::half_t* src = &feat[( static_cast<size_t>( y ) * img.grid_w + x ) * H];
					for ( int c = 0; c < H; ++c ) dst[static_cast<size_t>( c ) * R * R + ky * R + kx] = src[c];
				}
			}
		}
	}
	backend::memcpy_async( d_gathered_, cat.data(), cat.size() * sizeof( backend::half_t ), backend::Copy::H2D, stream );
	lin( d_gathered_, al_w1_, d_al1_, tokens, H * R * R, O );
	Vision::bias_act( d_al1_, al_b1_.ptr<const backend::half_t>(), tokens, O, Vision::ACT_GELU_ERF, stream );
	lin( d_al1_, al_w2_, d_out_, tokens, O, O );
	Vision::bias_act( d_out_, al_b2_.ptr<const backend::half_t>(), tokens, O, Vision::ACT_NONE, stream );
	std::vector<backend::half_t> rows( static_cast<size_t>( tokens ) * O );
	backend::stream_sync( stream );   // the unfold staging is consumed
	backend::memcpy_sync( rows.data(), d_out_, rows.size() * sizeof( backend::half_t ), backend::Copy::D2H );

	// [start] ([image] * llm_w + [newline]) * llm_h [end]
	out->clear();
	out->reserve( static_cast<size_t>( img.span_tokens() ) * O );
	out->insert( out->end(), start_.begin(), start_.end() );
	for ( int ty = 0; ty < img.llm_h; ++ty ) {
		out->insert( out->end(), rows.begin() + static_cast<size_t>( ty ) * img.llm_w * O, rows.begin() + static_cast<size_t>( ty + 1 ) * img.llm_w * O );
		out->insert( out->end(), newline_.begin(), newline_.end() );
	}
	out->insert( out->end(), end_.begin(), end_.end() );
	return true;
}

} // namespace engine::models
