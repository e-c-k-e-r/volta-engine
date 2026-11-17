#include "engine/core/arena.h"
#include <vector>
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
#include "host_vision_ops.h"
#include "engine/kernels/cpu/cpu_util.h"
#include <chrono>
#include <mutex>
#include <memory>
#include <string>

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

bool dsv41_vision_config( const char* model_dir, int max_image_tokens, DeepseekV41VisionConfig* out_cfg, engine::loader::VisionParams* out_params, int* out_max_tokens ) {
	DeepseekV41VisionConfig cfg;
	engine::loader::VisionParams params;
	const nlohmann::json cfg_json = read_json( std::string( model_dir ) + "/config.json" );
	if ( cfg_json.is_discarded() || !cfg_json.contains( "vision_config" ) ) return false;
	const auto& vc = cfg_json["vision_config"];
	cfg.depth = vc.value( "num_hidden_layers", cfg.depth );
	cfg.hidden = vc.value( "hidden_size", cfg.hidden );
	cfg.heads = vc.value( "num_attention_heads", cfg.heads );
	cfg.intermediate = vc.value( "intermediate_size", cfg.intermediate );
	cfg.patch_size = vc.value( "patch_size", cfg.patch_size );
	cfg.downsample = vc.value( "downsample_ratio", cfg.downsample );
	cfg.rope_theta = vc.value( "rope_theta", cfg.rope_theta );
	if ( cfg.head_dim() != 64 ) {
		LOG_WARN( "[vision] unsupported DeepSeek vision geometry (head_dim %d)\n", cfg.head_dim() );
		return false;
	}
	const int max_tokens = std::min( max_image_tokens, vc.value( "max_image_tokens", max_image_tokens ) );
	params.deepseek = true;
	params.patch_size = cfg.patch_size;
	params.downsample = cfg.downsample;
	params.max_tokens = max_tokens;
	params.min_pixels = vc.value( "min_pixels", 544 * 544 );
	*out_cfg = cfg;
	*out_params = params;
	*out_max_tokens = max_tokens;
	return true;
}

bool DeepseekV41VisionTower::load( const char* model_dir, pod::MemoryArena& arena, int max_image_tokens, backend::stream_t stream ) {
	if ( !dsv41_vision_config( model_dir, max_image_tokens, &cfg_, &params_, &max_tokens_ ) ) return false;
	max_patches_ = max_tokens_ * cfg_.downsample * cfg_.downsample;

	engine::loader::SafeTensorsRepo repo;
	if ( !repo.open( model_dir ) ) return false;
	if ( !repo.find( "vision.patch_embed.proj.weight" ) || aligner_out( repo ) == 0 ) {
		LOG_WARN( "[vision] checkpoint has no vision tower; image input disabled\n" );
		repo.close();
		return false;
	}
	cfg_.out_hidden = aligner_out( repo );
	const size_t M = static_cast<size_t>( max_patches_ ), H = cfg_.hidden, T = static_cast<size_t>( max_tokens_ ), O = cfg_.out_hidden, I = cfg_.intermediate;
	const size_t scratch = M * ( cfg_.patch_cols() * 2 + 64 * 8 + H * 4 + H * 2 * 6 + I * 2 * 3 ) + T * ( H * 9 * 2 + O * 2 * 2 ) + ( 16u << 20 );
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
	if ( cfg_.patch_cols() != cfg_.patch_dim() ) {
		// [hidden, 588] -> [hidden, 592] with zero columns (patch_cols)
		const int P = cfg_.patch_dim(), Pc = cfg_.patch_cols(), Hd = patch_w_.shape[0];
		std::vector<backend::half_t> src( static_cast<size_t>( Hd ) * P ), dst( static_cast<size_t>( Hd ) * Pc, backend::from_float_half( 0.0f ) );
		backend::memcpy_sync( src.data(), patch_w_.data, src.size() * sizeof( backend::half_t ), backend::Copy::D2H );
		for ( int r = 0; r < Hd; ++r ) std::copy( src.begin() + static_cast<size_t>( r ) * P, src.begin() + static_cast<size_t>( r + 1 ) * P, dst.begin() + static_cast<size_t>( r ) * Pc );
		int shape[2] = { Hd, Pc };
		patch_w_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, shape, 2 );
		backend::memcpy_sync( patch_w_.data, dst.data(), dst.size() * sizeof( backend::half_t ), backend::Copy::H2D );
	}
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

	d_patches_ = take<backend::half_t>( arena, M * cfg_.patch_cols() );
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
	// the patch rows at patch_cols columns, the tail zero
	const int Pc = cfg_.patch_cols();
	const size_t patch_floats = static_cast<size_t>( m ) * Pc;
	ENGINE_ASSERT( patch_floats * sizeof( float ) <= static_cast<size_t>( max_patches_ ) * I * sizeof( backend::half_t ), "vision: patch staging does not fit" );
	std::vector<float> padded;
	const float* patch_src = img.patches.data();
	if ( Pc != P ) {
		padded.assign( patch_floats, 0.0f );
		for ( int r = 0; r < m; ++r ) std::copy( img.patches.begin() + static_cast<size_t>( r ) * P, img.patches.begin() + static_cast<size_t>( r + 1 ) * P, padded.begin() + static_cast<size_t>( r ) * Pc );
		patch_src = padded.data();
	}
	backend::memcpy_async( d_mlp_, patch_src, patch_floats * sizeof( float ), backend::Copy::H2D, stream );
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
	Vision::patch_embed( d_patches_, patch_w_.ptr<const backend::half_t>(), patch_b_.ptr<const backend::half_t>(), nullptr, d_hidden_, m, H, Pc, stream );
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


// ---- the same tower on the host CPU (vision_cpu, host_vision.h): the arithmetic of encode() above over decoded panels or dense weights ----


namespace {

namespace cpu = engine::kernels::cpu;
using namespace hv;

class HostDsv41Tower : public VisionTower {
public:
	bool load( const char* model_dir, const char* mmproj_dir, int max_tokens, const std::vector<int>& cpus, std::string* error );
	bool loaded() const override { return loaded_; }
	const engine::loader::VisionParams& params() const override { return params_; }
	bool encode( const engine::loader::PreparedImage& img, std::vector<backend::half_t>* out, backend::stream_t stream ) override;

private:
	struct Block {
		std::vector<float> norm1_w, norm2_w, q_b, k_b, v_b, wo_b;
		Lin q, k, v, wo, gate, up, down;
	};

	bool loaded_ = false;
	DeepseekV41VisionConfig cfg_;
	engine::loader::VisionParams params_;
	int max_patches_ = 0, max_tokens_ = 0;
	std::unique_ptr<Par> par_;
	std::mutex mtx_;
	Scratch scratch_;
	std::vector<Block> blocks_;
	Dense patch_w_;
	std::vector<float> patch_b_, norm_w_, al_b1_, al_b2_;
	Lin al1_, al2_;
	std::vector<uint16_t> start_, end_, newline_;   // the delimiter embeddings of an image span
	std::vector<float> hidden_;
	std::vector<uint16_t> ln_, q_, k_, v_, attn_, tmp_, gate_, up_, mlp_, cat_, al_;
};

bool HostDsv41Tower::load( const char* model_dir, const char* mmproj_dir, int max_tokens, const std::vector<int>& cpus, std::string* error ) {
	const auto t0 = std::chrono::steady_clock::now();
	if ( !dsv41_vision_config( model_dir, max_tokens, &cfg_, &params_, &max_tokens_ ) ) {
		*error = "the checkpoint has no usable DeepSeek vision_config";
		return false;
	}
	max_patches_ = max_tokens_ * cfg_.downsample * cfg_.downsample;
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
		const std::vector<uint16_t> w = load_f16( repo, "vision.patch_embed.proj.weight", &err );
		if ( w.size() != static_cast<size_t>( H ) * cfg_.patch_dim() || H % 16 != 0 ) {
			*error = "vision: patch embedding shape " + err;
			return false;
		}
		patch_w_ = pack_nk( w.data(), H, cfg_.patch_dim() );
	}
	patch_b_ = vec( "vision.patch_embed.proj.bias" );
	norm_w_ = vec( "vision.norm.weight" );
	blocks_.resize( cfg_.depth );
	for ( int i = 0; i < cfg_.depth; ++i ) {
		const std::string p = "vision.blocks." + std::to_string( i ) + ".";
		Block& b = blocks_[i];
		b.norm1_w = vec( p + "norm1.weight" );
		b.norm2_w = vec( p + "norm2.weight" );
		b.wo_b = vec( p + "attn.wo.bias" );
		// the biases of q / k / v: separate in the EXL3 checkpoint, fused in the BF16 tensors and in a vision projector GGUF
		if ( repo.find( ( p + "attn.q_proj.bias" ).c_str() ) ) {
			b.q_b = vec( p + "attn.q_proj.bias" );
			b.k_b = vec( p + "attn.k_proj.bias" );
			b.v_b = vec( p + "attn.v_proj.bias" );
		}
		else {
			const std::vector<float> qkv = vec( repo.find( ( p + "attn.qkv.bias" ).c_str() ) ? p + "attn.qkv.bias" : p + "attn.wqkv.bias" );
			if ( qkv.size() == static_cast<size_t>( 3 ) * H ) {
				b.q_b.assign( qkv.begin(), qkv.begin() + H );
				b.k_b.assign( qkv.begin() + H, qkv.begin() + 2 * H );
				b.v_b.assign( qkv.begin() + 2 * H, qkv.end() );
			}
			else err += " (no q / k / v bias)";
		}
		if ( !want( p + "attn.q_proj", &b.q ) || !want( p + "attn.k_proj", &b.k ) || !want( p + "attn.v_proj", &b.v ) || !want( p + "attn.wo", &b.wo ) ||
		     !want( p + "mlp.gate_proj", &b.gate ) || !want( p + "mlp.up_proj", &b.up ) || !want( p + "mlp.w2", &b.down ) ) {
			*error = err;
			return false;
		}
	}
	al_b1_ = vec( "aligner.w1.bias" );
	al_b2_ = vec( "aligner.w2.bias" );
	start_ = load_f16( repo, "image_start", &err );
	end_ = load_f16( repo, "image_end", &err );
	newline_ = load_f16( repo, "image_newline", &err );
	if ( !want( "aligner.w1", &al1_ ) || !want( "aligner.w2", &al2_ ) || !err.empty() ) {
		*error = err;
		return false;
	}
	decode_all( *par_, sources, targets );
	repo.close();
	cfg_.out_hidden = al2_.d.n;
	if ( static_cast<int>( start_.size() ) != cfg_.out_hidden || end_.size() != start_.size() || newline_.size() != start_.size() || al1_.d.k != H * cfg_.downsample * cfg_.downsample ) {
		*error = "vision: the aligner or the delimiter embeddings do not fit the tower";
		return false;
	}

	size_t bytes = patch_w_.w.size() * 2 + ( al1_.d.w.size() + al2_.d.w.size() ) * 2;
	for ( const Block& b : blocks_ ) bytes += ( b.q.d.w.size() + b.k.d.w.size() + b.v.d.w.size() + b.wo.d.w.size() + b.gate.d.w.size() + b.up.d.w.size() + b.down.d.w.size() ) * 2;
	loaded_ = true;
	LOG_INFO( "[vision] DeepSeek tower on the host CPU: %d blocks, up to %d patches (%d tokens), %.2f GiB of decoded weights, %d threads, loaded in %.1f s\n", cfg_.depth,
		max_patches_, max_tokens_, bytes / 1073741824.0, par_->threads(), std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count() );
	return true;
}

bool HostDsv41Tower::encode( const engine::loader::PreparedImage& img, std::vector<backend::half_t>* out, backend::stream_t ) {
	if ( !loaded_ ) return false;
	const int m = img.num_patches();
	const int R = cfg_.downsample, tokens = img.llm_h * img.llm_w;
	if ( m < 1 || m > max_patches_ || img.llm_h < 1 || img.llm_w < 1 || img.span_tokens() > max_tokens_ ) {
		LOG_WARN( "[vision] image has %d patches and %d tokens, limits are %d and %d\n", m, img.span_tokens(), max_patches_, max_tokens_ );
		return false;
	}
	std::lock_guard<std::mutex> lock( mtx_ );
	Par& par = *par_;
	Par::Pin pin( par );
	const auto t0 = std::chrono::steady_clock::now();
	const int H = cfg_.hidden, I = cfg_.intermediate, O = cfg_.out_hidden, hd = cfg_.head_dim(), P = cfg_.patch_dim(), half = hd / 2;
	auto fit = []( auto& v, size_t n ) { if ( v.size() < n ) v.resize( n ); };
	const size_t mh = static_cast<size_t>( m ) * H, mi = static_cast<size_t>( m ) * I;
	fit( hidden_, mh );
	for ( auto* v : { &ln_, &q_, &k_, &v_, &attn_, &tmp_ } ) fit( *v, mh );
	for ( auto* v : { &gate_, &up_, &mlp_ } ) fit( *v, mi );

	// 2-D rotary of the raster grid: the first 16 of the 32 frequencies turn with the row, the last 16 with the column; both halves of the 64-wide
	// head use the same 32 (rotate_half pairs channel i with i + 32)
	std::vector<float> cs( static_cast<size_t>( m ) * hd ), sn( static_cast<size_t>( m ) * hd );
	{
		const int quarter = hd / 4;
		for ( int y = 0; y < img.grid_h; ++y )
			for ( int x = 0; x < img.grid_w; ++x ) {
				const size_t row = static_cast<size_t>( y ) * img.grid_w + x;
				for ( int j = 0; j < 2 * quarter; ++j ) {
					const int f = j % quarter;
					const double inv = 1.0 / std::pow( static_cast<double>( cfg_.rope_theta ), 2.0 * f / half );
					const double a = ( j < quarter ? y : x ) * inv;
					cs[row * hd + j] = cs[row * hd + j + half] = static_cast<float>( std::cos( a ) );
					sn[row * hd + j] = sn[row * hd + j + half] = static_cast<float>( std::sin( a ) );
				}
			}
	}

	// Patch embedding: the patches narrowed to FP16, times the kernel, plus its bias; the residual stream stays FP32
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
		// bias and the 2-D rotary on q and k (rotate_half: channel d pairs with d + 32 inside every head), bias only on v
		par.run( m, 4, [&]( int64_t r0, int64_t r1 ) {
			thread_local std::vector<float> t;
			t.resize( H );
			for ( int64_t row = r0; row < r1; ++row ) {
				const float* c = cs.data() + static_cast<size_t>( row ) * hd;
				const float* s = sn.data() + static_cast<size_t>( row ) * hd;
				for ( int which = 0; which < 3; ++which ) {
					uint16_t* p = ( which == 0 ? q_.data() : which == 1 ? k_.data() : v_.data() ) + static_cast<size_t>( row ) * H;
					const float* bias = which == 0 ? b.q_b.data() : which == 1 ? b.k_b.data() : b.v_b.data();
					cpu::half_to_float( p, t.data(), H );
					for ( int i = 0; i < H; ++i ) t[i] += bias[i];
					if ( which < 2 ) {
						for ( int head = 0; head < cfg_.heads; ++head ) {
							float* th = t.data() + head * hd;
							for ( int d = 0; d < half; ++d ) {
								const float a = th[d], bb = th[d + half];
								th[d] = a * c[d] - bb * s[d];
								th[d + half] = bb * c[d + half] + a * s[d + half];
							}
						}
					}
					cpu::float_to_half( t.data(), p, H );
				}
			}
		} );
		attention( par, scratch_, q_.data(), k_.data(), v_.data(), attn_.data(), m, cfg_.heads, scale );
		forward( par, scratch_, b.wo, attn_.data(), m, tmp_.data() );
		bias_residual( par, hidden_.data(), tmp_.data(), b.wo_b.data(), m, H );

		rms_norm( par, hidden_.data(), b.norm2_w.data(), ln_.data(), m, H, cfg_.eps );
		forward( par, scratch_, b.gate, ln_.data(), m, gate_.data() );
		forward( par, scratch_, b.up, ln_.data(), m, up_.data() );
		swiglu( par, gate_.data(), up_.data(), mlp_.data(), static_cast<int64_t>( m ) * I, 0.0f );
		forward( par, scratch_, b.down, mlp_.data(), m, tmp_.data() );
		// no bias on the down projection
		par.run( m, 8, [&]( int64_t r0, int64_t r1 ) {
			thread_local std::vector<float> t;
			t.resize( H );
			for ( int64_t r = r0; r < r1; ++r ) {
				cpu::half_to_float( tmp_.data() + static_cast<size_t>( r ) * H, t.data(), H );
				float* h = hidden_.data() + static_cast<size_t>( r ) * H;
				for ( int i = 0; i < H; ++i ) h[i] += t[i];
			}
		} );
	}
	rms_norm( par, hidden_.data(), norm_w_.data(), ln_.data(), m, H, cfg_.eps );

	// The aligner's R x R unfold (F.unfold on the zero-padded grid, kernel and stride R): token (ty, tx) takes channel c of patch (ty * R + ky, tx * R + kx)
	// at column c * R * R + ky * R + kx
	const int K = H * R * R;
	fit( cat_, static_cast<size_t>( tokens ) * K );
	fit( al_, static_cast<size_t>( tokens ) * O );
	std::fill( cat_.begin(), cat_.begin() + static_cast<size_t>( tokens ) * K, static_cast<uint16_t>( 0 ) );
	par.run( tokens, 4, [&]( int64_t t0r, int64_t t1r ) {
		for ( int64_t tk = t0r; tk < t1r; ++tk ) {
			const int ty = static_cast<int>( tk / img.llm_w ), tx = static_cast<int>( tk % img.llm_w );
			uint16_t* dst = cat_.data() + static_cast<size_t>( tk ) * K;
			for ( int ky = 0; ky < R; ++ky )
				for ( int kx = 0; kx < R; ++kx ) {
					const int y = ty * R + ky, x = tx * R + kx;
					if ( y >= img.grid_h || x >= img.grid_w ) continue;
					const uint16_t* src = ln_.data() + ( static_cast<size_t>( y ) * img.grid_w + x ) * H;
					for ( int c = 0; c < H; ++c ) dst[static_cast<size_t>( c ) * R * R + ky * R + kx] = src[c];
				}
		}
	} );
	forward( par, scratch_, al1_, cat_.data(), tokens, al_.data() );
	gelu_erf( par, al_.data(), al_b1_.data(), tokens, O );
	std::vector<uint16_t> rows( static_cast<size_t>( tokens ) * O );
	forward( par, scratch_, al2_, al_.data(), tokens, rows.data() );
	add_bias( par, rows.data(), al_b2_.data(), tokens, O );

	// [start] ([image] * llm_w + [newline]) * llm_h [end]
	const size_t span = static_cast<size_t>( img.span_tokens() );
	out->resize( span * O );
	uint16_t* o = reinterpret_cast<uint16_t*>( out->data() );
	std::memcpy( o, start_.data(), static_cast<size_t>( O ) * sizeof( uint16_t ) );
	o += O;
	for ( int ty = 0; ty < img.llm_h; ++ty ) {
		std::memcpy( o, rows.data() + static_cast<size_t>( ty ) * img.llm_w * O, static_cast<size_t>( img.llm_w ) * O * sizeof( uint16_t ) );
		o += static_cast<size_t>( img.llm_w ) * O;
		std::memcpy( o, newline_.data(), static_cast<size_t>( O ) * sizeof( uint16_t ) );
		o += O;
	}
	std::memcpy( o, end_.data(), static_cast<size_t>( O ) * sizeof( uint16_t ) );
	LOG_INFO( "[vision] host encode: %d patches in %.1f s\n", m, std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count() );
	return true;
}

} // namespace

std::unique_ptr<VisionTower> make_host_dsv41_tower( const char* model_dir, const char* mmproj_dir, int max_tokens, const std::vector<int>& cpus, std::string* error ) {
	auto t = std::make_unique<HostDsv41Tower>();
	if ( !t->load( model_dir, mmproj_dir, max_tokens, cpus, error ) ) return nullptr;
	return t;
}

} // namespace engine::models
