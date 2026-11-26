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
#include "host_vision_ops.h"
#include "engine/kernels/cpu/cpu_util.h"
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace engine::models {

bool qwen3_vision_config( const char* model_dir, int max_patches, Qwen3VisionConfig* out_cfg, engine::loader::VisionParams* out_params ) {
	Qwen3VisionConfig cfg;
	engine::loader::VisionParams params;
	const auto cfg_json = read_json( std::string( model_dir ) + "/config.json" );
	if ( cfg_json.is_discarded() || !cfg_json.contains( "vision_config" ) ) return false;
	const auto& vc = cfg_json["vision_config"];
	cfg.depth = vc.value( "depth", cfg.depth );
	cfg.hidden = vc.value( "hidden_size", cfg.hidden );
	cfg.heads = vc.value( "num_heads", cfg.heads );
	cfg.intermediate = vc.value( "intermediate_size", cfg.intermediate );
	cfg.out_hidden = vc.value( "out_hidden_size", cfg.out_hidden );
	cfg.patch_size = vc.value( "patch_size", cfg.patch_size );
	cfg.merge_size = vc.value( "spatial_merge_size", cfg.merge_size );
	cfg.temporal_patch_size = vc.value( "temporal_patch_size", cfg.temporal_patch_size );
	cfg.num_position_embeddings = vc.value( "num_position_embeddings", cfg.num_position_embeddings );
	cfg.deepstack_blocks.clear();
	for ( const auto& d : vc.value( "deepstack_visual_indexes", nlohmann::json::array() ) ) { cfg.deepstack_blocks.push_back( d.get<int>() ); }
	params.patch_size = cfg.patch_size;
	params.merge_size = cfg.merge_size;
	params.temporal_patch_size = cfg.temporal_patch_size;
	params.max_pixels = max_patches * cfg.patch_size * cfg.patch_size;
	*out_cfg = cfg;
	*out_params = params;
	return true;
}

bool Qwen3VisionTower::load( const char* model_dir, pod::MemoryArena& arena, int max_patches, backend::stream_t stream, bool allow_dense ) {
	if ( !qwen3_vision_config( model_dir, max_patches, &cfg_, &params_ ) ) return false;

	engine::loader::SafeTensorsRepo repo;
	if ( !repo.open( model_dir ) ) return false;
	const bool trellis = repo.find( "model.visual.merger.linear_fc1.trellis" ) && repo.find( "model.visual.blocks.0.attn.q_proj.trellis" );
	// an unquantized tower: dense `.weight` tensors, with either separate q / k / v projections or the fused `attn.qkv` of the original release
	fused_qkv_ = false;
	const bool dense = !trellis && allow_dense && repo.find( "model.visual.merger.linear_fc1.weight" ) &&
		( repo.find( "model.visual.blocks.0.attn.q_proj.weight" ) || ( fused_qkv_ = repo.find( "model.visual.blocks.0.attn.qkv.weight" ) != nullptr ) );
	if ( !trellis && !dense ) {
		LOG_WARN( "[vision] checkpoint has no vision tower; image input disabled\n" );
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

// ---- the same tower on the host CPU (vision_cpu, host_vision.h): the arithmetic of encode() above over decoded panels or dense weights ----

namespace {

namespace cpu = engine::kernels::cpu;
using namespace hv;

class HostQwen3Tower : public VisionTower {
public:
	bool load( const char* model_dir, const char* mmproj_dir, int max_tokens, const std::vector<int>& cpus, std::string* error );
	bool loaded() const override { return loaded_; }
	const engine::loader::VisionParams& params() const override { return params_; }
	bool encode( const engine::loader::PreparedImage& img, std::vector<backend::half_t>* out, backend::stream_t stream ) override;

private:
	struct Block {
		std::vector<float> norm1_w, norm1_b, norm2_w, norm2_b, q_b, k_b, v_b, proj_b, fc1_b, fc2_b;
		Lin q, k, v, proj, fc1, fc2;
	};

	bool loaded_ = false;
	Qwen3VisionConfig cfg_;
	engine::loader::VisionParams params_;
	int max_patches_ = 0;
	std::unique_ptr<Par> par_;
	std::mutex mtx_;
	Scratch scratch_;
	std::vector<Block> blocks_;
	Dense patch_w_;
	std::vector<float> patch_b_, merge_norm_w_, merge_norm_b_, merge_fc1_b_, merge_fc2_b_;
	int mlp_width_ = 0;                 // the MLP's width as the projections hold it (padded for EXL3)
	std::vector<uint16_t> pos_table_;   // [num_position_embeddings, hidden] FP16 bits
	Lin merge_fc1_, merge_fc2_;
	std::vector<float> hidden_, pos_;
	std::vector<uint16_t> ln_, q_, k_, v_, attn_, tmp_, mlp_, merged_, mfc1_, out_;
};

bool HostQwen3Tower::load( const char* model_dir, const char* mmproj_dir, int max_tokens, const std::vector<int>& cpus, std::string* error ) {
	const auto t0 = std::chrono::steady_clock::now();
	max_patches_ = max_tokens * 4;
	if ( !qwen3_vision_config( model_dir, max_patches_, &cfg_, &params_ ) ) {
		*error = "the checkpoint has no usable Qwen vision_config";
		return false;
	}
	if ( !cfg_.deepstack_blocks.empty() || cfg_.merge_size != 2 || cfg_.head_dim() % 8 != 0 || cfg_.head_dim() > 128 || cfg_.hidden % 16 != 0 ) {
		*error = "the host Qwen tower does not support DeepStack taps, a merge size other than 2 or this head width";
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
		if ( w.size() != static_cast<size_t>( H ) * cfg_.patch_dim() ) {
			*error = "vision: patch embedding shape " + err;
			return false;
		}
		patch_w_ = pack_nk( w.data(), H, cfg_.patch_dim() );
	}
	patch_b_ = vec( "model.visual.patch_embed.proj.bias" );
	pos_table_ = load_f16( repo, "model.visual.pos_embed.weight", &err );
	if ( pos_table_.size() != static_cast<size_t>( cfg_.num_position_embeddings ) * H ) {
		*error = "vision: pos_embed shape " + err;
		return false;
	}

	blocks_.resize( cfg_.depth );
	for ( int i = 0; i < cfg_.depth; ++i ) {
		const std::string p = "model.visual.blocks." + std::to_string( i ) + ".";
		Block& b = blocks_[i];
		b.norm1_w = vec( p + "norm1.weight" );
		b.norm1_b = vec( p + "norm1.bias" );
		b.norm2_w = vec( p + "norm2.weight" );
		b.norm2_b = vec( p + "norm2.bias" );
		const bool fused = !repo.find( ( p + "attn.q_proj.trellis" ).c_str() ) && !repo.find( ( p + "attn.q_proj.weight" ).c_str() ) && repo.find( ( p + "attn.qkv.weight" ).c_str() );
		if ( fused ) {   // a dense checkpoint of the original release: [3 hidden, hidden] and its bias, cut into thirds
			const pod::SafeTensorMeta* wm = repo.find( ( p + "attn.qkv.weight" ).c_str() );
			const char* raw = static_cast<const char*>( repo.cpu_data( ( p + "attn.qkv.weight" ).c_str() ) );
			const size_t esz = wm && wm->dtype == pod::DType::FP32 ? 4 : 2;
			if ( !wm || !raw || wm->ndim != 2 || wm->shape[0] != 3 * H || wm->shape[1] != H ) {
				*error = "vision: fused qkv shape in block " + std::to_string( i );
				return false;
			}
			Lin* part[3] = { &b.q, &b.k, &b.v };
			for ( int t = 0; t < 3; ++t ) {
				LinSource s;
				s.name = p + "attn.qkv";
				s.dense = raw + static_cast<size_t>( t ) * H * H * esz;
				s.dtype = wm->dtype;
				s.n = H;
				s.k = H;
				sources.push_back( s );
				targets.push_back( part[t] );
			}
			const std::vector<float> qkv = vec( p + "attn.qkv.bias" );
			if ( qkv.size() == static_cast<size_t>( 3 ) * H ) {
				b.q_b.assign( qkv.begin(), qkv.begin() + H );
				b.k_b.assign( qkv.begin() + H, qkv.begin() + 2 * H );
				b.v_b.assign( qkv.begin() + 2 * H, qkv.end() );
			}
		}
		else {
			if ( repo.find( ( p + "attn.q_proj.bias" ).c_str() ) ) {   // separate in the EXL3 checkpoint, fused in a vision projector GGUF
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
			if ( !want( p + "attn.q_proj", &b.q ) || !want( p + "attn.k_proj", &b.k ) || !want( p + "attn.v_proj", &b.v ) ) {
				*error = err;
				return false;
			}
		}
		b.proj_b = vec( p + "attn.proj.bias" );
		b.fc1_b = vec( p + "mlp.linear_fc1.bias" );
		b.fc2_b = vec( p + "mlp.linear_fc2.bias" );
		if ( !want( p + "attn.proj", &b.proj ) || !want( p + "mlp.linear_fc1", &b.fc1 ) || !want( p + "mlp.linear_fc2", &b.fc2 ) ) {
			*error = err;
			return false;
		}
	}
	merge_norm_w_ = vec( "model.visual.merger.norm.weight" );
	merge_norm_b_ = vec( "model.visual.merger.norm.bias" );
	merge_fc1_b_ = vec( "model.visual.merger.linear_fc1.bias" );
	merge_fc2_b_ = vec( "model.visual.merger.linear_fc2.bias" );
	if ( !want( "model.visual.merger.linear_fc1", &merge_fc1_ ) || !want( "model.visual.merger.linear_fc2", &merge_fc2_ ) || !err.empty() ) {
		*error = err;
		return false;
	}
	decode_all( *par_, sources, targets );
	repo.close();
	// EXL3 pads the MLP width (4304) to a multiple of 128 (4352): the first projection then produces the padded columns (zero: the padded weight rows are),
	// so the bias gets zeros there and the MLP buffers use the padded width
	mlp_width_ = blocks_[0].fc1.d.n;
	for ( Block& b : blocks_ ) {
		if ( b.fc1.d.n != mlp_width_ || b.fc2.d.k != mlp_width_ || mlp_width_ < cfg_.intermediate ) {
			*error = "vision: MLP projection widths do not agree";
			return false;
		}
		b.fc1_b.resize( mlp_width_, 0.0f );
	}

	size_t bytes = patch_w_.w.size() * 2 + pos_table_.size() * 2;
	for ( const Lin* l : targets ) bytes += l->d.w.size() * 2;
	loaded_ = true;
	LOG_INFO( "[vision] Qwen tower on the host CPU: %d blocks, up to %d patches (%d tokens), %.2f GiB of decoded weights, %d threads, loaded in %.1f s\n", cfg_.depth,
		max_patches_, max_tokens, bytes / 1073741824.0, par_->threads(), std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count() );
	return true;
}

bool HostQwen3Tower::encode( const engine::loader::PreparedImage& img, std::vector<backend::half_t>* out, backend::stream_t ) {
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
	const int H = cfg_.hidden, hd = cfg_.head_dim(), I = mlp_width_, O = cfg_.out_hidden, P = cfg_.patch_dim(), mrg = 4;
	const int tokens = m / mrg, mw = H * mrg;
	auto fit = []( auto& v, size_t n ) { if ( v.size() < n ) v.resize( n ); };
	const size_t mh = static_cast<size_t>( m ) * H, mi = static_cast<size_t>( m ) * I;
	fit( hidden_, mh );
	fit( pos_, mh );
	for ( auto* v : { &ln_, &q_, &k_, &v_, &attn_, &tmp_ } ) fit( *v, mh );
	fit( mlp_, mi );

	std::vector<float> cs, sn;
	vision_rotary_tables( img.grid_h, img.grid_w, cfg_.merge_size, hd, &cs, &sn );

	// The position table interpolated bilinearly to the image's grid, in the patches' (merge block) order, rounded to FP16
	{
		const int gh = img.grid_h, gw = img.grid_w, merge = cfg_.merge_size, mm = merge * merge, blocks_w = gw / merge;
		const int side = static_cast<int>( std::lround( std::sqrt( static_cast<double>( cfg_.num_position_embeddings ) ) ) );
		par.run( m, 16, [&]( int64_t r0, int64_t r1 ) {
			for ( int64_t row = r0; row < r1; ++row ) {
				const int blk = static_cast<int>( row / mm ), within = static_cast<int>( row % mm );
				const int by = blk / blocks_w, bx = blk % blocks_w;
				const int y = by * merge + within / merge, x = bx * merge + within % merge;
				const float hy = gh == 1 ? 0.0f : static_cast<float>( y ) * ( side - 1 ) / ( gh - 1 );
				const float wx = gw == 1 ? 0.0f : static_cast<float>( x ) * ( side - 1 ) / ( gw - 1 );
				const int hf = static_cast<int>( hy ), wf = static_cast<int>( wx );
				const int hc = std::min( hf + 1, side - 1 ), wc = std::min( wf + 1, side - 1 );
				const float dh = hy - hf, dw = wx - wf;
				const uint16_t* t00 = pos_table_.data() + ( static_cast<size_t>( hf ) * side + wf ) * H;
				const uint16_t* t01 = pos_table_.data() + ( static_cast<size_t>( hf ) * side + wc ) * H;
				const uint16_t* t10 = pos_table_.data() + ( static_cast<size_t>( hc ) * side + wf ) * H;
				const uint16_t* t11 = pos_table_.data() + ( static_cast<size_t>( hc ) * side + wc ) * H;
				for ( int d = 0; d < H; ++d ) {
					const float val = ( 1 - dh ) * ( 1 - dw ) * h2f( t00[d] ) + ( 1 - dh ) * dw * h2f( t01[d] ) + dh * ( 1 - dw ) * h2f( t10[d] ) + dh * dw * h2f( t11[d] );
					pos_[static_cast<size_t>( row ) * H + d] = h2f( f2h( val ) );
				}
			}
		} );
	}

	// Patch embedding: the patches narrowed to FP16, times the convolution kernel, plus its bias and the position rows; the residual stream stays FP32
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
			for ( int c = 0; c < H; ++c ) hidden_[static_cast<size_t>( r ) * H + c] = scratch_.raw[static_cast<size_t>( r ) * H + c] + patch_b_[c] + pos_[static_cast<size_t>( r ) * H + c];
	} );

	const float scale = 1.0f / std::sqrt( static_cast<float>( hd ) );
	const int half_dim = hd / 2;
	for ( const Block& b : blocks_ ) {
		layer_norm( par, hidden_.data(), b.norm1_w.data(), b.norm1_b.data(), ln_.data(), m, H, cfg_.eps );
		forward( par, scratch_, b.q, ln_.data(), m, q_.data() );
		forward( par, scratch_, b.k, ln_.data(), m, k_.data() );
		forward( par, scratch_, b.v, ln_.data(), m, v_.data() );
		// bias and the 2-D rotary (rotate-half layout) on q and k, the bias on v
		par.run( m, 4, [&]( int64_t r0, int64_t r1 ) {
			float t[128];
			for ( int64_t row = r0; row < r1; ++row ) {
				const float* c = cs.data() + static_cast<size_t>( row ) * hd;
				const float* s = sn.data() + static_cast<size_t>( row ) * hd;
				for ( int head = 0; head < cfg_.heads; ++head ) {
					const size_t base = static_cast<size_t>( row ) * H + static_cast<size_t>( head ) * hd;
					for ( int which = 0; which < 3; ++which ) {
						uint16_t* p = ( which == 0 ? q_.data() : which == 1 ? k_.data() : v_.data() ) + base;
						const float* bias = ( which == 0 ? b.q_b.data() : which == 1 ? b.k_b.data() : b.v_b.data() ) + static_cast<size_t>( head ) * hd;
						cpu::half_to_float( p, t, hd );
						for ( int d = 0; d < hd; ++d ) t[d] += bias[d];
						if ( which < 2 ) {
							for ( int d = 0; d < half_dim; ++d ) {
								const float a = t[d], bb = t[d + half_dim];
								t[d] = a * c[d] - bb * s[d];
								t[d + half_dim] = bb * c[d + half_dim] + a * s[d + half_dim];
							}
						}
						cpu::float_to_half( t, p, hd );
					}
				}
			}
		} );
		attention( par, scratch_, q_.data(), k_.data(), v_.data(), attn_.data(), m, cfg_.heads, scale, hd );
		forward( par, scratch_, b.proj, attn_.data(), m, tmp_.data() );
		bias_residual( par, hidden_.data(), tmp_.data(), b.proj_b.data(), m, H );

		layer_norm( par, hidden_.data(), b.norm2_w.data(), b.norm2_b.data(), ln_.data(), m, H, cfg_.eps );
		forward( par, scratch_, b.fc1, ln_.data(), m, mlp_.data() );
		gelu_tanh( par, mlp_.data(), b.fc1_b.data(), m, I );
		forward( par, scratch_, b.fc2, mlp_.data(), m, tmp_.data() );
		bias_residual( par, hidden_.data(), tmp_.data(), b.fc2_b.data(), m, H );
	}

	// The patch merger: LayerNorm over the width, four consecutive rows read as one row, two projections
	const size_t to = static_cast<size_t>( tokens ) * O, tw = static_cast<size_t>( tokens ) * mw;
	fit( merged_, tw );
	fit( mfc1_, tw );
	fit( out_, to );
	layer_norm( par, hidden_.data(), merge_norm_w_.data(), merge_norm_b_.data(), merged_.data(), m, H, cfg_.eps );
	forward( par, scratch_, merge_fc1_, merged_.data(), tokens, mfc1_.data() );
	gelu_erf( par, mfc1_.data(), merge_fc1_b_.data(), tokens, mw );
	forward( par, scratch_, merge_fc2_, mfc1_.data(), tokens, out_.data() );
	add_bias( par, out_.data(), merge_fc2_b_.data(), tokens, O );

	out->resize( to );
	std::memcpy( out->data(), out_.data(), to * sizeof( uint16_t ) );
	LOG_INFO( "[vision] host encode: %d patches in %.1f s\n", m, std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count() );
	return true;
}

} // namespace

std::unique_ptr<VisionTower> make_host_qwen3_tower( const char* model_dir, const char* mmproj_dir, int max_tokens, const std::vector<int>& cpus, std::string* error ) {
	auto t = std::make_unique<HostQwen3Tower>();
	if ( !t->load( model_dir, mmproj_dir, max_tokens, cpus, error ) ) return nullptr;
	return t;
}

} // namespace engine::models
