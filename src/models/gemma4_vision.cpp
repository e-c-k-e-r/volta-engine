#include "engine/models/gemma4_vision.h"
#include "engine/modules/vision.h"
#include "engine/modules/norm.h"
#include "engine/modules/swiglu.h"
#include "engine/loader/safetensors.h"
#include "engine/backend/numeric.h"
#include "engine/backend/runtime.h"
#include "engine/core/log.h"

#include <cmath>
#include <cstring>

namespace engine::models {

bool Gemma4VisionTower::load( const char* model_dir, pod::MemoryArena& arena, int max_soft_tokens, backend::stream_t stream ) {
	const nlohmann::json cfg_json = read_json( std::string( model_dir ) + "/config.json" );
	if ( cfg_json.is_discarded() || !cfg_json.contains( "vision_config" ) || !cfg_json["vision_config"].is_object() ) return false;
	const auto& vc = cfg_json["vision_config"];
	cfg_.depth = vc.value( "num_hidden_layers", cfg_.depth );
	cfg_.hidden = vc.value( "hidden_size", cfg_.hidden );
	cfg_.heads = vc.value( "num_attention_heads", cfg_.heads );
	cfg_.head_dim = vc.value( "head_dim", cfg_.head_dim );
	cfg_.intermediate = vc.value( "intermediate_size", cfg_.intermediate );
	cfg_.patch_size = vc.value( "patch_size", cfg_.patch_size );
	cfg_.pool = vc.value( "pooling_kernel_size", cfg_.pool );
	cfg_.pos_table = vc.value( "position_embedding_size", cfg_.pos_table );
	cfg_.eps = vc.value( "rms_norm_eps", cfg_.eps );
	if ( vc.contains( "rope_parameters" ) && vc["rope_parameters"].is_object() ) cfg_.rope_theta = vc["rope_parameters"].value( "rope_theta", cfg_.rope_theta );
	if ( cfg_json.contains( "text_config" ) && cfg_json["text_config"].is_object() ) {
		cfg_.out_hidden = cfg_json["text_config"].value( "hidden_size", cfg_.out_hidden );
		cfg_.text_eps = cfg_json["text_config"].value( "rms_norm_eps", cfg_.text_eps );
	}
	cfg_.max_soft_tokens = std::min( cfg_json.value( "vision_soft_tokens_per_image", cfg_.max_soft_tokens ), max_soft_tokens );
	if ( cfg_.heads * cfg_.head_dim != cfg_.hidden || cfg_.head_dim > 128 || cfg_.head_dim % 4 != 0 ) {
		LOG_WARN( "[vision] unsupported Gemma 4 vision geometry (%d heads of %d for %d)\n", cfg_.heads, cfg_.head_dim, cfg_.hidden );
		return false;
	}

	params_.patch_size = cfg_.patch_size;
	params_.merge_size = cfg_.pool;
	params_.gemma = true;
	params_.max_tokens = cfg_.max_soft_tokens;

	engine::loader::SafeTensorsRepo repo;
	if ( !repo.open( model_dir ) ) return false;
	const char* VT = "model.vision_tower.";
	auto has = [&]( const std::string& n ) { return repo.find( n.c_str() ) != nullptr; };
	if ( !has( std::string( VT ) + "patch_embedder.input_proj.weight" ) || !has( "model.embed_vision.embedding_projection.weight" ) ) {
		LOG_WARN( "[vision] checkpoint has no Gemma 4 vision tower; image input disabled\n" );
		repo.close();
		return false;
	}
	max_patches_ = cfg_.max_soft_tokens * cfg_.pool * cfg_.pool;
	const size_t M = static_cast<size_t>( max_patches_ ), H = cfg_.hidden, I = cfg_.intermediate, T = cfg_.max_soft_tokens, O = cfg_.out_hidden;
	const size_t scratch = M * ( cfg_.patch_dim() + 8 * H + 3 * I ) * 2 + M * H * 4 + T * ( 2 * H + O ) * 2 + ( 16u << 20 );
	if ( !vision_fits( arena, repo.bytes_with_prefix( VT ) + repo.bytes_with_prefix( "model.embed_vision." ) + scratch, "the Gemma 4 vision tower" ) ) {
		repo.close();
		return false;
	}

	auto fp16 = [&]( const std::string& name ) { return repo.load( name.c_str(), arena, stream, pod::DType::FP16 ); };
	// a dense [n, k] projection, tagged [k, n] for make_fp16
	auto dense = [&]( const std::string& name ) {
		pod::TensorView w = fp16( name );
		ENGINE_ASSERT( w.data != nullptr && w.ndim == 2, "vision: missing dense projection" );
		std::swap( w.shape[0], w.shape[1] );
		return pod::LinearWeight::make_fp16( w );
	};
	patch_proj_ = dense( std::string( VT ) + "patch_embedder.input_proj.weight" );
	out_proj_ = dense( "model.embed_vision.embedding_projection.weight" );
	std_bias_ = fp16( std::string( VT ) + "std_bias" );
	std_scale_ = fp16( std::string( VT ) + "std_scale" );
	{
		const std::string name = std::string( VT ) + "patch_embedder.position_embedding_table";
		pos_table_.resize( static_cast<size_t>( 2 ) * cfg_.pos_table * H );
		ENGINE_ASSERT( repo.copy_host_bf16( name.c_str(), pos_table_.data(), pos_table_.size(), stream ), "vision: position embedding table [2, size, hidden]" );
	}
	blocks_.resize( cfg_.depth );
	for ( int i = 0; i < cfg_.depth; ++i ) {
		const std::string p = std::string( VT ) + "encoder.layers." + std::to_string( i ) + ".";
		Block& b = blocks_[i];
		b.input_norm = fp16( p + "input_layernorm.weight" );
		b.post_attn_norm = fp16( p + "post_attention_layernorm.weight" );
		b.pre_ffn_norm = fp16( p + "pre_feedforward_layernorm.weight" );
		b.post_ffn_norm = fp16( p + "post_feedforward_layernorm.weight" );
		b.q_norm = fp16( p + "self_attn.q_norm.weight" );
		b.k_norm = fp16( p + "self_attn.k_norm.weight" );
		b.q = dense( p + "self_attn.q_proj.linear.weight" );
		b.k = dense( p + "self_attn.k_proj.linear.weight" );
		b.v = dense( p + "self_attn.v_proj.linear.weight" );
		b.o = dense( p + "self_attn.o_proj.linear.weight" );
		b.gate = dense( p + "mlp.gate_proj.linear.weight" );
		b.up = dense( p + "mlp.up_proj.linear.weight" );
		b.down = dense( p + "mlp.down_proj.linear.weight" );
	}
	repo.close();

	d_patches_ = take<backend::half_t>( arena, M * cfg_.patch_dim() );
	d_zero_bias_ = take<backend::half_t>( arena, H );
	backend::memset_sync( d_zero_bias_, 0, H * sizeof( backend::half_t ) );
	d_pos_ = take<backend::half_t>( arena, M * H );
	d_pos_xy_ = take<int>( arena, M * 2 );
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
	d_pooled_ = take<backend::half_t>( arena, T * H );
	d_normed_ = take<backend::half_t>( arena, T * H );
	d_out_ = take<backend::half_t>( arena, T * O );
	if ( stream ) backend::stream_sync( stream );
	loaded_ = true;
	LOG_INFO( "[vision] Gemma 4 tower loaded: %d blocks, up to %d soft tokens (%d patches)\n", cfg_.depth, cfg_.max_soft_tokens, max_patches_ );
	return true;
}

bool Gemma4VisionTower::encode( const engine::loader::PreparedImage& img, std::vector<backend::half_t>* out, backend::stream_t stream ) {
	if ( !loaded_ ) return false;
	const int m = img.num_patches(), gh = img.grid_h, gw = img.grid_w, k = cfg_.pool;
	if ( m < k * k || m > max_patches_ || gh % k || gw % k || static_cast<int>( img.patches.size() ) != m * cfg_.patch_dim() ) {
		LOG_WARN( "[vision] image of %dx%d patches does not fit the tower (at most %d patches, whole %dx%d cells)\n", gh, gw, max_patches_, k, k );
		return false;
	}
	const int H = cfg_.hidden, I = cfg_.intermediate, O = cfg_.out_hidden, tokens = m / ( k * k );

	// Patches go up as FP32 through the MLP scratch (unused until the first block); the position embeddings are the sum of the x and y table rows
	std::vector<int> xy( static_cast<size_t>( m ) * 2 );
	std::vector<backend::half_t> pos( static_cast<size_t>( m ) * H );
	for ( int r = 0; r < m; ++r ) {
		const int x = r % gw, y = r / gw;
		ENGINE_ASSERT( x < cfg_.pos_table && y < cfg_.pos_table, "vision: patch position beyond the position table" );
		xy[2 * r] = x;
		xy[2 * r + 1] = y;
		const uint16_t* tx = &pos_table_[static_cast<size_t>( x ) * H];
		const uint16_t* ty = &pos_table_[( static_cast<size_t>( cfg_.pos_table ) + y ) * H];
		for ( int c = 0; c < H; ++c ) {
			uint32_t a = static_cast<uint32_t>( tx[c] ) << 16, b = static_cast<uint32_t>( ty[c] ) << 16;
			float fa, fb;
			std::memcpy( &fa, &a, 4 );
			std::memcpy( &fb, &b, 4 );
			pos[static_cast<size_t>( r ) * H + c] = backend::from_float_half( fa + fb );
		}
	}
	const size_t patch_floats = static_cast<size_t>( m ) * cfg_.patch_dim();
	ENGINE_ASSERT( patch_floats * sizeof( float ) <= static_cast<size_t>( max_patches_ ) * I * sizeof( backend::half_t ), "vision: patch staging does not fit" );
	backend::memcpy_async( d_mlp_, img.patches.data(), patch_floats * sizeof( float ), backend::Copy::H2D, stream );
	backend::memcpy_async( d_pos_, pos.data(), pos.size() * sizeof( backend::half_t ), backend::Copy::H2D, stream );
	backend::memcpy_async( d_pos_xy_, xy.data(), xy.size() * sizeof( int ), backend::Copy::H2D, stream );
	backend::stream_sync( stream );   // the staging vectors die at return

	using engine::modules::Vision;
	using engine::modules::Linear;
	Vision::f32_to_f16( reinterpret_cast<const float*>( d_mlp_ ), d_patches_, patch_floats, stream );
	Vision::patch_embed( d_patches_, patch_proj_.w0.ptr<const backend::half_t>(), d_zero_bias_, d_pos_, d_hidden_, m, H, cfg_.patch_dim(), stream );
	pod::TensorView resid = pod::make_tensor_2d( d_hidden_, pod::DType::FP32, m, H );
	for ( const Block& b : blocks_ ) {
		Vision::rms_norm( d_hidden_, b.input_norm.ptr<const backend::half_t>(), d_ln_, m, H, cfg_.eps, stream );
		pod::TensorView ln = view2d( d_ln_, m, H ), qv = view2d( d_q_, m, H ), kv = view2d( d_k_, m, H ), vv = view2d( d_v_, m, H );
		Linear::forward( ln, b.q, qv, stream );
		Linear::forward( ln, b.k, kv, stream );
		Linear::forward( ln, b.v, vv, stream );
		Vision::qkv_norm_rope2d( d_q_, d_k_, d_v_, b.q_norm.ptr<const backend::half_t>(), b.k_norm.ptr<const backend::half_t>(), d_pos_xy_, m, cfg_.heads,
			cfg_.head_dim, cfg_.rope_theta, cfg_.eps, stream );
		Vision::attention( d_q_, d_k_, d_v_, d_attn_, m, cfg_.heads, cfg_.head_dim, 1.0f, stream );
		pod::TensorView at = view2d( d_attn_, m, H ), tmp = view2d( d_tmp_, m, H );
		Linear::forward( at, b.o, tmp, stream );
		engine::modules::Elementwise::add_normed( resid, tmp, b.post_attn_norm, cfg_.eps, 0.0f, 1.0f, stream );

		Vision::rms_norm( d_hidden_, b.pre_ffn_norm.ptr<const backend::half_t>(), d_ln_, m, H, cfg_.eps, stream );
		pod::TensorView gate = view2d( d_gate_, m, I ), up = view2d( d_up_, m, I ), mlp = view2d( d_mlp_, m, I );
		Linear::forward( ln, b.gate, gate, stream );
		Linear::forward( ln, b.up, up, stream );
		engine::modules::SwiGLU::forward( gate, up, mlp, stream, 0.0f, true );
		Linear::forward( mlp, b.down, tmp, stream );
		engine::modules::Elementwise::add_normed( resid, tmp, b.post_ffn_norm, cfg_.eps, 0.0f, 1.0f, stream );
	}

	// k x k average pooling with the standardization, the unweighted embedding norm, the projection into the language model
	Vision::pool_std( d_hidden_, std_bias_.ptr<const backend::half_t>(), std_scale_.ptr<const backend::half_t>(), d_pooled_, gh, gw, k, H, stream );
	pod::TensorView pooled = view2d( d_pooled_, tokens, H ), normed = view2d( d_normed_, tokens, H ), o = view2d( d_out_, tokens, O );
	engine::modules::RMSNorm::forward( pooled, pod::TensorView{}, normed, cfg_.text_eps, 0.0f, stream );
	Linear::forward( normed, out_proj_, o, stream );

	out->resize( static_cast<size_t>( tokens ) * O );
	backend::memcpy_async( out->data(), d_out_, out->size() * sizeof( backend::half_t ), backend::Copy::D2H, stream );
	backend::stream_sync( stream );
	return true;
}

} // namespace engine::models
