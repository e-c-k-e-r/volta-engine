#include "engine/models/gemma4_audio.h"
#include "engine/modules/vision.h"
#include "engine/modules/norm.h"
#include "engine/loader/safetensors.h"
#include "engine/backend/numeric.h"
#include "engine/backend/runtime.h"
#include "engine/core/log.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace engine::models {

namespace {

constexpr const char* AT = "model.audio_tower.";

} // namespace

bool Gemma4AudioTower::load( const char* model_dir, pod::MemoryArena& arena, int max_tokens, backend::stream_t stream ) {
	const nlohmann::json cfg_json = read_json( std::string( model_dir ) + "/config.json" );
	if ( cfg_json.is_discarded() || !cfg_json.contains( "audio_config" ) || !cfg_json["audio_config"].is_object() ) return false;
	const auto& ac = cfg_json["audio_config"];
	cfg_.layers = ac.value( "num_hidden_layers", cfg_.layers );
	cfg_.hidden = ac.value( "hidden_size", cfg_.hidden );
	cfg_.heads = ac.value( "num_attention_heads", cfg_.heads );
	cfg_.conv_kernel = ac.value( "conv_kernel_size", cfg_.conv_kernel );
	cfg_.context_left = ac.value( "attention_context_left", cfg_.context_left );
	cfg_.logit_cap = ac.value( "attention_logit_cap", cfg_.logit_cap );
	cfg_.residual_weight = ac.value( "residual_weight", cfg_.residual_weight );
	cfg_.eps = ac.value( "rms_norm_eps", cfg_.eps );
	cfg_.out_proj = ac.value( "output_proj_dims", cfg_.out_proj );
	cfg_.clipped = ac.value( "use_clipped_linears", cfg_.clipped );
	if ( ac.contains( "subsampling_conv_channels" ) && ac["subsampling_conv_channels"].size() == 2 ) {
		cfg_.sub_channels[0] = ac["subsampling_conv_channels"][0].get<int>();
		cfg_.sub_channels[1] = ac["subsampling_conv_channels"][1].get<int>();
	}
	if ( cfg_json.contains( "text_config" ) && cfg_json["text_config"].is_object() ) cfg_.out_hidden = cfg_json["text_config"].value( "hidden_size", cfg_.out_hidden );
	const int right = ac.value( "attention_context_right", 0 );
	const int chunk = ac.value( "attention_chunk_size", 12 );
	if ( right != 0 || cfg_.window() < 1 || cfg_.window() > 32 || cfg_.window() > chunk + cfg_.context_left || cfg_.hidden % cfg_.heads != 0 ||
	     ( params_.mel_bins / 4 ) * cfg_.sub_channels[1] <= 0 ) {
		LOG_WARN( "[audio] unsupported Gemma 4 audio geometry (context %d left / %d right, %d heads of %d)\n", cfg_.context_left, right, cfg_.heads, cfg_.hidden );
		return false;
	}

	engine::loader::SafeTensorsRepo repo;
	if ( !repo.open( model_dir ) ) return false;
	auto has = [&]( const std::string& n ) { return repo.find( n.c_str() ) != nullptr; };
	const std::string probe = std::string( AT ) + "layers.0.self_attn.q_proj.linear.weight";
	// a GGUF model keeps the tower in its mmproj file
	if ( !has( probe ) && !repo.open( model_dir, true ) ) return false;
	if ( !has( probe ) || !has( "model.embed_audio.embedding_projection.weight" ) ) {
		LOG_WARN( "[audio] checkpoint has no Gemma 4 audio tower; audio input disabled\n" );
		repo.close();
		return false;
	}
	const size_t H = cfg_.hidden, M = static_cast<size_t>( max_tokens ), F = 4 * H, P = cfg_.out_proj, O = cfg_.out_hidden;
	const size_t weights = repo.bytes_with_prefix( AT ) / 2 + repo.bytes_with_prefix( "model.embed_audio." ) / 2 + 1;   // FP16 of BF16 / FP32 tensors (at most)
	const size_t sub_bytes = ( 4 * M * params_.mel_bins + ( 2 * M + 1 ) * ( ( params_.mel_bins + 1 ) / 2 ) * cfg_.sub_channels[0] + M * params_.mel_bins / 4 * cfg_.sub_channels[1] ) * 4;
	const size_t scratch = sub_bytes + M * H * 4 + M * ( 6 * H + 2 * F + P + O ) * 2 + static_cast<size_t>( cfg_.layers ) * ( cfg_.window() + 1 ) * H * 4 + ( 16u << 20 );
	{
		const size_t left = arena.total_bytes > arena.offset_bytes ? arena.total_bytes - arena.offset_bytes : 0;
		const size_t need = std::max( weights, repo.bytes_with_prefix( AT ) ) + scratch;
		if ( left < need ) {
			LOG_WARN( "[audio] the Gemma 4 audio tower needs %.0f MiB but the arena has %.0f MiB left; audio input disabled\n", need / 1048576.0, left / 1048576.0 );
			repo.close();
			return false;
		}
	}

	auto fp16 = [&]( const std::string& name ) {
		pod::TensorView t = repo.load( name.c_str(), arena, stream, pod::DType::FP16 );
		ENGINE_ASSERT( t.data != nullptr, "audio: missing tensor" );
		return t;
	};
	// a dense [n, k] projection, tagged [k, n] for make_fp16
	auto dense = [&]( const std::string& name ) {
		pod::TensorView w = fp16( name );
		ENGINE_ASSERT( w.ndim == 2, "audio: a projection is not a matrix" );
		std::swap( w.shape[0], w.shape[1] );
		return pod::LinearWeight::make_fp16( w );
	};
	auto host = [&]( const std::string& name ) {
		std::vector<float> v;
		ENGINE_ASSERT( repo_host_floats( repo, name, &v ), "audio: missing or unreadable tensor" );
		return v;
	};
	auto upload_half = [&]( const std::vector<float>& v ) {
		std::vector<backend::half_t> h( v.size() );
		for ( size_t i = 0; i < v.size(); ++i ) h[i] = backend::from_float_half( v[i] );
		auto* d = take<backend::half_t>( arena, v.size() );
		backend::memcpy_sync( d, h.data(), h.size() * sizeof( backend::half_t ), backend::Copy::H2D );
		pod::TensorView t = pod::make_tensor_2d( d, pod::DType::FP16, 1, static_cast<int>( v.size() ) );
		t.ndim = 1;
		t.shape[0] = static_cast<int>( v.size() );
		return t;
	};
	auto upload_float = [&]( const std::vector<float>& v ) {
		auto* d = take<float>( arena, v.size() );
		backend::memcpy_sync( d, v.data(), v.size() * sizeof( float ), backend::Copy::H2D );
		return d;
	};
	auto clip = [&]( const std::string& base ) { return cfg_.clipped ? repo_clip( repo, base ) : ClipRange(); };

	// the subsampler in FP32
	const std::string sub = std::string( AT ) + "subsample_conv_projection.";
	for ( int i = 0; i < 2; ++i ) {
		const std::vector<float> w = host( sub + "layer" + std::to_string( i ) + ".conv.weight" ), g = host( sub + "layer" + std::to_string( i ) + ".norm.weight" );
		const size_t in_ch = i == 0 ? 1 : cfg_.sub_channels[0];
		ENGINE_ASSERT( w.size() == static_cast<size_t>( cfg_.sub_channels[i] ) * in_ch * 9 && g.size() == static_cast<size_t>( cfg_.sub_channels[i] ) &&
			cfg_.sub_channels[i] <= 256, "audio: subsampler convolution shapes" );
		d_conv_w_[i] = upload_float( w );
		d_conv_g_[i] = upload_float( g );
	}
	in_proj_ = dense( sub + "input_proj_linear.weight" );
	ENGINE_ASSERT( in_proj_.w0.shape[0] == ( params_.mel_bins / 4 ) * cfg_.sub_channels[1] && in_proj_.w0.shape[1] == static_cast<int>( H ), "audio: input projection shape" );

	// relative position keys: the sinusoid of each distance 0 .. window - 1 ([sin, cos] over hidden / 2 timescales) through relative_k_proj
	const int W = cfg_.window(), hd = cfg_.head_dim(), half_h = cfg_.hidden / 2;
	const double log_inc = std::log( 10000.0 ) / std::max( half_h - 1, 1 );
	std::vector<double> emb( static_cast<size_t>( W ) * H );
	for ( int d = 0; d < W; ++d ) {
		for ( int i = 0; i < half_h; ++i ) {
			const double a = d * std::exp( -i * log_inc );
			emb[static_cast<size_t>( d ) * H + i] = std::sin( a );
			emb[static_cast<size_t>( d ) * H + half_h + i] = std::cos( a );
		}
	}
	const double q_base = 1.0 / std::sqrt( static_cast<double>( hd ) ) / std::log( 2.0 );

	auto feed_forward = [&]( const std::string& p, FeedForward* f ) {
		f->pre = fp16( p + "pre_layer_norm.weight" );
		std::vector<float> post = host( p + "post_layer_norm.weight" );
		for ( float& v : post ) v *= cfg_.residual_weight;   // the half step folded into the norm's gain
		f->post = upload_half( post );
		f->up = dense( p + "ffw_layer_1.linear.weight" );
		f->down = dense( p + "ffw_layer_2.linear.weight" );
		f->cup = clip( p + "ffw_layer_1" );
		f->cdown = clip( p + "ffw_layer_2" );
	};
	blocks_.resize( cfg_.layers );
	for ( int l = 0; l < cfg_.layers; ++l ) {
		const std::string p = std::string( AT ) + "layers." + std::to_string( l ) + ".";
		Block& b = blocks_[l];
		feed_forward( p + "feed_forward1.", &b.ff1 );
		feed_forward( p + "feed_forward2.", &b.ff2 );
		b.attn_pre = fp16( p + "norm_pre_attn.weight" );
		b.attn_post = fp16( p + "norm_post_attn.weight" );
		b.q = dense( p + "self_attn.q_proj.linear.weight" );
		b.k = dense( p + "self_attn.k_proj.linear.weight" );
		b.v = dense( p + "self_attn.v_proj.linear.weight" );
		b.post = dense( p + "self_attn.post.linear.weight" );
		b.cq = clip( p + "self_attn.q_proj" );
		b.ck = clip( p + "self_attn.k_proj" );
		b.cv = clip( p + "self_attn.v_proj" );
		b.cpost = clip( p + "self_attn.post" );
		{
			// q's per-dimension scale: the converter stores softplus( per_dim_scale ), the checkpoint per_dim_scale itself
			std::vector<float> s;
			const bool softplused = repo_host_floats( repo, p + "self_attn.per_dim_scale_softplus", &s );
			if ( !softplused ) {
				s = host( p + "self_attn.per_dim_scale" );
				for ( float& v : s ) v = static_cast<float>( std::log1p( std::exp( static_cast<double>( v ) ) ) );
			}
			ENGINE_ASSERT( s.size() == static_cast<size_t>( hd ), "audio: per_dim_scale size" );
			for ( float& v : s ) v = static_cast<float>( v * q_base );
			b.q_scale = upload_float( s );
		}
		{
			const std::vector<float> w = host( p + "self_attn.relative_k_proj.weight" );   // [hidden out, hidden in]
			ENGINE_ASSERT( w.size() == H * H, "audio: relative_k_proj shape" );
			std::vector<float> rel( static_cast<size_t>( W ) * H );
			for ( int d = 0; d < W; ++d ) {
				for ( size_t o = 0; o < H; ++o ) {
					double acc = 0.0;
					for ( size_t j = 0; j < H; ++j ) acc += static_cast<double>( w[o * H + j] ) * emb[static_cast<size_t>( d ) * H + j];
					rel[static_cast<size_t>( d ) * H + o] = static_cast<float>( acc );
				}
			}
			b.rel = upload_float( rel );
		}
		b.conv_pre = fp16( p + "lconv1d.pre_layer_norm.weight" );
		b.conv_dw = fp16( p + "lconv1d.depthwise_conv1d.weight" );
		ENGINE_ASSERT( b.conv_dw.numel() == H * static_cast<size_t>( cfg_.conv_kernel ), "audio: depthwise convolution shape" );
		b.conv_norm = fp16( p + "lconv1d.conv_norm.weight" );
		b.pw1 = dense( p + "lconv1d.linear_start.linear.weight" );
		b.pw2 = dense( p + "lconv1d.linear_end.linear.weight" );
		b.cpw1 = clip( p + "lconv1d.linear_start" );
		b.cpw2 = clip( p + "lconv1d.linear_end" );
		b.out_norm = fp16( p + "norm_out.weight" );
	}
	out_proj_ = dense( std::string( AT ) + "output_proj.weight" );
	out_bias_ = fp16( std::string( AT ) + "output_proj.bias" );
	embed_proj_ = dense( "model.embed_audio.embedding_projection.weight" );
	ENGINE_ASSERT( out_proj_.w0.shape[1] == static_cast<int>( P ) && embed_proj_.w0.shape[0] == static_cast<int>( P ) && embed_proj_.w0.shape[1] == static_cast<int>( O ), "audio: output projections" );
	repo.close();

	max_tokens_ = max_tokens;
	{
		const size_t frames = 4 * M, mel = params_.mel_bins;
		d_mel_ = take<float>( arena, frames * mel );
		d_sub_[0] = take<float>( arena, ( frames / 2 + 1 ) * ( ( mel + 1 ) / 2 ) * cfg_.sub_channels[0] );
		d_sub_[1] = take<float>( arena, M * ( ( ( mel + 1 ) / 2 + 1 ) / 2 ) * cfg_.sub_channels[1] );
	}
	d_hidden_ = take<float>( arena, M * H );
	d_zero_bias_ = take<backend::half_t>( arena, H );
	backend::memset_sync( d_zero_bias_, 0, H * sizeof( backend::half_t ) );
	d_ln_ = take<backend::half_t>( arena, M * H );
	d_q_ = take<backend::half_t>( arena, M * H );
	d_k_ = take<backend::half_t>( arena, M * H );
	d_v_ = take<backend::half_t>( arena, M * H );
	d_attn_ = take<backend::half_t>( arena, M * H );
	d_tmp_ = take<backend::half_t>( arena, M * H );
	d_ffn_ = take<backend::half_t>( arena, M * F );
	d_proj_ = take<backend::half_t>( arena, M * P );
	d_normed_ = take<backend::half_t>( arena, M * P );
	d_out_ = take<backend::half_t>( arena, M * O );
	if ( stream ) backend::stream_sync( stream );
	loaded_ = true;
	LOG_INFO( "[audio] Gemma 4 tower loaded: %d blocks, up to %d tokens a chunk (%d s)\n", cfg_.layers, max_tokens_, params_.chunk_seconds );
	return true;
}

void Gemma4AudioTower::clipped( pod::TensorView& in, const pod::LinearWeight& w, const ClipRange& c, pod::TensorView& out, backend::half_t* spare,
	backend::stream_t stream ) {
	using engine::modules::Vision;
	pod::TensorView x = in;
	if ( c.in() ) {
		const size_t n = static_cast<size_t>( in.shape[0] ) * in.shape[1];
		if ( spare != nullptr ) {
			backend::memcpy_async( spare, in.data, n * sizeof( backend::half_t ), backend::Copy::D2D, stream );
			x = view2d( spare, in.shape[0], in.shape[1] );
		}
		Vision::clamp( x.ptr<backend::half_t>(), n, c.in_lo, c.in_hi, stream );
	}
	engine::modules::Linear::forward( x, w, out, stream );
	if ( c.out() ) Vision::clamp( out.ptr<backend::half_t>(), static_cast<size_t>( out.shape[0] ) * out.shape[1], c.out_lo, c.out_hi, stream );
}

// resid += residual_weight * post_norm( down( silu( up( pre_norm( resid ) ) ) ) ) (the weight folded into post_norm); next: the new rows
// normed into it (FP16)
void Gemma4AudioTower::feed_forward( const FeedForward& f, int T, backend::stream_t stream, pod::TensorView* next, const pod::TensorView* next_weight ) {
	using engine::modules::Vision;
	const int H = cfg_.hidden, F = 4 * H;
	pod::TensorView resid = pod::make_tensor_2d( d_hidden_, pod::DType::FP32, T, H ), ln = view2d( d_ln_, T, H ), ffn = view2d( d_ffn_, T, F ),
		tmp = view2d( d_tmp_, T, H );
	engine::modules::RMSNorm::forward( resid, f.pre, ln, cfg_.eps, 0.0f, stream );
	clipped( ln, f.up, f.cup, ffn, nullptr, stream );
	Vision::bias_act( d_ffn_, nullptr, T, F, Vision::ACT_SILU, stream );
	clipped( ffn, f.down, f.cdown, tmp, nullptr, stream );
	engine::modules::Elementwise::add_normed( resid, tmp, f.post, cfg_.eps, 0.0f, 1.0f, stream, next, next_weight );
}

bool Gemma4AudioTower::encode( const engine::loader::PreparedAudio& a, std::vector<backend::half_t>* out, backend::stream_t stream ) {
	if ( !loaded_ ) return false;
	const int T = a.tokens, H = cfg_.hidden, P = cfg_.out_proj, O = cfg_.out_hidden, hd = cfg_.head_dim();
	if ( T < 1 || T > max_tokens_ || a.mel.size() != static_cast<size_t>( a.frames ) * params_.mel_bins || params_.tokens( a.frames ) != T ) {
		LOG_WARN( "[audio] a chunk of %d tokens does not fit the tower (at most %d)\n", T, max_tokens_ );
		return false;
	}
	using engine::modules::Vision;
	using engine::modules::Linear;
	using engine::modules::RMSNorm;
	using engine::modules::Elementwise;

	// the log-mel frames up, the two subsampler layers ([rows, frequencies, channels]: the flattening order), narrowed, projected into the FP32
	// residual stream
	const int mel = params_.mel_bins, f1 = ( mel + 1 ) / 2, t1 = ( a.frames + 1 ) / 2, f2 = ( f1 + 1 ) / 2;
	ENGINE_ASSERT( ( t1 + 1 ) / 2 == T && f2 * cfg_.sub_channels[1] == in_proj_.w0.shape[0], "audio: subsampler shapes" );
	backend::memcpy_async( d_mel_, a.mel.data(), a.mel.size() * sizeof( float ), backend::Copy::H2D, stream );
	Vision::conv_ln_relu( d_mel_, d_conv_w_[0], d_conv_g_[0], d_sub_[0], a.frames, mel, 1, cfg_.sub_channels[0], cfg_.eps, stream );
	Vision::conv_ln_relu( d_sub_[0], d_conv_w_[1], d_conv_g_[1], d_sub_[1], t1, f1, cfg_.sub_channels[0], cfg_.sub_channels[1], cfg_.eps, stream );
	Vision::f32_to_f16( d_sub_[1], d_ln_, static_cast<size_t>( T ) * in_proj_.w0.shape[0], stream );
	pod::TensorView resid = pod::make_tensor_2d( d_hidden_, pod::DType::FP32, T, H ), ln = view2d( d_ln_, T, in_proj_.w0.shape[0] ), tmp = view2d( d_tmp_, T, H );
	Linear::forward( ln, in_proj_, tmp, stream );
	Elementwise::scale_to_f32( tmp, resid, 1.0f, stream );

	ln = view2d( d_ln_, T, H );
	pod::TensorView qv = view2d( d_q_, T, H ), kv = view2d( d_k_, T, H ), vv = view2d( d_v_, T, H ), at = view2d( d_attn_, T, H );
	pod::TensorView pw1 = view2d( d_ffn_, T, 2 * H );
	for ( const Block& b : blocks_ ) {
		feed_forward( b.ff1, T, stream );

		RMSNorm::forward( resid, b.attn_pre, ln, cfg_.eps, 0.0f, stream );
		const bool shared = b.cq.same_input( b.ck ) && b.cq.same_input( b.cv );
		clipped( ln, b.q, b.cq, qv, shared ? nullptr : d_attn_, stream );
		clipped( ln, b.k, b.ck, kv, shared ? nullptr : d_attn_, stream );
		clipped( ln, b.v, b.cv, vv, shared ? nullptr : d_attn_, stream );
		Vision::local_rel_attention( d_q_, d_k_, d_v_, b.q_scale, static_cast<float>( std::log1p( std::exp( 1.0 ) ) / std::log( 2.0 ) ), b.rel, d_attn_, T,
			cfg_.heads, hd, cfg_.window(), cfg_.logit_cap, stream );
		clipped( at, b.post, b.cpost, tmp, nullptr, stream );
		Elementwise::add_normed( resid, tmp, b.attn_post, cfg_.eps, 0.0f, 1.0f, stream );

		// the light convolution: GLU, the causal depthwise taps, a norm, SiLU, the projection back, added to the residual
		RMSNorm::forward( resid, b.conv_pre, ln, cfg_.eps, 0.0f, stream );
		clipped( ln, b.pw1, b.cpw1, pw1, nullptr, stream );
		Vision::glu_dwconv( d_ffn_, b.conv_dw.ptr<const backend::half_t>(), d_attn_, T, H, cfg_.conv_kernel, stream );
		RMSNorm::forward( at, b.conv_norm, ln, cfg_.eps, 0.0f, stream );
		Vision::bias_act( d_ln_, nullptr, T, H, Vision::ACT_SILU, stream );
		clipped( ln, b.pw2, b.cpw2, tmp, nullptr, stream );
		Vision::bias_residual( d_hidden_, d_tmp_, d_zero_bias_, T, H, stream );

		// the second half step, the block's norm as the next block's input
		feed_forward( b.ff2, T, stream, &ln, &b.out_norm );
		Elementwise::scale_to_f32( ln, resid, 1.0f, stream );
	}

	// the output projection, the unweighted norm, the projection into the language model
	pod::TensorView proj = view2d( d_proj_, T, P ), normed = view2d( d_normed_, T, P ), o = view2d( d_out_, T, O );
	Linear::forward( ln, out_proj_, proj, stream );
	Vision::bias_act( d_proj_, out_bias_.ptr<const backend::half_t>(), T, P, Vision::ACT_NONE, stream );
	RMSNorm::forward( proj, pod::TensorView{}, normed, cfg_.eps, 0.0f, stream );
	Linear::forward( normed, embed_proj_, o, stream );

	out->resize( static_cast<size_t>( T ) * O );
	backend::memcpy_async( out->data(), d_out_, out->size() * sizeof( backend::half_t ), backend::Copy::D2H, stream );
	backend::stream_sync( stream );
	return true;
}

} // namespace engine::models
