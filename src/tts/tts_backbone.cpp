// The TTS backbone (vall_e/model.py V3.hidden) for one unpadded row, or several rows of one pass: embeddings, pre-norm blocks of
// bidirectional gated attention (QK RMS norm, RoPE) and a SwiGLU MLP, then the final RMS norm. The operators are the backend's
// (engine/modules/tts.h), the rows' layout and the per-row attention host code.
#include "engine/tts/tts_model.h"
#include "engine/backend/runtime.h"
#include "engine/core/log.h"
#include "engine/modules/tts.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace engine::tts {

using engine::modules::TtsOps;

// nn.RMSNorm's default eps is the input dtype's machine epsilon: the residual stream is FP32 (1.19e-7), while under the
// bf16 autocast the model was trained and is served with, the q / k norms see bf16 projections (2^-7)
static constexpr float EPS_FP32 = 1.1920929e-7f;
static constexpr float EPS_QK = 0.0078125f;

int TtsRow::length() const {
	return static_cast<int>( text.size() ) + static_cast<int>( prompt.size() / 64 ) + ( len_query ? 1 : static_cast<int>( masked.size() ) );
}

void TtsModel::alloc_workspace() {
	const size_t L = MAX_LEN, d = cfg_.d_model, F = cfg_.ffn(), V = cfg_.values();
	const size_t attn = TtsOps::attention_scratch( cfg_.heads, cfg_.head_dim(), MAX_LEN );
	const size_t bytes = L * d * 4 * 2 + attn + L * d * 2 * 9 + L * 3 * d * 2 + L * 2 * F * 2 + L * F * 2 + L * V * 3 + L * 16 + ( 64 << 20 )
		+ 3 * L * ( d * 6 + cfg_.flow_width * 4 * 12 + V * 16 ) + ( 16 << 20 );  // + the flow head's (tts_flow.cpp)
	ENGINE_ASSERT( engine::core::arena_create( ws_arena_, device_, bytes ), "tts: can't allocate the workspace" );
	auto A = [&]( size_t n ) { return engine::core::arena_alloc( ws_arena_, n ); };
	ws_.x = static_cast<float*>( A( L * d * 4 ) );
	ws_.a = static_cast<float*>( A( L * d * 4 ) );
	if ( attn ) {
		ws_.attn = A( attn );
		ws_.attn_bytes = attn;
	}
	ws_.xn = static_cast<half*>( A( L * d * 2 ) );
	ws_.qkv = static_cast<half*>( A( L * 3 * d * 2 ) );
	ws_.q = static_cast<half*>( A( L * d * 2 ) );
	ws_.k = static_cast<half*>( A( L * d * 2 ) );
	ws_.v = static_cast<half*>( A( L * d * 2 ) );
	ws_.o = static_cast<half*>( A( L * d * 2 ) );
	ws_.g = static_cast<half*>( A( L * d * 2 ) );
	ws_.y = static_cast<half*>( A( L * d * 2 ) );
	ws_.up = static_cast<half*>( A( L * 2 * F * 2 ) );
	ws_.act = static_cast<half*>( A( L * F * 2 ) );
	ws_.vals = static_cast<half*>( A( L * V * 2 ) );
	ws_.digits = static_cast<uint8_t*>( A( L * V ) );
	ws_.ids = static_cast<int*>( A( L * 4 ) );
	ws_.pos = static_cast<int*>( A( L * 4 ) );
	ws_.kind = static_cast<uint8_t*>( A( L ) );
	ws_.masked = static_cast<uint8_t*>( A( L ) );
	const int rot = static_cast<int>( cfg_.head_dim() * cfg_.rot_frac ) / 2 * 2;
	ws_.rope_cs = static_cast<float*>( A( L * rot / 2 * 4 ) );
	ws_.rope_sn = static_cast<float*>( A( L * rot / 2 * 4 ) );
	if ( rot ) { TtsOps::rope_table( ws_.rope_cs, ws_.rope_sn, MAX_LEN, rot, cfg_.rope_theta, stream_ ); }
	alloc_flow_workspace();
}

void TtsModel::hidden( const TtsRow& row, float* out, std::vector<std::vector<float>>* taps ) {
	const TtsRow* r = &row;
	hidden( &r, 1, out, taps );
}

void TtsModel::hidden( const TtsRow* const* rows, int n, float* out, std::vector<std::vector<float>>* taps ) {
	using backend::Copy;
	const int d = cfg_.d_model, H = cfg_.heads, hd = cfg_.head_dim(), F = cfg_.ffn(), V = cfg_.values();
	// the rows one after another: the linears and norms see them as one block of positions, attention stays within each row
	std::vector<int> start( n + 1, 0 );
	for ( int r = 0; r < n; ++r ) {
		start[r + 1] = start[r] + rows[r]->length();
		ENGINE_ASSERT( rows[r]->mode == rows[0]->mode, "tts: rows of one pass share their mode" );
	}
	const int L = start[n];
	ENGINE_ASSERT( L <= MAX_LEN, "tts: rows longer than MAX_LEN" );

	// host-side layout of the rows
	std::vector<int> ids( L, 0 ), pos( L );
	std::vector<uint8_t> kind( L ), masked( L, 0 ), digits( static_cast<size_t>( L ) * V, 0 );
	for ( int r = 0; r < n; ++r ) {
		const TtsRow& row = *rows[r];
		const int s = start[r], Lt = static_cast<int>( row.text.size() ), Lp = static_cast<int>( row.prompt.size() / V );
		const int La = row.len_query ? 1 : static_cast<int>( row.masked.size() );
		for ( int i = 0; i < Lt + Lp + La; ++i ) { pos[s + i] = i; }
		for ( int i = 0; i < Lt; ++i ) { ids[s + i] = row.text[i]; kind[s + i] = 1; }
		for ( int i = 0; i < Lp; ++i ) { kind[s + Lt + i] = 2; }
		std::copy( row.prompt.begin(), row.prompt.end(), digits.begin() + static_cast<size_t>( s + Lt ) * V );
		if ( row.len_query ) {
			kind[s + Lt + Lp] = 5;
		} else {
			for ( int i = 0; i < La; ++i ) { kind[s + Lt + Lp + i] = 3; masked[s + Lt + Lp + i] = row.masked[i]; }
			std::copy( row.target.begin(), row.target.end(), digits.begin() + static_cast<size_t>( s + Lt + Lp ) * V );
		}
	}
	// the host vectors are copied out before each call returns
	backend::stream_t s = stream_;
	backend::memcpy_async( ws_.ids, ids.data(), L * 4, Copy::H2D, s );
	backend::memcpy_async( ws_.pos, pos.data(), L * 4, Copy::H2D, s );
	backend::memcpy_async( ws_.kind, kind.data(), L, Copy::H2D, s );
	backend::memcpy_async( ws_.masked, masked.data(), L, Copy::H2D, s );
	backend::memcpy_async( ws_.digits, digits.data(), digits.size(), Copy::H2D, s );

	// embeddings: audio_in(values) + bias everywhere (the text / mask / query positions are overwritten), then the mix
	TtsOps::digits_values( ws_.digits, ws_.vals, L * V, s );
	TtsOps::gemm( ws_.vals, w_.audio_in, ws_.a, L, d, V, true, s );
	TtsOps::add_bias( ws_.a, w_.audio_in_b, L, d, s );
	TtsOps::embed_mix( ws_.x, ws_.a, w_.text, ws_.ids, ws_.kind, ws_.masked, w_.mask_emb, w_.len_q, w_.segment, w_.mode + static_cast<size_t>( rows[0]->mode ) * d, L, d, s );

	auto tap = [&]( const float* src ) {
		if ( !taps ) { return; }
		taps->emplace_back( static_cast<size_t>( L ) * d );
		backend::memcpy_async( taps->back().data(), src, static_cast<size_t>( L ) * d * 4, Copy::D2H, s );
		backend::stream_sync( s );
	};
	const int rot = static_cast<int>( hd * cfg_.rot_frac ) / 2 * 2;
	const float scale = 1.0f / std::sqrt( static_cast<float>( hd ) );
	const bool head_major = TtsOps::head_major( hd );
	const float* branch = nullptr;   // the MLP's output, added into the residual stream by the next norm
	for ( const TtsBlock& b : w_.blocks ) {
		// attention
		TtsOps::add_rms_norm( ws_.x, branch, b.n1, L, d, EPS_FP32, ws_.xn, nullptr, s );
		tap( ws_.x );
		TtsOps::gemm( ws_.xn, b.qkv, ws_.qkv, L, 3 * d, d, false, s );
		TtsOps::qk_norm_rope( ws_.qkv, b.q_norm, b.k_norm, ws_.q, ws_.k, ws_.v, ws_.pos, L, H, hd, rot, ws_.rope_cs, ws_.rope_sn, EPS_QK, head_major, s );
		for ( int r = 0; r < n; ++r ) {
			TtsOps::attention( ws_.q, ws_.k, ws_.v, ws_.o, start[r], start[r + 1] - start[r], L, H, hd, scale, ws_.attn, ws_.attn_bytes, s );
		}
		if ( b.gate ) { TtsOps::gemm( ws_.xn, b.gate, ws_.g, L, d, d, false, s ); }
		TtsOps::heads_gate( ws_.o, b.gate ? ws_.g : nullptr, ws_.y, L, H, hd, head_major, s );
		TtsOps::gemm( ws_.y, b.out, ws_.a, L, d, d, true, s );
		// MLP (the attention's output added first)
		TtsOps::add_rms_norm( ws_.x, ws_.a, b.n2, L, d, EPS_FP32, ws_.xn, nullptr, s );
		TtsOps::gemm( ws_.xn, b.up, ws_.up, L, 2 * F, d, false, s );
		TtsOps::swiglu( ws_.up, ws_.act, L, F, s );
		TtsOps::gemm( ws_.act, b.down, ws_.a, L, d, F, true, s );
		branch = ws_.a;
	}
	TtsOps::add_rms_norm( ws_.x, branch, w_.norm, L, d, EPS_FP32, nullptr, out, s );
	backend::check_last_error();
	tap( out );
}

int TtsModel::predict_length( const std::vector<int>& text, const std::vector<uint8_t>& prompt, float* log_patches ) {
	using backend::Copy;
	ENGINE_ASSERT( w_.length_w, "tts: this export has no length head" );
	TtsRow row;
	row.text = text;
	row.prompt = prompt;
	row.mode = 2;
	row.len_query = true;
	hidden( row, fw_.h, nullptr );
	// one d-long dot product: the query's hidden state and the head's weights on the host, in double as the reference's FP32 sum
	const int d = cfg_.d_model;
	std::vector<float> h( d ), w( d );
	float bias = 0.0f;
	backend::memcpy_async( h.data(), fw_.h + static_cast<size_t>( row.length() - 1 ) * d, d * 4, Copy::D2H, stream_ );
	backend::memcpy_async( w.data(), w_.length_w, d * 4, Copy::D2H, stream_ );
	backend::memcpy_async( &bias, w_.length_b, 4, Copy::D2H, stream_ );
	backend::stream_sync( stream_ );
	double s = bias;
	for ( int i = 0; i < d; ++i ) { s += static_cast<double>( h[i] ) * w[i]; }
	if ( log_patches ) { *log_patches = static_cast<float>( s ); }
	return std::max( 1, static_cast<int>( std::nearbyint( std::exp( s ) * cfg_.patch ) ) );
}

} // namespace engine::tts
