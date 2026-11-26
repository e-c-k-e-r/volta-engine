// The TTS model's flow head (vall_e/model.py FlowHead + flow_sample) and the demasking loop (demask_flow): MAR-style
// generation where each step reveals the next positions of a reveal order, sampling their values with the flow head
// from the backbone's hidden states, guided (classifier-free and speaker guidance) and Euler-integrated, then snapped to
// the FSQ grid.
#include "engine/tts/tts_model.h"
#include "engine/backend/runtime.h"
#include "engine/core/log.h"
#include "engine/modules/tts.h"
#include <cmath>

namespace engine::tts {

using engine::modules::TtsOps;

void TtsModel::alloc_flow_workspace() {
	const size_t N = 3 * static_cast<size_t>( MAX_LEN ), d = cfg_.d_model, W = cfg_.flow_width, V = cfg_.values();
	auto A = [&]( size_t n ) { return engine::core::arena_alloc( ws_arena_, n ); };
	fw_.h = static_cast<float*>( A( N * d * 4 ) );
	fw_.hh = static_cast<half*>( A( N * d * 2 ) );
	fw_.cond = static_cast<float*>( A( N * W * 4 ) );
	fw_.c = static_cast<float*>( A( N * W * 4 ) );
	fw_.sc = static_cast<half*>( A( N * W * 2 ) );
	fw_.mod = static_cast<float*>( A( N * 3 * W * 4 ) );
	fw_.xf = static_cast<float*>( A( N * W * 4 ) );
	fw_.ln = static_cast<half*>( A( N * W * 2 ) );
	fw_.m1 = static_cast<float*>( A( N * 2 * W * 4 ) );
	fw_.m1h = static_cast<half*>( A( N * 2 * W * 2 ) );
	fw_.m2 = static_cast<float*>( A( N * W * 4 ) );
	fw_.v = static_cast<float*>( A( 2 * N * V * 4 ) );  // the per-row velocities, then the guided one (vtaps)
	fw_.x = static_cast<float*>( A( N * V * 4 ) );
	fw_.xin = static_cast<half*>( A( N * V * 2 ) );
	fw_.tvec = static_cast<half*>( A( 256 * 2 ) );
	fw_.t0 = static_cast<float*>( A( W * 4 ) );
	fw_.t0h = static_cast<half*>( A( W * 2 ) );
	fw_.temb = static_cast<float*>( A( W * 4 ) );
	fw_.snapped = static_cast<uint8_t*>( A( N * V ) );
	fw_.pos = static_cast<int*>( A( MAX_LEN * sizeof( int ) ) );
}

int TtsModel::revealed_after( int s, int steps, int La ) {
	return La - static_cast<int>( std::floor( std::cos( ( s + 1 ) / static_cast<double>( steps ) * M_PI / 2 ) * La ) );
}

void TtsModel::demask_step( const std::vector<int>& text, const std::vector<uint8_t>& prompt, std::vector<uint8_t>& digits, std::vector<uint8_t>& masked,
	const std::vector<int>& reveal, const float* noise, const Sampling& s, std::vector<std::vector<float>>* vtaps ) {
	using backend::Copy;
	const int d = cfg_.d_model, W = cfg_.flow_width, V = cfg_.values(), k = static_cast<int>( reveal.size() );
	if ( k == 0 ) { return; }
	const bool guided = s.cfg > 1.0f;
	const int R = guided ? ( s.cfg_prompt > 0.0f && !prompt.empty() ? 3 : 2 ) : 1;
	// the backbone over the guidance rows (as many in one pass as MAX_LEN holds), their hidden states at the revealed positions gathered
	// into fw_.hh [R * k, d]
	int* pos = fw_.pos;
	backend::memcpy_sync( pos, reveal.data(), k * sizeof( int ), Copy::H2D );
	TtsRow rows[3];
	for ( int r = 0; r < R; ++r ) {
		if ( r != 1 ) { rows[r].text = text; }
		if ( r == 0 ) { rows[r].prompt = prompt; }
		rows[r].target = digits;
		rows[r].masked = masked;
	}
	for ( int r0 = 0; r0 < R; ) {
		int r1 = r0 + 1, len = rows[r0].length();
		while ( r1 < R && len + rows[r1].length() <= MAX_LEN ) { len += rows[r1++].length(); }
		const TtsRow* pass[3];
		for ( int r = r0; r < r1; ++r ) { pass[r - r0] = &rows[r]; }
		hidden( pass, r1 - r0, fw_.h, nullptr );
		for ( int r = r0, at = 0; r < r1; at += rows[r++].length() ) {
			const int offset = at + static_cast<int>( rows[r].text.size() + rows[r].prompt.size() / V );
			TtsOps::gather_rows( fw_.h, pos, offset, fw_.hh + static_cast<size_t>( r ) * k * d, k, d );
		}
		r0 = r1;
	}
	const int N = R * k;
	// cond(h): the same for every flow step
	TtsOps::gemm( fw_.hh, w_.f_cond, fw_.cond, N, W, d, true );
	TtsOps::add_bias( fw_.cond, w_.f_cond_b, N, W );
	backend::memcpy_sync( fw_.x, noise, static_cast<size_t>( k ) * V * 4, Copy::H2D );
	// the sway grid of flow_sample
	std::vector<double> grid( s.flow_steps + 1 );
	for ( int i = 0; i <= s.flow_steps; ++i ) {
		const double u = static_cast<float>( i ) / s.flow_steps;  // torch.linspace in float32
		grid[i] = u + s.sway * ( std::cos( M_PI / 2 * u ) - 1 + u );
	}
	std::vector<float> vt( static_cast<size_t>( k ) * V );
	float* vt_dev = vtaps ? fw_.v + static_cast<size_t>( N ) * V : nullptr;  // the guided velocity, after the per-row ones
	for ( int j = 0; j < s.flow_steps; ++j ) {
		const float t = static_cast<float>( grid[j] );
		// time MLP -> temb [W]; c = cond + temb; sc = silu(c)
		TtsOps::time_embedding( t, fw_.tvec );
		TtsOps::gemm( fw_.tvec, w_.f_time0, fw_.t0, 1, W, 256, true );
		TtsOps::bias_act( fw_.t0, w_.f_time0_b, fw_.t0h, 1, W, true );
		TtsOps::gemm( fw_.t0h, w_.f_time2, fw_.temb, 1, W, W, true );
		TtsOps::add_bias( fw_.temb, w_.f_time2_b, 1, W );
		TtsOps::cond_silu( fw_.cond, fw_.temb, fw_.sc, N, W );
		// x -> inp(x) for every row
		TtsOps::replicate( fw_.x, fw_.xin, k * V, R );
		TtsOps::gemm( fw_.xin, w_.f_inp, fw_.xf, N, W, V, true );
		TtsOps::add_bias( fw_.xf, w_.f_inp_b, N, W );
		for ( const TtsFlowBlock& b : w_.flow_blocks ) {
			TtsOps::gemm( fw_.sc, b.mod, fw_.mod, N, 3 * W, W, true );
			TtsOps::add_bias( fw_.mod, b.mod_b, N, 3 * W );
			TtsOps::ln_mod( fw_.xf, fw_.mod, 3 * W, fw_.ln, N, W );
			TtsOps::gemm( fw_.ln, b.mlp0, fw_.m1, N, 2 * W, W, true );
			TtsOps::bias_act( fw_.m1, b.mlp0_b, fw_.m1h, N, 2 * W, true );
			TtsOps::gemm( fw_.m1h, b.mlp2, fw_.m2, N, W, 2 * W, true );
			TtsOps::gate_add( fw_.xf, fw_.m2, b.mlp2_b, fw_.mod, 3 * W, N, W );
		}
		TtsOps::gemm( fw_.sc, w_.f_final_mod, fw_.mod, N, 2 * W, W, true );
		TtsOps::add_bias( fw_.mod, w_.f_final_mod_b, N, 2 * W );
		TtsOps::ln_mod( fw_.xf, fw_.mod, 2 * W, fw_.ln, N, W );
		TtsOps::gemm( fw_.ln, w_.f_out, fw_.v, N, V, W, true );
		TtsOps::guide_step( fw_.v, w_.f_out_b, fw_.x, vt_dev, k * V, V, R, s.cfg, s.cfg_prompt, static_cast<float>( grid[j + 1] - grid[j] ) );
		if ( vtaps ) {
			backend::memcpy_sync( vt.data(), vt_dev, vt.size() * 4, Copy::D2H );
			vtaps->push_back( vt );
		}
	}
	TtsOps::snap( fw_.x, fw_.snapped, k * V );
	std::vector<uint8_t> snapped( static_cast<size_t>( k ) * V );
	backend::memcpy_sync( snapped.data(), fw_.snapped, snapped.size(), Copy::D2H );
	for ( int i = 0; i < k; ++i ) {
		std::copy( snapped.begin() + static_cast<size_t>( i ) * V, snapped.begin() + static_cast<size_t>( i + 1 ) * V, digits.begin() + static_cast<size_t>( reveal[i] ) * V );
		masked[reveal[i]] = 0;
	}
}

std::vector<uint8_t> TtsModel::generate( const std::vector<int>& text, const std::vector<uint8_t>& prompt, int n_frames, const std::vector<int>& order,
	const std::vector<std::vector<float>>& noise, const Sampling& s ) {
	const int V = cfg_.values(), La = ( n_frames + cfg_.patch - 1 ) / cfg_.patch;
	std::vector<uint8_t> digits( static_cast<size_t>( La ) * V, 0 ), masked( La, 1 );
	int done = 0;
	for ( int step = 0; step < s.steps; ++step ) {
		const int target = revealed_after( step, s.steps, La );
		if ( target == done ) { continue; }
		std::vector<int> reveal( order.begin() + done, order.begin() + target );
		demask_step( text, prompt, digits, masked, reveal, noise[step].data(), s );
		done = target;
	}
	digits.resize( static_cast<size_t>( n_frames ) * 32 );  // [La, patch * 32] -> frames
	return digits;
}

} // namespace engine::tts
