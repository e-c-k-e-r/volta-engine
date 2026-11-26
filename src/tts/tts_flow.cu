// The TTS model's flow head (vall_e/model.py FlowHead + flow_sample) and the demasking loop (demask_flow): MAR-style
// generation where each step reveals the next positions of a reveal order, sampling their values with the flow head
// from the backbone's hidden states, guided (classifier-free and speaker guidance) and Euler-integrated, then snapped to
// the FSQ grid.
#include "engine/tts/tts_model.h"
#include "engine/backend/cuda/activation.cuh"
#include "engine/core/blas_check.h"
#include "engine/core/log.h"
#include "engine/kernels/dense.cuh"
#include <cublas_v2.h>
#include <cmath>

namespace engine::tts {

namespace kern {

__global__ void k_gather_half( const float* src, const int* pos, int offset, half* dst, int d ) {
	const int i = blockIdx.x;
	const size_t s = static_cast<size_t>( offset + pos[i] ) * d;
	for ( int c = threadIdx.x; c < d; c += blockDim.x ) { dst[static_cast<size_t>( i ) * d + c] = __float2half( src[s + c] ); }
}

// timestep_embedding(t, 256): [cos(t * 1000 * f_i) | sin(...)], f_i = exp(-ln(1e4) i / 128)
__global__ void k_time_embedding( float t, half* out ) {
	const int i = threadIdx.x;  // 128 threads
	const float f = expf( -logf( 10000.0f ) * i / 128.0f );
	const float a = t * 1000.0f * f;
	out[i] = __float2half( cosf( a ) );
	out[128 + i] = __float2half( sinf( a ) );
}

// y = act(x + bias) as FP16 (act: 1 SiLU, 0 none)
__global__ void k_bias_act_half( const float* x, const float* b, half* y, int rows, int cols, int act ) {
	const size_t n = static_cast<size_t>( rows ) * cols;
	for ( size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>( gridDim.x ) * blockDim.x ) {
		float v = x[i] + ( b ? b[i % cols] : 0.0f );
		if ( act ) { v = engine::math::silu( v ); }
		y[i] = __float2half( v );
	}
}

// c = cond + temb (broadcast over rows); sc = silu(c) as FP16
__global__ void k_cond_silu( const float* cond, const float* temb, half* sc, int rows, int W ) {
	const size_t n = static_cast<size_t>( rows ) * W;
	for ( size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>( gridDim.x ) * blockDim.x ) {
		const float v = cond[i] + temb[i % W];
		sc[i] = __float2half( engine::math::silu( v ) );
	}
}

// out = LayerNorm(x) (no affine, eps 1e-6) * (1 + mod[scale]) + mod[shift], FP16; mod rows of mstride with shift at 0,
// scale at W. One block of 256 threads per row, W <= 2048
__global__ void k_ln_mod( const float* x, const float* mod, int mstride, half* out, int W ) {
	const int r = blockIdx.x;
	const float* xr = x + static_cast<size_t>( r ) * W;
	float s = 0.0f;
	for ( int c = threadIdx.x; c < W; c += blockDim.x ) { s += xr[c]; }
	const float mean = engine::math::block_sum<32>( s ) / W;
	float q = 0.0f;
	for ( int c = threadIdx.x; c < W; c += blockDim.x ) { const float dlt = xr[c] - mean; q += dlt * dlt; }
	__syncthreads();
	const float inv = rsqrtf( engine::math::block_sum<32>( q ) / W + 1e-6f );
	const float* m = mod + static_cast<size_t>( r ) * mstride;
	for ( int c = threadIdx.x; c < W; c += blockDim.x ) {
		out[static_cast<size_t>( r ) * W + c] = __float2half( ( xr[c] - mean ) * inv * ( 1.0f + m[W + c] ) + m[c] );
	}
}

// x += gate * (branch + bias), gate = mod[2W + c]
__global__ void k_gate_add( float* x, const float* branch, const float* bias, const float* mod, int mstride, int rows, int W ) {
	const size_t n = static_cast<size_t>( rows ) * W;
	for ( size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>( gridDim.x ) * blockDim.x ) {
		const size_t r = i / W, c = i % W;
		x[i] += mod[r * mstride + 2 * W + c] * ( branch[i] + bias[c] );
	}
}

// the sample x [k, V] replicated for R rows as FP16
__global__ void k_replicate_half( const float* x, half* y, int n, int R ) {
	for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n * R; i += gridDim.x * blockDim.x ) { y[i] = __float2half( x[i % n] ); }
}

// v [R, k, V] (+ bias) -> the guided velocity; x += v * dt; vt (optional) gets the guided velocity
__global__ void k_guide_step( const float* v, const float* bias, float* x, float* vt, int n, int V, int R, float cfg, float cfg_prompt, float dt ) {
	for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x ) {
		const float b = bias[i % V];
		const float vc = v[i] + b;
		float g = vc;
		if ( R >= 2 ) {
			const float vn = v[n + i] + b;
			g = vn + ( vc - vn ) * cfg;
			if ( R == 3 ) { g += ( vc - ( v[2 * n + i] + b ) ) * cfg_prompt; }
		}
		if ( vt ) { vt[i] = g; }
		x[i] += g * dt;
	}
}

// values -> FSQ level indices (round half to even, as torch.round), levels [8, 5, 5, 5] per group of 4 scalars
__global__ void k_snap( const float* x, uint8_t* out, int n ) {
	for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x ) {
		const int L = ( i % 4 == 0 ) ? 8 : 5;
		const float h = static_cast<float>( L / 2 );
		const float r = fmaxf( rintf( x[i] * h + h ), 0.0f );
		out[i] = static_cast<uint8_t>( fminf( r, static_cast<float>( L - 1 ) ) );
	}
}

} // namespace kern

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
	const int d = cfg_.d_model, W = cfg_.flow_width, V = cfg_.values(), k = static_cast<int>( reveal.size() );
	if ( k == 0 ) { return; }
	const bool guided = s.cfg > 1.0f;
	const int R = guided ? ( s.cfg_prompt > 0.0f && !prompt.empty() ? 3 : 2 ) : 1;
	// the backbone over the guidance rows (as many in one pass as MAX_LEN holds), their hidden states at the revealed positions gathered
	// into fw_.hh [R * k, d]
	int* pos = fw_.pos;
	CUDA_CHECK( cudaMemcpy( pos, reveal.data(), k * sizeof( int ), cudaMemcpyHostToDevice ) );
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
			kern::k_gather_half<<<k, 256>>>( fw_.h, pos, offset, fw_.hh + static_cast<size_t>( r ) * k * d, d );
		}
		r0 = r1;
	}
	const int N = R * k;
	// cond(h): the same for every flow step
	gemm( fw_.hh, w_.f_cond, fw_.cond, N, W, d, true );
	engine::dense::k_add_bias<<<256, 256>>>( fw_.cond, w_.f_cond_b, N, W );
	CUDA_CHECK( cudaMemcpy( fw_.x, noise, static_cast<size_t>( k ) * V * 4, cudaMemcpyHostToDevice ) );
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
		kern::k_time_embedding<<<1, 128>>>( t, fw_.tvec );
		gemm( fw_.tvec, w_.f_time0, fw_.t0, 1, W, 256, true );
		kern::k_bias_act_half<<<4, 256>>>( fw_.t0, w_.f_time0_b, fw_.t0h, 1, W, 1 );
		gemm( fw_.t0h, w_.f_time2, fw_.temb, 1, W, W, true );
		engine::dense::k_add_bias<<<4, 256>>>( fw_.temb, w_.f_time2_b, 1, W );
		kern::k_cond_silu<<<256, 256>>>( fw_.cond, fw_.temb, fw_.sc, N, W );
		// x -> inp(x) for every row
		kern::k_replicate_half<<<256, 256>>>( fw_.x, fw_.xin, k * V, R );
		gemm( fw_.xin, w_.f_inp, fw_.xf, N, W, V, true );
		engine::dense::k_add_bias<<<256, 256>>>( fw_.xf, w_.f_inp_b, N, W );
		for ( const TtsFlowBlock& b : w_.flow_blocks ) {
			gemm( fw_.sc, b.mod, fw_.mod, N, 3 * W, W, true );
			engine::dense::k_add_bias<<<256, 256>>>( fw_.mod, b.mod_b, N, 3 * W );
			kern::k_ln_mod<<<N, 256>>>( fw_.xf, fw_.mod, 3 * W, fw_.ln, W );
			gemm( fw_.ln, b.mlp0, fw_.m1, N, 2 * W, W, true );
			kern::k_bias_act_half<<<256, 256>>>( fw_.m1, b.mlp0_b, fw_.m1h, N, 2 * W, 1 );
			gemm( fw_.m1h, b.mlp2, fw_.m2, N, W, 2 * W, true );
			kern::k_gate_add<<<256, 256>>>( fw_.xf, fw_.m2, b.mlp2_b, fw_.mod, 3 * W, N, W );
		}
		gemm( fw_.sc, w_.f_final_mod, fw_.mod, N, 2 * W, W, true );
		engine::dense::k_add_bias<<<256, 256>>>( fw_.mod, w_.f_final_mod_b, N, 2 * W );
		kern::k_ln_mod<<<N, 256>>>( fw_.xf, fw_.mod, 2 * W, fw_.ln, W );
		gemm( fw_.ln, w_.f_out, fw_.v, N, V, W, true );
		kern::k_guide_step<<<64, 256>>>( fw_.v, w_.f_out_b, fw_.x, vt_dev, k * V, V, R, s.cfg, s.cfg_prompt, static_cast<float>( grid[j + 1] - grid[j] ) );
		if ( vtaps ) {
			CUDA_CHECK( cudaMemcpy( vt.data(), vt_dev, vt.size() * 4, cudaMemcpyDeviceToHost ) );
			vtaps->push_back( vt );
		}
	}
	kern::k_snap<<<64, 256>>>( fw_.x, fw_.snapped, k * V );
	std::vector<uint8_t> snapped( static_cast<size_t>( k ) * V );
	CUDA_CHECK( cudaMemcpy( snapped.data(), fw_.snapped, snapped.size(), cudaMemcpyDeviceToHost ) );
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
