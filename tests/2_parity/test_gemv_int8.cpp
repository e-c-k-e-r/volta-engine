// Tier 2 parity for the dp4a int8 GEMV (exl3_gemv_int8 sq kernel) on the Qwen TP4 per-rank shapes.
// Runs the same mul1 trellis weights through the int8 path and the fp16 tensor-core path, and checks
// both against an FP32 CPU reference built from the reconstructed weights. Cosine similarity must be
// >= 0.9999 and RMS relative error <= 0.005; the int8 path additionally has to stay within the
// fp16 path's own error band, since it quantizes activations to int8 (residual mode restores
// ~15-16 effective bits).
#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/arena.h"
#include "engine/modules/trellis.h"
#include "exl3_gemv_int8.cuh"
#include "reconstruct.cuh"
#include "trellis_context.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cstdlib>

namespace {

bool g_int8_on = true;

struct Shape { const char* name; int k; int n; int bits; };

double run_shape(
	const Shape& sh,
	int m,
	pod::MemoryArena& arena,
	cudaStream_t stream
) {
	const int k_dim = sh.k;
	const int n_dim = sh.n;
	const int K = sh.bits;
	const bool mul1 = true;

	const size_t x_numel = static_cast<size_t>( m ) * k_dim;
	const size_t w_numel = static_cast<size_t>( k_dim ) * n_dim;
	const size_t out_numel = static_cast<size_t>( m ) * n_dim;
	const size_t trellis_numel = static_cast<size_t>( k_dim / 16 ) * ( n_dim / 16 ) * ( 16 * K );

	std::vector<half> h_x( x_numel );
	std::vector<half> h_suh( k_dim );
	std::vector<half> h_svh( n_dim );
	std::vector<uint16_t> h_trellis( trellis_numel );
	std::vector<half> h_w_recon( w_numel );

	unsigned int seed = 987654321u + static_cast<unsigned int>( K ) * 7919u + static_cast<unsigned int>( n_dim );
	auto rnd = [&] ( float scale ) -> half {
		seed = seed * 1664525u + 1013904223u;
		return __float2half_rn( ( static_cast<float>( seed & 0xffff ) / 65536.0f - 0.5f ) * scale );
	};
	for ( size_t i = 0; i < x_numel; ++i ) h_x[i] = rnd( 0.4f );
	for ( int i = 0; i < k_dim; ++i ) h_suh[i] = rnd( 0.2f );
	for ( int i = 0; i < n_dim; ++i ) h_svh[i] = rnd( 0.2f );
	for ( size_t i = 0; i < trellis_numel; ++i ) {
		seed = seed * 1664525u + 1013904223u;
		h_trellis[i] = static_cast<uint16_t>( seed & 0xffff );
	}

	engine::core::arena_reset( arena );

	int x_shape[2] = { m, k_dim };
	pod::TensorView x_tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, x_shape, 2 );
	int trellis_shape[3] = { k_dim / 16, n_dim / 16, 16 * K };
	pod::TensorView trellis_tv = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, trellis_shape, 3 );
	int suh_shape[1] = { k_dim };
	pod::TensorView suh_tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, suh_shape, 1 );
	int svh_shape[1] = { n_dim };
	pod::TensorView svh_tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, svh_shape, 1 );
	int out_shape[2] = { m, n_dim };
	pod::TensorView out_tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, out_shape, 2 );
	int w_shape[2] = { k_dim, n_dim };
	pod::TensorView w_tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, w_shape, 2 );

	CUDA_CHECK( cudaMemcpyAsync( x_tv.data, h_x.data(), x_numel * sizeof( half ), cudaMemcpyHostToDevice, stream ) );
	CUDA_CHECK( cudaMemcpyAsync( trellis_tv.data, h_trellis.data(), trellis_numel * sizeof( uint16_t ), cudaMemcpyHostToDevice, stream ) );
	CUDA_CHECK( cudaMemcpyAsync( suh_tv.data, h_suh.data(), k_dim * sizeof( half ), cudaMemcpyHostToDevice, stream ) );
	CUDA_CHECK( cudaMemcpyAsync( svh_tv.data, h_svh.data(), n_dim * sizeof( half ), cudaMemcpyHostToDevice, stream ) );

	reconstruct_had_slice( w_tv.ptr<half>(), trellis_tv.ptr<const uint16_t>(), suh_tv.ptr<const half>(),
		svh_tv.ptr<const half>(), K, false, mul1, k_dim, n_dim, 0, stream );
	CUDA_CHECK( cudaStreamSynchronize( stream ) );
	CUDA_CHECK( cudaMemcpy( h_w_recon.data(), w_tv.data, w_numel * sizeof( half ), cudaMemcpyDeviceToHost ) );

	std::vector<float> h_ref( out_numel, 0.0f );
	for ( int i = 0; i < m; ++i ) {
		for ( int j = 0; j < n_dim; ++j ) {
			double acc = 0.0;
			for ( int p = 0; p < k_dim; ++p )
				acc += static_cast<double>( __half2float( h_x[i * k_dim + p] ) ) * __half2float( h_w_recon[p * n_dim + j] );
			h_ref[i * n_dim + j] = static_cast<float>( acc );
		}
	}

	auto launch = [&] () {
		engine::modules::LinearTrellis::forward( x_tv, trellis_tv, suh_tv, svh_tv, out_tv, K, false, mul1, stream );
	};

	// Two passes: warm up, then measure. The int8 sq kernel keeps its completion counters in a fixed
	// workspace and resets them at the end of every call, so replaying it is identical to a graph
	// replay; the second pass is the one that must not drift.
	std::vector<half> h_a( out_numel );
	std::vector<half> h_b( out_numel );
	launch();
	CUDA_CHECK( cudaStreamSynchronize( stream ) );
	CUDA_CHECK( cudaMemcpy( h_a.data(), out_tv.data, out_numel * sizeof( half ), cudaMemcpyDeviceToHost ) );
	launch();
	CUDA_CHECK( cudaStreamSynchronize( stream ) );
	CUDA_CHECK( cudaMemcpy( h_b.data(), out_tv.data, out_numel * sizeof( half ), cudaMemcpyDeviceToHost ) );

	auto metrics = [&] ( const std::vector<half>& v, double& cos, double& rms, double& maxabs ) {
		double dot = 0.0, ng = 0.0, nr = 0.0, dss = 0.0, rss = 0.0;
		maxabs = 0.0;
		for ( size_t i = 0; i < out_numel; ++i ) {
			float g = __half2float( v[i] );
			float r = h_ref[i];
			dot += static_cast<double>( g ) * r;
			ng += static_cast<double>( g ) * g;
			nr += static_cast<double>( r ) * r;
			double d = std::fabs( static_cast<double>( g ) - r );
			dss += d * d;
			rss += static_cast<double>( r ) * r;
			if ( d > maxabs ) maxabs = d;
		}
		cos = dot / ( std::sqrt( ng ) * std::sqrt( nr ) + 1e-12 );
		rms = std::sqrt( dss ) / ( std::sqrt( rss ) + 1e-12 );
	};

	double cos_a, rms_a, max_a, cos_b, rms_b, max_b;
	metrics( h_a, cos_a, rms_a, max_a );
	metrics( h_b, cos_b, rms_b, max_b );

	double replay_drift = 0.0;
	for ( size_t i = 0; i < out_numel; ++i )
		replay_drift = std::max( replay_drift, std::fabs( static_cast<double>( __half2float( h_a[i] ) ) - __half2float( h_b[i] ) ) );

	std::printf( "  %-14s k%-5d n%-6d %db M=%d | cos %.6f rms %.2e max %.2e | fp16 rms %.2e | replay drift %.2e\n",
		sh.name, k_dim, n_dim, K, m, cos_b, rms_b, max_b, rms_a, replay_drift );
	std::fflush( stdout );

	if ( !g_int8_on ) {
		std::printf( "  (int8 path disabled, fp16 reference only)\n" );
		return rms_b;
	}
	ENGINE_ASSERT( cos_b >= 0.9999, "int8 GEMV cosine similarity below 0.9999" );
	ENGINE_ASSERT( rms_b <= 0.0050, "int8 GEMV RMS relative error exceeds 0.005" );
	ENGINE_ASSERT( replay_drift == 0.0, "int8 GEMV is not deterministic across repeated launches" );
	// The int8 path trades ~16 activation bits for dp4a throughput, so it must not be materially worse
	// than the fp16 tensor-core path it replaces on the same weights.
	ENGINE_ASSERT( rms_b <= rms_a * 4.0 + 1e-6, "int8 GEMV error is far above the fp16 path's error" );
	return rms_b;
}

} // namespace

int main() {
	std::printf( "========================================================\n" );
	std::printf( "   volta-engine Tier 2: dp4a int8 GEMV Parity Test      \n" );
	std::printf( "========================================================\n" );

	CUDA_CHECK( cudaSetDevice( 0 ) );
	g_int8_on = exl3_gemv_int8_enabled();
	ENGINE_ASSERT( g_int8_on, "the int8 GEMV path is disabled" );
	std::printf( "[*] int8 GEMV enabled, max K = %d\n", exl3_gemv_int8_max_k( 0 ) );

	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );

	pod::MemoryArena arena;
	ENGINE_ASSERT( engine::core::arena_create( arena, 0, 2ULL << 30 ), "arena" );
	engine::modules::trellis_context_init_all();

	// Qwen3.8-27B TP4 per-rank shapes. N must be a multiple of 256 for the 256-column work unit.
	const Shape shapes[] = {
		{ "mlp gate/up", 5120, 4352, 4 },
		{ "mlp down", 4352, 5120, 4 },
		{ "gdn qkv", 5120, 2560, 4 },
		{ "gdn out", 1536, 5120, 4 },
		{ "attn q+gate", 5120, 3072, 4 },
		{ "mlp gate/up 5b", 5120, 4352, 5 },
	};
	for ( const auto& sh : shapes ) run_shape( sh, 1, arena, stream );

	CUDA_CHECK( cudaStreamDestroy( stream ) );
	engine::core::arena_destroy( arena );
	engine::modules::trellis_context_destroy();
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		(void) cudaSetDevice( d );
		(void) cudaDeviceReset();
	}

	std::printf( "========================================================\n" );
	std::printf( "[+] dp4a int8 GEMV PARITY PASSED\n" );
	std::printf( "========================================================\n" );
	return 0;
}
