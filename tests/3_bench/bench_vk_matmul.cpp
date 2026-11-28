// FP16 matmul of the Vulkan backend (out [m, n] = x [m, k] W^T, W [n, k]): time per call and TFLOP/s on the device's matmul path (cooperative matrices where the device
// has them). bench_vk_matmul M N K
#include "engine/backend/runtime.h"
#include "../../src/backend/vulkan/mod_common.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
int main( int argc, char** argv ) {
	if ( argc < 4 ) { std::printf( "usage: bench_vk_matmul M N K\n" ); return 2; }
	const int m = std::atoi( argv[1] ), n = std::atoi( argv[2] ), k = std::atoi( argv[3] );
	void* w = engine::backend::device_malloc( size_t( n ) * k * 2 );
	void* x = engine::backend::device_malloc( size_t( m ) * k * 2 );
	void* o = engine::backend::device_malloc( size_t( m ) * n * 2 );
	engine::backend::memset_sync( w, 0, size_t( n ) * k * 2 );
	engine::backend::memset_sync( x, 0, size_t( m ) * k * 2 );
	for ( int i = 0; i < 5; ++i ) engine::backend::impl::ggml_f16_matmul( w, x, o, m, n, k, false, nullptr );
	engine::backend::device_sync();
	const int reps = 40;
	const auto t0 = std::chrono::steady_clock::now();
	for ( int i = 0; i < reps; ++i ) engine::backend::impl::ggml_f16_matmul( w, x, o, m, n, k, false, nullptr );
	engine::backend::device_sync();
	const double ms = std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t0 ).count() / reps;
	std::printf( "%.4f ms %.1f TFLOP/s\n", ms, 2.0 * m * n * k / ms / 1e9 );
	return 0;
}
