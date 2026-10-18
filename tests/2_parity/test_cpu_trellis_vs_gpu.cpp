// The CPU trellis linear (kernels/cpu/trellis_linear) against the GPU Linear::forward on real GLM expert weights: argv[1] checkpoint dir, [2] layer, [3] expert.
#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/loader/safetensors.h"
#include "engine/modules/linear.h"
#include "engine/kernels/cpu/cpu_util.h"
#include "engine/kernels/cpu/trellis_linear.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace engine::kernels::cpu;

int main( int argc, char** argv ) {
	const std::string dir = argc > 1 ? argv[1] : "/home/mrq/Documents/models/GLM-5.3-Flash-exl3-3.05bpw";
	const int layer = argc > 2 ? std::atoi( argv[2] ) : 10, expert = argc > 3 ? std::atoi( argv[3] ) : 0;
	cudaSetDevice( 0 );
	engine::loader::SafeTensorsRepo repo;
	if ( !repo.open( dir.c_str() ) ) { std::printf( "cannot open %s\n", dir.c_str() ); return 1; }
	pod::MemoryArena arena;
	engine::core::arena_create( arena, 0, 2ULL * 1024 * 1024 * 1024 );
	std::mt19937 rng( 5 );
	std::normal_distribution<float> nd( 0.0f, 1.0f );
	int fails = 0;
	for ( const char* part : { "gate_proj", "up_proj", "down_proj" } ) {
		char name[256];
		std::snprintf( name, sizeof name, "model.language_model.layers.%d.mlp.experts.%d.%s", layer, expert, part );
		pod::LinearWeight lw = repo.load_linear( name, arena, 4 );
		if ( lw.w0.data == nullptr ) { std::printf( "%s not found\n", name ); return 1; }
		const int kt = lw.w0.shape[0], nt = lw.w0.shape[1], bits = lw.bits, k = kt * 16, n = nt * 16;
		std::vector<uint16_t> codes( static_cast<size_t>( kt ) * nt * 16 * bits ), suh( k ), svh( n );
		cudaMemcpy( codes.data(), lw.w0.data, codes.size() * 2, cudaMemcpyDeviceToHost );
		cudaMemcpy( suh.data(), lw.w1.data, suh.size() * 2, cudaMemcpyDeviceToHost );
		cudaMemcpy( svh.data(), lw.w2.data, svh.size() * 2, cudaMemcpyDeviceToHost );
		TrellisLinear cpu( codes.data(), kt, nt, bits, suh.data(), svh.data() );
		for ( int m : { 1, 4 } ) {
			std::vector<half> hx( static_cast<size_t>( m ) * k );
			for ( auto& v : hx ) v = __float2half( nd( rng ) );
			int xs[2] = { m, k }, os[2] = { m, n };
			pod::TensorView xd = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, xs, 2 );
			pod::TensorView od = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, os, 2 );
			cudaMemcpy( xd.data, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice );
			pod::TensorView x = pod::make_tensor_2d( xd.data, pod::DType::FP16, m, k ), o = pod::make_tensor_2d( od.data, pod::DType::FP16, m, n );
			engine::modules::Linear::forward( x, lw, o, nullptr );
			cudaDeviceSynchronize();
			std::vector<half> gpu( static_cast<size_t>( m ) * n );
			cudaMemcpy( gpu.data(), od.data, gpu.size() * 2, cudaMemcpyDeviceToHost );
			// CPU
			std::vector<float> xf( static_cast<size_t>( m ) * k ), xin( xf.size() ), rs( m ), raw( static_cast<size_t>( m ) * n );
			for ( size_t i = 0; i < xf.size(); ++i ) xf[i] = __half2float( hx[i] );
			for ( int r = 0; r < m; ++r ) {
				double s = 0;
				for ( int b = 0; b < k / kBlock; ++b ) { cpu.prepare_block( &xf[static_cast<size_t>( r ) * k + b * kBlock], b, &xin[static_cast<size_t>( r ) * k + b * kBlock] ); for ( int i = 0; i < kBlock; ++i ) s += xin[static_cast<size_t>( r ) * k + b * kBlock + i]; }
				rs[r] = static_cast<float>( s );
			}
			cpu.run( m, xin.data(), rs.data(), 0, cpu.units(), raw.data() );
			for ( int r = 0; r < m; ++r ) for ( int b = 0; b < n / kBlock; ++b ) { cpu.finish_block( &raw[static_cast<size_t>( r ) * n + b * kBlock], b ); round_to_half( &raw[static_cast<size_t>( r ) * n + b * kBlock], kBlock ); }
			double num = 0, den = 0;
			for ( size_t i = 0; i < raw.size(); ++i ) { const double g = __half2float( gpu[i] ); num += ( g - raw[i] ) * ( g - raw[i] ); den += g * g; }
			const double rel = std::sqrt( num / ( den + 1e-30 ) );
			std::printf( "%-9s bits %d m=%d: CPU vs GPU relative error %.4g\n", part, bits, m, rel );
			if ( !( rel < 3e-3 ) ) ++fails;
		}
	}
	std::printf( fails ? "%d FAILED\n" : "all passed\n", fails );
	return fails ? 1 : 0;
}
