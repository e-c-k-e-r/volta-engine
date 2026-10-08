// Bridge to the engine's trellis Linear / reconstruct for reference comparison (no Slang prelude in this translation unit)
#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/modules/trellis.h"
#include "reconstruct.cuh"
#include "ref_cuda.h"

void ref_engine_trellis_linear( void* x, void* codes, void* suh, void* svh, void* out, int m, int k, int n, int bits, int cb, int recon, cudaStream_t s ) {
	auto xt = pod::make_tensor_2d( x, pod::DType::FP16, m, k );
	auto ct = pod::make_tensor_3d( codes, pod::DType::UINT16, k / 16, n / 16, 16 * bits );
	auto st = pod::make_tensor_1d( suh, pod::DType::FP16, k );
	auto vt = pod::make_tensor_1d( svh, pod::DType::FP16, n );
	auto ot = pod::make_tensor_2d( out, pod::DType::FP16, m, n );
	if ( recon ) {
		engine::modules::LinearTrellis::forward_recon( xt, ct, st, vt, ot, bits, cb == 1, cb == 2, s );
	}
	else {
		engine::modules::LinearTrellis::forward( xt, ct, st, vt, ot, bits, cb == 1, cb == 2, s );
	}
}

void ref_engine_reconstruct( void* codes, void* w, int k, int n, int bits, int cb, cudaStream_t s ) {
	reconstruct_slice( (half*) w, (const uint16_t*) codes, bits, cb == 1, cb == 2, k / 16, n, 0, s );
}
