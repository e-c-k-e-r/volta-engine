#include "engine/modules/trellis.h"
#include "exl3_gemm.cuh"
#include "exl3_gemv.cuh"
#include "trellis_context.h"

namespace engine::modules {

void LinearTrellis::forward(
	const pod::TensorView& x,
	const pod::TensorView& trellis,
	const pod::TensorView& suh,
	const pod::TensorView& svh,
	pod::TensorView& out,
	int bits,
	bool mcg,
	bool mul1,
	cudaStream_t stream
) {
	const int m = x.shape[0];
	const int k = trellis.shape[0] * 16;
	const int n = trellis.shape[1] * 16;

	int device = 0;
	cudaGetDevice( &device );
	half* A_had = trellis_get_workspace( device, static_cast<size_t>( m ) * k * sizeof( half ) );

	bool c_fp32 = ( out.dtype == pod::DType::FP32 );
	const half* x_ptr = x.ptr<const half>();
	const uint16_t* trellis_ptr = trellis.ptr<const uint16_t>();
	void* out_ptr = out.data;
	const half* suh_ptr = suh.ptr<const half>();
	const half* svh_ptr = svh.ptr<const half>();

	if ( m <= 8 && ( m == 1 || m == 2 || m == 4 || m == 8 ) && ( k % 128 == 0 ) && ( n % 128 == 0 ) ) {
		exl3_gemv(
			x_ptr,
			trellis_ptr,
			out_ptr,
			suh_ptr,
			A_had,
			svh_ptr,
			m,
			k,
			n,
			bits,
			mcg,
			mul1,
			c_fp32,
			stream
		);
	}
	else {
		exl3_gemm(
			x_ptr,
			trellis_ptr,
			out_ptr,
			suh_ptr,
			A_had,
			svh_ptr,
			m,
			k,
			n,
			bits,
			mcg,
			mul1,
			c_fp32,
			stream
		);
	}
}

} // namespace engine::modules
