#include "exl3_encode_kernel.cuh"
#include "exl3_encode_inst.cuh"

exl3e_kernel_fn exl3e_kernel_hi( int K, int cb ) {
	#define EXL3E_CASE( k ) \
		case k: { return cb == 0 ? exl3_encode_tiles_kernel<k, 0> : cb == 1 ? exl3_encode_tiles_kernel<k, 1> : exl3_encode_tiles_kernel<k, 2>; }
	switch ( K ) {
		EXL3E_CASE( 5 )
		EXL3E_CASE( 6 )
		EXL3E_CASE( 7 )
		EXL3E_CASE( 8 )
		default: break;
	}
	#undef EXL3E_CASE
	return nullptr;
}
