// Issue cost of the integer operations in the EXL3 codebook decode, in add-equivalents (standalone; no engine headers):
//   nvcc -O3 -arch=sm_70 tests/tools/int_rates.cu -o int_rates && ./int_rates        (CUDA only: the SAD and multiply variants are PTX)
// Each kernel runs 8 independent chains of `op, then an xor that feeds the next iteration`, so the loop is bound by the issue rate of the
// operation and not by its latency. units = (add rate / op rate) * 2 - 1: an add-and-xor iteration is 2 units, so an operation that costs
// k adds reads k. On a GPU where the byte sum (dp4a) costs the same as an add, replacing it gains nothing; on RDNA3 v_dot4_u32_u8 reads
// about 3.8 and v_sad_u8 about 1.6, which is what the ROCm build's __dp4a (src/backend/rocm/compat.h) exploits.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

enum Op { ADD, MUL32, MUL24X2, DP4A, SAD4, SWAR, NOPS };
static const char* kNames[NOPS] = {
	"add (reference)", "32-bit multiply (imad.lo)", "mul24.lo x2 + shl-add (the multiply as two 24-bit)",
	"__dp4a byte sum", "vabsdiff4 .add byte sum (SIMD video)", "byte sum by SWAR (and, shift, add)",
};

template <int OP>
__global__ void chains( uint32_t* out, int iters, uint32_t c ) {
	uint32_t x0 = threadIdx.x, x1 = x0 + 1, x2 = x0 + 2, x3 = x0 + 3, x4 = x0 + 4, x5 = x0 + 5, x6 = x0 + 6, x7 = x0 + 7;
	uint32_t a0 = 1, a1 = 2, a2 = 3, a3 = 4, a4 = 5, a5 = 6, a6 = 7, a7 = 8;
	const uint32_t cl = c & 0xffffu, ch = c >> 16;
#define STEP( x, a ) { \
		if constexpr ( OP == ADD ) a = a + x; \
		else if constexpr ( OP == MUL32 ) a = x * c; \
		else if constexpr ( OP == MUL24X2 ) { uint32_t lo, hi; \
			asm volatile( "mul24.lo.u32 %0, %1, %2;" : "=r"( lo ) : "r"( x ), "r"( cl ) ); \
			asm volatile( "mul24.lo.u32 %0, %1, %2;" : "=r"( hi ) : "r"( x ), "r"( ch ) ); \
			a = lo + ( hi << 16 ); } \
		else if constexpr ( OP == DP4A ) a = __dp4a( x, 0x01010101u, a ); \
		else if constexpr ( OP == SAD4 ) asm volatile( "vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;" : "=r"( a ) : "r"( x ), "r"( 0u ), "r"( a ) ); \
		else { const uint32_t t = ( x & 0x00ff00ffu ) + ( ( x >> 8 ) & 0x00ff00ffu ); a = a + ( ( t + ( t >> 16 ) ) & 0xffffu ); } \
	}
	for ( int it = 0; it < iters; ++it ) {
		STEP( x0, a0 ) STEP( x1, a1 ) STEP( x2, a2 ) STEP( x3, a3 ) STEP( x4, a4 ) STEP( x5, a5 ) STEP( x6, a6 ) STEP( x7, a7 )
		x0 ^= a0; x1 ^= a1; x2 ^= a2; x3 ^= a3; x4 ^= a4; x5 ^= a5; x6 ^= a6; x7 ^= a7;
	}
	out[blockIdx.x * blockDim.x + threadIdx.x] = x0 ^ x1 ^ x2 ^ x3 ^ x4 ^ x5 ^ x6 ^ x7;
}

template <int OP>
static double rate( uint32_t* out, int blocks, int iters ) {
	cudaEvent_t a, b;
	cudaEventCreate( &a );
	cudaEventCreate( &b );
	chains<OP><<<blocks, 256>>>( out, 100, 0x83DCD12Du );
	cudaDeviceSynchronize();
	double best = 0;
	for ( int rep = 0; rep < 3; ++rep ) {
		cudaEventRecord( a );
		chains<OP><<<blocks, 256>>>( out, iters, 0x83DCD12Du );
		cudaEventRecord( b );
		cudaEventSynchronize( b );
		float ms = 0;
		cudaEventElapsedTime( &ms, a, b );
		const double r = static_cast<double>( blocks ) * 256 * iters * 8 / ( ms * 1e-3 );   // chain iterations per second
		if ( r > best ) best = r;
	}
	return best;
}

int main() {
	cudaDeviceProp p;
	cudaGetDeviceProperties( &p, 0 );
	std::printf( "%s, %d SMs\n", p.name, p.multiProcessorCount );
	uint32_t* out = nullptr;
	cudaMalloc( &out, 1 << 24 );
	const int blocks = p.multiProcessorCount * 8, iters = 20000;
	double r[NOPS];
	r[ADD] = rate<ADD>( out, blocks, iters );
	r[MUL32] = rate<MUL32>( out, blocks, iters );
	r[MUL24X2] = rate<MUL24X2>( out, blocks, iters );
	r[DP4A] = rate<DP4A>( out, blocks, iters );
	r[SAD4] = rate<SAD4>( out, blocks, iters );
	r[SWAR] = rate<SWAR>( out, blocks, iters );
	for ( int i = 0; i < NOPS; ++i ) std::printf( "%-58s %8.1f G iter/s  = %5.2f add-equivalents\n", kNames[i], r[i] / 1e9, ( r[ADD] / r[i] ) * 2.0 - 1.0 );
	return 0;
}
