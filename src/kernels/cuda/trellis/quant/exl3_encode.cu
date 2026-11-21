// Host side of the EXL3 trellis encoder: launch geometry, scratch, the batched launches, and the tile packers (exllamav3's pack.cu / frac.cu,
// MIT, see LICENSE).
#include "engine/quant/exl3_encode.h"
#include "engine/modules/trellis_bits.h"
#include "engine/config.h"
#include "exl3_encode_kernel.cuh"
#include "exl3_encode_inst.cuh"
#include <cuda_runtime.h>
#include <mutex>

namespace {

using engine::modules::trellis_base_bits;
using engine::modules::trellis_is_half;
using engine::modules::trellis_tile_u16;

// Only the alternating pattern is instantiated: half-integer tiles stay whole 32-bit words
constexpr uint32_t HALF_MASK = 0xAAAAu;

struct Launch {
	exl3e_kernel_fn kernel = nullptr;
	int threads = 0;
	int shmem = 0;
	int blocks_per_sm = 0;
	int sms = 0;
};

exl3e_kernel_fn kernel_for( int code, int cb ) {
	if ( trellis_is_half( code ) ) {
		switch ( trellis_base_bits( code ) ) {
			case 1: {
				return exl3_encode_tiles_half_kernel<1, HALF_MASK>;
			} break;
			case 2: {
				return exl3_encode_tiles_half_kernel<2, HALF_MASK>;
			} break;
			case 3: {
				return exl3_encode_tiles_half_kernel<3, HALF_MASK>;
			} break;
			default: break;
		}
		return nullptr;
	}
	return code <= 4 ? exl3e_kernel_lo( code, cb ) : exl3e_kernel_hi( code, cb );
}

// Per device, width code and codebook: the kernel and how many of its blocks one SM keeps resident (shared memory opted in once)
Launch launch_for( int code, int cb ) {
	constexpr int MAX_DEV = 16;
	static Launch table[MAX_DEV][12][3];
	static std::mutex mu;
	int dev = 0;
	CUDA_CHECK( cudaGetDevice( &dev ) );
	ENGINE_ASSERT( dev < MAX_DEV && code >= 1 && code <= 11 && cb >= 0 && cb <= 2, "exl3_encode: bad device, width code or codebook" );
	ENGINE_ASSERT( !trellis_is_half( code ) || cb == 2, "exl3_encode: half-integer rates need the mul1 codebook" );
	std::lock_guard<std::mutex> lock( mu );
	Launch& l = table[dev][code][cb];
	if ( l.kernel ) return l;
	l.kernel = kernel_for( code, cb );
	ENGINE_ASSERT( l.kernel, "exl3_encode: no kernel for this width code" );
	if ( trellis_is_half( code ) ) {
		l.threads = exl3e_half_threads( trellis_base_bits( code ) );
		l.shmem = exl3e_half_shmem( trellis_base_bits( code ) );
	} else {
		l.threads = exl3e_num_threads( code );
		l.shmem = exl3e_shmem( code );
	}
	CUDA_CHECK( cudaFuncSetAttribute( (const void*) l.kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, l.shmem ) );
	CUDA_CHECK( cudaOccupancyMaxActiveBlocksPerMultiprocessor( &l.blocks_per_sm, (const void*) l.kernel, l.threads, l.shmem ) );
	CUDA_CHECK( cudaDeviceGetAttribute( &l.sms, cudaDevAttrMultiProcessorCount, dev ) );
	ENGINE_ASSERT( l.blocks_per_sm > 0, "exl3_encode: kernel does not fit on an SM" );
	return l;
}

// Out-states of a step: 65536 >> K (half-integer rates: of the narrower shift, the widest node space)
size_t states( int code ) { return static_cast<size_t>( 65536 >> trellis_base_bits( code ) ); }

// The history: a byte per pair of out-states (the half-integer rates and K <= 4, exl3e_packed_hist), else a 16-bit in-state per out-state
size_t edges_bytes( int code, int tiles ) {
	const bool packed = trellis_is_half( code ) || exl3e_packed_hist( code );
	return static_cast<size_t>( tiles ) * 256 * ( packed ? states( code ) / 2 : states( code ) * sizeof( uint16_t ) );
}

size_t costs_bytes( int code, int tiles ) {
	const bool global = trellis_is_half( code ) || code < EXL3E_SHARED_COSTS_MIN_K;
	return global ? static_cast<size_t>( tiles ) * 2 * states( code ) * sizeof( half ) : 0;
}

// 16 spans of 16 windows per tile, K bits each, MSB-first into uint16 words, then the halves of each 32-bit word swapped (the stream order the
// decode kernels read)
template <int K>
__global__ __launch_bounds__( 128 )
void pack_kernel( uint16_t* __restrict__ g_packed, const uint16_t* __restrict__ g_unpacked ) {
	constexpr int packed_size = 256 * K / 16;
	__shared__ uint16_t s_unpacked[256];
	__shared__ uint16_t s_packed[packed_size];
	const int t = threadIdx.x;
	g_packed += static_cast<size_t>( blockIdx.x ) * packed_size;
	g_unpacked += static_cast<size_t>( blockIdx.x ) * 256;
	( (uint32_t*) s_unpacked )[t] = ( (const uint32_t*) g_unpacked )[t];
	__syncthreads();
	if ( t < 16 ) {
		int i = 16 * t;
		int j = K * t;
		int k = 32;
		uint32_t buf = 0;
		for ( int n = 0; n < 16; ++n ) {
			uint32_t v = (uint32_t) s_unpacked[i] & ( ( 1u << K ) - 1 );
			k -= K;
			buf |= ( v << k );
			if ( k <= 16 ) {
				s_packed[j] = (uint16_t) ( buf >> 16 );
				buf <<= 16;
				k += 16;
				j++;
			}
			i++;
		}
	}
	__syncthreads();
	if ( t < packed_size / 2 ) ( (uint32_t*) g_packed )[t] = __byte_perm( ( (uint32_t*) s_packed )[t], 0, 0x1032 );
}

// Half-integer tiles: window i's D(i) new bits (D(i) = KA + bit (i mod 16) of the mask) at ring positions S(i) - D(i) .. S(i), MSB-first in
// 32-bit words; one thread per tile
__global__ __launch_bounds__( 128 )
void pack_half_kernel( uint16_t* __restrict__ g_packed, const uint16_t* __restrict__ g_unpacked, int num_tiles, int ka, uint32_t mask, int bpb ) {
	const int tile = blockIdx.x * 128 + threadIdx.x;
	if ( tile >= num_tiles ) return;
	const int nw = bpb / 2;
	uint32_t words[64];
	for ( int w = 0; w < nw; ++w ) words[w] = 0;
	const uint16_t* idx = g_unpacked + static_cast<size_t>( tile ) * 256;
	int pos = 0;
	for ( int i = 0; i < 256; ++i ) {
		const int d = ka + ( ( mask >> ( i & 15 ) ) & 1 );
		const uint32_t v = (uint32_t) idx[i] & ( ( 1u << d ) - 1 );
		for ( int b = d - 1; b >= 0; --b, ++pos ) {
			if ( ( v >> b ) & 1 ) words[pos >> 5] |= 1u << ( 31 - ( pos & 31 ) );
		}
	}
	uint32_t* out = (uint32_t*) ( g_packed + static_cast<size_t>( tile ) * bpb );
	for ( int w = 0; w < nw; ++w ) out[w] = words[w];
}

} // namespace

namespace engine::quant {

int exl3_encode_wave_tiles( int code, int cb ) {
	const Launch l = launch_for( code, cb );
	return l.blocks_per_sm * l.sms;
}

size_t exl3_encode_scratch_bytes( int code, int tiles ) { return edges_bytes( code, tiles ) + costs_bytes( code, tiles ); }

bool exl3_encode_scratch_alloc( Exl3EncodeScratch& s, int code, int cb, int tiles ) {
	exl3_encode_scratch_free( s );
	if ( tiles <= 0 ) tiles = exl3_encode_wave_tiles( code, cb );
	if ( cudaMalloc( &s.edges, edges_bytes( code, tiles ) ) != cudaSuccess ) {
		(void) cudaGetLastError();
		s.edges = nullptr;
		return false;
	}
	const size_t cbytes = costs_bytes( code, tiles );
	if ( cbytes && cudaMalloc( &s.costs, cbytes ) != cudaSuccess ) {
		(void) cudaGetLastError();
		exl3_encode_scratch_free( s );
		return false;
	}
	s.code = code;
	s.tiles = tiles;
	return true;
}

void exl3_encode_scratch_free( Exl3EncodeScratch& s ) {
	if ( s.edges ) CUDA_CHECK( cudaFree( s.edges ) );
	if ( s.costs ) CUDA_CHECK( cudaFree( s.costs ) );
	s = Exl3EncodeScratch{};
}

void exl3_encode_tiles( const float* tiles, float* decoded, uint16_t* windows, int num_tiles, int code, int cb, const Exl3EncodeScratch& s,
	backend::stream_t stream, const float* weights, bool open ) {
	ENGINE_ASSERT( s.edges && s.code == code && s.tiles > 0, "exl3_encode_tiles: scratch was allocated for another width code" );
	const Launch l = launch_for( code, cb );
	// One resident wave at most per launch: more blocks would only queue on the same scratch
	const int batch = s.tiles < l.blocks_per_sm * l.sms ? s.tiles : l.blocks_per_sm * l.sms;
	auto st = static_cast<cudaStream_t>( stream );
	for ( int b0 = 0; b0 < num_tiles; b0 += batch ) {
		const int n = num_tiles - b0 < batch ? num_tiles - b0 : batch;
		const size_t off = static_cast<size_t>( b0 ) * 256;
		l.kernel<<<n, l.threads, l.shmem, st>>>( tiles + off, weights ? weights + off : nullptr, decoded + off, windows + off,
			static_cast<half*>( s.costs ), static_cast<uint16_t*>( s.edges ), open ? 1 : 0 );
		CUDA_CHECK( cudaPeekAtLastError() );
	}
}

void exl3_pack_tiles( uint16_t* packed, const uint16_t* windows, int num_tiles, int code, backend::stream_t stream ) {
	if ( num_tiles <= 0 ) return;
	auto st = static_cast<cudaStream_t>( stream );
	if ( trellis_is_half( code ) ) {
		const int bpb = trellis_tile_u16( code );
		pack_half_kernel<<<( num_tiles + 127 ) / 128, 128, 0, st>>>( packed, windows, num_tiles, trellis_base_bits( code ), HALF_MASK, bpb );
	} else {
		switch ( code ) {
			case 1: { pack_kernel<1><<<num_tiles, 128, 0, st>>>( packed, windows ); } break;
			case 2: { pack_kernel<2><<<num_tiles, 128, 0, st>>>( packed, windows ); } break;
			case 3: { pack_kernel<3><<<num_tiles, 128, 0, st>>>( packed, windows ); } break;
			case 4: { pack_kernel<4><<<num_tiles, 128, 0, st>>>( packed, windows ); } break;
			case 5: { pack_kernel<5><<<num_tiles, 128, 0, st>>>( packed, windows ); } break;
			case 6: { pack_kernel<6><<<num_tiles, 128, 0, st>>>( packed, windows ); } break;
			case 7: { pack_kernel<7><<<num_tiles, 128, 0, st>>>( packed, windows ); } break;
			case 8: { pack_kernel<8><<<num_tiles, 128, 0, st>>>( packed, windows ); } break;
			default: {
				ENGINE_ASSERT( false, "exl3_pack_tiles: bad width code" );
			} break;
		}
	}
	CUDA_CHECK( cudaPeekAtLastError() );
}

} // namespace engine::quant
