#pragma once

// The EXL3 trellis encoder (src/kernels/cuda/trellis/quant): 16x16 tiles in stream order (the order the decode kernels read a tile's
// elements in: the stock or the Volta permutation of the row-major tile) to 16-bit windows, and windows to the packed words of a .trellis
// tensor. Width codes as in engine/modules/trellis_bits.h (1..8 bits, 9..11 = 1.5 / 2.5 / 3.5 bpw, mul1 only); codebooks 0 3INST, 1 mcg,
// 2 mul1.
#include "engine/backend/types.h"
#include <cstddef>
#include <cstdint>

namespace engine::quant {

// Device scratch for one width code: the Viterbi history of `tiles` tiles at once (and below 2 bits, and for half-integer rates, their costs)
struct Exl3EncodeScratch {
	void* costs = nullptr;
	void* edges = nullptr;
	int code = 0;
	int tiles = 0;
};

// Tiles one launch keeps resident on the current device (blocks per SM times SMs)
int exl3_encode_wave_tiles( int code, int cb );
size_t exl3_encode_scratch_bytes( int code, int tiles );
// tiles = 0: one resident wave. False when the device memory is not there
bool exl3_encode_scratch_alloc( Exl3EncodeScratch& s, int code, int cb, int tiles = 0 );
void exl3_encode_scratch_free( Exl3EncodeScratch& s );

// num_tiles x 256 floats (stream order) to the decoded values and the windows of the cheapest tail-biting path, num_tiles x 256 each, in
// launches of the scratch's tile count. The path's cost is the sum of squared errors, or with weights (num_tiles x 256 positive floats, stream
// order) the weighted sum. open: no tail-biting ring, a free start state (offline only: the windows of a tile's first positions then hold
// bits from before it, which the packed format cannot store)
void exl3_encode_tiles( const float* tiles, float* decoded, uint16_t* windows, int num_tiles, int code, int cb, const Exl3EncodeScratch& s,
	backend::stream_t stream, const float* weights = nullptr, bool open = false );

// num_tiles x 256 windows to num_tiles x trellis_tile_u16( code ) packed words
void exl3_pack_tiles( uint16_t* packed, const uint16_t* windows, int num_tiles, int code, backend::stream_t stream );

// Stream position p of a 16x16 tile in the stock or the Volta order: its row r and column c in the tile
constexpr void exl3_tile_pos( bool volta, int p, int* r, int* c ) {
	const int L = p >> 3, i = p & 7;
	if ( volta ) {
		const int q = ( L >> 2 ) & 3;
		*r = 4 * ( 2 * ( q >> 1 ) + ( i >> 2 ) ) + ( i & 3 );
		*c = 8 * ( q & 1 ) + ( L & 3 ) + 4 * ( L >> 4 );
	} else {
		*r = ( L & 3 ) * 2 + ( i & 1 ) + 8 * ( ( i >> 1 ) & 1 );
		*c = 2 * ( ( L >> 3 ) + 4 * ( i >> 2 ) ) + ( ( L >> 2 ) & 1 );
	}
}

} // namespace engine::quant
