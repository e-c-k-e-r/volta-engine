#pragma once

#include "engine/backend/types.h"
#include <cstddef>
#include <cstdint>

namespace pod {

// Storage format of one KV cache tensor (K or V), element index e within a slot:
//   F16   2 bytes per element
//   E5M2  1 byte per element, the high byte of the FP16 value (round to nearest even)
//   I8    1 byte per element, one FP16 scale per 64 elements (amax / 127)
//   Q4    4 bits per element, one FP16 scale per 64 elements; values are Walsh-Hadamard rotated per 64 before rounding
//   FP4   4 bits (E2M1) per element, one E4M3 scale byte per 16 elements (amax / 6)
enum class KVQ : uint8_t { F16, E5M2, I8, Q4, FP4 };

// Whole-cache selection: fp16 = (F16, F16), fp8 = (E5M2, E5M2), k8v4 = (I8, Q4), nvfp4 = (FP4, FP4).
enum class KVFormat : uint8_t { FP16, FP8, K8V4, NVFP4 };

inline KVQ kv_k_quant( KVFormat f ) {
	switch ( f ) {
		case KVFormat::FP8: return KVQ::E5M2;
		case KVFormat::K8V4: return KVQ::I8;
		case KVFormat::NVFP4: return KVQ::FP4;
		default: return KVQ::F16;
	}
}

inline KVQ kv_v_quant( KVFormat f ) {
	switch ( f ) {
		case KVFormat::FP8: return KVQ::E5M2;
		case KVFormat::K8V4: return KVQ::Q4;
		case KVFormat::NVFP4: return KVQ::FP4;
		default: return KVQ::F16;
	}
}

// Bytes of element data for `elems` elements of format q (elems a multiple of 64).
inline size_t kv_data_bytes( KVQ q, size_t elems ) {
	switch ( q ) {
		case KVQ::F16: return elems * 2;
		case KVQ::E5M2:
		case KVQ::I8: return elems;
		default: return elems / 2;
	}
}

// Bytes of scale data for `elems` elements (0 for the unscaled formats).
inline size_t kv_scale_bytes( KVQ q, size_t elems ) {
	switch ( q ) {
		case KVQ::I8:
		case KVQ::Q4: return elems / 64 * 2;
		case KVQ::FP4: return elems / 16;
		default: return 0;
	}
}

// Shared token pool (a model option): the cache of every layer is one [layer][pool_tokens] array, and the logical page
// pos >> shift of a slot lives in the physical page table[slot * per_slot + (pos >> shift)]. A null table is the per-slot
// layout, [slot][layer][max_seq], addressed by the callers' slot stride.
struct KVPages {
	const int* table = nullptr;
	int shift = 0;            // log2(tokens per page)
	int per_slot = 0;         // logical pages of one slot
	size_t pool_tokens = 0;   // tokens of one layer in the pool (the layer stride)

	// Token index of (slot, pos) within one layer's array
	__host__ __device__ size_t token( int slot, int pos ) const {
		return ( static_cast<size_t>( table[static_cast<size_t>( slot ) * per_slot + ( pos >> shift )] ) << shift ) + ( pos & ( ( 1 << shift ) - 1 ) );
	}
};

// Device-visible view of the KV pool. Element offsets (slot stride, layer, position) are format independent;
// k / v point at slot 0, ks / vs at the matching scales (null for the unscaled formats).
struct KVRef {
	void* k = nullptr;
	void* v = nullptr;
	void* ks = nullptr;
	void* vs = nullptr;
	KVQ kq = KVQ::F16;
	KVQ vq = KVQ::F16;
	KVPages pages;            // the latent cache of GLM only
};

inline KVRef kv_ref_f16( const void* k, const void* v ) {
	KVRef r;
	r.k = const_cast<void*>( k );
	r.v = const_cast<void*>( v );
	return r;
}

} // namespace pod
