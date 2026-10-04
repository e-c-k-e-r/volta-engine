#include "engine/runtime/kv_cache.h"
#include <cuda_fp16.h>
#include <cstdio>

namespace engine::runtime {

bool KVCachePool::init(
	int device_id,
	pod::MemoryArena& arena,
	int max_slots_in,
	int max_seq_len_in,
	int num_layers_in,
	int num_heads_in,
	int head_dim_in
) {
	CUDA_CHECK( cudaSetDevice( device_id ) );

	max_slots = max_slots_in;
	max_seq_len = max_seq_len_in;
	num_layers = num_layers_in;
	num_heads = num_heads_in;
	head_dim = head_dim_in;

	size_t pos_stride_bytes = static_cast<size_t>( num_heads ) * head_dim * sizeof( half );
	size_t layer_stride_bytes = static_cast<size_t>( max_seq_len ) * pos_stride_bytes;
	slot_stride_bytes = static_cast<size_t>( num_layers ) * layer_stride_bytes;

	size_t total_bytes_per_pool = static_cast<size_t>( max_slots ) * slot_stride_bytes;

	k_data = engine::core::arena_alloc( arena, total_bytes_per_pool, 128 );
	v_data = engine::core::arena_alloc( arena, total_bytes_per_pool, 128 );

	if ( !k_data || !v_data ) {
		return false;
	}

	CUDA_CHECK( cudaMemset( k_data, 0, total_bytes_per_pool ) );
	CUDA_CHECK( cudaMemset( v_data, 0, total_bytes_per_pool ) );

	return true;
}

void* KVCachePool::get_k_ptr( int slot_id, int layer_idx, int pos ) {
	if ( !k_data || slot_id >= max_slots || layer_idx >= num_layers || pos >= max_seq_len ) {
		return nullptr;
	}

	size_t pos_stride_bytes = static_cast<size_t>( num_heads ) * head_dim * sizeof( half );
	size_t layer_stride_bytes = static_cast<size_t>( max_seq_len ) * pos_stride_bytes;
	size_t offset = static_cast<size_t>( slot_id ) * slot_stride_bytes +
		static_cast<size_t>( layer_idx ) * layer_stride_bytes +
		static_cast<size_t>( pos ) * pos_stride_bytes;

	return static_cast<char*>( k_data ) + offset;
}

void* KVCachePool::get_v_ptr( int slot_id, int layer_idx, int pos ) {
	if ( !v_data || slot_id >= max_slots || layer_idx >= num_layers || pos >= max_seq_len ) {
		return nullptr;
	}

	size_t pos_stride_bytes = static_cast<size_t>( num_heads ) * head_dim * sizeof( half );
	size_t layer_stride_bytes = static_cast<size_t>( max_seq_len ) * pos_stride_bytes;
	size_t offset = static_cast<size_t>( slot_id ) * slot_stride_bytes +
		static_cast<size_t>( layer_idx ) * layer_stride_bytes +
		static_cast<size_t>( pos ) * pos_stride_bytes;

	return static_cast<char*>( v_data ) + offset;
}

void KVCachePool::reset_slot( int slot_id, cudaStream_t stream ) {
	if ( !k_data || !v_data || slot_id >= max_slots ) {
		return;
	}

	size_t offset = static_cast<size_t>( slot_id ) * slot_stride_bytes;
	if ( stream ) {
		CUDA_CHECK( cudaMemsetAsync( static_cast<char*>( k_data ) + offset, 0, slot_stride_bytes, stream ) );
		CUDA_CHECK( cudaMemsetAsync( static_cast<char*>( v_data ) + offset, 0, slot_stride_bytes, stream ) );
	}
	else {
		CUDA_CHECK( cudaMemset( static_cast<char*>( k_data ) + offset, 0, slot_stride_bytes ) );
		CUDA_CHECK( cudaMemset( static_cast<char*>( v_data ) + offset, 0, slot_stride_bytes ) );
	}
}

} // namespace engine::runtime
