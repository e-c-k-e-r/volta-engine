#include "engine/runtime/kv_cache.h"
#include "engine/runtime/options.h"
#include "engine/backend/runtime.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace engine::runtime {

bool KVCachePool::init(
	int device_id,
	pod::MemoryArena& arena,
	int max_slots_in,
	int max_seq_len_in,
	int num_layers_in,
	int num_heads_in,
	int head_dim_in,
	pod::KVFormat format_in
) {
	backend::set_device( device_id );

	max_slots = max_slots_in;
	max_seq_len = max_seq_len_in;
	num_layers = num_layers_in;
	num_heads = num_heads_in;
	head_dim = head_dim_in;

	format = format_in;
	if ( const char* env = runtime::options().kv_format.c_str(); *env ) {
		if ( !std::strcmp( env, "fp8" ) ) format = pod::KVFormat::FP8;
		else if ( !std::strcmp( env, "k8v4" ) ) format = pod::KVFormat::K8V4;
		else if ( !std::strcmp( env, "nvfp4" ) ) format = pod::KVFormat::NVFP4;
		else if ( !std::strcmp( env, "fp16" ) ) format = pod::KVFormat::FP16;
	}
	const pod::KVQ kq = pod::kv_k_quant( format ), vq = pod::kv_v_quant( format );

	slot_stride_elems = static_cast<size_t>( num_layers ) * max_seq_len * num_heads * head_dim;
	slot_stride_bytes = slot_stride_elems * sizeof( backend::half_t );
	k_slot_bytes = pod::kv_data_bytes( kq, slot_stride_elems );
	v_slot_bytes = pod::kv_data_bytes( vq, slot_stride_elems );
	k_scale_slot_bytes = pod::kv_scale_bytes( kq, slot_stride_elems );
	v_scale_slot_bytes = pod::kv_scale_bytes( vq, slot_stride_elems );

	k_data = engine::core::arena_alloc( arena, static_cast<size_t>( max_slots ) * k_slot_bytes, 128 );
	v_data = engine::core::arena_alloc( arena, static_cast<size_t>( max_slots ) * v_slot_bytes, 128 );
	if ( k_scale_slot_bytes ) k_scale = engine::core::arena_alloc( arena, static_cast<size_t>( max_slots ) * k_scale_slot_bytes, 128 );
	if ( v_scale_slot_bytes ) v_scale = engine::core::arena_alloc( arena, static_cast<size_t>( max_slots ) * v_scale_slot_bytes, 128 );
	if ( !k_data || !v_data || ( k_scale_slot_bytes && !k_scale ) || ( v_scale_slot_bytes && !v_scale ) ) {
		return false;
	}

	backend::memset_sync( k_data, 0, static_cast<size_t>( max_slots ) * k_slot_bytes );
	backend::memset_sync( v_data, 0, static_cast<size_t>( max_slots ) * v_slot_bytes );
	if ( k_scale ) backend::memset_sync( k_scale, 0, static_cast<size_t>( max_slots ) * k_scale_slot_bytes );
	if ( v_scale ) backend::memset_sync( v_scale, 0, static_cast<size_t>( max_slots ) * v_scale_slot_bytes );
	return true;
}

pod::KVRef KVCachePool::ref() const {
	pod::KVRef r;
	r.k = k_data;
	r.v = v_data;
	r.ks = k_scale;
	r.vs = v_scale;
	r.kq = pod::kv_k_quant( format );
	r.vq = pod::kv_v_quant( format );
	return r;
}

// Base of the slot's tensors; only slot 0, layer 0, position 0 is meaningful for the quantized formats.
void* KVCachePool::get_k_ptr( int slot_id, int layer_idx, int pos ) {
	if ( !k_data || slot_id >= max_slots || layer_idx >= num_layers || pos >= max_seq_len ) {
		return nullptr;
	}
	ENGINE_ASSERT( format == pod::KVFormat::FP16 || ( slot_id == 0 && layer_idx == 0 && pos == 0 ), "KVCachePool::get_k_ptr: base pointer only" );
	const size_t elem = static_cast<size_t>( slot_id ) * slot_stride_elems + ( static_cast<size_t>( layer_idx ) * max_seq_len + pos ) * num_heads * head_dim;
	return static_cast<char*>( k_data ) + elem * sizeof( backend::half_t );
}

void* KVCachePool::get_v_ptr( int slot_id, int layer_idx, int pos ) {
	if ( !v_data || slot_id >= max_slots || layer_idx >= num_layers || pos >= max_seq_len ) {
		return nullptr;
	}
	ENGINE_ASSERT( format == pod::KVFormat::FP16 || ( slot_id == 0 && layer_idx == 0 && pos == 0 ), "KVCachePool::get_v_ptr: base pointer only" );
	const size_t elem = static_cast<size_t>( slot_id ) * slot_stride_elems + ( static_cast<size_t>( layer_idx ) * max_seq_len + pos ) * num_heads * head_dim;
	return static_cast<char*>( v_data ) + elem * sizeof( backend::half_t );
}

void KVCachePool::reset_slot( int slot_id, backend::stream_t stream ) {
	if ( !k_data || !v_data || slot_id >= max_slots ) {
		return;
	}
	auto clear = [&]( void* base, size_t slot_bytes ) {
		if ( !base || slot_bytes == 0 ) return;
		char* p = static_cast<char*>( base ) + static_cast<size_t>( slot_id ) * slot_bytes;
		if ( stream ) backend::memset_async( p, 0, slot_bytes, stream );
		else backend::memset_sync( p, 0, slot_bytes );
	};
	clear( k_data, k_slot_bytes );
	clear( v_data, v_slot_bytes );
	clear( k_scale, k_scale_slot_bytes );
	clear( v_scale, v_scale_slot_bytes );
}

} // namespace engine::runtime
