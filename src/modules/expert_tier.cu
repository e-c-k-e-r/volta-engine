#include "engine/modules/expert_tier.h"
#include <algorithm>
#include <cstdlib>
#include <cstdio>

namespace engine::modules {

double ExpertTier::env_fraction() {
	const char* e = std::getenv( "EXL3_EXPERT_CACHE" );
	if ( e == nullptr ) return 1.0;
	const double f = std::atof( e );
	return f > 0.0 && f < 1.0 ? f : 1.0;
}

void ExpertTier::configure( int experts_per_layer, double fraction, const size_t bytes[3] ) {
	experts = experts_per_layer;
	resident = std::max( 1, std::min( experts, static_cast<int>( fraction * experts_per_layer + 0.5 ) ) );
	slot_bytes = 0;
	for ( int i = 0; i < 3; ++i ) {
		part_bytes[i] = bytes[i];
		slot_bytes += bytes[i];
	}
	ENGINE_ASSERT( slot_bytes % 256 == 0, "ExpertTier: expert code bytes must be a multiple of 256" );
	if ( const char* e = std::getenv( "EXL3_EXPERT_SWAPS" ) ) max_swaps = std::max( 0, std::atoi( e ) );
}

void ExpertTier::add_layer( pod::MoELayerWeights* mw, pod::MemoryArena& device, pod::MemoryArena& host, std::vector<int> resident_set ) {
	Layer l;
	l.mw = mw;
	l.slots = static_cast<uint8_t*>( engine::core::arena_alloc( device, static_cast<size_t>( resident ) * slot_bytes, 256 ) );
	l.slot_of.assign( experts, -1 );
	l.owner.assign( resident, -1 );
	l.host.assign( experts, nullptr );
	l.score.assign( experts, 0 );
	ENGINE_ASSERT( static_cast<int>( resident_set.size() ) == resident, "ExpertTier::add_layer: resident set size" );
	for ( int s = 0; s < resident; ++s ) {
		l.slot_of[resident_set[s]] = s;
		l.owner[s] = resident_set[s];
	}
	for ( int e = 0; e < experts; ++e ) {
		if ( l.slot_of[e] < 0 ) l.host[e] = static_cast<uint8_t*>( engine::core::arena_alloc( host, slot_bytes, 256 ) );
	}
	layers.push_back( std::move( l ) );
}

void ExpertTier::code_arena( int layer, int expert, int part, pod::MemoryArena& out ) const {
	const Layer& l = layers[layer];
	uint8_t* base = l.slot_of[expert] >= 0 ? l.slots + static_cast<size_t>( l.slot_of[expert] ) * slot_bytes : l.host[expert];
	size_t off = 0;
	for ( int i = 0; i < part; ++i ) off += part_bytes[i];
	out.base_ptr = base + off;
	out.total_bytes = part_bytes[part];
	out.offset_bytes = 0;
	out.pinned = l.slot_of[expert] < 0;
}

void ExpertTier::allocate_staging( pod::MemoryArena& host ) {
	staging = static_cast<uint8_t*>( engine::core::arena_alloc( host, slot_bytes, 256 ) );
}

int ExpertTier::rebalance( int layer, cudaStream_t stream ) {
	Layer& l = layers[layer];
	std::vector<uint32_t> heat( experts );
	CUDA_CHECK( cudaStreamSynchronize( stream ) );
	CUDA_CHECK( cudaMemcpy( heat.data(), l.mw->heat, static_cast<size_t>( experts ) * sizeof( uint32_t ), cudaMemcpyDeviceToHost ) );
	CUDA_CHECK( cudaMemset( l.mw->heat, 0, static_cast<size_t>( experts ) * sizeof( uint32_t ) ) );
	for ( int e = 0; e < experts; ++e ) {
		window_routes += heat[e];
		if ( l.slot_of[e] >= 0 ) window_hits += heat[e];
		l.score[e] = l.score[e] / 2 + heat[e];
	}
	int done = 0;
	void* table = l.mw->device_ptrs;
	for ( ; done < max_swaps; ++done ) {
		int hot = -1, cold_res = -1; // hottest cold expert, coldest resident one
		for ( int e = 0; e < experts; ++e ) {
			if ( l.slot_of[e] < 0 ) { if ( hot < 0 || l.score[e] > l.score[hot] ) hot = e; }
			else { if ( cold_res < 0 || l.score[e] < l.score[cold_res] ) cold_res = e; }
		}
		if ( hot < 0 || cold_res < 0 || l.score[hot] < l.score[cold_res] + l.score[cold_res] / 2 + 8 ) break;
		const int slot = l.slot_of[cold_res];
		uint8_t* slot_ptr = l.slots + static_cast<size_t>( slot ) * slot_bytes;
		uint8_t* hot_buf = l.host[hot];
		// Exchange: the slot's bytes into the staging buffer, the hot expert's bytes into the slot
		CUDA_CHECK( cudaMemcpyAsync( staging, slot_ptr, slot_bytes, cudaMemcpyDeviceToHost, stream ) );
		CUDA_CHECK( cudaMemcpyAsync( slot_ptr, hot_buf, slot_bytes, cudaMemcpyHostToDevice, stream ) );
		CUDA_CHECK( cudaStreamSynchronize( stream ) );
		l.host[cold_res] = staging;
		l.host[hot] = nullptr;
		staging = hot_buf;
		l.slot_of[hot] = slot;
		l.slot_of[cold_res] = -1;
		l.owner[slot] = hot;
		// Point the kernels' tables at the new homes (gate, up, down codes are rows 0, 3, 6 of the 9 x E pointer table)
		const void* ptrs_hot[3], *ptrs_cold[3];
		size_t off = 0;
		for ( int p = 0; p < 3; ++p ) {
			ptrs_hot[p] = slot_ptr + off;
			ptrs_cold[p] = l.host[cold_res] + off;
			off += part_bytes[p];
			const size_t row = static_cast<size_t>( 3 * p ) * experts;
			CUDA_CHECK( cudaMemcpy( static_cast<const void**>( table ) + row + hot, &ptrs_hot[p], sizeof( void* ), cudaMemcpyHostToDevice ) );
			CUDA_CHECK( cudaMemcpy( static_cast<const void**>( table ) + row + cold_res, &ptrs_cold[p], sizeof( void* ), cudaMemcpyHostToDevice ) );
		}
		// Host mirrors of the weights
		off = 0;
		for ( int p = 0; p < 3; ++p ) {
			pod::LinearWeight* set[3] = { l.mw->gate, l.mw->up, l.mw->down };
			set[p][hot].w0.data = slot_ptr + off;
			set[p][cold_res].w0.data = l.host[cold_res] + off;
			off += part_bytes[p];
		}
		++swaps;
	}
	return done;
}

} // namespace engine::modules
