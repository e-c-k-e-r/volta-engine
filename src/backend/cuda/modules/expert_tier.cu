#include "engine/modules/expert_tier.h"
#include "engine/runtime/options.h"
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <thread>
#include <unistd.h>

namespace engine::modules {

double ExpertTier::resident_fraction() {
	return runtime::options().expert_cache;
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
	if ( runtime::options().expert_swaps >= 0 ) max_swaps = runtime::options().expert_swaps;
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
	if ( stream_mode() && ring == nullptr ) ring = static_cast<uint8_t*>( engine::core::arena_alloc( host, cold_bytes_per_layer(), 256 ) );
	l.src.assign( static_cast<size_t>( experts ) * 3, Layer::Src{} );
	size_t cold_i = 0;
	for ( int e = 0; e < experts; ++e ) {
		if ( l.slot_of[e] >= 0 || runtime::options().expert_synth ) continue;
		l.host[e] = stream_mode() ? ring + cold_i * slot_bytes : static_cast<uint8_t*>( engine::core::arena_alloc( host, slot_bytes, 256 ) );
		++cold_i;
	}
	layers.push_back( std::move( l ) );
}

void ExpertTier::code_arena( int layer, int expert, int part, pod::MemoryArena& out ) const {
	const Layer& l = layers[layer];
	// ENGINE_DEBUG_EXPERT_SYNTH: a cold expert shares the device slot of resident slot (expert mod resident), so every read is a device read
	const bool synth = runtime::options().expert_synth && l.slot_of[expert] < 0;
	uint8_t* base = l.slot_of[expert] >= 0 ? l.slots + static_cast<size_t>( l.slot_of[expert] ) * slot_bytes
		: synth ? l.slots + static_cast<size_t>( expert % resident ) * slot_bytes : l.host[expert];
	size_t off = 0;
	for ( int i = 0; i < part; ++i ) off += part_bytes[i];
	out.base_ptr = base + off;
	out.total_bytes = part_bytes[part];
	out.offset_bytes = 0;
	out.pinned = l.slot_of[expert] < 0 && !synth;
	out.stream_skip = out.pinned && runtime::options().expert_stream;
}

bool ExpertTier::stream_mode() {
	return runtime::options().expert_stream;
}

void ExpertTier::set_source( int layer, int expert, int part, int fd, size_t off, size_t bytes ) {
	// The repository that opened the shard closes it when loading ends: keep one duplicate per shard for the whole run
	static std::vector<std::pair<int, int>> dups;
	int mine = -1;
	for ( const auto& d : dups ) if ( d.first == fd ) mine = d.second;
	if ( mine < 0 ) { mine = dup( fd ); dups.emplace_back( fd, mine ); }
	layers[layer].src[static_cast<size_t>( expert ) * 3 + part] = Layer::Src{ mine, off, bytes };
}

void ExpertTier::stream_fill( int layer, cudaStream_t stream ) {
	// The previous layer's kernels still read the shared buffer
	CUDA_CHECK( cudaStreamSynchronize( stream ) );
	const Layer& l = layers[layer];
	std::vector<int> cold;
	for ( int e = 0; e < experts; ++e ) if ( l.host[e] != nullptr ) cold.push_back( e );
	const int nthreads = 8;
	std::vector<std::thread> pool;
	for ( int t = 0; t < nthreads; ++t ) {
		pool.emplace_back( [&, t]() {
			for ( size_t i = t; i < cold.size(); i += nthreads ) {
				const int e = cold[i];
				size_t dst_off = 0;
				for ( int p = 0; p < 3; ++p ) {
					const Layer::Src& s = l.src[static_cast<size_t>( e ) * 3 + p];
					ENGINE_ASSERT( s.fd >= 0 && s.bytes == part_bytes[p], "ExpertTier::stream_fill: no source for a cold expert" );
					size_t done = 0;
					while ( done < s.bytes ) {
						const ssize_t n = pread( s.fd, l.host[e] + dst_off + done, s.bytes - done, static_cast<off_t>( s.off + done ) );
						if ( n <= 0 ) std::fprintf( stderr, "stream_fill: pread n=%zd errno=%d fd=%d off=%zu bytes=%zu done=%zu\n", n, errno, s.fd, s.off, s.bytes, done );
						ENGINE_ASSERT( n > 0, "ExpertTier::stream_fill: read failed" );
						done += static_cast<size_t>( n );
					}
					dst_off += part_bytes[p];
				}
			}
		} );
	}
	for ( auto& th : pool ) th.join();
}

void ExpertTier::allocate_staging( pod::MemoryArena& host ) {
	staging = static_cast<uint8_t*>( engine::core::arena_alloc( host, slot_bytes, 256 ) );
}

int ExpertTier::rebalance( int layer, cudaStream_t stream ) {
	if ( runtime::options().expert_synth || runtime::options().expert_stream ) return 0;   // the cold experts alias resident slots: nothing to swap
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
		// Point the kernels' tables at the new homes (gate, up, down codes: rows 0, 3, 6 of the trellis 9 x E pointer table, rows 0, 2, 4 of the
		// NVFP4 6 x E table, where the scales are the rows between)
		const int row_step = l.mw->type == pod::QuantType::NVFP4 ? 2 : 3;
		const void* ptrs_hot[3], *ptrs_cold[3];
		size_t off = 0;
		for ( int p = 0; p < 3; ++p ) {
			ptrs_hot[p] = slot_ptr + off;
			ptrs_cold[p] = l.host[cold_res] + off;
			off += part_bytes[p];
			const size_t row = static_cast<size_t>( row_step * p ) * experts;
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
