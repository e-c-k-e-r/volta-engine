#include "engine/modules/expert_tier.h"
#include "engine/core/log.h"
#include "engine/runtime/options.h"
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <thread>
#include <unistd.h>
#include <fstream>
#include <numeric>

namespace engine::modules {

double ExpertTier::resident_fraction() {
	return runtime::options().expert_cache;
}

void ExpertTier::configure( int experts_per_layer, double fraction, const size_t bytes[3], const std::string& heat_file ) {
	experts = experts_per_layer;
	resident = std::max( 1, std::min( experts, static_cast<int>( fraction * experts_per_layer + 0.5 ) ) );
	slot_bytes = 0;
	for ( int i = 0; i < 3; ++i ) {
		part_bytes[i] = bytes[i];
		slot_bytes += bytes[i];
	}
	ENGINE_ASSERT( slot_bytes % 256 == 0, "ExpertTier: expert code bytes must be a multiple of 256" );
	max_swaps = runtime::options().expert_swaps;
	heat_path = heat_file;
	seed.clear();
	if ( !heat_path.empty() ) {
		// File: "VEH1", layers, experts (int32 each), then layers x experts uint64 counts. Older runs count half, so the latest workload leads
		std::ifstream f( heat_path, std::ios::binary );
		char magic[4] = {};
		int32_t nl = 0, ne = 0;
		if ( f && f.read( magic, 4 ) && std::string( magic, 4 ) == "VEH1" && f.read( reinterpret_cast<char*>( &nl ), 4 ) && f.read( reinterpret_cast<char*>( &ne ), 4 ) && ne == experts_per_layer && nl > 0 ) {
			seed.resize( static_cast<size_t>( nl ) * ne );
			if ( !f.read( reinterpret_cast<char*>( seed.data() ), static_cast<std::streamsize>( seed.size() * sizeof( uint64_t ) ) ) ) seed.clear();
			for ( auto& v : seed ) v /= 2;
		}
		LOG_INFO( "[experts] heat file %s: %s\n", heat_path.c_str(), seed.empty() ? "none usable, starting from experts 0..N-1" : "seeding the resident sets" );
	}
}

int ExpertTier::swap_cap() const {
	if ( max_swaps >= 0 ) return max_swaps;
	// A swap moves two experts' bytes: the budget per rebalance pass is spread over the layers
	const size_t budget = static_cast<size_t>( std::max( 1, runtime::options().expert_swap_mib ) ) << 20;
	const size_t per_layer = budget / ( std::max<size_t>( 1, layers.size() ) * 2 * slot_bytes );
	return static_cast<int>( std::max<size_t>( 1, std::min<size_t>( 64, per_layer ) ) );
}

std::vector<int> ExpertTier::initial_set( int layer ) const {
	std::vector<int> set( resident );
	if ( seed.size() >= static_cast<size_t>( layer + 1 ) * experts ) {
		const uint64_t* c = seed.data() + static_cast<size_t>( layer ) * experts;
		std::vector<int> order( experts );
		std::iota( order.begin(), order.end(), 0 );
		std::stable_sort( order.begin(), order.end(), [&]( int a, int b ) { return c[a] > c[b]; } );
		order.resize( resident );
		std::sort( order.begin(), order.end() );
		return order;
	}
	std::iota( set.begin(), set.end(), 0 );
	return set;
}

void ExpertTier::save_heat() {
	if ( heat_path.empty() || layers.empty() || !heat_dirty ) return;
	const std::string tmp = heat_path + ".tmp";
	{
		std::ofstream f( tmp, std::ios::binary );
		const int32_t nl = static_cast<int32_t>( layers.size() ), ne = experts;
		f.write( "VEH1", 4 );
		f.write( reinterpret_cast<const char*>( &nl ), 4 );
		f.write( reinterpret_cast<const char*>( &ne ), 4 );
		for ( const Layer& l : layers ) f.write( reinterpret_cast<const char*>( l.total.data() ), static_cast<std::streamsize>( l.total.size() * sizeof( uint64_t ) ) );
		if ( !f ) return;
	}
	std::rename( tmp.c_str(), heat_path.c_str() );
	heat_dirty = false;
}

ExpertTier::~ExpertTier() {
	save_heat();
}

void ExpertTier::add_layer( pod::MoELayerWeights* mw, pod::MemoryArena& device, pod::MemoryArena& host, std::vector<int> resident_set ) {
	Layer l;
	l.mw = mw;
	l.slots = static_cast<uint8_t*>( engine::core::arena_alloc( device, static_cast<size_t>( resident ) * slot_bytes, 256 ) );
	l.slot_of.assign( experts, -1 );
	l.owner.assign( resident, -1 );
	l.host.assign( experts, nullptr );
	l.score.assign( experts, 0 );
	l.total.assign( experts, 0 );
	if ( seed.size() >= ( layers.size() + 1 ) * static_cast<size_t>( experts ) ) std::copy_n( seed.begin() + layers.size() * experts, experts, l.total.begin() );
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
		l.total[e] += heat[e];
	}
	int done = 0;
	void* table = l.mw->device_ptrs;
	heat_dirty = true;
	const int cap = swap_cap();
	for ( ; done < cap; ++done ) {
		int hot = -1, cold_res = -1; // hottest cold expert, coldest resident one
		for ( int e = 0; e < experts; ++e ) {
			if ( l.slot_of[e] < 0 ) { if ( hot < 0 || l.score[e] > l.score[hot] ) hot = e; }
			else { if ( cold_res < 0 || l.score[e] < l.score[cold_res] ) cold_res = e; }
		}
		if ( hot < 0 || cold_res < 0 || l.score[hot] < l.score[cold_res] + l.score[cold_res] / 2 + static_cast<uint64_t>( std::max( 0, runtime::options().expert_swap_margin ) ) ) break;
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
