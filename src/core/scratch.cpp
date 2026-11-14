#include "engine/core/scratch.h"
#include "engine/backend/runtime.h"
#include "engine/core/log.h"
#include "engine/runtime/options.h"
#include <map>
#include <mutex>
#include <utility>
#include <vector>

namespace engine::core {

namespace {

struct Block {
	void* ptr = nullptr;
	size_t bytes = 0;
};

struct DeviceScratch {
	std::mutex mu;   // per device: a direct allocation may wait for its device, whose queue can hold a collective another rank's thread has to join
	pod::MemoryArena* arena = nullptr;
	char* region = nullptr;      // the budget taken from the arena at attach
	size_t region_bytes = 0, region_used = 0;
	std::map<std::pair<const char*, uintptr_t>, Block> blocks;
	std::vector<void*> direct;   // allocated without an arena
	size_t taken = 0;
};

DeviceScratch g_dev[ENGINE_NUM_GPUS];

DeviceScratch& device_scratch( int device ) {
	ENGINE_ASSERT( device >= 0 && device < ENGINE_NUM_GPUS, "scratch: device out of range" );
	return g_dev[device];
}

}   // namespace

void scratch_attach( int device, pod::MemoryArena* arena, size_t budget ) {
	DeviceScratch& d = device_scratch( device );
	std::lock_guard<std::mutex> lk( d.mu );
	d.arena = arena;
	d.region_bytes = budget & ~static_cast<size_t>( 255 );
	d.region = d.region_bytes ? static_cast<char*>( arena_alloc( *arena, d.region_bytes ) ) : nullptr;
	d.region_used = 0;
	d.taken = 0;
}

void scratch_detach( int device ) {
	DeviceScratch& d = device_scratch( device );
	std::lock_guard<std::mutex> lk( d.mu );
	for ( void* p : d.direct ) backend::device_free( p );
	d.arena = nullptr;
	d.region = nullptr;
	d.region_bytes = d.region_used = 0;
	d.blocks.clear();
	d.direct.clear();
	d.taken = 0;
}

void* scratch( const char* site, size_t bytes, uintptr_t sub ) {
	const int device = backend::get_device();
	DeviceScratch& d = device_scratch( device );
	std::lock_guard<std::mutex> lk( d.mu );
	Block& b = d.blocks[{ site, sub }];
	if ( b.ptr != nullptr && b.bytes >= bytes ) return b.ptr;
	const size_t aligned = ( bytes + 255 ) & ~static_cast<size_t>( 255 );
	if ( d.region != nullptr && d.region_used + aligned <= d.region_bytes ) {
		b.ptr = d.region + d.region_used;
		d.region_used += aligned;
		d.taken += aligned;
	}
	else if ( d.arena != nullptr ) {
		LOG_INFO( "[scratch] GPU %d: %s %.2f MiB does not fit the %.0f MiB budget (scratch_mib), taken from the arena's free space\n", device, site,
			bytes / 1048576.0, d.region_bytes / 1048576.0 );
		b.ptr = arena_alloc( *d.arena, bytes );
		d.taken += bytes;
	}
	else {
		b.ptr = backend::device_malloc( bytes );
		d.direct.push_back( b.ptr );
	}
	b.bytes = bytes;
	if ( runtime::options().mem_log ) {
		LOG_INFO( "[scratch] GPU %d: %s %.2f MiB%s (blocks %.2f MiB of the arena, budget %.0f MiB)\n", device, site, bytes / 1048576.0,
			d.arena ? "" : " outside the arena", d.taken / 1048576.0, d.region_bytes / 1048576.0 );
	}
	return b.ptr;
}

size_t scratch_taken( int device ) {
	DeviceScratch& d = device_scratch( device );
	std::lock_guard<std::mutex> lk( d.mu );
	return d.taken;
}

} // namespace engine::core
