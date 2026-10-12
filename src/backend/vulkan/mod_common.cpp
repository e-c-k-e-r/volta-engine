// Vulkan module layer: SPIR-V kernel cache
#include "mod_common.h"
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#ifndef SLANG_SPV_DIR
	#error "SLANG_SPV_DIR must point at the generated .spv directory"
#endif

namespace engine::backend::impl {

vk::Pipeline* kernel( const char* name, uint32_t push_bytes ) {
	// Pipelines belong to a VkDevice: one cache per device (the rank threads call this concurrently)
	static std::map<std::string, vk::Pipeline*> caches[ENGINE_NUM_GPUS];
	static std::mutex mu;
	std::lock_guard<std::mutex> lock( mu );
	auto& cache = caches[get_device()];
	auto it = cache.find( name );
	if ( it != cache.end() ) return it->second;
	const char* dir = std::getenv( "ENGINE_SPV_DIR" );
	const std::string path = std::string( dir ? dir : SLANG_SPV_DIR ) + "/" + name + ".spv";
	vk::Pipeline* p = vk::pipeline_create( path.c_str(), push_bytes );
	ENGINE_ASSERT( p != nullptr, "Vulkan backend: cannot create the compute pipeline from the .spv" );
	cache[name] = p;
	return p;
}

void* scratch( int slot, size_t bytes, stream_t stream ) {
	struct Slot { void* p = nullptr; size_t bytes = 0; };
	static std::vector<Slot> all_slots[ENGINE_NUM_GPUS];   // per device: a scratch buffer lives on the device that was current when it was made
	auto& slots = all_slots[get_device()];
	if ( slot >= static_cast<int>( slots.size() ) ) slots.resize( slot + 1 );
	Slot& s = slots[slot];
	if ( bytes > s.bytes ) {
		if ( s.p ) {
			stream_sync( stream );   // recorded commands may reference the old buffer
			device_free( s.p );
		}
		s.p = device_malloc( bytes );
		s.bytes = bytes;
	}
	return s.p;
}

} // namespace engine::backend::impl
