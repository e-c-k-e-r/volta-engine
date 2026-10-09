// Vulkan module layer: SPIR-V kernel cache
#include "mod_common.h"
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#ifndef SLANG_SPV_DIR
	#error "SLANG_SPV_DIR must point at the generated .spv directory"
#endif

namespace engine::backend::impl {

vk::Pipeline* kernel( const char* name, uint32_t push_bytes ) {
	static std::map<std::string, vk::Pipeline*> cache;
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
	static std::vector<Slot> slots;
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
