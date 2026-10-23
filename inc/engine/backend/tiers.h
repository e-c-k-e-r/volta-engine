#pragma once

#include "engine/config.h"
#include <cstring>

// Capability registry for the computing backends. Every module entry point has a tier (inc/engine/backend/tier_table.inc);
// a backend that cannot run a CudaSm70 entry point provides a stub that calls ENGINE_UNSUPPORTED, so a missing kernel is a named,
// greppable abort and not a link error or a silent wrong answer. The CUDA backend implements everything and ignores the tags.

namespace engine::backend {

enum class Tier {
	Portable,         // a Slang kernel exists (src/kernels/slang)
	PortablePending,  // portable in kind, no Slang kernel yet
	CudaSm70          // hand-tuned CUDA/SM70 only
};

struct TierEntry {
	const char* module;
	const char* function;
	Tier tier;
	const char* note;
};

inline const TierEntry* tier_entries( int* count ) {
	static const TierEntry entries[] = {
#define ENGINE_TIER_ENTRY( module, function, tier, note ) { #module, #function, Tier::tier, note },
#include "engine/backend/tier_table.inc"
#undef ENGINE_TIER_ENTRY
	};
	*count = static_cast<int>( sizeof( entries ) / sizeof( entries[0] ) );
	return entries;
}

#if defined(ENGINE_BACKEND_CUDA) && ENGINE_BACKEND_CUDA == 1
	#define ENGINE_BACKEND_NAME "CUDA"
#elif defined(ENGINE_BACKEND_VULKAN) && ENGINE_BACKEND_VULKAN == 1
	#define ENGINE_BACKEND_NAME "Vulkan"
#elif defined(ENGINE_BACKEND_CPU) && ENGINE_BACKEND_CPU == 1
	#define ENGINE_BACKEND_NAME "CPU"
#else
	#define ENGINE_BACKEND_NAME "unknown"
#endif

} // namespace engine::backend

// Body of a stub for an entry point this backend does not implement
#define ENGINE_UNSUPPORTED( what ) \
	ENGINE_ASSERT( false, ENGINE_BACKEND_NAME " backend: " what " is not implemented (CUDA/SM70 only or not yet ported)" )
