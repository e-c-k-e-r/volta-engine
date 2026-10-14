// Vulkan backend: weak stubs for the expert_tier entry points that have no Vulkan implementation (CUDA/SM70 only or not yet ported).
// A strong definition in another translation unit overrides these. Declarations come from engine/modules/expert_tier.h.
#include "engine/modules/expert_tier.h"
#include "engine/backend/tiers.h"

namespace engine::modules {

// No host-memory tier on this backend: every expert is resident (the models ask before they decide how to place the expert codes)
double ExpertTier::resident_fraction() {
	return 1.0;
}

__attribute__(( weak )) void ExpertTier::configure( int experts_per_layer, double fraction, const size_t bytes[3] ) {
	ENGINE_UNSUPPORTED( "ExpertTier::configure" );
	__builtin_unreachable();
}

__attribute__(( weak )) void ExpertTier::add_layer( pod::MoELayerWeights* mw, pod::MemoryArena& device, pod::MemoryArena& host, std::vector<int> resident_set ) {
	ENGINE_UNSUPPORTED( "ExpertTier::add_layer" );
	__builtin_unreachable();
}

__attribute__(( weak )) void ExpertTier::code_arena( int layer, int expert, int part, pod::MemoryArena& out ) const {
	ENGINE_UNSUPPORTED( "ExpertTier::code_arena" );
	__builtin_unreachable();
}

__attribute__(( weak )) void ExpertTier::allocate_staging( pod::MemoryArena& host ) {
	ENGINE_UNSUPPORTED( "ExpertTier::allocate_staging" );
	__builtin_unreachable();
}

__attribute__(( weak )) int ExpertTier::rebalance( int layer, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "ExpertTier::rebalance" );
	__builtin_unreachable();
}

// No cold experts to stream on this backend
bool ExpertTier::stream_mode() {
	return false;
}

__attribute__(( weak )) void ExpertTier::set_source( int layer, int expert, int part, int fd, size_t off, size_t bytes ) {
	ENGINE_UNSUPPORTED( "ExpertTier::set_source" );
	__builtin_unreachable();
}

__attribute__(( weak )) void ExpertTier::stream_fill( int layer, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "ExpertTier::stream_fill" );
	__builtin_unreachable();
}

} // namespace engine::modules
