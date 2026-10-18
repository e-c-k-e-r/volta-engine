// Vulkan backend: weak stubs for the expert_tier entry points that have no Vulkan implementation (CUDA/SM70 only or not yet ported).
// A strong definition in another translation unit overrides these. Declarations come from engine/modules/expert_tier.h.
#include "engine/modules/expert_tier.h"
#include "engine/modules/cpu_experts.h"
#include "engine/backend/tiers.h"

namespace engine::modules {

// No host-memory tier on this backend: every expert is resident (the models ask before they decide how to place the expert codes)
double ExpertTier::resident_fraction() {
	return 1.0;
}

__attribute__(( weak )) void ExpertTier::configure( int experts_per_layer, double fraction, const size_t bytes[3], const std::string& heat_file ) {
	ENGINE_UNSUPPORTED( "ExpertTier::configure" );
	__builtin_unreachable();
}

__attribute__(( weak )) std::vector<int> ExpertTier::initial_set( int layer, int count ) const {
	ENGINE_UNSUPPORTED( "ExpertTier::initial_set" );
	__builtin_unreachable();
}

__attribute__(( weak )) int ExpertTier::swap_cap() const {
	ENGINE_UNSUPPORTED( "ExpertTier::swap_cap" );
	__builtin_unreachable();
}

// Nothing is tiered on this backend, so there is nothing to save
__attribute__(( weak )) void ExpertTier::save_heat() {}

__attribute__(( weak )) int ExpertTier::assign_cpu( pod::MemoryArena& ) { return 0; }

// Staging is part of the host-memory tier: nothing to stage here
__attribute__(( weak )) void ExpertTier::stage_init( pod::MemoryArena& ) {}
__attribute__(( weak )) void ExpertTier::stage_prefetch( int, backend::stream_t ) {}
__attribute__(( weak )) bool ExpertTier::stage_begin( int, backend::stream_t ) { return false; }
__attribute__(( weak )) void ExpertTier::stage_end( int, backend::stream_t ) {}

__attribute__(( weak )) ExpertTier::~ExpertTier() {}

__attribute__(( weak )) void ExpertTier::start( int experts_per_layer, double fraction, const size_t bytes[3], int rank, int tp, int layers, bool streamed, pod::MemoryArena& host ) {
	ENGINE_UNSUPPORTED( "ExpertTier::start" );
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

// The CPU expert path needs a GPU backend
__attribute__(( weak )) void CpuExperts::post( const backend::half_t*, int*, const backend::half_t*, int, int, int, int, int, const uint8_t*, float, uint32_t*, cpu_moe::Mailbox*, uint64_t*, int, backend::stream_t ) {
	ENGINE_UNSUPPORTED( "CpuExperts::post" );
	__builtin_unreachable();
}
__attribute__(( weak )) void CpuExperts::wait_add( backend::half_t*, int, int, cpu_moe::Mailbox*, const uint64_t*, backend::stream_t ) {
	ENGINE_UNSUPPORTED( "CpuExperts::wait_add" );
	__builtin_unreachable();
}

} // namespace engine::modules
