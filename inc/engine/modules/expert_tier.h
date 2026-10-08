#pragma once

#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/modules/moe.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace engine::modules {

// Two-tier storage of the routed experts' trellis codes (the bulk of a MoE checkpoint). A fraction of each layer's local experts
// is resident in device slots; the rest lives in pinned, mapped host memory and is read by the expert kernels over PCIe in place
// (zero-copy: no host synchronisation, so decode graphs are unchanged; a miss costs 12 GB/s instead of HBM speed). Scale vectors
// and Hadamard signs always stay on the device. The kernels reach the codes through the layer's device pointer table, so moving an
// expert is: copy its bytes, then rewrite three table entries.
//
// The residency adapts to the routing seen so far (Strata's online scoring): per-expert route counters (MoELayerWeights::heat)
// are folded into a decayed score, and `rebalance` swaps the hottest cold expert with the coldest resident one while the cold
// one scores clearly higher. A swap is an exchange: the slot's bytes go to a staging buffer, the cold expert's bytes into the
// slot, and the staging buffer and the cold expert's old buffer trade roles, so no host copy is needed.
struct ExpertTier {
	struct Layer {
		pod::MoELayerWeights* mw = nullptr;
		uint8_t* slots = nullptr;            // resident * slot_bytes, device
		std::vector<int> slot_of;            // expert -> slot, -1 when cold
		std::vector<int> owner;              // slot -> expert
		std::vector<uint8_t*> host;          // expert -> its pinned buffer when cold
		std::vector<uint64_t> score;         // decayed route counts
	};

	int experts = 0;
	int resident = 0;
	size_t part_bytes[3] = {};
	size_t slot_bytes = 0;
	std::vector<Layer> layers;
	uint8_t* staging = nullptr;              // one pinned expert-sized buffer that rotates through the swaps
	uint64_t swaps = 0, window_routes = 0, window_hits = 0;
	int max_swaps = 4;

	// EXL3_EXPERT_CACHE: fraction of each layer's experts kept resident (1 or unset: tiering off)
	static double env_fraction();

	bool active() const { return resident < experts; }
	size_t cold_bytes_per_layer() const { return static_cast<size_t>( experts - resident ) * slot_bytes; }

	// Sizes the tier for `experts` local experts per layer and the byte sizes of one expert's gate / up / down codes
	void configure( int experts_per_layer, double fraction, const size_t bytes[3] );
	// Backing for one layer: its device slots (from `device`) and the pinned buffers of its cold experts (from `host`); the
	// returned views are where expert `e`'s three code tensors are to be loaded
	void add_layer( pod::MoELayerWeights* mw, pod::MemoryArena& device, pod::MemoryArena& host, std::vector<int> resident_set );
	void code_arena( int layer, int expert, int part, pod::MemoryArena& out ) const;
	// After a layer's experts are loaded and its device pointer table built
	void allocate_staging( pod::MemoryArena& host );
	// Reads and clears the layer counters, updates the scores and performs up to max_swaps exchanges; returns swaps done
	int rebalance( int layer, cudaStream_t stream );
	double hit_rate() const { return window_routes ? static_cast<double>( window_hits ) / static_cast<double>( window_routes ) : 1.0; }
	void reset_window() { window_routes = window_hits = 0; }
};

} // namespace engine::modules
