#pragma once

#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/modules/moe.h"
#include "engine/backend/types.h"
#include <cstdint>
#include <string>
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
		std::vector<uint64_t> total;         // route counts since the first run (saved to / seeded from expert_heat_file)
		struct Src { int fd = -1; size_t off = 0, bytes = 0; };
		std::vector<Src> src;                // expert * 3 + part: where a cold expert's codes are in the checkpoint (expert_stream)
		void* staged_dev = nullptr;          // prefill: the 9 x experts pointer table with the cold experts' codes pointing into the stage buffer
		int version = 0, built = -1;         // residency changes (swaps) against the version staged_dev was built for
		bool queued = false;                 // the stage copy of this layer is issued and not yet consumed
	};

	int experts = 0;
	int resident = 0;                        // the base count per layer; add_layer takes the size of the set it is given
	size_t part_bytes[3] = {};
	size_t slot_bytes = 0;
	std::vector<Layer> layers;
	uint8_t* ring = nullptr;                 // expert_stream: the one pinned buffer every layer's cold experts are read into before the layer runs
	uint8_t* staging = nullptr;              // one pinned expert-sized buffer that rotates through the swaps
	uint64_t swaps = 0, window_routes = 0, window_hits = 0;
	int max_swaps = -1;                      // per layer per rebalance; <0 = derived from the byte budget expert_swap_mib
	std::string heat_path;                   // expert_heat_file of this rank (empty: no persistence)
	std::vector<uint64_t> seed;              // [layer][expert] counts read from heat_path
	int rebalances = 0;
	bool heat_dirty = false;
	// Prefill staging: a prefill chunk's MoE kernel reads every cold expert's codes once per row tile, over the shared PCIe link. With staging a
	// layer's cold experts are copied once into one of two device buffers (on a copy stream, a layer ahead of the compute) and the kernel reads
	// them from there. stage_init after all layers are added and their pointer tables built; it turns itself off when the arena has no room
	std::vector<void*> ev_ready, ev_free;
	void* copy_stream = nullptr;
	uint8_t* stage_buf[2] = {};
	int cold_max = 0;
	bool staging_on = false;
	~ExpertTier();
	void stage_init( pod::MemoryArena& device );
	bool stage_auto_off = false;   // the owner's default for expert_stage = -1 (auto): staging off
	bool prefill_staging() const { return staging_on; }
	void stage_prefetch( int layer, backend::stream_t stream );   // issue the copy of the layer's cold experts (idempotent until stage_end)
	bool stage_begin( int layer, backend::stream_t stream );      // the layer's MoE then reads the staged copy; false: nothing cold, run as usual
	void stage_end( int layer, backend::stream_t stream );        // after the layer's MoE kernel

	// ENGINE_EXPERT_CACHE: fraction of each layer's experts kept resident (1 or unset: tiering off)
	static double resident_fraction();

	bool active() const { return resident < experts; }
	size_t cold_bytes_per_layer() const { return static_cast<size_t>( experts - resident ) * slot_bytes; }

	// Sizes the tier for `experts` local experts per layer and the byte sizes of one expert's gate / up / down codes
	void configure( int experts_per_layer, double fraction, const size_t bytes[3], const std::string& heat_file = std::string() );
	// configure() for rank `rank` of `tp` (the heat file gets a .r<rank> suffix with more than one rank), plus the pinned host buffer for
	// `layers` MoE layers' cold experts (one layer's worth when `streamed`) and the staging; host is created here
	void start( int experts_per_layer, double fraction, const size_t bytes[3], int rank, int tp, int layers, bool streamed, pod::MemoryArena& host );
	// The experts a layer starts with: the hottest ones of the saved counts when there are any, experts 0..resident - 1 otherwise
	// (`count` of them; the tier's `resident` when 0 - a layer may hold more than the base count, never fewer)
	std::vector<int> initial_set( int layer, int count = 0 ) const;
	void save_heat();
	int swap_cap() const;
	// Backing for one layer: its device slots (from `device`) and the pinned buffers of its cold experts (from `host`); the
	// returned views are where expert `e`'s three code tensors are to be loaded
	void add_layer( pod::MoELayerWeights* mw, pod::MemoryArena& device, pod::MemoryArena& host, std::vector<int> resident_set );
	void code_arena( int layer, int expert, int part, pod::MemoryArena& out ) const;
	// ENGINE_DEBUG_EXPERT_STREAM: the cold experts are not kept in host memory: before each layer's MoE `stream_fill` reads them from the
	// checkpoint (parallel pread, so NVMe speed) into one pinned buffer shared by all layers, where the kernels read them in place
	static bool stream_mode();
	void set_source( int layer, int expert, int part, int fd, size_t off, size_t bytes );
	void stream_fill( int layer, backend::stream_t stream );
	// After a layer's experts are loaded and its device pointer table built
	void allocate_staging( pod::MemoryArena& host );
	// Reads and clears the layer counters, updates the scores and performs up to max_swaps exchanges; returns swaps done
	int rebalance( int layer, backend::stream_t stream );
	double hit_rate() const { return window_routes ? static_cast<double>( window_hits ) / static_cast<double>( window_routes ) : 1.0; }
	void reset_window() { window_routes = window_hits = 0; }
};

} // namespace engine::modules
