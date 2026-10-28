#pragma once

#include "engine/modules/steer.h"
#include "engine/core/arena.h"
#include "engine/backend/types.h"
#include <vector>

namespace engine::models {

// Host side of the runtime steering of one rank (option steer): the device tables of ResidualSteer, the per-slot control blocks and the probe's
// capture of the residual after chosen layers. A model owns one, calls init() while it allocates (nothing is allocated, and on() is false, while the
// option is off), and calls the kernels with tables() and index() / capture_dst() from its layer loop.
class SteerHost {
public:
	bool on() const { return t_.layers > 0; }
	const engine::modules::SteerTables& tables() const { return t_; }
	int first() const { return first_; }
	int layers() const { return t_.layers; }
	// Position of a model layer in the steered range, -1 outside it
	int index( int layer ) const { return layer >= first_ && layer < first_ + t_.layers ? layer - first_ : -1; }

	void init( pod::MemoryArena& arena, int hidden, int num_layers, int slots, int rank );
	void set( int slot, const std::vector<engine::modules::SteerEntry>& entries, backend::stream_t stream );
	void control( int slot, int scope, int prompt_len, int think_end, backend::stream_t stream );
	void penalty( int slot, const std::vector<int>& ids, float penalty, backend::stream_t stream );

	// Probe capture: while on, the model runs eagerly and copies the last row of the residual (fp16) after each listed layer to capture_dst( layer )
	void capture( const std::vector<int>& layers, bool on );
	bool capturing() const { return cap_on_; }
	backend::half_t* capture_dst( int layer ) const { return cap_on_ && cap_layer_[layer] >= 0 ? cap_ + static_cast<size_t>( cap_layer_[layer] ) * t_.hidden : nullptr; }
	bool read( std::vector<float>& out, backend::stream_t stream );

private:
	engine::modules::SteerTables t_;
	int first_ = 0;
	int rank_ = 0;
	std::vector<engine::modules::SteerSlotCtl> ctl_host_;
	bool cap_on_ = false;
	std::vector<int> cap_layer_;   // per model layer: row of cap_ (-1 = not kept)
	std::vector<int> cap_order_;   // the captured layers in the caller's order
	backend::half_t* cap_ = nullptr;
};

} // namespace engine::models
