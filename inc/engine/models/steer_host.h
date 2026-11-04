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

	// mean_rows > 0 (the four-stream models): a [mean_rows, hidden] fp16 buffer for the stream mean a probe captures
	// route_experts > 0: a router bias table for every layer (MoE models whose router takes it)
	void init( pod::MemoryArena& arena, int hidden, int num_layers, int slots, int rank, int mean_rows = 0, int route_experts = 0 );
	// the router bias of `layer` for slot 0 (slots are route_stride() floats apart), nullptr when the table is off
	const float* route_bias( int layer ) const { return t_.route && layer >= 0 && layer < t_.route_layers ? t_.route + static_cast<size_t>( layer ) * t_.route_experts : nullptr; }
	int route_stride() const { return t_.route_layers * t_.route_experts; }
	int route_experts() const { return t_.route_experts; }
	void set( int slot, const std::vector<engine::modules::SteerEntry>& entries, backend::stream_t stream );
	void control( int slot, int scope, int prompt_len, int think_end, backend::stream_t stream );
	void penalty( int slot, const std::vector<int>& ids, float penalty, backend::stream_t stream );

	// Probe capture: while on, the model runs eagerly and copies the last row of the residual (fp16) after each listed layer to capture_dst( layer )
	void capture( const std::vector<int>& layers, bool on );
	bool capturing() const { return cap_on_; }
	backend::half_t* capture_dst( int layer ) const { return cap_on_ && cap_layer_[layer] >= 0 ? cap_ + static_cast<size_t>( cap_layer_[layer] ) * t_.hidden : nullptr; }
	bool read( std::vector<float>& out, backend::stream_t stream );
	// probe scores of positions [first_pos, first_pos + count) of a slot, out [kProbes][count] (count <= kProbeRing)
	bool read_probes( int slot, int first_pos, int count, float* out, backend::stream_t stream );

	// After model layer `layer`: its edits on the rows' fp16 residual [m, hidden], and the probe's capture of the last row
	void layer_rows( backend::half_t* hidden, const pod::BatchRows* d_rows, int m, int layer, backend::stream_t stream ) const;
	// The same for a residual of four fp32 streams per row [m, 4, hidden]: the edits act on the streams' mean (ResidualSteer::apply_streams), the
	// capture reads the mean
	void layer_streams( float* streams, const pod::BatchRows* d_rows, int m, int layer, backend::stream_t stream ) const;
	// The reflection-marker penalty on logit rows [row0, row0 + n) of this rank's vocabulary slice [shard_lo, shard_lo + shard_cols); nothing while off
	void penalize( backend::half_t* logits, int logit_cols, const pod::BatchRows* d_rows, int row0, int n, int shard_lo, int shard_cols, backend::stream_t stream ) const;

private:
	engine::modules::SteerTables t_;
	int first_ = 0;
	int rank_ = 0;
	std::vector<engine::modules::SteerSlotCtl> ctl_host_;
	bool cap_on_ = false;
	std::vector<int> cap_layer_;   // per model layer: row of cap_ (-1 = not kept)
	std::vector<int> cap_order_;   // the captured layers in the caller's order
	backend::half_t* cap_ = nullptr;
	backend::half_t* mean_ = nullptr;   // [mean_rows, hidden]
};

// The Model steering overrides of a pipeline that owns a SteerHost named `host`
#define ENGINE_STEER_HOOKS( host ) \
	bool steer_enabled() const override { return host.on(); } \
	void steer_set( int slot_id, const std::vector<engine::modules::SteerEntry>& entries, backend::stream_t stream = nullptr ) override { host.set( slot_id, entries, stream ); } \
	void steer_control( int slot_id, int scope, int prompt_len, int think_end, backend::stream_t stream = nullptr ) override { host.control( slot_id, scope, prompt_len, think_end, stream ); } \
	void steer_penalty( int slot_id, const std::vector<int>& ids, float penalty, backend::stream_t stream = nullptr ) override { host.penalty( slot_id, ids, penalty, stream ); } \
	void steer_capture( const std::vector<int>& layers, bool on ) override { host.capture( layers, on ); } \
	bool steer_read( std::vector<float>& out, backend::stream_t stream = nullptr ) override { return host.read( out, stream ); } \
	bool probe_read( int slot_id, int first_pos, int count, float* out, backend::stream_t stream = nullptr ) override { return host.read_probes( slot_id, first_pos, count, out, stream ); } \
	void steer_range( int* first_layer, int* num_layers ) const override { *first_layer = host.first(); *num_layers = host.layers(); } \
	int steer_route_experts() const override { return host.route_experts(); }

} // namespace engine::models
