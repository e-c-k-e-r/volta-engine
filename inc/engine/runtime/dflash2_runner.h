#pragma once

#include "engine/config.h"
#include "engine/core/batch.h"
#include "engine/models/model.h"
#include "engine/spec/dflash.h"
#include <cuda_runtime.h>
#include <string>

namespace engine::runtime {

class ModelCluster;

// Outcome of one speculative round on a single slot.
struct SpecResult {
	int proposed = 0;                  // draft tokens submitted for verification
	int accepted = 0;                  // draft tokens matching the target
	int emitted[pod::MAX_BATCH] = {};  // tokens committed this round (accepted drafts + correction/bonus)
	int emitted_count = 0;
	int next_pos = 0;                  // position of the next anchor (== tokens committed to KV)
};

// Per-stage wall time of a speculative round (host clock: several stages are host syncs whose cost a
// device event would not attribute to them). Accumulated since the last dump.
struct SpecProfile {
	enum Stage : int { DRAFT = 0, WALK, CHECKPOINT, VERIFY, REPAIR, COUNT };
	double ms[COUNT] = {};
	int rounds = 0;
	int commits = 0;
};

// One DFlash2 round over a ModelCluster: draft (backbone on rank 0, head shard + top-k on every
// rank, merge) -> selector walk -> batched target verify -> greedy accept -> state repair.
//
// TP-readiness: every stage is driven through the cluster's rank pool and the per-rank DraftHooks,
// so sharding the drafter backbone only fills in work for ranks != 0 without changing this control
// flow. The speculator itself lives on rank 0 until then.
class DFlash2Runner {
public:
	bool init( ModelCluster& mc, const std::string& draft_dir );
	void shutdown();

	// Drafter context belongs to one slot at a time: the target publishes taps for the owner slot
	// only (device scalar), so a slot that does not own the ring has no context to draft from.
	// Claiming a slot resets the drafter's projected context and redirects tap publication.
	void claim( int slot );
	bool owns( int slot ) const { return owner_ == slot; }

	// Requires owns( slot ). `anchor` is the last committed, not yet forwarded token at `pos`.
	SpecResult speculate( int slot, int anchor, int pos, float depth_threshold = -1.0f );

	// Per-step draft accuracy where the step was reached (the loop breaks at the first mismatch, so
	// step k is sampled exactly when steps 0..k-1 were accepted).
	void step_stats( int* attempts, int* hits ) const;
	void profile_dump();
	void set_profile( bool on ) { profile_on_ = on; }

private:
	void mark( SpecProfile::Stage stage );

	ModelCluster* mc_ = nullptr;
	engine::spec::DFlashSpeculator spec_;
	int owner_ = -1;

	int att_[8] = {};
	int hit_[8] = {};
	float cond_[8] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };  // survival EMA (depth policy)

	bool profile_on_ = false;
	SpecProfile prof_;
	int prof_last_ = 0;
	double prof_t0_ = 0.0;
};

} // namespace engine::runtime
