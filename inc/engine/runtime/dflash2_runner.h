#pragma once

#include "engine/config.h"
#include "engine/core/batch.h"
#include "engine/models/model.h"
#include "engine/spec/dflash.h"
#include "engine/runtime/sampling.h"
#include "engine/runtime/speculator.h"
#include "engine/backend/types.h"
#include <string>

namespace engine::runtime {

class ModelCluster;

// Per-stage wall time of a speculative round (host clock: several stages are host syncs whose cost a
// device event would not attribute to them). Accumulated since the last dump.
struct SpecProfile {
	enum Stage : int { DRAFT = 0, WALK, CHECKPOINT, VERIFY, SAMPLE, REPAIR, COUNT };
	double ms[COUNT] = {};
	int rounds = 0;
	int commits = 0;
};

// One DFlash2 round over a ModelCluster: draft (backbone, head shard + top-k on every rank, merge) ->
// selector walk -> batched target verify -> greedy accept -> state repair.
//
// The drafter backbone is tensor-parallel across the ranks by default: every rank holds a shard of each
// projection, runs the whole block in lockstep (all-reduce after o_proj and down_proj, fc all-gather) and
// ends with the full drafter hidden replicated, which feeds its own head shard with no peer copy. The walk
// runs on rank 0. ENGINE_DRAFT_TP=0 selects the single-GPU drafter on rank 0 (A/B knob).
class DFlash2Runner : public Speculator {
public:
	bool init( ModelCluster& mc, const std::string& draft_dir );
	void shutdown() override;

	// Drafter context belongs to one slot at a time: the target publishes taps for the owner slot
	// only (device scalar), so a slot that does not own the ring has no context to draft from.
	// Claiming a slot resets the drafter's projected context and redirects tap publication.
	// Where the model publishes a tap ring per slot (DraftHooks::taps_slot_stride) every slot keeps its own drafter context and ring, so any number
	// of slots can speculate (per_slot()); otherwise one slot owns the single context at a time.
	void claim( int slot ) override;
	bool owns( int slot ) const override { return per_slot_ ? ( slot >= 0 && slot < kMaxSlots && owned_[slot] ) : owner_ == slot; }
	bool per_slot() const override { return per_slot_; }
	// The slot stops owning the drafter context (it was overwritten by a restored snapshot, which carries no taps).
	void release( int slot ) override;

	// Requires owns( slot ). `anchor` is the last committed, not yet forwarded token at `pos`.
	// With a non-greedy `sp` the round samples: each draft token is accepted with the target's probability of it
	// (exact for a deterministic draft) and the correction / bonus token is drawn from the target's distribution.
	SpecResult speculate( int slot, int anchor, int pos, float depth_threshold = -1.0f,
		const SamplingParams* sp = nullptr, Rng* rng = nullptr ) override;

	// Per-step draft accuracy where the step was reached (the loop breaks at the first mismatch, so
	// step k is sampled exactly when steps 0..k-1 were accepted).
	// A joint round: every slot's block is drafted in turn (the drafter scratch is shared), then one target forward verifies all the blocks, so the
	// experts and weights that rows of different slots share are read once. Needs the journaled KDA commit (spec_commit_always).
	bool joint() const override;
	bool joint_default() const override { return true; }
	int joint_max() const override { return pod::MAX_BATCH / models::kDraftM; }
	void speculate_joint( int n, const int* slots, const int* anchors, const int* poss, const SamplingParams* const* sps, Rng* const* rngs, SpecResult* out ) override;

	void step_stats( int* attempts, int* hits ) const override;
	void profile_dump() override;
	void set_profile( bool on ) { profile_on_ = on; }

	// Candidate-quality diagnostics (off by default; each round then pays one extra readback). For draft
	// position k, counted only where the target's own token at that position is known to be on its true
	// continuation (k <= first mismatch): rows seen, rows where the drafter's best candidate was the target's
	// token, rows where it was anywhere in the top-k set, and rows where the selector walk chose it.
	struct DiagStats {
		int seen[8] = {};
		int top1[8] = {};
		int in_topk[8] = {};
		int walk[8] = {};
	};
	void set_diag( bool on ) { diag_on_ = on; }
	const DiagStats& diag() const { return diag_; }
	void diag_reset() { diag_ = DiagStats{}; }

private:
	int draft_block( int slot, int anchor, int pos, float depth_threshold, int* h_draft );
	void mark( SpecProfile::Stage stage );
	bool diag_on_ = false;
	DiagStats diag_;

	ModelCluster* mc_ = nullptr;
	engine::spec::DFlashSpeculator spec_[ENGINE_NUM_GPUS];  // [0] only when tp_ == 1
	int tp_ = 1;                                            // drafter tensor-parallel width
	int owner_ = -1;
	static constexpr int kMaxSlots = 8;
	bool per_slot_ = false;
	bool owned_[kMaxSlots] = {};

	int att_[8] = {};
	int hit_[8] = {};
	float cond_[8] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };  // survival EMA (depth policy)

	bool profile_on_ = false;
	SpecProfile prof_;
	int prof_last_ = 0;
	double prof_t0_ = 0.0;
};

} // namespace engine::runtime
