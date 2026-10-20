#pragma once

#include "engine/core/batch.h"
#include "engine/runtime/sampling.h"

namespace engine::runtime {

// Outcome of one speculative round on a single slot.
struct SpecResult {
	int proposed = 0;                  // draft tokens submitted for verification
	int accepted = 0;                  // draft tokens matching the target
	int emitted[pod::MAX_BATCH] = {};  // tokens committed this round (accepted drafts + correction/bonus)
	int emitted_count = 0;
	int next_pos = 0;                  // position of the next anchor (== tokens committed to KV)
};

// A source of draft tokens for the target's batched verify (DFlash2: a block drafter reading tapped hidden states; MTP: the
// checkpoint's own next-token head). One slot at a time owns the speculator's context; other slots decode normally.
class Speculator {
public:
	virtual ~Speculator() = default;
	virtual void shutdown() = 0;

	virtual void claim( int slot ) = 0;     // the slot takes over the context (a cold start at position 0)
	virtual void release( int slot ) = 0;   // the slot loses it
	virtual bool owns( int slot ) const = 0;
	// True when every slot keeps its own context (claim does not take it from another slot), so the generator may interleave rounds of
	// several slots. DFlash2's tap ring / drafter KV is a single context.
	virtual bool per_slot() const { return false; }

	// Requires owns( slot ). `anchor` is the last committed, not yet forwarded token at `pos`. With a non-greedy `sp` the round
	// samples: each draft token is accepted with the target's probability of it (exact for a deterministic draft) and the
	// correction / bonus token is drawn from the target's distribution.
	virtual SpecResult speculate( int slot, int anchor, int pos, float depth_threshold, const SamplingParams* sp, Rng* rng ) = 0;

	// Draft tokens from outside the speculator (a suffix match over the request's own text) for the slot's next round: used in place of drafting
	// when n >= 1, then dropped. `n` is at most the speculator's block length
	virtual void offer_draft( int slot, const int* tokens, int n ) { (void) slot; (void) tokens; (void) n; }
	virtual bool takes_offered_draft() const { return false; }

	// Joint rounds are on by default for this speculator (the option spec_joint overrides), and at most this many slots share one
	virtual bool joint_default() const { return false; }
	virtual int joint_max() const { return 1 << 20; }
	// One round for several owned slots at once (a single draft chain and a single verify forward over all their blocks). Only where joint().
	virtual bool joint() const { return false; }
	virtual void speculate_joint( int n, const int* slots, const int* anchors, const int* poss, const SamplingParams* const* sps, Rng* const* rngs, SpecResult* out ) {
		(void) n; (void) slots; (void) anchors; (void) poss; (void) sps; (void) rngs; (void) out;
		ENGINE_ASSERT( false, "speculator has no joint rounds" );
	}

	// Called after every prefill chunk of the owning slot: `rows` prompt tokens starting at position pos0 were just forwarded
	virtual void on_prefill_chunk( int slot, const int* tokens, int rows, int pos0 ) { (void) slot; (void) tokens; (void) rows; (void) pos0; }
	// Called after a plain (non-speculative) decode step of the owning slot: `token` was forwarded at `pos` as row `row` of the batch
	virtual void on_dense_row( int slot, int token, int pos, int row ) { (void) slot; (void) token; (void) pos; (void) row; }
	// Whether the target must publish drafter taps (and snapshots carry them)
	virtual bool needs_taps() const { return true; }
	// The tap rows of a snapshot are small, so every prefix checkpoint may carry them (a resume from a periodic checkpoint keeps speculating)
	virtual bool cheap_snapshot_taps() const { return false; }
	// The slot's context was restored from a snapshot of `tokens` positions that carried taps (after claim, before the rest of the prompt prefills)
	virtual void on_restore( int slot, int tokens ) { (void) slot; (void) tokens; }

	// Per-step draft accuracy where the step was reached
	virtual void step_stats( int* attempts, int* hits ) const = 0;
	virtual void profile_dump() {}
};

} // namespace engine::runtime
