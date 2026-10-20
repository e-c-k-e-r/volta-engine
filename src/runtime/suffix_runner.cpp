#include "engine/runtime/suffix_runner.h"
#include "engine/runtime/cluster.h"
#include "engine/runtime/options.h"
#include "engine/core/log.h"

namespace engine::runtime {

bool SuffixRunner::init( ModelCluster& mc ) {
	mc_ = &mc;
	if ( runtime::options().spec_suffix <= 0 ) {
		LOG_ERROR( "SuffixRunner: speculator=suffix needs spec_suffix > 0 (the match length that makes a draft)\n" );
		return false;
	}
	LOG_INFO( "Suffix speculator: drafts from the request's own text, matches of %d tokens\n", runtime::options().spec_suffix );
	return true;
}

void SuffixRunner::claim( int slot ) {
	ENGINE_ASSERT( slot >= 0 && slot < kMaxSlots, "SuffixRunner::claim: slot out of range" );
	st_[slot] = SlotState();
	st_[slot].owned = true;
}

void SuffixRunner::release( int slot ) {
	if ( slot >= 0 && slot < kMaxSlots ) st_[slot] = SlotState();
}

void SuffixRunner::offer_draft( int slot, const int* tokens, int n ) {
	if ( !owns( slot ) ) return;
	SlotState& S = st_[slot];
	S.n = n < 0 ? 0 : n > 7 ? 7 : n;
	for ( int i = 0; i < S.n; ++i ) S.tok[i] = tokens[i];
}

void SuffixRunner::step_stats( int* attempts, int* hits ) const {
	for ( int i = 0; i < 8; ++i ) {
		attempts[i] = att_[i];
		hits[i] = hit_[i];
	}
}

SpecResult SuffixRunner::speculate( int slot, int anchor, int pos, float depth_threshold, const SamplingParams* sp, Rng* rng ) {
	( void ) depth_threshold;
	ENGINE_ASSERT( owns( slot ), "SuffixRunner::speculate: slot does not own the speculator" );
	SlotState& S = st_[slot];
	ModelCluster& mc = *mc_;
	SpecResult res;
	const bool sampled = sp != nullptr && !sp->greedy();
	ENGINE_ASSERT( !sampled || rng != nullptr, "SuffixRunner::speculate: sampling needs an Rng" );
	const int active = MIN( S.n, mc.metadata().max_seq_len - pos - 2 );
	int cands[8];
	for ( int i = 0; i < active; ++i ) cands[i] = S.tok[i];
	S.n = 0;
	if ( active < 1 ) {
		// nothing offered: one plain step
		mc.forward( &anchor, 1, pos, slot );
		if ( sampled ) {
			Candidates c1[1];
			mc.sample_topk( 1, c1 );
			res.emitted[res.emitted_count++] = sample_token( c1[0], *sp, *rng );
		}
		else {
			res.emitted[res.emitted_count++] = mc.sample_one();
		}
		res.next_pos = pos + 1;
		return res;
	}
	res.proposed = active;
	// Batched verify of [anchor, d0 .. d_{active-1}] at [pos, pos + active]: row j's logits predict position pos + j + 1
	int block[pod::MAX_BATCH];
	block[0] = anchor;
	for ( int i = 0; i < active; ++i ) block[1 + i] = cands[i];
	mc.forward( block, active + 1, pos, slot );
	int preds[pod::MAX_BATCH];
	Candidates cs[pod::MAX_BATCH];
	if ( sampled ) mc.sample_topk( active + 1, cs );
	else mc.sample( preds, active + 1 );
	for ( int k = 0; k < active; ++k ) {
		++att_[k];
		if ( sampled ) {
			bool acc = false;
			const int tok = spec_step( cs[k], *sp, cands[k], *rng, &acc );
			if ( !acc ) {
				res.emitted[res.emitted_count++] = tok;
				break;
			}
		}
		else if ( cands[k] != preds[k] ) {
			res.emitted[res.emitted_count++] = preds[k];
			break;
		}
		++hit_[k];
		res.emitted[res.emitted_count++] = cands[k];
		res.accepted++;
	}
	if ( res.accepted == active ) res.emitted[res.emitted_count++] = sampled ? sample_token( cs[active], *sp, *rng ) : preds[active];
	res.next_pos = pos + 1 + res.accepted;
	return res;
}

} // namespace engine::runtime
