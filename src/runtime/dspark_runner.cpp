#include <chrono>
#include <cmath>
#include "engine/runtime/dspark_runner.h"
#include "engine/runtime/cluster.h"
#include "engine/runtime/options.h"
#include "engine/core/log.h"

namespace engine::runtime {

bool DSparkRunner::init( ModelCluster& mc ) {
	mc_ = &mc;
	block_ = mc.model( 0 ).dspark_block();
	ENGINE_ASSERT( block_ >= 1 && block_ <= 7, "DSparkRunner: the drafter's block must be 1 to 7 tokens" );
	conf_th_ = static_cast<float>( runtime::options().dspark_conf );
	LOG_INFO( "DSpark speculator: blocks of %d tokens, confidence threshold %.2f\n", block_, conf_th_ );
	return true;
}

void DSparkRunner::claim( int slot ) {
	ENGINE_ASSERT( slot >= 0 && slot < kMaxSlots, "DSparkRunner::claim: slot out of range" );
	st_[slot] = SlotState();
	st_[slot].owned = true;
}

void DSparkRunner::release( int slot ) {
	if ( slot >= 0 && slot < kMaxSlots ) st_[slot] = SlotState();
}

void DSparkRunner::offer_draft( int slot, const int* tokens, int n ) {
	if ( !owns( slot ) ) return;
	SlotState& S = st_[slot];
	S.n = n < 0 ? 0 : n > 7 ? 7 : n;
	for ( int i = 0; i < S.n; ++i ) S.tok[i] = tokens[i];
}

// The taps of a resumed prefill start at its first position
void DSparkRunner::on_prefill_chunk( int slot, const int* tokens, int rows, int pos0 ) {
	( void ) tokens;
	( void ) rows;
	if ( !owns( slot ) ) return;
	SlotState& S = st_[slot];
	if ( S.fresh ) {
		S.fresh = false;
		S.valid_from = pos0;
		S.ingested = pos0;
	}
}

// The taps of the last WINDOW positions before `tokens` are back in the ring: the window KV is derived from them at the first round
void DSparkRunner::on_restore( int slot, int tokens ) {
	if ( !owns( slot ) ) return;
	SlotState& S = st_[slot];
	S.fresh = false;
	S.valid_from = MAX( 0, tokens - 128 );
	S.ingested = S.valid_from;
}

void DSparkRunner::step_stats( int* attempts, int* hits ) const {
	for ( int i = 0; i < 8; ++i ) {
		attempts[i] = att_[i];
		hits[i] = hit_[i];
	}
}

void DSparkRunner::profile_dump() {
	if ( rounds_ == 0 ) return;
	LOG_INFO( "DSpark: %ld rounds, %ld drafted (%.1f ms each), %ld from an offer, %ld proposed, %ld accepted (%.2f per round), verify %.1f ms per round\n", rounds_, drafted_rounds_,
		drafted_rounds_ ? draft_ms_ / drafted_rounds_ : 0.0, offered_rounds_, proposed_, accepted_, static_cast<double>( accepted_ ) / rounds_, verify_ms_ / rounds_ );
}

SpecResult DSparkRunner::speculate( int slot, int anchor, int pos, float depth_threshold, const SamplingParams* sp, Rng* rng ) {
	( void ) depth_threshold;
	ENGINE_ASSERT( owns( slot ), "DSparkRunner::speculate: slot does not own the speculator" );
	SlotState& S = st_[slot];
	ModelCluster& mc = *mc_;
	SpecResult res;
	const bool sampled = sp != nullptr && !sp->greedy();
	ENGINE_ASSERT( !sampled || rng != nullptr, "DSparkRunner::speculate: sampling needs an Rng" );
	const int room = mc.metadata().max_seq_len - pos - 2;
	int active = MIN( S.n, room );
	int cands[8];
	const auto t0 = std::chrono::steady_clock::now();
	if ( active >= 1 ) {
		for ( int i = 0; i < active; ++i ) cands[i] = S.tok[i];
		++offered_rounds_;
	}
	else if ( room >= block_ + 1 && pos >= 1 ) {
		int drafts[8] = {};
		float conf[8] = {};
		mc.dspark_draft( slot, pos, MAX( S.ingested, S.valid_from ), S.valid_from, anchor, drafts, conf );
		S.ingested = pos;
		int n = 0;
		while ( n < block_ && 1.0f / ( 1.0f + std::exp( -conf[n] ) ) >= conf_th_ ) ++n;
		for ( int i = 0; i < n; ++i ) cands[i] = drafts[i];
		active = n;
		++drafted_rounds_;
		draft_ms_ += std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t0 ).count();
	}
	S.n = 0;
	++rounds_;
	if ( active < 1 ) {
		// nothing worth verifying: one plain step
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
	proposed_ += active;
	// Batched verify of [anchor, d0 .. d_{active-1}] at [pos, pos + active]: row j's logits predict position pos + j + 1
	const auto t1 = std::chrono::steady_clock::now();
	int block[pod::MAX_BATCH];
	block[0] = anchor;
	for ( int i = 0; i < active; ++i ) block[1 + i] = cands[i];
	mc.forward( block, active + 1, pos, slot );
	int preds[pod::MAX_BATCH];
	Candidates cs[pod::MAX_BATCH];
	if ( sampled ) mc.sample_topk( active + 1, cs );
	else mc.sample( preds, active + 1 );
	verify_ms_ += std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t1 ).count();
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
	accepted_ += res.accepted;
	res.next_pos = pos + 1 + res.accepted;
	return res;
}

} // namespace engine::runtime
