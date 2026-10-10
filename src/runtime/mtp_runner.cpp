#include "engine/runtime/mtp_runner.h"
#include "engine/runtime/cluster.h"
#include "engine/runtime/options.h"
#include "engine/core/log.h"

namespace engine::runtime {

namespace {
constexpr int kHalf = models::kDraftM + 1;   // rows per half of the model's saved-hidden buffer
}

bool MtpRunner::init( ModelCluster& mc ) {
	mc_ = &mc;
	depth_ = MIN( MAX( runtime::options().mtp_depth, 1 ), models::kDraftM - 1 );
	LOG_INFO( "MTP speculator: depth %d\n", depth_ );
	return true;
}

void MtpRunner::claim( int slot ) {
	ENGINE_ASSERT( slot >= 0 && slot < kMaxSlots, "MtpRunner::claim: slot out of range" );
	st_[slot] = SlotState();
	st_[slot].owned = true;
}

void MtpRunner::release( int slot ) {
	if ( slot >= 0 && slot < kMaxSlots ) st_[slot] = SlotState();
}

void MtpRunner::step_stats( int* attempts, int* hits ) const {
	for ( int i = 0; i < 8; ++i ) {
		attempts[i] = att_[i];
		hits[i] = hit_[i];
	}
}

// Brings the head's KV along with a prompt: rows for the positions of this chunk (and the previous chunk's last position, whose
// hidden was kept), the chunk's own last hidden becomes the pending row.
void MtpRunner::on_prefill_chunk( int slot, const int* tokens, int count, int pos0 ) {
	if ( !owns( slot ) || count < 1 ) return;
	SlotState& S = st_[slot];
	const int off = hist_off( slot );
	const int other = kHalf - S.hist_base;
	mc_->mtp_save_hist( count - 1, 1, off + other );   // before the step, which reuses the target's hidden rows as scratch
	const bool carry = S.pend_n == 1 && S.pend_pos == pos0 - 1;
	const int rows = carry ? count : count - 1;
	if ( rows >= 1 ) {
		mc_->mtp_step( slot, carry ? pos0 - 1 : pos0, rows, carry ? tokens : tokens + 1, carry ? 1 : 0, off + S.hist_base, false );
	}
	S.hist_base = other;
	S.pend_pos = pos0 + count - 1;
	S.pend_n = 1;
}

// A plain decode step of the slot (speculation is paused): feed the previous position's row now that its following token
// is known, and keep this step's hidden as the pending row.
void MtpRunner::on_dense_row( int slot, int token, int pos, int row ) {
	if ( !owns( slot ) ) return;
	SlotState& S = st_[slot];
	const int off = hist_off( slot );
	const int other = kHalf - S.hist_base;
	mc_->mtp_save_hist( row, 1, off + other );
	if ( S.pend_n == 1 && S.pend_pos + 1 == pos ) mc_->mtp_step( slot, S.pend_pos, 1, &token, 2, off + S.hist_base, false );
	S.hist_base = other;
	S.pend_pos = pos;
	S.pend_n = 1;
}

SpecResult MtpRunner::speculate( int slot, int anchor, int pos, float depth_threshold, const SamplingParams* sp, Rng* rng ) {
	( void ) depth_threshold;
	ENGINE_ASSERT( owns( slot ), "MtpRunner::speculate: slot does not own the speculator" );
	SlotState& S = st_[slot];
	const int off = hist_off( slot );
	ModelCluster& mc = *mc_;
	const auto& meta = mc.metadata();
	SpecResult res;
	const bool sampled = sp != nullptr && !sp->greedy();
	ENGINE_ASSERT( !sampled || rng != nullptr, "MtpRunner::speculate: sampling needs an Rng" );

	int active = MIN( depth_, meta.max_seq_len - pos - 2 );
	if ( S.pend_n < 1 || S.pend_pos + S.pend_n != pos || active < 1 ) {
		// No pending hidden to draft from (cold start, a restored prefix, or no room): one plain target step for the anchor, which
		// leaves its hidden as the pending row of the next round
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
		S.hist_base = kHalf - S.hist_base;
		mc.mtp_save_hist( 0, 1, off + S.hist_base );
		S.pend_pos = pos;
		S.pend_n = 1;
		return res;
	}

	// 1. Catch-up over the pending rows (true hidden), the last of which takes the anchor as its following token; its output
	// predicts the successor of the anchor. Then chain the head on its own output for the rest of the block.
	int toks[models::kDraftM + 1];
	for ( int j = 0; j < S.pend_n - 1; ++j ) toks[j] = S.pend_tok[j];
	toks[S.pend_n - 1] = anchor;
	mc.mtp_step( slot, S.pend_pos, S.pend_n, toks, 2, off + S.hist_base, true );
	int cands[pod::MAX_BATCH];
	cands[0] = mc.sample_one();
	// A draft step costs about as much as 1/6 of a token; stop chaining once the chance that the block is still alive drops below that
	constexpr float kMinSurvival = 0.3f;
	float alive = 1.0f;
	int drafted = 1;
	for ( int k = 1; k < active; ++k ) {
		alive *= cond_[k - 1];
		if ( alive < kMinSurvival ) break;
		mc.mtp_step( slot, pos - 1 + k, 1, &cands[k - 1], 3, 0, true );
		cands[k] = mc.sample_one();
		++drafted;
	}
	active = drafted;
	res.proposed = active;

	// 2. Batched verify of [anchor, c0 .. c_{active-1}] at [pos, pos + active]: row j's logits predict position pos + j + 1
	int block[pod::MAX_BATCH];
	block[0] = anchor;
	for ( int i = 0; i < active; ++i ) block[1 + i] = cands[i];
	mc.set_spec_snapshot( true );
	mc.forward( block, active + 1, pos, slot );
	int preds[pod::MAX_BATCH];
	Candidates cs[pod::MAX_BATCH];
	if ( sampled ) mc.sample_topk( active + 1, cs );
	else mc.sample( preds, active + 1 );

	// 3. Acceptance (see DFlash2Runner::speculate)
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
	if ( res.accepted == active ) {
		res.emitted[res.emitted_count++] = sampled ? sample_token( cs[active], *sp, *rng ) : preds[active];
	}

	// Survival of draft step k given that the steps before it were accepted (observed only where it was reached)
	constexpr float kEma = 0.2f;
	for ( int k = 0; k <= res.accepted && k < active; ++k ) cond_[k] = ( 1.0f - kEma ) * cond_[k] + kEma * ( k < res.accepted ? 1.0f : 0.0f );

	// 4. Keep the committed rows' hidden for the next round's catch-up; repair the recurrent state when a draft was rejected
	S.hist_base = kHalf - S.hist_base;
	mc.mtp_save_hist( 0, res.accepted + 1, off + S.hist_base );
	S.pend_pos = pos;
	S.pend_n = res.accepted + 1;
	for ( int j = 0; j < res.accepted; ++j ) S.pend_tok[j] = cands[j];
	if ( res.accepted < active ) mc.commit_spec_row( slot, res.accepted );
	mc.set_spec_snapshot( false );
	res.next_pos = pos + 1 + res.accepted;
	return res;
}

bool MtpRunner::joint() const {
	return mc_ != nullptr && mc_->mtp_joint();
}

// A round for several slots: one catch-up step over all their pending rows, one draft-chain step per depth for all of them, and one verify
// forward over their blocks (mixed-slot rows; the recurrent states are snapshotted per row and committed per slot). Slots without a pending
// hidden (cold start, a restored prefix) take their plain step through speculate().
void MtpRunner::speculate_joint( int n, const int* slots, const int* anchors, const int* poss, const SamplingParams* const* sps, Rng* const* rngs, SpecResult* out ) {
	ModelCluster& mc = *mc_;
	const auto& meta = mc.metadata();
	ENGINE_ASSERT( n >= 1 && n <= kMaxSlots, "MtpRunner::speculate_joint: slot count out of range" );
	int J[kMaxSlots], k = 0, catch_rows = 0, depth = depth_;
	for ( int i = 0; i < n; ++i ) {
		ENGINE_ASSERT( owns( slots[i] ), "MtpRunner::speculate_joint: slot does not own the speculator" );
		const SlotState& S = st_[slots[i]];
		const int active = MIN( depth_, meta.max_seq_len - poss[i] - 2 );
		if ( S.pend_n >= 1 && S.pend_pos + S.pend_n == poss[i] && active >= 1 ) {
			J[k++] = i;
			catch_rows += S.pend_n;
			depth = MIN( depth, active );
		}
		else out[i] = SpecResult();
	}
	// Block rows: k blocks of depth + 1 in one forward
	depth = MIN( depth, pod::MAX_BATCH / MAX( 1, k ) - 1 );
	const bool joint_ok = k >= 2 && depth >= 1 && catch_rows <= pod::MAX_BATCH;
	for ( int i = 0; i < n; ++i ) {
		bool in_joint = false;
		for ( int j = 0; j < k && joint_ok; ++j ) in_joint = in_joint || J[j] == i;
		if ( !in_joint ) out[i] = speculate( slots[i], anchors[i], poss[i], -1.0f, sps[i], rngs[i] );
	}
	if ( !joint_ok ) return;

	// 1. catch-up over every slot's pending rows (true hidden); the last row of a run takes the anchor as its following token and predicts its successor
	pod::BatchRows crows;
	crows.count = catch_rows;
	int ctok[pod::MAX_BATCH], csrc[pod::MAX_BATCH], cout_rows[kMaxSlots], r = 0;
	for ( int j = 0; j < k; ++j ) {
		const int i = J[j];
		SlotState& S = st_[slots[i]];
		for ( int t = 0; t < S.pend_n; ++t, ++r ) {
			crows.slot[r] = slots[i];
			crows.pos[r] = S.pend_pos + t;
			ctok[r] = ( t < S.pend_n - 1 ) ? S.pend_tok[t] : anchors[i];
			csrc[r] = hist_off( slots[i] ) + S.hist_base + t;
		}
		cout_rows[j] = r - 1;
	}
	mc.mtp_gather_hist( csrc, catch_rows );
	mc.mtp_step_rows( crows, ctok, 4, 0, cout_rows, k );
	int cands[kMaxSlots][pod::MAX_BATCH], preds[pod::MAX_BATCH];
	mc.sample( preds, k );
	for ( int j = 0; j < k; ++j ) cands[j][0] = preds[j];

	// 2. chain the head on its own output for the rest of the block (the depth is the same for every slot: the survival estimate is shared)
	constexpr float kMinSurvival = 0.3f;
	float alive = 1.0f;
	int drafted = 1;
	int ident[kMaxSlots];
	for ( int j = 0; j < k; ++j ) ident[j] = j;
	for ( int step = 1; step < depth; ++step ) {
		alive *= cond_[step - 1];
		if ( alive < kMinSurvival ) break;
		pod::BatchRows drows;
		drows.count = k;
		int dtok[kMaxSlots];
		for ( int j = 0; j < k; ++j ) {
			drows.slot[j] = slots[J[j]];
			drows.pos[j] = poss[J[j]] - 1 + step;
			dtok[j] = cands[j][step - 1];
		}
		mc.mtp_step_rows( drows, dtok, 3, 0, ident, k );
		mc.sample( preds, k );
		for ( int j = 0; j < k; ++j ) cands[j][step] = preds[j];
		++drafted;
	}
	depth = drafted;

	// 3. one verify forward over [anchor, c0 .. c_{depth-1}] of every slot
	const int B = depth + 1, total = k * B;
	int vtok[pod::MAX_BATCH], vslot[pod::MAX_BATCH], vpos[pod::MAX_BATCH];
	bool any_sampled = false;
	for ( int j = 0; j < k; ++j ) {
		const int i = J[j];
		vtok[j * B] = anchors[i];
		for ( int t = 0; t < depth; ++t ) vtok[j * B + 1 + t] = cands[j][t];
		for ( int t = 0; t < B; ++t ) {
			vslot[j * B + t] = slots[i];
			vpos[j * B + t] = poss[i] + t;
		}
		any_sampled = any_sampled || ( sps[i] != nullptr && !sps[i]->greedy() );
	}
	mc.set_spec_snapshot( true );
	mc.forward( vtok, pod::BatchRows::batched( total, vslot, vpos ) );
	int vpred[pod::MAX_BATCH];
	Candidates cs[pod::MAX_BATCH];
	mc.sample( vpred, total );
	if ( any_sampled ) mc.sample_topk( total, cs );

	// 4. acceptance per slot (see speculate)
	for ( int j = 0; j < k; ++j ) {
		const int i = J[j];
		SpecResult& res = out[i];
		res = SpecResult();
		const bool sampled = sps[i] != nullptr && !sps[i]->greedy();
		ENGINE_ASSERT( !sampled || rngs[i] != nullptr, "MtpRunner::speculate_joint: sampling needs an Rng" );
		const int base = j * B;
		res.proposed = depth;
		for ( int t = 0; t < depth; ++t ) {
			++att_[t];
			if ( sampled ) {
				bool acc = false;
				const int tok = spec_step( cs[base + t], *sps[i], cands[j][t], *rngs[i], &acc );
				if ( !acc ) {
					res.emitted[res.emitted_count++] = tok;
					break;
				}
			}
			else if ( cands[j][t] != vpred[base + t] ) {
				res.emitted[res.emitted_count++] = vpred[base + t];
				break;
			}
			++hit_[t];
			res.emitted[res.emitted_count++] = cands[j][t];
			res.accepted++;
		}
		if ( res.accepted == depth ) {
			res.emitted[res.emitted_count++] = sampled ? sample_token( cs[base + depth], *sps[i], *rngs[i] ) : vpred[base + depth];
		}
		constexpr float kEma = 0.2f;
		for ( int t = 0; t <= res.accepted && t < depth; ++t ) cond_[t] = ( 1.0f - kEma ) * cond_[t] + kEma * ( t < res.accepted ? 1.0f : 0.0f );
		res.next_pos = poss[i] + 1 + res.accepted;
	}

	// 5. keep the committed rows' hidden for the next round's catch-up; repair the recurrent state of slots that rejected a draft
	for ( int j = 0; j < k; ++j ) {
		const int i = J[j];
		SlotState& S = st_[slots[i]];
		const SpecResult& res = out[i];
		S.hist_base = kHalf - S.hist_base;
		mc.mtp_save_hist( j * B, res.accepted + 1, hist_off( slots[i] ) + S.hist_base );
		S.pend_pos = poss[i];
		S.pend_n = res.accepted + 1;
		for ( int t = 0; t < res.accepted; ++t ) S.pend_tok[t] = cands[j][t];
		if ( res.accepted < depth ) mc.commit_spec_row( slots[i], j * B + res.accepted );
	}
	mc.set_spec_snapshot( false );
}

} // namespace engine::runtime
