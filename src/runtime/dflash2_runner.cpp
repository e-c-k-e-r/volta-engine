#include "engine/runtime/dflash2_runner.h"
#include "engine/core/log.h"
#include "engine/runtime/options.h"
#include "engine/backend/runtime.h"
#include "engine/runtime/cluster.h"
#include <chrono>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>

namespace engine::runtime {

namespace impl {

double now_ms() {
	return std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now().time_since_epoch() ).count();
}

// Geometry from the drafter checkpoint's own config.json (DFlashConfig's defaults describe the Qwen drafter).
// Missing keys keep the defaults.
void load_draft_config( const std::string& dir, pod::DFlashConfig& cfg ) {
	std::ifstream f( dir + "/config.json" );
	if ( !f ) return;
	const nlohmann::json j = nlohmann::json::parse( f, nullptr, false );
	if ( j.is_discarded() || !j.is_object() ) return;
	const auto geti = [&]( const nlohmann::json& o, const char* key, int& out ) {
		if ( o.contains( key ) && o[key].is_number_integer() ) out = o[key].get<int>();
	};
	geti( j, "hidden_size", cfg.hidden_size );
	geti( j, "intermediate_size", cfg.intermediate_size );
	geti( j, "num_attention_heads", cfg.num_q_heads );
	geti( j, "num_key_value_heads", cfg.num_kv_heads );
	geti( j, "head_dim", cfg.head_dim );
	geti( j, "num_hidden_layers", cfg.num_hidden_layers );
	geti( j, "num_target_layers", cfg.num_target_layers );
	geti( j, "sliding_window", cfg.sliding_window );
	if ( j.contains( "rms_norm_eps" ) && j["rms_norm_eps"].is_number() ) cfg.rms_norm_eps = j["rms_norm_eps"].get<float>();
	if ( j.contains( "rope_parameters" ) && j["rope_parameters"].is_object() && j["rope_parameters"].contains( "rope_theta" ) ) {
		cfg.rope_theta = j["rope_parameters"]["rope_theta"].get<float>();
	}
	if ( j.contains( "dflash_config" ) && j["dflash_config"].is_object() ) {
		const auto& d = j["dflash_config"];
		geti( d, "block_size", cfg.block_size );
		geti( d, "conv_group_size", cfg.conv_group_size );
		geti( d, "conv_kernel_size", cfg.conv_kernel_size );
		geti( d, "mask_token_id", cfg.mask_token_id );
		geti( d, "selector_rank", cfg.selector_rank );
		geti( d, "selector_top_k", cfg.selector_top_k );
		if ( d.contains( "target_layer_ids" ) && d["target_layer_ids"].is_array() ) {
			int n = 0;
			for ( const auto& v : d["target_layer_ids"] ) {
				if ( n < 8 && v.is_number_integer() ) cfg.target_layer_ids[n++] = v.get<int>();
			}
			cfg.num_target_layer_ids = n;
		}
	}
}

} // namespace impl

bool DFlash2Runner::init( ModelCluster& mc, const std::string& draft_dir ) {
	mc_ = &mc;
	for ( int d = 0; d < mc.tp(); ++d ) {
		ENGINE_ASSERT( mc.model( d ).draft_hooks().valid(), "DFlash2Runner: model exposes no draft hooks" );
	}
	const auto& meta = mc.metadata();
	pod::DFlashConfig cfg;
	impl::load_draft_config( draft_dir, cfg );
	ENGINE_ASSERT( cfg.hidden_size == meta.hidden_size, "DFlash2Runner: drafter hidden size does not match the target" );
	cfg.hidden_size = meta.hidden_size;
	cfg.vocab_size = meta.vocab_size;
	ENGINE_ASSERT( cfg.block_size == models::kDraftM, "DFlash2Runner: drafter block size != kDraftM" );
	cfg.block_size = models::kDraftM;
	// Full rotary over head_dim: the drafter checkpoint declares rope_type "default" with no
	// partial_rotary_factor, and a partial-rotation default would quietly poison every prediction.
	cfg.rope_dim_sections[0] = cfg.head_dim;
	cfg.rope_dim_sections[1] = 0;
	cfg.rope_dim_sections[2] = 0;
	cfg.rope_dim_sections[3] = 0;
	ENGINE_ASSERT( cfg.selector_top_k == models::kDraftTopK, "DFlash2Runner: selector_top_k != kDraftTopK" );

	tp_ = runtime::options().draft_tp ? mc.tp() : 1;
	bool loaded = true;
	for ( int d = 0; d < tp_; ++d ) {
		backend::set_device( d );
		spec_[d].set_candidate_ranks( mc.tp() );
		if ( tp_ > 1 ) spec_[d].set_ring( mc.ring( d ) );
		if ( !spec_[d].init( cfg, d, mc.arena( d ), mc.stream( d ), d, tp_ ) ) return false;
		loaded = spec_[d].load_weights( draft_dir.c_str(), mc.stream( d ) ) && loaded;
	}
	LOG_INFO( "DFlash2 speculator: %s (%s), drafter tp=%d\n", loaded ? "loaded" : "STANDALONE", draft_dir.c_str(), tp_ );
	return true;
}

void DFlash2Runner::shutdown() {
	for ( int d = 0; d < tp_; ++d ) {
		backend::set_device( d );
		spec_[d].destroy();
	}
	mc_ = nullptr;
	owner_ = -1;
}

void DFlash2Runner::claim( int slot ) {
	owner_ = slot;
	for ( int d = 0; d < tp_; ++d ) spec_[d].reset_context();
	for ( int i = 0; i < 8; ++i ) att_[i] = hit_[i] = 0;
	// cond_ survives: per-step survival is a property of the drafter against this target, not of the
	// sequence being decoded, so resetting it per request would re-learn it at full depth every time.
	mc_->run_ranks( [&]( int d ) {
		const int v = slot;
		const auto& hk = mc_->model( d ).draft_hooks();
		// Pageable source: staged before return, and the sync below orders it before any forward.
		backend::memcpy_async( hk.taps_owner, &v, sizeof( int ), backend::Copy::H2D, mc_->stream( d ) );
		backend::stream_sync( mc_->stream( d ) );
	} );
}

void DFlash2Runner::release( int slot ) {
	if ( owner_ != slot ) return;
	owner_ = -1;
	mc_->run_ranks( [&]( int d ) {
		const int v = -1;
		const auto& hk = mc_->model( d ).draft_hooks();
		backend::memcpy_async( hk.taps_owner, &v, sizeof( int ), backend::Copy::H2D, mc_->stream( d ) );
		backend::stream_sync( mc_->stream( d ) );
	} );
}

void DFlash2Runner::step_stats( int* attempts, int* hits ) const {
	for ( int i = 0; i < 8; ++i ) {
		attempts[i] = att_[i];
		hits[i] = hit_[i];
	}
}

SpecResult DFlash2Runner::speculate( int slot, int anchor, int pos, float depth_threshold,
	const SamplingParams* sp, Rng* rng ) {
	ENGINE_ASSERT( owner_ == slot, "DFlash2Runner::speculate: slot does not own the drafter context" );
	constexpr int M = models::kDraftM;
	constexpr int K = M - 1;
	ModelCluster& mc = *mc_;
	const auto& meta = mc.metadata();
	const backend::stream_t s0 = mc.stream( 0 );
	SpecResult res;
	const bool sampled = sp != nullptr && !sp->greedy();
	ENGINE_ASSERT( !sampled || rng != nullptr, "DFlash2Runner::speculate: sampling needs an Rng" );
	prof_last_ = SpecProfile::DRAFT;
	prof_t0_ = impl::now_ms();

	// 1. Drafter backbone over the block [anchor, mask x K] and the target's tapped context, then each rank's
	// head shard + local top-k over the drafter hidden. Every rank's candidate set is the top-k of its
	// vocabulary shard (exact: a rank's local top-k is a superset of the global top-k restricted to its shard).
	const auto& hk0 = mc.model( 0 ).draft_hooks();
	const size_t hid_bytes = static_cast<size_t>( M ) * mc.metadata().hidden_size * sizeof( backend::half_t );
	if ( tp_ > 1 ) {
		// Sharded drafter: all ranks run the backbone in lockstep and end with the same full hidden.
		mc.run_ranks( [&]( int d ) {
			const auto& hk = mc.model( d ).draft_hooks();
			spec_[d].draft_backbone( *hk.embedding, anchor, hk.taps, pos, hk.taps_cap, mc.stream( d ) );
			backend::memcpy_async( hk.lm_in, spec_[d].draft_hidden().data, hid_bytes, backend::Copy::D2D, mc.stream( d ) );
			mc.model( d ).draft_head_topk( M, mc.stream( d ) );
			backend::stream_sync( mc.stream( d ) );
		} );
	}
	else {
		backend::set_device( 0 );
		spec_[0].draft_backbone( *hk0.embedding, anchor, hk0.taps, pos, hk0.taps_cap, s0 );
		// Peer the hidden rows to every rank, then run each rank's head shard on its own worker/stream.
		const pod::TensorView& dh = spec_[0].draft_hidden();
		for ( int d = 0; d < mc.tp(); ++d ) {
			backend::peer_copy_async( mc.model( d ).draft_hooks().lm_in, d, dh.ptr<const backend::half_t>(), 0, hid_bytes, s0 );
		}
		backend::stream_sync( s0 );
		mc.run_ranks( [&]( int d ) {
			mc.model( d ).draft_head_topk( M, mc.stream( d ) );
			backend::stream_sync( mc.stream( d ) );
		} );
	}

	// Diagnostic: ENGINE_DEBUG_DUMP_DRAFT_ROUND=<round> writes that round's inputs and the drafter hidden to
	// /tmp/draft_dump.bin for comparison against an independent implementation. Header (int32): pos, anchor,
	// hidden, taps_cap; then the tap ring rows for positions [0, pos) as FP16 [pos, 5 * hidden]
	// (only valid while pos <= taps_cap), then the drafter hidden FP16 [M, hidden].
	{
		const int dump_round = runtime::options().dump_draft_round;
		static int round_no = 0;
		if ( dump_round >= 0 && round_no++ == dump_round ) {
			const int H = mc.metadata().hidden_size;
			ENGINE_ASSERT( pos <= hk0.taps_cap, "dump: context longer than the tap ring" );
			std::vector<backend::half_t> taps( static_cast<size_t>( pos ) * 5 * H ), hid( static_cast<size_t>( M ) * H );
			backend::set_device( 0 );
			backend::stream_sync( s0 );
			backend::memcpy_sync( taps.data(), hk0.taps, taps.size() * sizeof( backend::half_t ), backend::Copy::D2H );
			backend::memcpy_sync( hid.data(), spec_[0].draft_hidden().data, hid.size() * sizeof( backend::half_t ), backend::Copy::D2H );
			FILE* f = std::fopen( "/tmp/draft_dump.bin", "wb" );
			const int hdr[4] = { pos, anchor, H, hk0.taps_cap };
			std::fwrite( hdr, sizeof( int ), 4, f );
			std::fwrite( taps.data(), sizeof( backend::half_t ), taps.size(), f );
			std::fwrite( hid.data(), sizeof( backend::half_t ), hid.size(), f );
			std::fclose( f );
			LOG_INFO( "[dump] round %d: pos %d anchor %d\n", dump_round, pos, anchor );
		}
	}

	// Bring the small [M, k] results to rank 0 and merge.
	const size_t val_bytes = static_cast<size_t>( M ) * models::kDraftTopK * sizeof( float );
	const size_t idx_bytes = static_cast<size_t>( M ) * models::kDraftTopK * sizeof( int );
	backend::set_device( 0 );
	for ( int d = 0; d < mc.tp(); ++d ) {
		const auto& hk = mc.model( d ).draft_hooks();
		backend::peer_copy_async( spec_[0].merged_cand_values_ptr( d ), 0, hk.cand_vals, d, val_bytes, s0 );
		backend::peer_copy_async( spec_[0].merged_cand_indices_ptr( d ), 0, hk.cand_idx, d, idx_bytes, s0 );
	}
	engine::spec::dflash2_topk_merge(
		spec_[0].merged_cand_values_ptr( 0 ), spec_[0].merged_cand_indices_ptr( 0 ),
		mc.tp(), M, models::kDraftTopK,
		spec_[0].merged_final_values_ptr(), spec_[0].merged_final_indices_ptr(), M, s0 );

	// 3. Selector gate + bilinear lattice walk over the merged candidates.
	mark( SpecProfile::WALK );
	spec_[0].project_gate( M, s0 );
	spec_[0].set_walk_anchor( anchor, s0 );
	spec_[0].draft_select( spec_[0].merged_final_values_ptr(), spec_[0].merged_final_indices_ptr(), K, s0 );
	int h_draft[M];
	backend::memcpy_async( h_draft, spec_[0].d_draft_tokens(), sizeof( h_draft ), backend::Copy::D2H, s0 );
	backend::stream_sync( s0 );
	backend::check_last_error();

	// Depth: the whole block by default (what DFlash2 does natively). depth_threshold > 0 enables the
	// survival-EMA truncation; it is a throughput option, off by default so acceptance is measurable.
	int active = K;
	if ( depth_threshold > 0.0f ) {
		active = 1;
		float run = cond_[0];
		for ( int k = 1; k < K && run >= depth_threshold; ++k ) {
			run *= cond_[k];
			active = k + 1;
		}
	}
	active = MIN( MAX( active, 1 ), meta.max_seq_len - pos - 2 );
	if ( active < 1 ) {
		// No room to verify a block: fall back to the target's own prediction for the anchor.
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
	const int* cands = h_draft + 1;

	// 4. Batched verify of [anchor, c1..c_active] at positions [pos, pos + active]. Row 0 re-feeds the
	// anchor (exllamav3 generator/job.py:384), so one forward yields both the draft verification and
	// the anchor's own successor: row j's logits predict position pos+j+1, so cands[k] is checked
	// against preds[k] with no shift.
	int block[pod::MAX_BATCH];
	block[0] = anchor;
	for ( int i = 0; i < active; ++i ) block[1 + i] = cands[i];

	mark( SpecProfile::CHECKPOINT );
	mc.set_spec_snapshot( true );
	mark( SpecProfile::VERIFY );
	mc.forward( block, active + 1, pos, slot );
	mark( SpecProfile::SAMPLE );
	int preds[pod::MAX_BATCH];
	Candidates cs[pod::MAX_BATCH];
	if ( sampled ) mc.sample_topk( active + 1, cs );
	else mc.sample( preds, active + 1 );
	mark( SpecProfile::REPAIR );
	++prof_.rounds;

	// 5. Acceptance. Greedy: the draft must equal the target's argmax. Sampled: accept with the target's
	// probability of the draft token (speculative sampling against a point-mass proposal), else draw the
	// correction from the target's distribution with that token removed.
	for ( int k = 0; k < active; ++k ) {
		++att_[k];
		if ( sampled ) {
			bool acc = false;
			const int tok = spec_step( cs[k], *sp, cands[k], *rng, &acc );
			if ( !acc ) {
				res.emitted[res.emitted_count++] = tok; // correction token
				break;
			}
		}
		else if ( cands[k] != preds[k] ) {
			res.emitted[res.emitted_count++] = preds[k]; // correction token
			break;
		}
		++hit_[k];
		res.emitted[res.emitted_count++] = cands[k];
		res.accepted++;
	}
	if ( diag_on_ && !sampled ) {
		constexpr int kTop = models::kDraftTopK;
		float hv[M * kTop];
		int hi[M * kTop];
		backend::set_device( 0 );
		backend::memcpy_sync( hv, spec_[0].merged_final_values_ptr(), sizeof( hv ), backend::Copy::D2H );
		backend::memcpy_sync( hi, spec_[0].merged_final_indices_ptr(), sizeof( hi ), backend::Copy::D2H );
		for ( int k = 0; k < active && k <= res.accepted; ++k ) {
			// Draft token k is chosen from candidate row k + 1 (row 0 is the anchor and is never scored).
			const int row = k + 1;
			int best = 0;
			bool any = false;
			for ( int j = 0; j < kTop; ++j ) {
				if ( hv[row * kTop + j] > hv[row * kTop + best] ) best = j;
				any = any || ( hi[row * kTop + j] == preds[k] );
			}
			++diag_.seen[k];
			diag_.top1[k] += ( hi[row * kTop + best] == preds[k] ) ? 1 : 0;
			diag_.in_topk[k] += any ? 1 : 0;
			diag_.walk[k] += ( cands[k] == preds[k] ) ? 1 : 0;
		}
	}
	// Per-step survival EMA, observed only where the step was proposed.
	constexpr float kEma = 0.25f;
	for ( int k = 0; k < res.proposed; ++k ) {
		cond_[k] = ( 1.0f - kEma ) * cond_[k] + kEma * ( k < res.accepted ? 1.0f : 0.0f );
	}
	// Bonus token: the target's prediction just past the accepted prefix.
	if ( res.accepted == active ) {
		res.emitted[res.emitted_count++] = sampled ? sample_token( cs[active], *sp, *rng ) : preds[active];
	}

	// 6. State repair. The next round expects exactly accepted + 1 rows consumed ([anchor, c1..c_accepted]),
	// which is the snapshot published after row `accepted` of the verify forward. A full accept already
	// leaves the live state there, so only a partial accept copies it in. No re-forward.
	if ( res.accepted < active ) {
		++prof_.commits;
		mc.commit_spec_row( slot, res.accepted );
	}
	mc.set_spec_snapshot( false );
	res.next_pos = pos + 1 + res.accepted;
	// Close the round: account the trailing segment and rewind the cursor.
	if ( profile_on_ ) {
		const double now = impl::now_ms();
		prof_.ms[prof_last_] += now - prof_t0_;
		prof_last_ = SpecProfile::DRAFT;
		prof_t0_ = now;
	}
	return res;
}

// Closes the segment that began at the previous mark, attributing it to the stage just finished, and
// arms the clock for `stage`. Only accumulates when profiling is on.
void DFlash2Runner::mark( SpecProfile::Stage stage ) {
	if ( !profile_on_ ) return;
	const double now = impl::now_ms();
	if ( stage != prof_last_ ) prof_.ms[prof_last_] += now - prof_t0_;
	prof_last_ = stage;
	prof_t0_ = now;
}

void DFlash2Runner::profile_dump() {
	static const char* names[SpecProfile::COUNT] = {
		"draft(backbone+head+merge)", "walk(gate+lattice)", "snapshot_on", "verify_fwd", "sample(sync+merge)", "accept+commit"
	};
	double total = 0.0;
	for ( int i = 0; i < SpecProfile::COUNT; ++i ) total += prof_.ms[i];
	const int n = prof_.rounds;
	LOG_INFO( "\n--- speculative round profile: %d rounds, %d commits (%.0f%%) ---\n",
		n, prof_.commits, n ? 100.0 * prof_.commits / n : 0.0 );
	for ( int i = 0; i < SpecProfile::COUNT; ++i ) {
		LOG_INFO( "%-26s %8.2f ms total  %7.3f ms/round  %5.1f%%\n", names[i], prof_.ms[i],
			n ? prof_.ms[i] / n : 0.0, total > 0.0 ? 100.0 * prof_.ms[i] / total : 0.0 );
	}
	LOG_INFO( "%-26s %8.2f ms total  %7.3f ms/round\n", "TOTAL", total, n ? total / n : 0.0 );
	std::fflush( stdout );
	prof_ = SpecProfile{};
}

} // namespace engine::runtime
