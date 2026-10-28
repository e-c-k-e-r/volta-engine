// Steering probe: per request (steer_probe), before the real prefill. The prompt is prefilled unedited; a few forced openings after it
// (accepting, declining, plain, hedged) give the residual states the directions are built from; a short greedy look-ahead says whether this
// prompt drifts towards the declining side. Past the gate the slot gets direction clamps (state - relu(state . dir - ref) * dir) on the edit
// layers, active on the prompt and the generated rows, and the real prefill then runs with them. The prompt's KV is built once for the probe
// (the openings run as short blocks that are rolled back) and once for real.
#include "engine/runtime/generator.h"
#include "engine/runtime/steer_probe.h"
#include "engine/core/log.h"
#include "engine/runtime/options.h"
#include <chrono>
#include <climits>
#include <cmath>

namespace engine::runtime {

namespace {

// Openings forced after the prompt's generation prefix. English and short by design; the sets only need to span the behaviour, not to be exhaustive.
const char* const kAccept[] = { "Sure, here is", "Certainly! Here is", "Of course. Here are", "Absolutely, here's", "Here is" };
const char* const kDecline[] = { "I'm sorry, but I can't help with that", "I cannot assist with that request", "I can't provide that",
	"Sorry, but I won't be able to help with this", "I'm unable to help with that", "I apologize, but I can't assist with this",
	"I won't help with that", "As an AI, I can't help with that" };
const char* const kPlain[] = { "Here is the answer:", "Sure. Step 1:", "Here are the steps:", "The answer is" };
const char* const kHedged[] = { "Here is the answer, but note that this may be risky and you should", "Sure, but please be aware that",
	"Here is, with the caveat that", "I can help; however, please note that this is for educational purposes only and" };

constexpr int kLookahead = 24;

bool layer_range( const std::string& spec, int lo_default, int hi_default, int num_layers, int* lo, int* hi ) {
	*lo = lo_default;
	*hi = hi_default;
	return spec.empty() || engine::modules::parse_steer_layers( spec, num_layers, lo, hi );
}

}   // namespace

void Generator::probe_slot( int s ) {
	Slot& slot = slots_[s];
	const auto& opt = runtime::options();
	const auto t_start = std::chrono::steady_clock::now();
	const int NL = cluster_.metadata().num_layers, H = cluster_.metadata().hidden_size;
	const std::vector<int>& prompt = slot.req.prompt_tokens;
	const int n = static_cast<int>( prompt.size() );
	int b0, b1, e0, e1, sfirst = 0, scount = 0;
	cluster_.steer_range( &sfirst, &scount );
	if ( !layer_range( opt.steer_band, NL / 3, 2 * NL / 3, NL, &b0, &b1 ) || !layer_range( !slot.req.probe_edit_layers.empty() ? slot.req.probe_edit_layers : opt.steer_edit_layers, 17 * NL / 32, 21 * NL / 32, NL, &e0, &e1 ) ||
		e0 < b0 || e1 > b1 || e0 < sfirst || e1 >= sfirst + scount ) {
		LOG_ERROR( "[steer] probe skipped: the edit layers %d-%d must lie inside the band and inside steer_layers %d-%d\n", e0, e1, sfirst, sfirst + scount - 1 );
		return;
	}
	if ( n + kLookahead + 64 >= cluster_.metadata().max_seq_len ) {
		LOG_ERROR( "[steer] probe skipped: the prompt leaves no room for the openings\n" );
		return;
	}
	std::vector<int> band;
	for ( int l = b0; l <= b1; ++l ) band.push_back( l );
	const int nb = static_cast<int>( band.size() );

	cluster_.steer_control( s, 0, n, 0 );   // the probe reads the unedited model
	cluster_.reset_slot( s );
	int cur = cluster_.prefill( prompt.data(), n, s );
	cluster_.checkpoint( s );
	cluster_.steer_capture( band, true );
	auto read_state = [&]() { std::vector<float> st; ENGINE_ASSERT( cluster_.steer_read( st ), "probe: capture failed" ); return st; };
	auto states_of = [&]( const char* const* openings, size_t count ) {
		std::vector<std::vector<float>> out;
		for ( size_t i = 0; i < count; ++i ) {
			const std::vector<int> ids = tokenizer_.encode( openings[i], false );
			cluster_.forward( ids.data(), pod::BatchRows::causal( static_cast<int>( ids.size() ), n, s, pod::BatchRows::LOGITS_NONE ) );   // no head: a block above 16 rows would dequantize the whole LM head
			out.push_back( read_state() );
			cluster_.rollback( s );
		}
		return out;
	};
	const auto accept = states_of( kAccept, std::size( kAccept ) ), decline = states_of( kDecline, std::size( kDecline ) );
	const auto plain = states_of( kPlain, std::size( kPlain ) ), hedged = states_of( kHedged, std::size( kHedged ) );
	// look-ahead: the states of the first tokens the unedited model would write
	std::vector<std::vector<float>> ahead;
	for ( int i = 0, pos = n; i < kLookahead && !cluster_.is_eos( cur ); ++i, ++pos ) {
		cluster_.forward( &cur, 1, pos, s );
		ahead.push_back( read_state() );
		cur = cluster_.sample_one();
	}
	cluster_.steer_capture( {}, false );

	auto set_at = [&]( const std::vector<std::vector<float>>& v, int bi ) {
		StateSet r;
		for ( const auto& st : v ) r.push_back( st.data() + static_cast<size_t>( bi ) * H );
		return r;
	};
	std::vector<ProbeAxis> axes( nb );
	for ( int bi = 0; bi < nb; ++bi ) axes[bi] = probe_axis( set_at( accept, bi ), set_at( decline, bi ), H, true );
	float score = 0.0f;
	for ( const auto& st : ahead ) {
		float f = 0.0f;
		for ( int bi = 0; bi < nb; ++bi ) f += probe_fraction( axes[bi], st.data() + static_cast<size_t>( bi ) * H, H );
		score = std::max( score, f / nb );
	}
	auto pick = [&]( float request, double option ) { return std::isnan( request ) ? static_cast<float>( option ) : request; };
	const float threshold = pick( slot.req.probe_threshold, opt.steer_gate_threshold ), edit_strength = pick( slot.req.probe_edit_strength, opt.steer_edit_strength ),
		hedge_strength = pick( slot.req.probe_hedge_strength, opt.steer_hedge_strength );
	const bool fired = score > threshold;
	slot.metrics.gate_score = score;
	slot.metrics.gate_fired = fired;

	cluster_.reset_slot( s );   // the real prefill starts from nothing
	if ( !fired ) {
		steer_admit( s );       // back to what the request asked for without the probe
	}
	else {
		std::vector<engine::modules::SteerEntry> entries;
		for ( int l = e0; l <= e1; ++l ) {
			const int bi = l - b0;
			engine::modules::SteerEntry c;
			c.layer = l;
			c.kind = engine::modules::STEER_CLAMP;
			c.vec = axes[bi].dir;
			c.par = axes[bi].pa - edit_strength * axes[bi].sd;
			entries.push_back( std::move( c ) );
			if ( hedge_strength >= 0.0f ) {
				const ProbeAxis hx = probe_axis( set_at( plain, bi ), set_at( hedged, bi ), H, false );
				engine::modules::SteerEntry d;
				d.layer = l;
				d.kind = engine::modules::STEER_CLAMP;
				d.vec = hx.dir;
				d.par = hx.pa - hedge_strength * ( hx.pb - hx.pa );
				entries.push_back( std::move( d ) );
			}
		}
		cluster_.steer_set( s, entries );
		int scope = engine::modules::STEER_ALL;
		if ( !slot.req.steer_scope.empty() ) parse_steer_scope( slot.req.steer_scope, &scope );   // a request may restrict the clamps (prompt, generated, ...)
		slot.steer_scope = scope;
		slot.steer_prompt = ( scope & engine::modules::STEER_PROMPT ) != 0;
		slot.steer_think_open = slot.in_reasoning;
		cluster_.steer_control( s, slot.steer_scope, n, slot.in_reasoning ? INT_MAX : 0 );
	}
	LOG_INFO( "[steer] req %llu gate %s score=%.3f (threshold %.2f, %.0f ms)\n", static_cast<unsigned long long>( slot.req.req_id ), fired ? "fired" : "clean", score,
		threshold, std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t_start ).count() );
}

} // namespace engine::runtime
