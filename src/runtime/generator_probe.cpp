// Steering probe: per request (steer_probe), before the real prefill. The prompt is prefilled unedited; a few forced openings after it
// (accepting, declining, plain, hedged) give the residual states the directions are built from; a short greedy look-ahead says whether this
// prompt drifts towards the declining side. Past the gate the slot gets direction clamps (state - relu(state . dir - ref) * dir) on the edit
// layers, active on the prompt and the generated rows, and the real prefill then runs with them. The prompt's KV is built once for the probe
// (the openings run as short blocks that are rolled back) and once for real.
#include "engine/runtime/generator.h"
#include "engine/runtime/steer_probe.h"
#include "engine/core/log.h"
#include "engine/runtime/options.h"
#include "engine/models/steer_defaults.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
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

// The forced openings of the four kinds: the built-in ones, or those of the steer_openings file ( `kind: text` per line )
struct Openings {
	std::vector<std::string> kind[4];   // accept, decline, plain, hedged
};

const Openings& openings() {
	static Openings built_in, loaded;
	static std::string loaded_from;
	static bool init = false;
	if ( !init ) {
		init = true;
		for ( const char* o : kAccept ) built_in.kind[0].push_back( o );
		for ( const char* o : kDecline ) built_in.kind[1].push_back( o );
		for ( const char* o : kPlain ) built_in.kind[2].push_back( o );
		for ( const char* o : kHedged ) built_in.kind[3].push_back( o );
	}
	const std::string& path = runtime::options().steer_openings;
	if ( path.empty() ) return built_in;
	if ( path != loaded_from ) {
		loaded_from = path;
		loaded = built_in;
		Openings file;
		std::ifstream in( path );
		ENGINE_ASSERT( in.good(), "steer_openings: cannot read the file" );
		static const char* const names[4] = { "accept:", "decline:", "plain:", "hedged:" };
		for ( std::string line; std::getline( in, line ); ) {
			for ( int k = 0; k < 4; ++k ) {
				const size_t len = std::strlen( names[k] );
				if ( line.compare( 0, len, names[k] ) != 0 ) continue;
				size_t b = len;
				while ( b < line.size() && line[b] == ' ' ) ++b;
				if ( b < line.size() ) file.kind[k].push_back( line.substr( b ) );
			}
		}
		for ( int k = 0; k < 4; ++k )
			if ( !file.kind[k].empty() ) loaded.kind[k] = std::move( file.kind[k] );
	}
	return loaded;
}

// An option that says "auto" (or nothing) takes the model's value
double option_or( const std::string& text, double fallback ) {
	if ( text.empty() || text == "auto" ) return fallback;
	char* end = nullptr;
	const double v = std::strtod( text.c_str(), &end );
	return end != text.c_str() && *end == '\0' ? v : fallback;
}

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
	const engine::models::SteerDefaults sd = engine::models::steer_defaults( config_.model_type );
	const std::vector<int>& prompt = slot.req.prompt_tokens;
	const int n = static_cast<int>( prompt.size() );
	int b0, b1, e0, e1, sfirst = 0, scount = 0;
	cluster_.steer_range( &sfirst, &scount );
	if ( !layer_range( opt.steer_band, NL / 3, 2 * NL / 3, NL, &b0, &b1 ) || !layer_range( !slot.req.probe_edit_layers.empty() ? slot.req.probe_edit_layers : opt.steer_edit_layers, sd.lo_num * NL / sd.lo_den, sd.hi_num * NL / sd.hi_den, NL, &e0, &e1 ) ||
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
	auto states_of = [&]( const std::vector<std::string>& list ) {
		std::vector<std::vector<float>> out;
		for ( const std::string& text : list ) {
			const std::vector<int> ids = tokenizer_.encode( text, false );
			cluster_.forward( ids.data(), pod::BatchRows::causal( static_cast<int>( ids.size() ), n, s, pod::BatchRows::LOGITS_NONE ) );   // no head: a block above 16 rows would dequantize the whole LM head
			out.push_back( read_state() );
			cluster_.rollback( s );
		}
		return out;
	};
	const Openings& ops = openings();
	const auto accept = states_of( ops.kind[0] ), decline = states_of( ops.kind[1] );
	const auto plain = states_of( ops.kind[2] ), hedged = states_of( ops.kind[3] );
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
	const float threshold = pick( slot.req.probe_threshold, option_or( opt.steer_gate_threshold, sd.gate_threshold ) ), edit_strength = pick( slot.req.probe_edit_strength, option_or( opt.steer_edit_strength, sd.edit_strength ) ),
		hedge_strength = pick( slot.req.probe_hedge_strength, opt.steer_hedge_strength ), push = pick( slot.req.probe_push, opt.steer_push );
	const bool fired = score > threshold;
	slot.metrics.gate_score = score;
	slot.metrics.gate_fired = fired;

	cluster_.reset_slot( s );   // the real prefill starts from nothing
	if ( !fired && push == 0.0f ) {
		steer_admit( s );       // back to what the request asked for without the probe
	}
	else {
		std::vector<engine::modules::SteerEntry> entries;
		for ( int l = e0; l <= e1; ++l ) {
			const int bi = l - b0;
			if ( fired ) {
				engine::modules::SteerEntry c;
				c.layer = l;
				c.kind = engine::modules::STEER_CLAMP;
				c.vec = axes[bi].dir;
				c.par = axes[bi].pa - edit_strength * axes[bi].sd;
				entries.push_back( std::move( c ) );
			}
			if ( push != 0.0f ) {
				// towards the declining side: k gaps along the axis (gap = mean projection of the declining openings minus the accepting ones)
				engine::modules::SteerEntry a;
				a.layer = l;
				a.kind = engine::modules::STEER_ADD;
				a.vec = axes[bi].dir;
				a.par = push * ( axes[bi].pb - axes[bi].pa );
				entries.push_back( std::move( a ) );
			}
			if ( fired && push == 0.0f && hedge_strength >= 0.0f ) {   // at most kSteerEntries edits per layer: a push takes the hedge clamp's place
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
	LOG_INFO( "[steer] req %llu gate %s score=%.3f (threshold %.2f; clamp layers %d-%d strength %g; push %g; %.0f ms)\n", static_cast<unsigned long long>( slot.req.req_id ), fired ? "fired" : "clean", score,
		threshold, e0, e1, edit_strength, push, std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t_start ).count() );
}

} // namespace engine::runtime
