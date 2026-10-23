// The dashboard's frame renderer: every tab draws from a full /api/stats document and from an empty or partial one, at normal and tiny
// terminal sizes, without a crash and with the expected rows; keys move between tabs and scroll within the clamps.
#include "engine/server/tui/term.h"
#include "engine/server/tui/view.h"
#include <cstdio>
#include <string>

using namespace engine::server::tui;

static int g_failed = 0;
#define CHECK( cond, ... ) do { if ( !( cond ) ) { ++g_failed; std::printf( "[FAIL] " __VA_ARGS__ ); std::printf( "\n" ); } } while ( 0 )

static int count_rows( const std::string& frame ) {
	int n = 1;
	for ( char c : frame ) n += c == '\n';
	return n;
}

int main() {
	const nlohmann::json full = nlohmann::json::parse( R"({
		"server":{"model":"m","model_type":"qwen4_exp","uptime_s":3725,"slots":2,"max_context":32768,"tp":4,"queued":0,"speculation":true,"speculator":"dflash2","kv_format":"fp8"},
		"totals":{"requests":10,"completed":9,"tokens_out":1000,"prompt_tokens":5000,"cached_tokens":2500,"finished":{"stop":8,"length":1,"stop_string":0,"cancelled":0}},
		"averages":{"ttft_ms":300,"prefill_tok_s":5000,"decode_tok_s":100},
		"latency":{"gap_ms":[5,8,20,40,90],"ttft_ms":[200,400,900]},
		"speculation":{"rounds":10,"accept_rate":0.5,"tokens_per_round":2.5},
		"slots":[{"slot":0,"active":true,"phase":"decode","req_id":5,"pos":100,"prompt_tokens":80,"cached_tokens":10,"generated":20,"max_tokens":256,"age_s":3,"decode_tok_s":40},
		         {"slot":1,"active":false,"phase":"idle"}],
		"recent":[{"req_id":4,"age_s":9,"prompt_tokens":10,"cached_tokens":0,"generated":5,"queue_ms":1,"ttft_ms":50,"prefill_ms":40,"total_ms":200,"decode_tok_s":30,"reason":"stop"}],
		"gpus":[{"index":0,"name":"gpu","power_w":100,"power_limit_w":150,"energy_kj":1,"temp_c":60,"util_gpu":50,"util_mem":10,"sm_mhz":1300,"mem_mhz":877,"mem_used":1073741824,"mem_total":34359738368,"arena_bytes":0,"throttle":["sw_power_cap"]}],
		"host":{"cpu_pct":10,"mem_used":1073741824,"mem_total":68719476736,"rss":1073741824,"load1":1.5}})" );
	const nlohmann::json partial = nlohmann::json::parse( R"({"server":{"model":"m"}})" );
	const std::vector<std::string> log = { "[*] one", "[!] two" };

	const int sizes[][2] = { { 40, 120 }, { 24, 80 }, { 6, 20 }, { 3, 5 } };
	for ( const nlohmann::json* doc : { &full, &partial, static_cast<const nlohmann::json*>( nullptr ) } ) {
		for ( int tab = 0; tab < TAB_COUNT; ++tab ) {
			for ( const auto& sz : sizes ) {
				UiState ui;
				ui.tab = tab;
				ViewData d;
				d.stats = doc;
				d.connected = doc != nullptr;
				d.error = "refused";
				d.source = "host:8000";
				d.log = &log;
				const std::string frame = render_frame( d, ui, sz[0], sz[1] );
				const int want = sz[0] < 6 ? 6 : sz[0];
				CHECK( count_rows( frame ) == want, "tab %d size %dx%d: %d rows, expected %d", tab, sz[0], sz[1], count_rows( frame ), want );
			}
		}
	}

	UiState ui;
	ViewData d;
	d.stats = &full;
	d.connected = true;
	std::string frame = render_frame( d, ui, 40, 120 );
	CHECK( frame.find( "dflash2" ) != std::string::npos, "overview shows the speculator" );
	CHECK( frame.find( "from cache" ) != std::string::npos, "overview shows the cached share" );
	ui.tab = TAB_GPUS;
	CHECK( render_frame( d, ui, 40, 120 ).find( "sw_power_cap" ) != std::string::npos, "gpu tab lists the throttle reason" );
	ui.tab = TAB_SLOTS;
	CHECK( render_frame( d, ui, 40, 120 ).find( "#5" ) != std::string::npos, "slot tab lists the running request" );

	UiState k;
	CHECK( apply_key( k, '3', 10 ) && k.tab == TAB_GPUS, "digit selects a tab" );
	CHECK( apply_key( k, KEY_TAB, 10 ) && k.tab == TAB_LOG, "tab key advances" );
	CHECK( apply_key( k, KEY_TAB, 10 ) && k.tab == TAB_OVERVIEW, "tab key wraps" );
	CHECK( !apply_key( k, 'q', 10 ), "q quits" );
	UiState s;
	s.tab = TAB_SLOTS;
	ViewData sd;
	sd.stats = &full;
	sd.connected = true;
	apply_key( s, KEY_END, 10 );
	render_frame( sd, s, 8, 100 );
	CHECK( s.scroll[TAB_SLOTS] >= 0 && s.scroll[TAB_SLOTS] < 1000, "scroll is clamped to the body (%d)", s.scroll[TAB_SLOTS] );

	std::printf( g_failed ? "tui view: %d failure(s)\n" : "tui view: ok\n", g_failed );
	return g_failed ? 1 : 0;
}
