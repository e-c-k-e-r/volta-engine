// LoopDetector (runtime/loop_detect, a port of exllamav3's generator/loop_detect.py) on synthetic token streams: what it must find (an exact
// loop, a loop whose copies differ by a few tokens) and what it must leave alone (varied text, text that reuses phrases, answer options
// transcribed from the prompt, a list of items that differ by an index, below the near-repetition length). No GPU. The port
// reproduces upstream's detections token for token (checked against loop_detect.py on 40 mixed streams); like upstream it reports random text
// over a handful of tokens as a loop.
#include "engine/runtime/loop_detect.h"
#include <cstdio>
#include <vector>

using engine::runtime::LoopDetector;

static uint32_t g_seed = 271828u;
static int rnd( int n ) { g_seed = g_seed * 1664525u + 1013904223u; return static_cast<int>( ( g_seed >> 8 ) % static_cast<uint32_t>( n ) ); }
static std::vector<int> random_tokens( int n ) { std::vector<int> v( n ); for ( int& t : v ) t = 100 + rnd( 30000 ); return v; }

struct Result { bool found; long long at; int period; };

static Result run( const std::vector<int>& prompt, const std::vector<int>& gen ) {
	LoopDetector d;
	d.prime( prompt.data(), static_cast<int>( prompt.size() ) );
	for ( int t : gen ) if ( d.feed( t ) ) break;
	return { d.detected(), d.detected_at(), d.period() };
}

static bool g_pass = true;
static void expect( const char* name, const Result& r, bool want, int period = 0 ) {
	bool ok = r.found == want;
	if ( ok && want && period > 0 ) ok = r.period >= period * 95 / 100 && r.period <= period * 105 / 100;
	std::printf( "  %-58s %s", name, r.found ? "loop" : "no loop" );
	if ( r.found ) std::printf( " at token %lld, period %d", r.at, r.period );
	std::printf( "%s\n", ok ? "" : "  FAIL" );
	g_pass &= ok;
}

int main() {
	const std::vector<int> prompt = random_tokens( 2000 );

	{   // exact repetition of a 200-token unit after 300 tokens of new text
		std::vector<int> gen = random_tokens( 300 ), unit = random_tokens( 200 );
		for ( int c = 0; c < 12; ++c ) gen.insert( gen.end(), unit.begin(), unit.end() );
		const Result r = run( prompt, gen );
		expect( "exact loop, period 200", r, true, 200 );
		if ( r.found && ( r.at < 300 + 1024 || r.at > 300 + 6 * 200 + 200 ) ) { std::printf( "  FAIL: found at %lld, expected after 4 copies and 1024 tokens\n", r.at ); g_pass = false; }
	}
	{   // near repetition: 1200-token unit, 2% of the tokens changed in every copy
		std::vector<int> gen = random_tokens( 500 ), unit = random_tokens( 1200 );
		for ( int c = 0; c < 12; ++c ) {
			std::vector<int> copy = unit;
			for ( int i = 0; i < 24; ++i ) copy[rnd( 1200 )] = 100 + rnd( 30000 );
			gen.insert( gen.end(), copy.begin(), copy.end() );
		}
		expect( "near loop, period 1200, 2% of each copy changed", run( prompt, gen ), true, 1200 );
	}
	expect( "20000 varied tokens", run( prompt, random_tokens( 20000 ) ), false );
	{   // a 50-token phrase reused every ~300 tokens with new text between
		std::vector<int> gen, phrase = random_tokens( 50 );
		for ( int c = 0; c < 60; ++c ) {
			const std::vector<int> fresh = random_tokens( 200 + rnd( 150 ) );
			gen.insert( gen.end(), fresh.begin(), fresh.end() );
			gen.insert( gen.end(), phrase.begin(), phrase.end() );
		}
		expect( "a phrase reused between new text (60 times)", run( prompt, gen ), false );
	}
	{   // the prompt lists 12 distinct items; the answer transcribes each one in the same frame
		std::vector<int> p2 = prompt, frame = random_tokens( 60 ), gen;
		std::vector<std::vector<int>> items;
		for ( int i = 0; i < 12; ++i ) { items.push_back( random_tokens( 150 ) ); p2.insert( p2.end(), items.back().begin(), items.back().end() ); }
		for ( int i = 0; i < 12; ++i ) { gen.insert( gen.end(), frame.begin(), frame.end() ); gen.insert( gen.end(), items[i].begin(), items[i].end() ); }
		expect( "12 prompt items transcribed in one frame", run( p2, gen ), false );
	}
	{   // 40 items of 120 tokens that differ only by an index token (a list, JSON objects): near-exact, 4800 tokens < 8192
		std::vector<int> gen = random_tokens( 200 ), item = random_tokens( 120 );
		for ( int i = 0; i < 40; ++i ) { std::vector<int> c = item; c[5] = 50 + i; gen.insert( gen.end(), c.begin(), c.end() ); }
		expect( "40 items differing by an index (4800 tokens)", run( prompt, gen ), false );
	}
	{   // the same list running on: past 8 copies and 8192 tokens it is a loop (the documented trade-off)
		std::vector<int> gen = random_tokens( 200 ), item = random_tokens( 120 );
		for ( int i = 0; i < 100; ++i ) { std::vector<int> c = item; c[5] = 50 + i % 40; gen.insert( gen.end(), c.begin(), c.end() ); }
		expect( "100 such items (12000 tokens)", run( prompt, gen ), true, 120 );
	}
	std::printf( "test_loop_detect: %s\n", g_pass ? "OK" : "FAIL" );
	return g_pass ? 0 : 1;
}
