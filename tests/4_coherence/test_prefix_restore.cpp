// Tier 2: a slot restored from a sequence snapshot continues exactly like a cold prefill. Prefills the first `cut`
// tokens, snapshots the slot (recurrent state + KV rows), restores into other slots and prefills only the tail; the
// next-token candidates and 16 greedy continuation tokens must match the cold run. An aligned cut (a multiple of the
// prefill chunk) reproduces the cold chunk boundaries and must match bit for bit.
//   test_prefix_restore [--tokens N]
#include "engine/runtime/test_driver.h"
#include "engine/runtime/prefix_cache.h"
#include "engine/loader/tokenizer.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace engine::runtime;

static int g_proposed = 0, g_accepted = 0;

// Greedy continuation of `count` tokens; speculative rounds (DFlash2) when the cluster has a drafter that owns the slot
static std::vector<int> decode( ModelCluster& mc, int first, int pos, int slot, int count ) {
	std::vector<int> out = { first };
	int tok = first;
	while ( static_cast<int>( out.size() ) < count ) {
		if ( mc.spec_owns( slot ) ) {
			const SpecResult r = mc.speculate( slot, tok, pos );
			g_proposed += r.proposed;
			g_accepted += r.accepted;
			pos = r.next_pos;
			for ( int i = 0; i < r.emitted_count && static_cast<int>( out.size() ) < count; ++i ) {
				tok = r.emitted[i];
				out.push_back( tok );
			}
		}
		else {
			mc.forward( &tok, 1, pos++, slot );
			tok = mc.sample_one();
			out.push_back( tok );
		}
	}
	return out;
}

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	int n_tokens = 6000;
	bool spec = false;
	for ( int i = 1; i < argc; ++i ) {
		if ( !std::strcmp( argv[i], "--spec" ) ) spec = true;
		if ( !std::strcmp( argv[i], "--tokens" ) && i + 1 < argc ) n_tokens = std::atoi( argv[++i] );
	}
	engine::loader::Tokenizer tok;
	ENGINE_ASSERT( tok.load( test::QWEN_DIR ), "tokenizer load failed" );
	std::vector<int> ids = { 248045, 846, 198 };
	for ( int i = 0; static_cast<int>( ids.size() ) < n_tokens; ++i ) {
		const std::string s = "Note " + std::to_string( i ) + ": the weather in district " + std::to_string( i * 7 % 101 ) + " was mild on day " + std::to_string( i % 365 ) + ". ";
		const std::vector<int> t = tok.encode( s, false );
		ids.insert( ids.end(), t.begin(), t.end() );
	}
	ids.resize( n_tokens );
	const std::vector<int> end = { 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 };
	ids.insert( ids.end(), end.begin(), end.end() );
	const int n = static_cast<int>( ids.size() );

	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = "qwen3_5";
	cfg.model_dir = test::QWEN_DIR;
	if ( spec ) cfg.draft_model_dir = test::QWEN_DRAFT_DIR;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	ENGINE_ASSERT( mc.snapshot_bytes( 0, 1024 ) > 0, "the model cannot snapshot" );
	mc.warm_graphs( 8 );
	const int chunk = mc.metadata().max_rows;
	std::printf( "[*] prompt %d tokens, prefill chunk %d, snapshot of 4096 tokens = %.1f MiB per rank\n", n, chunk, mc.snapshot_bytes( 0, 4096 ) / 1048576.0 );

	// Cold reference
	mc.reset_slot( 0 );
	Candidates cold_c;
	const int cold_first = mc.prefill( ids.data(), n, 0, 0, &cold_c );
	const std::vector<int> cold = decode( mc, cold_first, n, 0, 48 );
	std::printf( "[*] cold run: %s, proposed %d accepted %d\n", spec ? "speculative" : "plain", g_proposed, g_accepted );
	g_proposed = g_accepted = 0;

	int failures = 0;
	for ( int cut : { chunk * ( ( n / 2 ) / chunk ), n / 2 + 37 } ) {
		const bool aligned = cut % chunk == 0;
		mc.reset_slot( 0 );
		mc.prefill( ids.data(), cut, 0 );
		ModelCluster::SeqSnapshot snap;
		auto t0 = std::chrono::steady_clock::now();
		ENGINE_ASSERT( mc.save_snapshot( 0, 0, cut, &snap ), "save_snapshot failed" );
		const double save_ms = test::ms_since( t0 );
		for ( int slot : { 1, 2, 1 } ) {
			mc.reset_slot( slot );
			t0 = std::chrono::steady_clock::now();
			ENGINE_ASSERT( mc.load_snapshot( slot, snap ), "load_snapshot failed" );
			const double load_ms = test::ms_since( t0 );
			Candidates warm_c;
			const int first = mc.prefill( ids.data() + cut, n - cut, slot, 0, &warm_c, cut );
			const std::vector<int> warm = decode( mc, first, n, slot, 48 );
			float max_diff = 0.0f;
			for ( int j = 0; j < warm_c.n; ++j ) max_diff = std::max( max_diff, std::fabs( warm_c.logits[j] - cold_c.logits[j] ) );
			const bool same = warm == cold;
			const bool ok = same && ( aligned ? max_diff == 0.0f : max_diff < 0.25f );
			std::printf( "%s cut %d (%s) -> slot %d: save %.1f ms, load %.1f ms, max |dlogit| %.4g, tokens %s%s\n", ok ? "[+]" : "[-]", cut,
				aligned ? "aligned" : "unaligned", slot, save_ms, load_ms, max_diff, same ? "identical" : "DIFFER",
				spec ? ( mc.spec_owns( slot ) ? ", drafter owns the slot" : ", DRAFTER LOST" ) : "" );
			if ( spec && !mc.spec_owns( slot ) ) failures++;
			failures += ok ? 0 : 1;
		}
		mc.free_snapshot( &snap );
	}

	// A snapshot without taps resumes a slot that decodes without the drafter, with the same tokens
	if ( spec ) {
		const int cut = chunk * ( ( n / 2 ) / chunk );
		mc.reset_slot( 0 );
		mc.prefill( ids.data(), cut, 0 );
		ModelCluster::SeqSnapshot nt;
		ENGINE_ASSERT( mc.save_snapshot( 0, 0, cut, &nt, false ), "save_snapshot failed" );
		mc.reset_slot( 1 );
		ENGINE_ASSERT( mc.load_snapshot( 1, nt ), "load_snapshot failed" );
		const int first = mc.prefill( ids.data() + cut, n - cut, 1, 0, nullptr, cut );
		const bool plain = !mc.spec_owns( 1 );
		const std::vector<int> t = decode( mc, first, n, 1, 48 );
		std::printf( "%s snapshot without taps: drafter %s, tokens %s\n", plain && t == cold ? "[+]" : "[-]", plain ? "released" : "STILL OWNS", t == cold ? "identical" : "DIFFER" );
		failures += plain && t == cold ? 0 : 1;
		mc.free_snapshot( &nt );
	}

	// Steady-state save cost: the second save of a size reuses the pooled pinned buffers
	{
		ModelCluster::SeqSnapshot a, b;
		auto t0 = std::chrono::steady_clock::now();
		ENGINE_ASSERT( mc.save_snapshot( 0, 0, 3000, &a ), "save_snapshot failed" );
		const double cold_ms = test::ms_since( t0 );
		mc.free_snapshot( &a );
		t0 = std::chrono::steady_clock::now();
		ENGINE_ASSERT( mc.save_snapshot( 0, 0, 3000, &b ), "save_snapshot failed" );
		std::printf( "[*] save 3000 tokens: %.1f ms with fresh pinned buffers, %.1f ms pooled\n", cold_ms, test::ms_since( t0 ) );
		mc.free_snapshot( &b );
	}

	// PrefixCache as a tree of chained snapshots: ancestors hold only their own KV segment
	{
		auto prefix = [&]( int tokens ) { return std::vector<int>( ids.begin(), ids.begin() + tokens ); };
		auto expect = [&]( bool cond, const char* what ) {
			std::printf( "%s PrefixCache: %s\n", cond ? "[+]" : "[-]", what );
			failures += cond ? 0 : 1;
		};
		// One chain root(1024) -> mid(2560) -> leaf(4096) built along one prefill like the generator does
		PrefixCache cache( mc, size_t( 64 ) << 30 );
		mc.reset_slot( 0 );
		PrefixCache::Entry* parent = nullptr;
		int pos = 0;
		for ( int cp : { 1024, 2560, 4096 } ) {
			mc.prefill( ids.data() + pos, cp - pos, 0, 0, nullptr, pos );
			ModelCluster::SeqSnapshot sn;
			ENGINE_ASSERT( mc.save_snapshot( 0, parent ? static_cast<int>( parent->tokens.size() ) : 0, cp, &sn ), "save_snapshot failed" );
			parent = cache.insert( parent, prefix( cp ), sn, cp == 1024 );
			ENGINE_ASSERT( parent, "insert failed" );
			pos = cp;
		}
		const PrefixCache::Stats chain_stats = cache.stats();
		const size_t flat = 3 * mc.snapshot_bytes( 0, 4096, true ) * mc.tp();
		std::printf( "[*] chain of 3 snapshots holds %.0f MiB (3 flat snapshots of 4096 tokens: %.0f MiB)\n", chain_stats.bytes / 1048576.0, flat / 1048576.0 );
		expect( chain_stats.bytes < flat, "chained storage is smaller than flat snapshots" );

		PrefixCache::Entry* leaf = cache.find( ids );
		expect( leaf && leaf->tokens.size() == 4096, "deepest prefix wins" );
		mc.reset_slot( 3 );
		expect( leaf && cache.restore( 3, leaf ), "restore the chain into another slot" );
		Candidates chain_c;
		const int chain_first = mc.prefill( ids.data() + 4096, n - 4096, 3, 0, &chain_c, 4096 );
		const std::vector<int> chain_tokens = decode( mc, chain_first, n, 3, 48 );
		float chain_diff = 0.0f;
		for ( int j = 0; j < chain_c.n; ++j ) chain_diff = std::max( chain_diff, std::fabs( chain_c.logits[j] - cold_c.logits[j] ) );
		expect( chain_tokens == cold && chain_diff < 0.25f, "a chain restore continues like the cold run" );
		std::printf( "    chain restore: max |dlogit| %.4g\n", chain_diff );

		std::vector<int> other = prefix( 4096 );
		other[3000] ^= 1;
		PrefixCache::Entry* e = cache.find( other );
		expect( e && e->tokens.size() == 2560, "a diverging prompt falls back to the deepest matching ancestor" );
		other = prefix( 2560 );
		e = cache.find( other );
		expect( e && e->tokens.size() == 1024, "an entry equal to the whole prompt is not a hit (no token left to prefill)" );

		// eviction: a small budget keeps the pinned root and its in-use parent chain, drops leaves LRU first
		cache.clear();
		auto rounded = [&]( int from, int to ) {
			const size_t r = size_t( 8 ) << 20;
			return ( mc.snapshot_bytes( from, to, true ) + r - 1 ) / r * r * mc.tp();
		};
		// room for the root and two children, not for a fourth node
		PrefixCache small( mc, rounded( 0, 900 ) + rounded( 900, 950 ) + rounded( 900, 960 ) + rounded( 960, 1000 ) / 2 );
		auto snap_at = [&]( int from, int to ) {
			mc.reset_slot( 0 );
			mc.prefill( ids.data(), to, 0 );
			ModelCluster::SeqSnapshot sn;
			ENGINE_ASSERT( mc.save_snapshot( 0, from, to, &sn ), "save_snapshot failed" );
			return sn;
		};
		PrefixCache::Entry* root = small.insert( nullptr, prefix( 900 ), snap_at( 0, 900 ), true );
		PrefixCache::Entry* a1 = small.insert( root, prefix( 950 ), snap_at( 900, 950 ), false );
		PrefixCache::Entry* b1 = small.insert( root, prefix( 960 ), snap_at( 900, 960 ), false );
		expect( root && a1 && b1, "root and two children fit" );
		small.find( ids ); // touches the deepest (960) chain: a1 is now the older leaf
		PrefixCache::Entry* c1 = small.insert( b1, prefix( 1000 ), snap_at( 960, 1000 ), false );
		expect( c1 != nullptr, "a fourth node fits by evicting the oldest leaf" );
		const PrefixCache::Stats st = small.stats();
		std::printf( "    evictions %llu; present: 900 %d, 950 %d, 960 %d, 1000 %d; bytes %.0f of %.0f MiB\n", static_cast<unsigned long long>( st.evictions ),
			small.exact( prefix( 900 ) ) != nullptr, small.exact( prefix( 950 ) ) != nullptr, small.exact( prefix( 960 ) ) != nullptr, small.exact( prefix( 1000 ) ) != nullptr,
			st.bytes / 1048576.0, small.budget() / 1048576.0 );
		expect( st.evictions >= 1 && small.exact( prefix( 900 ) ) && small.exact( prefix( 960 ) ) && !small.exact( prefix( 950 ) ), "the older leaf went; the pinned root and the used chain stayed" );
		small.clear();
	}
	if ( spec ) {
		std::printf( "[*] warm runs: proposed %d accepted %d (acceptance %.2f)\n", g_proposed, g_accepted, g_proposed ? static_cast<double>( g_accepted ) / g_proposed : 0.0 );
		if ( g_proposed == 0 ) failures++;
	}
	std::printf( "%s\n", failures ? "[-] FAIL" : "[+] PASS prefix restore" );
	mc.shutdown();
	return failures ? 1 : 0;
}
