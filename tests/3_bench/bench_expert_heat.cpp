// Routed-expert skew on GLM-5.3-Flash: how well would a resident subset of each layer's local experts serve the routing of
// real prompts? Per prompt the per-(rank, layer, expert) route counts are read back, then for a resident fraction f the share of a
// prompt's routes that land in (a) its own top-f experts (an adaptive cache's ceiling), (b) the top-f of all the OTHER prompts
// (a static profile built elsewhere) and (c) a random f of the experts (the uniform baseline).
//   bench_expert_heat [--tokens N]
#include "engine/runtime/test_driver.h"
#include "engine/loader/tokenizer.h"
#include "engine/loader/chat_template.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <numeric>

using namespace engine::runtime;

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	int n_tokens = 192, n_prompts = 16;
	for ( int i = 1; i < argc; ++i ) {
		if ( !std::strcmp( argv[i], "--tokens" ) && i + 1 < argc ) n_tokens = std::atoi( argv[++i] );
		else if ( !std::strcmp( argv[i], "--prompts" ) && i + 1 < argc ) n_prompts = std::atoi( argv[++i] );
	}
	const char* prompts[] = {
		"Write a Python function that returns the n-th Fibonacci number using memoization, and explain the complexity.",
		"Write a C function to reverse a singly linked list in place, then walk through it on a 4-node example.",
		"Write a SQL query to find the second highest salary in an employees table and explain how it works.",
		"Explain how a hash table handles collisions, and compare chaining with open addressing.",
		"I have a Rust project that fails with 'cannot borrow as mutable more than once'. Show a minimal example and three ways to fix it.",
		"Write a bash script that finds the 10 largest files under a directory and prints their sizes in human readable form.",
		"Convert this JSON to YAML and explain the differences: {\"name\": \"svc\", \"ports\": [80, 443], \"env\": {\"MODE\": \"prod\"}}",
		"Summarize the plot of Romeo and Juliet in five sentences.",
		"What are the main differences between TCP and UDP? Give use cases for each.",
		"Prove that the square root of 2 is irrational.",
		"A train leaves at 3pm going 60 mph, another leaves at 4pm going 80 mph on the same track. When does the second catch up?",
		"Write a git command sequence to undo the last commit but keep the changes staged, then force push safely.",
		"Draft a polite email asking a colleague to review a pull request by Friday.",
		"Write a regular expression that matches ISO 8601 dates and explain each part.",
		"Describe how a B-tree insert works when a leaf node is full.",
		"Translate 'The quick brown fox jumps over the lazy dog' into French, Spanish and German.",
	};
	const int total = std::min( n_prompts, static_cast<int>( sizeof( prompts ) / sizeof( prompts[0] ) ) );
	engine::loader::Tokenizer tok;
	ENGINE_ASSERT( tok.load( test::GLM_DIR ), "tokenizer load failed" );
	engine::loader::ChatTemplate chat;
	ENGINE_ASSERT( chat.load( test::GLM_DIR ), "chat template load failed" );
	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = "glm53";
	cfg.model_dir = test::GLM_DIR;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	mc.warm_graphs( 8 );

	std::vector<std::vector<uint32_t>> heat( total ); // per prompt: [rank][layer][expert] flattened
	int layers = 0, experts = 0;
	for ( int pi = 0; pi < total; ++pi ) {
		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) { std::vector<uint32_t> t; mc.model( d ).expert_heat( t, layers, experts, true ); }
		engine::loader::ChatTemplateOptions opts;
		opts.enable_thinking = false;
		const std::vector<int> ids = tok.encode( chat.render( { { "user", prompts[pi] } }, opts ), false );
		const auto st = test::generate( mc, ids, n_tokens, false );
		std::printf( "  prompt %2d: decode %.1f tok/s\n", pi, st.decode_tok_s() );
		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
			std::vector<uint32_t> t;
			ENGINE_ASSERT( mc.model( d ).expert_heat( t, layers, experts, false ), "no expert heat" );
			heat[pi].insert( heat[pi].end(), t.begin(), t.end() );
		}
		mc.rebalance_experts(); // no-op unless EXL3_EXPERT_CACHE tiers the experts
	}
	const size_t per_rank = static_cast<size_t>( layers ) * experts;
	std::printf( "\n%d prompts, %d tokens each, %d MoE layers x %d local experts x %d ranks\n", total, n_tokens, layers, experts, ENGINE_NUM_GPUS );
	std::printf( "  resident fraction | own top-f (adaptive ceiling) | top-f of other prompts (static profile) | uniform\n" );
	for ( double f : { 0.25, 0.5, 0.75, 0.9 } ) {
		const int keep = static_cast<int>( f * experts + 0.5 );
		double own = 0, other = 0, routes = 0;
		for ( int pi = 0; pi < total; ++pi ) {
			for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
				for ( int l = 0; l < layers; ++l ) {
					const size_t base = d * per_rank + static_cast<size_t>( l ) * experts;
					std::vector<uint32_t> mine( heat[pi].begin() + base, heat[pi].begin() + base + experts ), rest( experts, 0 );
					for ( int pj = 0; pj < total; ++pj ) if ( pj != pi ) for ( int e = 0; e < experts; ++e ) rest[e] += heat[pj][base + e];
					double tot = 0;
					for ( int e = 0; e < experts; ++e ) tot += mine[e];
					std::vector<int> ord( experts );
					std::iota( ord.begin(), ord.end(), 0 );
					std::sort( ord.begin(), ord.end(), [&]( int a, int b ) { return mine[a] > mine[b]; } );
					double a = 0;
					for ( int k = 0; k < keep; ++k ) a += mine[ord[k]];
					std::sort( ord.begin(), ord.end(), [&]( int a2, int b2 ) { return rest[a2] > rest[b2]; } );
					double b = 0;
					for ( int k = 0; k < keep; ++k ) b += mine[ord[k]];
					own += a; other += b; routes += tot;
				}
			}
		}
		std::printf( "        %.2f        |            %.3f             |                 %.3f                  |  %.3f\n", f, own / routes, other / routes, f );
	}
	mc.shutdown();
	return 0;
}
