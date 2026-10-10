// End-to-end single-rank (tp=1) model forward, cross-backend: `--dump <dir>` (CUDA build) runs a short prefill and writes the last
// row's logits; `--check <dir>` (any build, in particular CPU) repeats it and compares against the golden file (tier 2 tolerance,
// plus argmax equality). Model directory from argv `--model <dir>` (default: the Qwen EXL3 checkpoint) and `--type <model_type>` (default qwen3_5).
// ENGINE_TP overrides the single rank (a checkpoint too big for one GPU, e.g. Flash-Next, is dumped from a CUDA tp=4 build).
#include "engine/runtime/test_paths.h"
#include "engine/runtime/cluster.h"
#include "engine/backend/numeric.h"
#include "golden_io.h"
#include <cstdlib>
#include <vector>

using namespace engine::runtime;
namespace backend = engine::backend;

int main( int argc, char** argv ) {
	GoldenArgs args( argc, argv );
	const char* model_dir = engine::runtime::test::model_path( "Qwen3.8-27B-exl3" );
	const char* model_type = "qwen3_5";
	int steps = 0;
	int ntokens = 4;
	for ( int i = 1; i + 1 < argc; ++i ) {
		if ( !std::strcmp( argv[i], "--model" ) ) model_dir = argv[i + 1];
		if ( !std::strcmp( argv[i], "--type" ) ) model_type = argv[i + 1];
		if ( !std::strcmp( argv[i], "--steps" ) ) steps = std::atoi( argv[i + 1] );
		if ( !std::strcmp( argv[i], "--tokens" ) ) ntokens = std::atoi( argv[i + 1] );
	}
	if ( !args.dump && !args.check ) {
		std::printf( "usage: test_cpu_forward --dump|--check <dir> [--model <dir>] [--type <model_type>] [--steps <greedy decode steps after the prefill>] [--tokens <prompt length, default 4>]\n" );
		return 2;
	}
	::setenv( "ENGINE_TP", "1", 0 );
#if ENGINE_BACKEND_VULKAN
	// The reference trellis linear dequantizes a whole weight into device scratch (2.4 GiB for the 5120 x 248320 head)
	::setenv( "ENGINE_VRAM_RESERVE_MIB", "6000", 0 );
#endif
	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = model_type;
	cfg.model_dir = model_dir;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	const int vocab = mc.metadata().vocab_size;

	std::vector<int> prompt;
	for ( int i = 0; i < ntokens; ++i ) prompt.push_back( 15 + i );
	mc.reset_slot( 0 );
	const int pred = mc.prefill( prompt.data(), static_cast<int>( prompt.size() ), 0, 0 );
	std::vector<backend::half_t> logits_h( static_cast<size_t>( vocab ) );
	mc.read_logits( 1, logits_h.data() );
	std::vector<float> logits( logits_h.size() );
	for ( size_t i = 0; i < logits.size(); ++i ) logits[i] = backend::to_float( logits_h[i] );
	std::printf( "  prefill %zu tokens -> next token %d\n", prompt.size(), pred );

	// Greedy decode steps (single-row forwards: the decode paths) after the prefill
	std::vector<int> decoded;
	int tok = pred;
	for ( int i = 0; i < steps; ++i ) {
		mc.forward( &tok, 1, static_cast<int>( prompt.size() ) + i, 0 );
		tok = mc.sample_one();
		decoded.push_back( tok );
	}
	if ( steps > 0 ) {
		std::printf( "  decoded:" );
		for ( int t : decoded ) std::printf( " %d", t );
		std::printf( "\n" );
	}

	bool ok = true;
	if ( args.dump ) {
		ok = golden_write( args.dir, "forward_logits", logits.data(), logits.size() * sizeof( float ) )
			&& golden_write( args.dir, "forward_pred", &pred, sizeof( pred ) )
			&& ( steps == 0 || golden_write( args.dir, "forward_decoded", decoded.data(), decoded.size() * sizeof( int ) ) );
		std::printf( "  dumped to %s: %s\n", args.dir.c_str(), ok ? "ok" : "FAIL" );
	}
	else {
		std::vector<uint8_t> raw_l, raw_p;
		ENGINE_ASSERT( golden_read( args.dir, "forward_logits", raw_l ) && golden_read( args.dir, "forward_pred", raw_p ), "missing golden files" );
		std::vector<float> ref( logits.size() );
		std::memcpy( ref.data(), raw_l.data(), MIN( raw_l.size(), ref.size() * sizeof( float ) ) );
		int ref_pred = 0;
		std::memcpy( &ref_pred, raw_p.data(), sizeof( ref_pred ) );
		ok = golden_compare( "logits", ref, logits );
		const bool same = ( ref_pred == pred );
		std::printf( "  argmax %d vs golden %d  %s\n", pred, ref_pred, same ? "PASS" : "FAIL" );
		ok = ok && same;
		if ( steps > 0 ) {
			std::vector<uint8_t> raw_d;
			ENGINE_ASSERT( golden_read( args.dir, "forward_decoded", raw_d ) && raw_d.size() == decoded.size() * sizeof( int ), "missing or mismatched decode golden" );
			const bool same_d = std::memcmp( raw_d.data(), decoded.data(), raw_d.size() ) == 0;
			std::printf( "  decoded tokens vs golden  %s\n", same_d ? "PASS" : "FAIL" );
			ok = ok && same_d;
		}
	}
	mc.shutdown();
	std::printf( "[%s] test_cpu_forward\n", ok ? "PASS" : "FAIL" );
	return ok ? 0 : 1;
}
