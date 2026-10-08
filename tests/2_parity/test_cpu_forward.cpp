// End-to-end single-rank (tp=1) model forward, cross-backend: `--dump <dir>` (CUDA build) runs a short prefill and writes the last
// row's logits; `--check <dir>` (any build, in particular CPU) repeats it and compares against the golden file (tier 2 tolerance,
// plus argmax equality). Model directory from argv `--model <dir>` (default: the Qwen EXL3 checkpoint).
#include "engine/runtime/cluster.h"
#include "engine/backend/numeric.h"
#include "golden_io.h"
#include <cstdlib>
#include <vector>

using namespace engine::runtime;
namespace backend = engine::backend;

int main( int argc, char** argv ) {
	GoldenArgs args( argc, argv );
	const char* model_dir = "/home/mrq/Documents/models/Qwen3.8-27B-exl3";
	for ( int i = 1; i + 1 < argc; ++i ) {
		if ( !std::strcmp( argv[i], "--model" ) ) model_dir = argv[i + 1];
	}
	if ( !args.dump && !args.check ) {
		std::printf( "usage: test_cpu_forward --dump|--check <dir> [--model <dir>]\n" );
		return 2;
	}
	::setenv( "EXL3_TP", "1", 1 );
#if ENGINE_BACKEND_VULKAN
	// The reference trellis linear dequantizes a whole weight into device scratch (2.4 GiB for the 5120 x 248320 head)
	::setenv( "EXL3_VRAM_RESERVE_MIB", "6000", 0 );
#endif
	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = "qwen38";
	cfg.model_dir = model_dir;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	const int vocab = mc.metadata().vocab_size;

	const std::vector<int> prompt = { 15, 16, 17, 18 };
	mc.reset_slot( 0 );
	const int pred = mc.prefill( prompt.data(), static_cast<int>( prompt.size() ), 0, 0 );
	std::vector<backend::half_t> logits_h( static_cast<size_t>( vocab ) );
	mc.read_logits( 1, logits_h.data() );
	std::vector<float> logits( logits_h.size() );
	for ( size_t i = 0; i < logits.size(); ++i ) logits[i] = backend::to_float( logits_h[i] );
	std::printf( "  prefill %zu tokens -> next token %d\n", prompt.size(), pred );

	bool ok = true;
	if ( args.dump ) {
		ok = golden_write( args.dir, "forward_logits", logits.data(), logits.size() * sizeof( float ) )
			&& golden_write( args.dir, "forward_pred", &pred, sizeof( pred ) );
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
	}
	mc.shutdown();
	std::printf( "[%s] test_cpu_forward\n", ok ? "PASS" : "FAIL" );
	return ok ? 0 : 1;
}
