#pragma once

// Shared harness for tier-3 benchmarks and tier-4 coherence gates.
// All model plumbing (arenas, NVLink mesh, pinned rank pool, weight loading, speculation)
// lives in ModelCluster; this header only adds generation loops and timing helpers.

#include "engine/runtime/cluster.h"
#include "engine/runtime/test_paths.h"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace engine::runtime::test {

// ENGINE_TEST_QWEN_DIR selects another Qwen checkpoint (the NVFP4 one, for instance); the loader detects the format.
static const char* QWEN_DIR = model_path( "Qwen3.8-27B-exl3", "ENGINE_TEST_QWEN_DIR" );
// Overridable so the unquantized drafter can be A/B'd against the quantized one without a rebuild:
// load_linear falls back to a dense <base>.weight when no trellis group is present, so
// QWEN_DRAFT_BF16_DIR drops straight into cfg.draft_model_dir.
static const char* QWEN_DRAFT_DIR = model_path( "Qwen3.8-27B-DFlash2-EXL3-5.0bpw", "ENGINE_TEST_DRAFT_DIR" );
// The same drafter before quantization: 81 BF16 tensors, `fc.weight` and the 35 block projections
// unquantized. It is the ground truth the Trellis weights are checked against.
static const char* QWEN_DRAFT_BF16_DIR = model_path( "Qwen3.8-27B-DFlash2" );
static const char* GLM_DIR = model_path( "GLM-5.3-Flash-exl3-3.05bpw", "ENGINE_TEST_GLM_DIR" );
static const char* FLASH_DIR = model_path( "Swift-Qwen3.8-Flash-Next-exl3-4.00bpw", "ENGINE_TEST_FLASH_DIR" );
static const char* GLM_DRAFT_DIR = model_path( "GLM-5.3-Flash-DFlash2", "ENGINE_TEST_GLM_DRAFT_DIR" );

struct GenStats {
	std::vector<int> tokens; // generated tokens (first = prefill prediction)
	double prefill_ms = 0.0;
	double decode_ms = 0.0;
	int proposed = 0;
	int accepted = 0;
	int rounds = 0;
	double decode_tok_s() const {
		const int n = static_cast<int>( tokens.size() ) - 1;
		return ( decode_ms > 0.0 && n > 0 ) ? n * 1000.0 / decode_ms : 0.0;
	}
};

inline double ms_since( std::chrono::steady_clock::time_point t0 ) {
	return std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t0 ).count();
}

// Greedy (T=0) generation of up to max_new tokens on `slot`; stops at EOS.
// speculative=true drives DFlash2 rounds; output must be identical to plain greedy decoding.
inline GenStats generate( ModelCluster& mc, const std::vector<int>& prompt, int max_new, bool speculative, int slot = 0, int prefill_chunk = 0 ) {
	GenStats st;
	mc.reset_slot( slot );
	auto t0 = std::chrono::steady_clock::now();
	int tok = mc.prefill( prompt.data(), static_cast<int>( prompt.size() ), slot, prefill_chunk );
	st.prefill_ms = ms_since( t0 );
	st.tokens.push_back( tok );
	int pos = static_cast<int>( prompt.size() );

	t0 = std::chrono::steady_clock::now();
	while ( static_cast<int>( st.tokens.size() ) < max_new && !mc.is_eos( tok ) ) {
		if ( speculative ) {
			const SpecResult r = mc.speculate( slot, tok, pos );
			st.rounds++;
			st.proposed += r.proposed;
			st.accepted += r.accepted;
			pos = r.next_pos;
			for ( int i = 0; i < r.emitted_count && static_cast<int>( st.tokens.size() ) < max_new; ++i ) {
				tok = r.emitted[i];
				st.tokens.push_back( tok );
				if ( mc.is_eos( tok ) ) break;
			}
		}
		else {
			mc.forward( &tok, 1, pos++, slot );
			tok = mc.sample_one();
			st.tokens.push_back( tok );
		}
	}
	st.decode_ms = ms_since( t0 );
	return st;
}

inline std::string format_tokens( const std::vector<int>& toks, size_t max_n = 64 ) {
	std::string s;
	for ( size_t i = 0; i < toks.size() && i < max_n; ++i ) {
		s += ( i ? "," : "" ) + std::to_string( toks[i] );
	}
	return s;
}

} // namespace engine::runtime::test
