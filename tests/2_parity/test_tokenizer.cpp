#include "engine/config.h"
#include "engine/loader/tokenizer.h"
#include "engine/loader/chat_template.h"
#include <cstdio>
#include <string>
#include <vector>

int main() {
	std::printf( "========================================================\n" );
	std::printf( "  VOLTA ENGINE: BPE TOKENIZER & CHAT TEMPLATE PARITY\n" );
	std::printf( "========================================================\n\n" );

	const char* qwen_dir = "/home/mrq/Documents/models/Qwen3.8-27B-exl3";
	const char* glm_dir = "/home/mrq/Documents/models/GLM-5.3-Flash-exl3-3.05bpw";

	bool all_passed = true;

	// =======================================================================
	// PART 1: Qwen3.8-27B Tokenizer & Template Parity
	// =======================================================================
	std::printf( "[*] SECTION 1: Qwen3.8-27B BPE Tokenizer & Chat Template\n" );

	engine::loader::Tokenizer qwen_tok;
	bool qwen_load_ok = qwen_tok.load( qwen_dir );
	if ( !qwen_load_ok ) {
		std::printf( "[-] FAILED to load Qwen tokenizer from: %s\n", qwen_dir );
		return 1;
	}
	std::printf( "[+] Qwen tokenizer loaded successfully. Vocab size: %zu\n", qwen_tok.vocab_size() );

	// Verify special token lookups
	int qwen_im_start = qwen_tok.im_start_id();
	int qwen_im_end = qwen_tok.im_end_id();
	int qwen_think = qwen_tok.think_start_id();
	int qwen_think_end = qwen_tok.think_end_id();
	int qwen_eos = qwen_tok.eos_id();
	int qwen_pad = qwen_tok.pad_id();

	std::printf( "    Special tokens: im_start=%d, im_end=%d, think=%d, /think=%d, eos=%d, pad=%d\n",
		qwen_im_start, qwen_im_end, qwen_think, qwen_think_end, qwen_eos, qwen_pad );

	if ( qwen_im_start != 248045 || qwen_im_end != 248046 || qwen_think != 248068 || qwen_think_end != 248069 || qwen_eos != 248046 || qwen_pad != 248044 ) {
		std::printf( "[-] Qwen special token ID mismatch!\n" );
		all_passed = false;
	} else {
		std::printf( "[+] Qwen special token IDs verified.\n" );
	}

	engine::loader::ChatTemplate qwen_tmpl;
	bool qwen_tmpl_ok = qwen_tmpl.load( qwen_dir );
	if ( !qwen_tmpl_ok ) {
		std::printf( "[-] FAILED to load Qwen chat template!\n" );
		return 1;
	}
	std::printf( "[+] Qwen chat template loaded.\n" );

	// Ground truth coherence prompts for Qwen (with enable_thinking=false to match coherence test suite)
	struct TestCase {
		const char* prompt;
		std::vector<int> expected_tokens;
	};

	TestCase qwen_coherence_cases[] = {
		{
			"What is the capital of France?",
			{ 248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 }
		},
		{
			"List the first 5 prime numbers:",
			{ 248045, 846, 198, 826, 279, 1118, 220, 20, 9944, 4947, 25, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 }
		},
		{
			"Passage: The secret access code is 84920. Question: What is the secret access code?",
			{ 248045, 846, 198, 11839, 416, 25, 561, 6038, 2528, 1970, 369, 220, 23, 19, 24, 17, 15, 13, 15380, 25, 3437, 369, 279, 6038, 2528, 1970, 30, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 }
		},
		{
			"What is the capital of Japan?",
			{ 248045, 846, 198, 3710, 369, 279, 6511, 314, 6124, 30, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 }
		}
	};

	engine::loader::ChatTemplateOptions qwen_no_think;
	qwen_no_think.add_generation_prompt = true;
	qwen_no_think.enable_thinking = false;

	for ( size_t i = 0; i < sizeof( qwen_coherence_cases ) / sizeof( qwen_coherence_cases[0] ); ++i ) {
		const auto& tc = qwen_coherence_cases[i];
		std::vector<engine::loader::ChatMessage> msgs = { { "user", tc.prompt } };
		std::string rendered = qwen_tmpl.render( msgs, qwen_no_think );
		std::vector<int> tokens = qwen_tok.encode( rendered );

		bool match = ( tokens == tc.expected_tokens );
		std::printf( "    Qwen Coherence Prompt %zu: %s (tokens: %zu)\n", i + 1, match ? "PASS" : "FAIL", tokens.size() );
		if ( !match ) {
			all_passed = false;
			std::printf( "      Rendered: %s\n", rendered.c_str() );
			std::printf( "      Got: " );
			for ( int t : tokens ) std::printf( "%d ", t );
			std::printf( "\n      Exp: " );
			for ( int t : tc.expected_tokens ) std::printf( "%d ", t );
			std::printf( "\n" );
		}

		// Verify round-trip decode
		std::string decoded = qwen_tok.decode( tokens );
		if ( decoded != rendered ) {
			std::printf( "[-] Qwen decode round-trip mismatch!\n" );
			all_passed = false;
		}
	}

	// Verify standard thinking prompt rendering
	engine::loader::ChatTemplateOptions qwen_think_opts;
	qwen_think_opts.add_generation_prompt = true;
	qwen_think_opts.enable_thinking = true;
	qwen_think_opts.reasoning_effort = "low";

	std::vector<engine::loader::ChatMessage> test_msg = { { "user", "What is the capital of France?" } };
	std::string qwen_rendered_think = qwen_tmpl.render( test_msg, qwen_think_opts );
	std::string expected_qwen_think = "<|im_start|>system\nReasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n<think>\n";
	if ( qwen_rendered_think == expected_qwen_think ) {
		std::printf( "[+] Qwen thinking template rendered correctly.\n" );
	} else {
		std::printf( "[-] Qwen thinking template mismatch!\n" );
		all_passed = false;
	}

	// =======================================================================
	// PART 2: GLM-5.3-Flash Tokenizer & Template Parity
	// =======================================================================
	std::printf( "\n[*] SECTION 2: GLM-5.3-Flash BPE Tokenizer & Chat Template\n" );

	engine::loader::Tokenizer glm_tok;
	bool glm_load_ok = glm_tok.load( glm_dir );
	if ( !glm_load_ok ) {
		std::printf( "[-] FAILED to load GLM tokenizer from: %s\n", glm_dir );
		return 1;
	}
	std::printf( "[+] GLM tokenizer loaded successfully. Vocab size: %zu\n", glm_tok.vocab_size() );

	int glm_gmask = glm_tok.gmask_id();
	int glm_sop = glm_tok.sop_id();
	int glm_think = glm_tok.think_start_id();
	int glm_think_end = glm_tok.think_end_id();
	int glm_eos = glm_tok.eos_id();

	std::printf( "    Special tokens: gmask=%d, sop=%d, think=%d, /think=%d, eos=%d\n",
		glm_gmask, glm_sop, glm_think, glm_think_end, glm_eos );

	if ( glm_gmask != 154822 || glm_sop != 154824 || glm_think != 154841 || glm_think_end != 154842 || glm_eos != 154820 ) {
		std::printf( "[-] GLM special token ID mismatch!\n" );
		all_passed = false;
	} else {
		std::printf( "[+] GLM special token IDs verified.\n" );
	}

	engine::loader::ChatTemplate glm_tmpl;
	bool glm_tmpl_ok = glm_tmpl.load( glm_dir );
	if ( !glm_tmpl_ok ) {
		std::printf( "[-] FAILED to load GLM chat template!\n" );
		return 1;
	}
	std::printf( "[+] GLM chat template loaded.\n" );

	TestCase glm_coherence_cases[] = {
		{
			"What is the capital of France? Answer in one sentence.",
			{ 154822, 154824, 154826, 25062, 287, 29905, 371, 25, 12035, 154827, 3838, 374, 279, 6722, 315, 9621, 30, 21754, 304, 825, 11646, 13, 154828, 154841 }
		},
		{
			"List the first 5 prime numbers:",
			{ 154822, 154824, 154826, 25062, 287, 29905, 371, 25, 12035, 154827, 852, 279, 1156, 220, 20, 10245, 5109, 25, 154828, 154841 }
		},
		{
			"Passage: The secret access code is 84920. Question: What is the secret access code?",
			{ 154822, 154824, 154826, 25062, 287, 29905, 371, 25, 12035, 154827, 12181, 424, 25, 576, 6234, 2615, 2038, 374, 220, 23, 101474, 98360, 13, 15834, 25, 3555, 374, 279, 6234, 2615, 2038, 30, 154828, 154841 }
		},
		{
			"What is the capital of Japan?",
			{ 154822, 154824, 154826, 25062, 287, 29905, 371, 25, 12035, 154827, 3838, 374, 279, 6722, 315, 6323, 30, 154828, 154841 }
		}
	};

	engine::loader::ChatTemplateOptions glm_opts;
	glm_opts.add_generation_prompt = true;
	glm_opts.reasoning_effort = "low";

	for ( size_t i = 0; i < sizeof( glm_coherence_cases ) / sizeof( glm_coherence_cases[0] ); ++i ) {
		const auto& tc = glm_coherence_cases[i];
		std::vector<engine::loader::ChatMessage> msgs = { { "user", tc.prompt } };
		std::string rendered = glm_tmpl.render( msgs, glm_opts );
		std::vector<int> tokens = glm_tok.encode( rendered );

		bool match = ( tokens == tc.expected_tokens );
		std::printf( "    GLM Coherence Prompt %zu: %s (tokens: %zu)\n", i + 1, match ? "PASS" : "FAIL", tokens.size() );
		if ( !match ) {
			all_passed = false;
			std::printf( "      Rendered: %s\n", rendered.c_str() );
			std::printf( "      Got: " );
			for ( int t : tokens ) std::printf( "%d ", t );
			std::printf( "\n      Exp: " );
			for ( int t : tc.expected_tokens ) std::printf( "%d ", t );
			std::printf( "\n" );
		}

		// Verify round-trip decode
		std::string decoded = glm_tok.decode( tokens );
		if ( decoded != rendered ) {
			std::printf( "[-] GLM decode round-trip mismatch!\n" );
			all_passed = false;
		}
	}

	std::printf( "\n========================================================\n" );
	if ( all_passed ) {
		std::printf( "  PARITY SUITE: ALL TESTS PASSED (100%% EXACT MATCH)\n" );
	} else {
		std::printf( "  PARITY SUITE: FAILURES DETECTED\n" );
	}
	std::printf( "========================================================\n" );

	return all_passed ? 0 : 1;
}
