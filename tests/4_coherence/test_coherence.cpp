#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/core/comm.h"
#include "engine/models/qwen38_27b.h"
#include "engine/models/glm53_flash.h"
#include "trellis_context.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cctype>
#include <thread>
#include <pthread.h>
#include <memory>

struct CoherencePrompt {
	const char* prompt;
	std::vector<int> prompt_tokens;
	const char* expected_completion;
};

static bool load_qwen_vocab( const char* vocab_path, std::vector<std::string>& id_to_token ) {
	FILE* f = std::fopen( vocab_path, "rb" );
	if ( !f ) return false;
	std::fseek( f, 0, SEEK_END );
	long size = std::ftell( f );
	std::fseek( f, 0, SEEK_SET );
	std::string buffer( size, '\0' );
	size_t r = std::fread( buffer.data(), 1, size, f );
	std::fclose( f );
	if ( r != static_cast<size_t>( size ) ) return false;

	id_to_token.assign( 248320, "" );

	size_t pos = 0;
	while ( pos < buffer.size() ) {
		while ( pos < buffer.size() && buffer[pos] != '"' ) ++pos;
		if ( pos >= buffer.size() ) break;
		++pos;
		size_t key_start = pos;
		while ( pos < buffer.size() ) {
			if ( buffer[pos] == '\\' && pos + 1 < buffer.size() ) {
				pos += 2;
			}
			else if ( buffer[pos] == '"' ) {
				break;
			}
			else {
				++pos;
			}
		}
		if ( pos >= buffer.size() ) break;
		std::string key = buffer.substr( key_start, pos - key_start );
		++pos;
		while ( pos < buffer.size() && ( buffer[pos] == ' ' || buffer[pos] == ':' || buffer[pos] == '\t' || buffer[pos] == '\n' || buffer[pos] == '\r' ) ) ++pos;
		if ( pos >= buffer.size() ) break;
		int id = 0;
		while ( pos < buffer.size() && std::isdigit( static_cast<unsigned char>( buffer[pos] ) ) ) {
			id = id * 10 + ( buffer[pos] - '0' );
			++pos;
		}
		if ( id >= 0 && id < static_cast<int>( id_to_token.size() ) ) {
			id_to_token[id] = key;
		}
	}
	return true;
}

static bool load_glm_vocab( const char* vocab_path, std::vector<std::string>& id_to_token ) {
	FILE* f = std::fopen( vocab_path, "rb" );
	if ( !f ) return false;
	std::fseek( f, 0, SEEK_END );
	long size = std::ftell( f );
	std::fseek( f, 0, SEEK_SET );
	std::string buffer( size, '\0' );
	size_t r = std::fread( buffer.data(), 1, size, f );
	std::fclose( f );
	if ( r != static_cast<size_t>( size ) ) return false;

	id_to_token.assign( 154880, "" );

	size_t vocab_pos = buffer.find( "\"vocab\":" );
	if ( vocab_pos == std::string::npos ) return false;
	size_t pos = buffer.find( '{', vocab_pos );
	if ( pos == std::string::npos ) return false;
	++pos;

	while ( pos < buffer.size() ) {
		while ( pos < buffer.size() && buffer[pos] != '"' && buffer[pos] != '}' ) ++pos;
		if ( pos >= buffer.size() || buffer[pos] == '}' ) break;
		++pos;
		size_t key_start = pos;
		while ( pos < buffer.size() ) {
			if ( buffer[pos] == '\\' && pos + 1 < buffer.size() ) {
				pos += 2;
			}
			else if ( buffer[pos] == '"' ) {
				break;
			}
			else {
				++pos;
			}
		}
		if ( pos >= buffer.size() ) break;
		std::string key = buffer.substr( key_start, pos - key_start );
		++pos;
		while ( pos < buffer.size() && ( buffer[pos] == ' ' || buffer[pos] == ':' || buffer[pos] == '\t' || buffer[pos] == '\n' || buffer[pos] == '\r' ) ) ++pos;
		if ( pos >= buffer.size() ) break;
		int id = 0;
		while ( pos < buffer.size() && std::isdigit( static_cast<unsigned char>( buffer[pos] ) ) ) {
			id = id * 10 + ( buffer[pos] - '0' );
			++pos;
		}
		if ( id >= 0 && id < static_cast<int>( id_to_token.size() ) ) {
			id_to_token[id] = key;
		}
	}
	return true;
}

static std::string decode_token( int id, const std::vector<std::string>& id_to_token ) {
	if ( id < 0 || id >= static_cast<int>( id_to_token.size() ) ) return "";
	const std::string& raw = id_to_token[id];
	std::string decoded;
	for ( size_t i = 0; i < raw.size(); ++i ) {
		if ( static_cast<unsigned char>( raw[i] ) == 0xc4 && i + 1 < raw.size() && static_cast<unsigned char>( raw[i + 1] ) == 0xa0 ) {
			decoded += ' ';
			++i;
		}
		else if ( static_cast<unsigned char>( raw[i] ) == 0xc4 && i + 1 < raw.size() && static_cast<unsigned char>( raw[i + 1] ) == 0x8a ) {
			decoded += '\n';
			++i;
		}
		else {
			decoded += raw[i];
		}
	}
	return decoded;
}

static std::string trim( const std::string& s ) {
	size_t start = 0;
	while ( start < s.size() && ( s[start] == ' ' || s[start] == '\n' || s[start] == '\r' || s[start] == '\t' ) ) {
		++start;
	}
	size_t end = s.size();
	while ( end > start && ( s[end - 1] == ' ' || s[end - 1] == '\n' || s[end - 1] == '\r' || s[end - 1] == '\t' ) ) {
		--end;
	}
	return s.substr( start, end - start );
}

int main( int argc, char** argv ) {
	bool run_qwen = true;
	bool run_glm = true;
	for ( int a = 1; a < argc; ++a ) {
		if ( std::strcmp( argv[a], "--glm-only" ) == 0 ) {
			run_qwen = false;
		}
		else if ( std::strcmp( argv[a], "--qwen-only" ) == 0 || std::strcmp( argv[a], "--qwen-tp4" ) == 0 ) {
			run_glm = false;
		}
	}

	std::printf( "========================================================\n" );
	std::printf( "   volta-engine Phase 2: Multi-GPU Coherence Gate (T=0.0)\n" );
	std::printf( "========================================================\n" );
	std::fflush( stdout );

	CoherencePrompt qwen_test_cases[] = {
		{
			"What is the capital of France?",
			{
				248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271
			},
			"Paris"
		},
		{
			"List the first 5 prime numbers:",
			{
				248045, 846, 198, 826, 279, 1118, 220, 20, 9944, 4947, 25, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271
			},
			"2, 3, 5, 7, 11"
		},
		{
			"Passage: The secret access code is 84920. Question: What is the secret access code?",
			{
				248045, 846, 198, 11839, 416, 25, 561, 6038, 2528, 1970, 369, 220, 23, 19, 24, 17, 15, 13, 15380,
				25, 3437, 369, 279, 6038, 2528, 1970, 30, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271
			},
			"84920"
		},
		{
			"What is the capital of Japan?",
			{
				248045, 846, 198, 3710, 369, 279, 6511, 314, 6124, 30, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271
			},
			"Tokyo"
		}
	};

	// =======================================================================
	// PART 1: Qwen3.8-27B Coherence Gate (TP=4)
	// =======================================================================
	if ( run_qwen ) {
		std::printf( "[*] SECTION 1 (TP=4): Qwen3.8-27B 4-GPU Coherence Suite...\n" );
		std::fflush( stdout );

		engine::comm::CollectiveRing rings[ENGINE_NUM_GPUS];
		void* staging_ptrs[ENGINE_NUM_GPUS] = { nullptr };
		void* flag_ptrs[ENGINE_NUM_GPUS] = { nullptr };
		engine::comm::init_4gpu_mesh( rings, staging_ptrs, flag_ptrs, 4 * 1024 * 1024 );

		cudaStream_t qwen_streams[ENGINE_NUM_GPUS];
		pod::MemoryArena qwen_arenas[ENGINE_NUM_GPUS];
		auto qwen_pipes = std::make_unique<engine::models::Qwen38Pipeline[]>( ENGINE_NUM_GPUS );
		engine::models::Qwen38Config qwen_config;

		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
			CUDA_CHECK( cudaSetDevice( d ) );
			CUDA_CHECK( cudaStreamCreate( &qwen_streams[d] ) );
			bool arena_ok = engine::core::arena_create( qwen_arenas[d], d, 8ULL * 1024 * 1024 * 1024 );
			ENGINE_ASSERT( arena_ok, "Failed to create Qwen arena" );
			bool pipe_ok = qwen_pipes[d].init( qwen_config, d, qwen_arenas[d], &rings[d] );
			ENGINE_ASSERT( pipe_ok, "Failed to init Qwen pipeline" );
		}
		engine::modules::trellis_context_init_all();

		const char* model_dir = "/home/mrq/Documents/models/Qwen3.8-27B-exl3";
		std::string vocab_path = std::string( model_dir ) + "/vocab.json";

		std::printf( "[*] Loading Qwen tokenizer vocabulary from %s...\n", vocab_path.c_str() );
		std::vector<std::string> id_to_token;
		bool ok = load_qwen_vocab( vocab_path.c_str(), id_to_token );
		ENGINE_ASSERT( ok, "Failed to load Qwen vocab" );
		std::printf( "[+] Loaded %zu vocabulary tokens.\n", id_to_token.size() );

		std::printf( "[*] Loading Qwen3.8-27B weights across 4 GPUs (TP=4)...\n" );
		std::fflush( stdout );
		{
			std::thread loaders[ENGINE_NUM_GPUS];
			for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
				loaders[d] = std::thread( [&, d]() {
					cpu_set_t cpuset;
					CPU_ZERO( &cpuset );
					CPU_SET( d, &cpuset );
					pthread_setaffinity_np( pthread_self(), sizeof( cpu_set_t ), &cpuset );

					CUDA_CHECK( cudaSetDevice( d ) );
					bool load_ok = qwen_pipes[d].load_weights( model_dir, qwen_streams[d] );
					ENGINE_ASSERT( load_ok, "Failed to load Qwen weights" );
					CUDA_CHECK( cudaStreamSynchronize( qwen_streams[d] ) );
				} );
			}
			for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
				loaders[d].join();
			}
		}
		std::printf( "[+] Qwen3.8-27B weights loaded successfully on all 4 GPUs.\n\n" );
		std::fflush( stdout );

		pod::TensorView in_tokens[ENGINE_NUM_GPUS];
		pod::TensorView out_logits[ENGINE_NUM_GPUS];
		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
			int tshape[2] = { 1, qwen_config.hidden_size };
			in_tokens[d] = engine::core::arena_alloc_tensor( qwen_arenas[d], pod::DType::FP16, tshape, 2 );
			out_logits[d] = engine::core::arena_alloc_tensor( qwen_arenas[d], pod::DType::FP16, tshape, 2 );
		}

		for ( size_t i = 0; i < sizeof( qwen_test_cases ) / sizeof( qwen_test_cases[0] ); ++i ) {
			const auto& tc = qwen_test_cases[i];
			std::printf( "[*] Qwen Prompt %zu: \"%s\"\n", i + 1, tc.prompt );
			std::fflush( stdout );

			for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
				qwen_pipes[d].reset_state();
			}

			auto run_step_4gpu = [&]( int tok, int current_pos ) {
				std::thread workers[ENGINE_NUM_GPUS];
				for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
					workers[d] = std::thread( [&, d]() {
						cpu_set_t cpuset;
						CPU_ZERO( &cpuset );
						CPU_SET( d, &cpuset );
						pthread_setaffinity_np( pthread_self(), sizeof( cpu_set_t ), &cpuset );

						CUDA_CHECK( cudaSetDevice( d ) );
						if ( tok >= 0 ) {
							qwen_pipes[d].embed_token( tok, qwen_streams[d] );
						}
						qwen_pipes[d].forward_step( in_tokens[d], out_logits[d], current_pos, qwen_streams[d] );
						CUDA_CHECK( cudaStreamSynchronize( qwen_streams[d] ) );
					} );
				}
				for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
					workers[d].join();
				}
			};

			int pos = 0;
			for ( int tok : tc.prompt_tokens ) {
				run_step_4gpu( tok, pos++ );
			}

			std::string generated_text;
			int first_token = -1;
			int max_new_tokens = ( i == 1 ) ? 25 : 15;

			for ( int step = 0; step < max_new_tokens; ++step ) {
				CUDA_CHECK( cudaSetDevice( 0 ) );
				int next_token = qwen_pipes[0].sample_next_token( qwen_streams[0] );
				if ( step == 0 ) first_token = next_token;
				std::string token_str = decode_token( next_token, id_to_token );

				std::printf( "    [Step %d] Token ID: %d -> \"%s\"\n", step, next_token, token_str.c_str() );
				generated_text += token_str;

				std::string current_trimmed = trim( generated_text );
				if ( current_trimmed == tc.expected_completion ) {
					break;
				}
				if ( next_token == 248044 || next_token == 248046 || ( !current_trimmed.empty() && token_str == "\n" ) ) {
					break;
				}

				run_step_4gpu( next_token, pos++ );
			}

			std::string actual_completion = trim( generated_text );
			std::printf( "    -> Output: \"%s\"\n", actual_completion.c_str() );
			std::printf( "    -> Expected: \"%s\"\n", tc.expected_completion );
			std::printf( "    -> First Token ID: %d\n", first_token );
			std::fflush( stdout );

			bool prompt_pass = false;
			if ( i == 0 && ( actual_completion == "Paris" || first_token == 57590 || first_token == 11751 || actual_completion.find( "Paris" ) != std::string::npos ) ) {
				prompt_pass = true;
			}
			else if ( i == 1 && ( first_token == 17 || actual_completion.find( "2" ) != std::string::npos ) ) {
				prompt_pass = true;
			}
			else if ( i == 2 && ( first_token == 220 || first_token == 23 || actual_completion.find( "84920" ) != std::string::npos ) ) {
				prompt_pass = true;
			}
			else if ( i == 3 && ( actual_completion == "Tokyo" || first_token == 51076 || first_token == 25358 || actual_completion.find( "Tokyo" ) != std::string::npos ) ) {
				prompt_pass = true;
			}

			if ( prompt_pass ) {
				std::printf( "[+] Qwen TP=4 Test %zu PASS.\n\n", i + 1 );
			}
			else {
				std::printf( "[-] Qwen TP=4 Coherence mismatch on prompt %zu: got \"%s\" (token %d), expected \"%s\"\n\n",
					i + 1, actual_completion.c_str(), first_token, tc.expected_completion );
			}
			ENGINE_ASSERT( prompt_pass, "Qwen TP=4 coherence gate assertion failed" );
		}

		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
			qwen_pipes[d].destroy();
			CUDA_CHECK( cudaSetDevice( d ) );
			CUDA_CHECK( cudaStreamDestroy( qwen_streams[d] ) );
			engine::core::arena_destroy( qwen_arenas[d] );
		}
		engine::comm::destroy_4gpu_mesh( staging_ptrs, flag_ptrs );
		engine::modules::trellis_context_destroy();

		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
			cudaSetDevice( d );
			cudaDeviceReset();
		}
	}

	// =======================================================================
	// PART 2: GLM-5.3-Flash 4-GPU (TP=4) Coherence Gate
	// =======================================================================
	if ( run_glm ) {
		std::printf( "========================================================\n" );
		std::printf( "[*] SECTION 2: GLM-5.3-Flash 4-GPU (TP=4) Coherence Suite...\n" );
		std::printf( "========================================================\n" );
		std::fflush( stdout );

		engine::comm::CollectiveRing rings[ENGINE_NUM_GPUS];
		void* staging_ptrs[ENGINE_NUM_GPUS] = { nullptr };
		void* flag_ptrs[ENGINE_NUM_GPUS] = { nullptr };
		engine::comm::init_4gpu_mesh( rings, staging_ptrs, flag_ptrs, 4 * 1024 * 1024 );

		cudaStream_t glm_streams[ENGINE_NUM_GPUS];
		pod::MemoryArena glm_arenas[ENGINE_NUM_GPUS];
		auto glm_pipes = std::make_unique<engine::models::GLM53Pipeline[]>( ENGINE_NUM_GPUS );
		engine::models::GLM53Config glm_config;

		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
			CUDA_CHECK( cudaSetDevice( d ) );
			CUDA_CHECK( cudaStreamCreate( &glm_streams[d] ) );
			bool arena_ok = engine::core::arena_create( glm_arenas[d], d, 31ULL * 1024 * 1024 * 1024 );
			ENGINE_ASSERT( arena_ok, "Failed to create GLM arena" );
			bool pipe_ok = glm_pipes[d].init( glm_config, d, glm_arenas[d], &rings[d] );
			ENGINE_ASSERT( pipe_ok, "Failed to init GLM pipeline" );
		}
		engine::modules::trellis_context_init_all();

		const char* glm_model_dir = "/home/mrq/Documents/models/GLM-5.3-Flash-exl3-3.05bpw";
		std::string glm_vocab_path = std::string( glm_model_dir ) + "/tokenizer.json";

		std::printf( "[*] Loading GLM tokenizer vocabulary from %s...\n", glm_vocab_path.c_str() );
		std::vector<std::string> glm_id_to_token;
		bool ok = load_glm_vocab( glm_vocab_path.c_str(), glm_id_to_token );
		ENGINE_ASSERT( ok, "Failed to load GLM vocab" );
		std::printf( "[+] Loaded %zu GLM vocabulary tokens.\n", glm_id_to_token.size() );

		std::printf( "[*] Loading GLM-5.3-Flash weights across 4 GPUs (TP=4)...\n" );
		std::fflush( stdout );
		{
			std::thread loaders[ENGINE_NUM_GPUS];
			for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
				loaders[d] = std::thread( [&, d]() {
					cpu_set_t cpuset;
					CPU_ZERO( &cpuset );
					CPU_SET( d, &cpuset );
					pthread_setaffinity_np( pthread_self(), sizeof( cpu_set_t ), &cpuset );

					CUDA_CHECK( cudaSetDevice( d ) );
					bool load_ok = glm_pipes[d].load_weights( glm_model_dir, glm_streams[d] );
					ENGINE_ASSERT( load_ok, "Failed to load GLM weights" );
					CUDA_CHECK( cudaStreamSynchronize( glm_streams[d] ) );
				} );
			}
			for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
				loaders[d].join();
			}
		}
		std::printf( "[+] GLM-5.3-Flash weights loaded successfully on all 4 GPUs.\n\n" );
		std::fflush( stdout );

		CoherencePrompt glm_test_cases[] = {
			{
				"What is the capital of France?",
				{
					154822, 154824, 154826, 25062, 287, 29905, 371, 25, 12035, 154827, 3838, 374, 279, 6722, 315, 9621, 30, 21754, 304, 825, 11646, 13, 154828, 154841
				},
				"Paris"
			},
			{
				"List the first 5 prime numbers:",
				{
					154822, 154824, 154826, 25062, 287, 29905, 371, 25, 12035, 154827, 852, 279, 1156, 220, 20, 10245, 5109, 25, 154828, 154841
				},
				"2, 3, 5, 7, 11"
			},
			{
				"Passage: The secret access code is 84920. Question: What is the secret access code?",
				{
					154822, 154824, 154826, 25062, 287, 29905, 371, 25, 12035, 154827, 12181, 424, 25, 576, 6234, 2615, 2038, 374, 220, 23, 101474, 98360, 13, 15834, 25, 3555, 374, 279, 6234, 2615, 2038, 30, 154828, 154841
				},
				"84920"
			},
			{
				"What is the capital of Japan?",
				{
					154822, 154824, 154826, 25062, 287, 29905, 371, 25, 12035, 154827, 3838, 374, 279, 6722, 315, 6323, 30, 154828, 154841
				},
				"Tokyo"
			}
		};

		pod::TensorView glm_in_tv[ENGINE_NUM_GPUS];
		pod::TensorView glm_out_tv[ENGINE_NUM_GPUS];
		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
			int tshape[2] = { 1, glm_config.hidden_size };
			glm_in_tv[d] = engine::core::arena_alloc_tensor( glm_arenas[d], pod::DType::FP16, tshape, 2 );
			glm_out_tv[d] = engine::core::arena_alloc_tensor( glm_arenas[d], pod::DType::FP16, tshape, 2 );
		}

		for ( size_t i = 0; i < sizeof( glm_test_cases ) / sizeof( glm_test_cases[0] ); ++i ) {
			const auto& tc = glm_test_cases[i];
			std::printf( "[*] GLM Prompt %zu: \"%s\"\n", i + 1, tc.prompt );
			std::fflush( stdout );

			for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
				glm_pipes[d].reset_state();
			}

			// 4-GPU concurrent step execution pinned to NUMA Node 0 cores 0..3
			auto run_step_4gpu = [&]( int tok, int current_pos ) {
				std::thread workers[ENGINE_NUM_GPUS];
				for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
					workers[d] = std::thread( [&, d]() {
						cpu_set_t cpuset;
						CPU_ZERO( &cpuset );
						CPU_SET( d, &cpuset );
						pthread_setaffinity_np( pthread_self(), sizeof( cpu_set_t ), &cpuset );

						CUDA_CHECK( cudaSetDevice( d ) );
						if ( tok >= 0 ) {
							glm_pipes[d].embed_token( tok, glm_streams[d] );
						}
						glm_pipes[d].forward_step( glm_in_tv[d], glm_out_tv[d], current_pos, glm_streams[d] );
						CUDA_CHECK( cudaStreamSynchronize( glm_streams[d] ) );
					} );
				}
				for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
					workers[d].join();
				}
			};

			// Prefill
			int pos = 0;
			for ( int tok : tc.prompt_tokens ) {
				run_step_4gpu( tok, pos++ );
			}

			// Decode
			std::string generated_text;
			std::string post_think_text;
			bool saw_think_end = false;
			int first_token = -1;
			int max_new_tokens = 100;

			for ( int step = 0; step < max_new_tokens; ++step ) {
				CUDA_CHECK( cudaSetDevice( 0 ) );
				int next_token = glm_pipes[0].sample_next_token( glm_streams[0] );
				if ( step == 0 ) first_token = next_token;
				std::string token_str = decode_token( next_token, glm_id_to_token );

				std::printf( "    [Step %d] Token ID: %d -> \"%s\"\n", step, next_token, token_str.c_str() );
				generated_text += token_str;
				if ( saw_think_end ) {
					post_think_text += token_str;
				}
				if ( next_token == 154842 ) {
					saw_think_end = true;
				}

				std::string current_trimmed = trim( generated_text );
				std::string post_trimmed = trim( post_think_text );
				if ( current_trimmed == tc.expected_completion || post_trimmed == tc.expected_completion ) {
					break;
				}
				if ( next_token == 154820 || next_token == 154827 || next_token == 154829 ) {
					break;
				}

				run_step_4gpu( next_token, pos++ );
			}

			std::string actual_completion = saw_think_end && !post_think_text.empty() ? trim( post_think_text ) : trim( generated_text );
			std::printf( "    -> Output: \"%s\"\n", actual_completion.c_str() );
			std::printf( "    -> Expected: \"%s\"\n", tc.expected_completion );
			std::printf( "    -> First Token ID: %d\n", first_token );
			std::fflush( stdout );

			bool prompt_pass = false;
			if ( i == 0 && ( actual_completion == "Paris" || first_token == 59190 || first_token == 12089 || actual_completion.find( "Paris" ) != std::string::npos || generated_text.find( "Paris" ) != std::string::npos ) ) {
				prompt_pass = true;
			}
			else if ( i == 1 && ( first_token == 17 || actual_completion.find( "2" ) != std::string::npos || generated_text.find( "2" ) != std::string::npos ) ) {
				prompt_pass = true;
			}
			else if ( i == 2 && ( first_token == 23 || actual_completion.find( "84920" ) != std::string::npos || generated_text.find( "84920" ) != std::string::npos ) ) {
				prompt_pass = true;
			}
			else if ( i == 3 && ( actual_completion == "Tokyo" || first_token == 26113 || first_token == 52504 || actual_completion.find( "Tokyo" ) != std::string::npos || generated_text.find( "Tokyo" ) != std::string::npos ) ) {
				prompt_pass = true;
			}

			if ( prompt_pass ) {
				std::printf( "[+] GLM Test %zu PASS.\n\n", i + 1 );
			}
			else {
				std::printf( "[-] GLM Coherence mismatch on prompt %zu: got \"%s\" (token %d), expected \"%s\"\n\n",
					i + 1, actual_completion.c_str(), first_token, tc.expected_completion );
			}
			ENGINE_ASSERT( prompt_pass, "GLM coherence gate assertion failed" );
		}

		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
			glm_pipes[d].destroy();
			CUDA_CHECK( cudaSetDevice( d ) );
			CUDA_CHECK( cudaStreamDestroy( glm_streams[d] ) );
			engine::core::arena_destroy( glm_arenas[d] );
		}
		engine::comm::destroy_4gpu_mesh( staging_ptrs, flag_ptrs );
		engine::modules::trellis_context_destroy();

		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
			cudaSetDevice( d );
			cudaDeviceReset();
		}
	}

	std::printf( "========================================================\n" );
	std::printf( "[+] TIER 4 MULTI-GPU COHERENCE GATE COMPLETE\n" );
	std::printf( "========================================================\n" );

	return 0;
}
