#include "engine/runtime/cluster.h"
#include "trellis_context.h"
#include <cstdio>
#include <chrono>
#include <cstring>
#include <thread>

namespace engine::runtime {

bool ModelCluster::init( const ClusterConfig& cfg ) {
	ENGINE_ASSERT( !initialized_, "ModelCluster::init: already initialized" );
	const bool is_glm = ( cfg.model_type == "glm53" || cfg.model_type == "glm" );
	const size_t arena_bytes = cfg.arena_bytes ? cfg.arena_bytes : ( is_glm ? 31ULL : 10ULL ) * 1024 * 1024 * 1024;

	created_ = true;
	engine::comm::init_4gpu_mesh( rings_, staging_ptrs_, flag_ptrs_, 4 * 1024 * 1024 );
	engine::modules::trellis_context_init_all();

	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		CUDA_CHECK( cudaSetDevice( d ) );
		CUDA_CHECK( cudaStreamCreate( &streams_[d] ) );
		if ( !engine::core::arena_create( arenas_[d], d, arena_bytes ) ) return false;
		models_[d] = engine::models::create_model_pipeline( cfg.model_type );
		if ( !models_[d] || !models_[d]->init( d, arenas_[d], &rings_[d] ) ) return false;
	}

	std::printf( "[*] Loading %s weights across %d GPUs from %s...\n", cfg.model_type.c_str(), ENGINE_NUM_GPUS, cfg.model_dir.c_str() );
	std::fflush( stdout );
	pool_.start( ENGINE_NUM_GPUS );
	bool ok[ENGINE_NUM_GPUS] = {};
	pool_.run( [&]( int d ) {
		ok[d] = models_[d]->load_weights( cfg.model_dir.c_str(), streams_[d] );
		CUDA_CHECK( cudaStreamSynchronize( streams_[d] ) );
	} );
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		if ( !ok[d] ) return false;
		std::printf( "    [GPU %d] arena: %.3f / %.3f GiB\n", d,
			arenas_[d].offset_bytes / 1073741824.0, arenas_[d].total_bytes / 1073741824.0 );
	}

	if ( !cfg.draft_model_dir.empty() ) {
		const auto& meta = models_[0]->get_metadata();
		pod::DFlashConfig spec_cfg;
		spec_cfg.hidden_size = meta.hidden_size;
		spec_cfg.vocab_size = meta.vocab_size;
		spec_cfg.block_size = 8;
		CUDA_CHECK( cudaSetDevice( 0 ) );
		if ( !speculator_.init( spec_cfg, 0, arenas_[0], streams_[0] ) ) return false;
		const bool loaded = speculator_.load_weights( cfg.draft_model_dir.c_str(), streams_[0] );
		std::printf( "[+] DFlash2 speculator: %s (%s)\n", loaded ? "loaded" : "STANDALONE", cfg.draft_model_dir.c_str() );
		has_spec_ = true;
	}

	initialized_ = true;
	return true;
}

void ModelCluster::shutdown() {
	if ( !created_ ) return;
	created_ = false;
	initialized_ = false;
	pool_.stop();
	if ( has_spec_ ) {
		CUDA_CHECK( cudaSetDevice( 0 ) );
		speculator_.destroy();
		has_spec_ = false;
	}
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		CUDA_CHECK( cudaSetDevice( d ) );
		models_[d].reset();
		if ( streams_[d] ) {
			cudaStreamSynchronize( streams_[d] );
			cudaStreamDestroy( streams_[d] );
			streams_[d] = nullptr;
		}
		engine::core::arena_destroy( arenas_[d] );
	}
	engine::comm::destroy_4gpu_mesh( staging_ptrs_, flag_ptrs_ );
	engine::modules::trellis_context_destroy();
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		cudaSetDevice( d );
		cudaDeviceReset();
	}
}

bool ModelCluster::is_eos( int token ) const {
	const auto& meta = models_[0]->get_metadata();
	for ( int id : meta.eos_ids ) {
		if ( id >= 0 && id == token ) return true;
	}
	return false;
}

void ModelCluster::forward( const int* tokens, const pod::BatchRows& rows ) {
	pool_.run( [&]( int d ) {
		models_[d]->embed_tokens( tokens, rows.count, streams_[d] );
		models_[d]->forward( rows, streams_[d] );
		CUDA_CHECK( cudaStreamSynchronize( streams_[d] ) );
	} );
}

void ModelCluster::forward( const int* tokens, int count, int pos, int slot ) {
	forward( tokens, pod::BatchRows::causal( count, pos, slot ) );
}

int ModelCluster::prefill( const int* tokens, int count, int slot, int chunk ) {
	ENGINE_ASSERT( count >= 1, "ModelCluster::prefill: empty prompt" );
	chunk = MAX( 1, MIN( chunk, metadata().max_rows ) );
	int last_rows = 1;
	for ( int pos = 0; pos < count; pos += chunk ) {
		last_rows = MIN( chunk, count - pos );
		forward( tokens + pos, last_rows, pos, slot );
	}
	int preds[pod::MAX_BATCH];
	sample( preds, last_rows );
	return preds[last_rows - 1];
}

void ModelCluster::sample( int* out, int count ) {
	CUDA_CHECK( cudaSetDevice( 0 ) );
	models_[0]->sample_tokens( out, count, streams_[0] );
}

int ModelCluster::sample_one() {
	int tok = -1;
	sample( &tok, 1 );
	return tok;
}

void ModelCluster::warm_graphs( int max_m, int slot ) {
	max_m = MIN( max_m, metadata().max_rows );
	const int tokens[pod::MAX_BATCH] = {};
	auto t0 = std::chrono::steady_clock::now();
	for ( int m = 1; m <= max_m; ++m ) {
		forward( tokens, m, 0, slot ); // eager warm-up
		forward( tokens, m, 0, slot ); // capture + first replay
	}
	reset_slot( slot );
	std::printf( "[+] Decode graphs warmed for m=1..%d in %.0f ms\n", max_m,
		std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t0 ).count() );
}

void ModelCluster::reset_slot( int slot ) {
	pool_.run( [&]( int d ) {
		models_[d]->reset_state( slot );
	} );
}

void ModelCluster::checkpoint( int slot ) {
	pool_.run( [&]( int d ) {
		models_[d]->checkpoint_state( slot, streams_[d] );
	} );
}

void ModelCluster::rollback( int slot ) {
	pool_.run( [&]( int d ) {
		models_[d]->rollback_state( slot, streams_[d] );
	} );
}

SpecResult ModelCluster::speculate( int slot, int anchor, int pos, const std::vector<int>& context, float conf_threshold ) {
	ENGINE_ASSERT( has_spec_, "ModelCluster::speculate: no speculator loaded" );
	const auto& meta = metadata();
	SpecResult res;

	// 1. Anchor forward: logits row 0 predicts the token after the anchor; tapped states feed the drafter
	forward( &anchor, 1, pos, slot );
	const int target_pred = sample_one();

	// 2. Draft K tokens conditioned on target logits, context and tapped hidden states
	CUDA_CHECK( cudaSetDevice( 0 ) );
	speculator_.draft( anchor, pos + 1, streams_[0], models_[0]->logits().ptr<const half>(),
		context.data(), static_cast<int>( context.size() ), models_[0]->tapped_states() );
	int h_draft[8];
	float h_conf[8];
	CUDA_CHECK( cudaMemcpyAsync( h_draft, speculator_.d_draft_tokens(), sizeof( h_draft ), cudaMemcpyDeviceToHost, streams_[0] ) );
	CUDA_CHECK( cudaMemcpyAsync( h_conf, speculator_.d_conf(), sizeof( h_conf ), cudaMemcpyDeviceToHost, streams_[0] ) );
	CUDA_CHECK( cudaStreamSynchronize( streams_[0] ) );

	// Confidence truncation; keep >= 1 and stay inside the context window
	int active = 7;
	for ( int k = 1; k <= 7; ++k ) {
		if ( h_conf[k] < conf_threshold ) {
			active = k - 1;
			break;
		}
	}
	active = MIN( MAX( active, 1 ), meta.max_seq_len - pos - 2 );
	if ( active < 1 ) {
		res.emitted[res.emitted_count++] = target_pred;
		res.next_pos = pos + 1;
		return res;
	}
	res.proposed = active;
	const int* cands = h_draft + 1;

	// 3. Batched verify of the candidate block [pos + 1, pos + 1 + active)
	checkpoint( slot );
	forward( cands, active, pos + 1, slot );
	int preds[pod::MAX_BATCH];
	sample( preds, active );

	// 4. Greedy acceptance: candidate k is checked against the prediction of the previous position
	for ( int k = 0; k < active; ++k ) {
		const int expected = ( k == 0 ) ? target_pred : preds[k - 1];
		if ( cands[k] != expected ) {
			res.emitted[res.emitted_count++] = expected; // correction token
			break;
		}
		res.emitted[res.emitted_count++] = cands[k];
		res.accepted++;
	}
	if ( res.accepted == active ) {
		res.emitted[res.emitted_count++] = preds[active - 1]; // bonus token
	}

	// 5. State repair: recurrent state holds all candidates; restore and replay the accepted prefix
	if ( res.accepted < active ) {
		rollback( slot );
		if ( res.accepted > 0 ) {
			forward( cands, res.accepted, pos + 1, slot );
		}
	}
	res.next_pos = pos + 1 + res.accepted;
	return res;
}

} // namespace engine::runtime
