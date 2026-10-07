#include "engine/runtime/cluster.h"
#include "trellis_context.h"
#include <cmath>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <thread>

namespace engine::runtime {

bool ModelCluster::init( const ClusterConfig& cfg ) {
	ENGINE_ASSERT( !initialized_, "ModelCluster::init: already initialized" );
	const bool is_glm = ( cfg.model_type == "glm53" || cfg.model_type == "glm" );
	// EXL3_ARENA_GIB overrides the per-GPU arena. The default of 10 GiB is sized for the quantized
	// drafter; loading the unquantized one needs roughly twice the weight bytes.
	const char* arena_env = std::getenv( "EXL3_ARENA_GIB" );
	const size_t arena_bytes = cfg.arena_bytes ? cfg.arena_bytes
		: ( arena_env ? strtoull( arena_env, nullptr, 10 ) : ( is_glm ? 31ULL : 10ULL ) )
			* 1024 * 1024 * 1024;

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
		runner_ = std::make_unique<DFlash2Runner>();
		if ( !runner_->init( *this, cfg.draft_model_dir ) ) return false;
	}

	// arena_create on the third device leaves cudaErrorInvalidValue pending on this thread although every
	// call it makes reports success (bisected with cudaPeekAtLastError; source inside the driver, not
	// identified). Clear it so a later CUDA_CHECK( cudaGetLastError() ) on this thread, such as the
	// speculative round's, reports only errors raised after init.
	(void) cudaGetLastError();
	initialized_ = true;
	return true;
}

void ModelCluster::shutdown() {
	if ( !created_ ) return;
	created_ = false;
	initialized_ = false;
	pool_.stop();
	if ( runner_ ) {
		runner_->shutdown();
		runner_.reset();
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

	// Vocabulary-parallel path: every rank picks the best token inside its own shard, and the winner
	// is the largest logit across ranks. Each rank syncs its own stream inside sample_local, so by the
	// time the fan-out returns all four (token, logit) pairs are in host memory. Ties keep the lower
	// token id, matching the single-shard argmax tie-break.
	int toks[ENGINE_NUM_GPUS][pod::MAX_BATCH];
	float vals[ENGINE_NUM_GPUS][pod::MAX_BATCH];
	int sharded[ENGINE_NUM_GPUS] = {};
	pool_.run( [&]( int d ) {
		toks[d][0] = -1;
		sharded[d] = models_[d]->sample_local( toks[d], vals[d], count, streams_[d] ) ? 1 : 0;
	} );
	if ( !sharded[0] ) {
		models_[0]->sample_tokens( out, count, streams_[0] );
		return;
	}
	for ( int d = 1; d < ENGINE_NUM_GPUS; ++d ) {
		ENGINE_ASSERT( sharded[d] != 0, "ModelCluster::sample: vocabulary sharding must be uniform across ranks" );
	}
	for ( int r = 0; r < count; ++r ) {
		int best = toks[0][r];
		float best_val = vals[0][r];
		for ( int d = 1; d < ENGINE_NUM_GPUS; ++d ) {
			if ( vals[d][r] > best_val || ( vals[d][r] == best_val && toks[d][r] < best ) ) {
				best_val = vals[d][r];
				best = toks[d][r];
			}
		}
		out[r] = best;
	}
}

void ModelCluster::read_logits( int rows, half* host_dst ) {
	CUDA_CHECK( cudaSetDevice( 0 ) );
	const half* peer_shards[ENGINE_NUM_GPUS] = { nullptr };
	int peer_cols = 0;
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		peer_shards[d] = models_[d]->shard_logits_ptr();
		peer_cols = models_[d]->shard_logits_cols();
	}
	const pod::TensorView& full = models_[0]->gather_logits( rows, peer_shards, peer_cols, streams_[0] );
	const size_t bytes = static_cast<size_t>( rows ) * metadata().vocab_size * sizeof( half );
	CUDA_CHECK( cudaMemcpyAsync( host_dst, full.data, bytes, cudaMemcpyDeviceToHost, streams_[0] ) );
	CUDA_CHECK( cudaStreamSynchronize( streams_[0] ) );
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
	// The drafter's context (tap ring, projected KV, fill level) describes one slot at a time. The
	// slot being recycled takes it over, so its prefill publishes taps from position 0.
	if ( runner_ ) runner_->claim( slot );
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

void ModelCluster::set_spec_snapshot( bool on ) {
	pool_.run( [&]( int d ) {
		models_[d]->set_spec_snapshot( on, streams_[d] );
	} );
}

void ModelCluster::commit_spec_row( int slot, int row ) {
	pool_.run( [&]( int d ) {
		models_[d]->commit_spec_row( slot, row, streams_[d] );
	} );
}

} // namespace engine::runtime
