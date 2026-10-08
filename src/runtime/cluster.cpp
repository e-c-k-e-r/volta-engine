#include "engine/runtime/cluster.h"
#include "engine/runtime/options.h"
#include "engine/backend/runtime.h"
#if ENGINE_BACKEND_CUDA
#include "trellis_context.h"
#endif
#include <cmath>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <thread>

namespace engine::runtime {

bool ModelCluster::init( const ClusterConfig& cfg ) {
	ENGINE_ASSERT( !initialized_, "ModelCluster::init: already initialized" );
	if ( !engine::models::is_known_model_type( cfg.model_type ) ) {
		std::fprintf( stderr, "[-] ModelCluster: unsupported model_type '%s'\n", cfg.model_type.c_str() );
		return false;
	}
	const bool is_glm = ( cfg.model_type == "glm5_next" );
	// EXL3_ARENA_GIB overrides the per-GPU arena. The default of 10 GiB is sized for the quantized
	// drafter; loading the unquantized one needs roughly twice the weight bytes.
	const auto& opt = runtime::options();
	const bool arena_env = opt.arena_gib >= 0;
	// A tiered GLM (EXL3_EXPERT_CACHE < 1) keeps only that fraction of the routed experts (~26 GiB per rank in all) in the arena
	size_t glm_gib = 31;
	if ( is_glm ) {
		const double f = opt.expert_cache;
		if ( f > 0.0 && f < 1.0 ) glm_gib = static_cast<size_t>( std::ceil( 31.0 - ( 1.0 - f ) * 25.0 ) );
	}
	tp_ = opt.tp > 0 ? opt.tp : cfg.tp;
	// A single-rank group holds the whole model, so the 4-rank defaults are meaningless: take the device's free memory
	// less a reserve for what lives outside the arena (graphs, trellis workspace, collective scratch, CUDA context).
	// EXL3_VRAM_RESERVE_MIB overrides the reserve; EXL3_ARENA_GIB or cfg.arena_bytes still win.
	size_t arena_bytes = cfg.arena_bytes ? cfg.arena_bytes
		: ( arena_env ? static_cast<size_t>( opt.arena_gib ) : ( is_glm ? glm_gib : 10ULL ) )
			* 1024 * 1024 * 1024;
	if ( tp_ == 1 && !cfg.arena_bytes && !arena_env ) {
		const size_t reserve = ( opt.vram_reserve_mib >= 0 ? static_cast<size_t>( opt.vram_reserve_mib ) : 1536ULL ) * 1024 * 1024;
		const size_t free_bytes = backend::device_free_bytes( 0 );
		if ( free_bytes > reserve ) {
			arena_bytes = ( free_bytes - reserve ) & ~( ( size_t ) 0xFFFFF );
			std::printf( "[*] Arena sized from free memory: %.2f GiB (%.2f GiB free, %.2f GiB reserve)\n",
				arena_bytes / 1073741824.0, free_bytes / 1073741824.0, reserve / 1073741824.0 );
		}
	}
	ENGINE_ASSERT( tp_ == 1 || tp_ == ENGINE_NUM_GPUS, "ModelCluster::init: the collective supports tp 1 or 4 ranks" );
	created_ = true;
	if ( tp_ > 1 ) engine::comm::init_4gpu_mesh( rings_, staging_ptrs_, flag_ptrs_, engine::comm::kMeshStagingBytes );
#if ENGINE_BACKEND_CUDA
	engine::modules::trellis_context_init_all();
#endif

	for ( int d = 0; d < tp_; ++d ) {
		backend::set_device( d );
		streams_[d] = backend::stream_create();
		if ( !engine::core::arena_create( arenas_[d], d, arena_bytes ) ) return false;
		models_[d] = engine::models::create_model_pipeline( cfg.model_type );
		if ( !models_[d] ) return false;
		models_[d]->set_tp_size( tp_ );
		if ( !models_[d]->init( d, arenas_[d], ( tp_ > 1 ) ? &rings_[d] : nullptr ) ) return false;
	}

	std::printf( "[*] Loading %s weights across %d GPUs from %s...\n", cfg.model_type.c_str(), tp_, cfg.model_dir.c_str() );
	std::fflush( stdout );
	pool_.start( tp_ );
	bool ok[ENGINE_NUM_GPUS] = {};
	pool_.run( [&]( int d ) {
		ok[d] = models_[d]->load_weights( cfg.model_dir.c_str(), streams_[d] );
		backend::stream_sync( streams_[d] );
	} );
	for ( int d = 0; d < tp_; ++d ) {
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
	backend::clear_last_error();
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
	for ( int d = 0; d < tp_; ++d ) {
		backend::set_device( d );
		models_[d].reset();
		if ( streams_[d] ) {
			backend::stream_sync( streams_[d] );
			backend::stream_destroy( streams_[d] );
			streams_[d] = nullptr;
		}
		engine::core::arena_destroy( arenas_[d] );
	}
	if ( tp_ > 1 ) engine::comm::destroy_4gpu_mesh( staging_ptrs_, flag_ptrs_ );
#if ENGINE_BACKEND_CUDA
	engine::modules::trellis_context_destroy();
#endif
	for ( int d = 0; d < tp_; ++d ) {
		backend::set_device( d );
		backend::device_reset();
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
		backend::stream_sync( streams_[d] );
	} );
}

void ModelCluster::forward( const int* tokens, int count, int pos, int slot ) {
	forward( tokens, pod::BatchRows::causal( count, pos, slot ) );
}

int ModelCluster::prefill( const int* tokens, int count, int slot, int chunk, Candidates* last_cands ) {
	ENGINE_ASSERT( count >= 1, "ModelCluster::prefill: empty prompt" );
	const int max_rows = metadata().max_rows;
	chunk = ( chunk <= 0 ) ? max_rows : MAX( 1, MIN( chunk, max_rows ) );
	int last_rows = 1;
	for ( int pos = 0; pos < count; pos += chunk ) {
		last_rows = MIN( chunk, count - pos );
		const bool last = ( pos + last_rows >= count );
		forward( tokens + pos, pod::BatchRows::causal( last_rows, pos, slot,
			last ? pod::BatchRows::LOGITS_LAST : pod::BatchRows::LOGITS_NONE ) );
	}
	// A final chunk above the decode sizes computed logits for its last row only (in row 0).
	const int rows = ( last_rows > pod::MAX_BATCH ) ? 1 : last_rows;
	int preds[pod::MAX_BATCH];
	sample( preds, rows );
	if ( last_cands != nullptr ) {
		Candidates cs[pod::MAX_BATCH];
		sample_topk( rows, cs );
		*last_cands = cs[rows - 1];
	}
	return preds[rows - 1];
}

void ModelCluster::sample_topk( int count, Candidates* out ) {
	ENGINE_ASSERT( count >= 1 && count <= pod::MAX_BATCH, "ModelCluster::sample_topk: row count out of range" );
	constexpr int K = kSampleTopK;
	float vals[ENGINE_NUM_GPUS][pod::MAX_BATCH * K];
	int ids[ENGINE_NUM_GPUS][pod::MAX_BATCH * K];
	int ok[ENGINE_NUM_GPUS] = {};
	pool_.run( [&]( int d ) {
		ok[d] = models_[d]->topk_local( vals[d], ids[d], count, streams_[d] ) ? 1 : 0;
	} );
	for ( int d = 0; d < tp_; ++d ) {
		ENGINE_ASSERT( ok[d] != 0, "ModelCluster::sample_topk: the model has no vocabulary-parallel top-k" );
	}
	// A rank's local top-16 is a superset of the global top-16 restricted to its shard, so the union holds the
	// answer. Ties keep the lower token id (argmax convention).
	for ( int r = 0; r < count; ++r ) {
		float v[ENGINE_NUM_GPUS * K];
		int t[ENGINE_NUM_GPUS * K];
		int n = 0;
		for ( int d = 0; d < tp_; ++d ) {
			for ( int j = 0; j < K; ++j ) {
				v[n] = vals[d][r * K + j];
				t[n] = ids[d][r * K + j];
				++n;
			}
		}
		Candidates& c = out[r];
		c.n = K;
		for ( int j = 0; j < K; ++j ) {
			int best = j;
			for ( int i = j + 1; i < n; ++i ) {
				if ( v[i] > v[best] || ( v[i] == v[best] && t[i] < t[best] ) ) best = i;
			}
			std::swap( v[j], v[best] );
			std::swap( t[j], t[best] );
			c.logits[j] = v[j];
			c.ids[j] = t[j];
		}
	}
}

void ModelCluster::sample( int* out, int count ) {
	backend::set_device( 0 );

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
	for ( int d = 1; d < tp_; ++d ) {
		ENGINE_ASSERT( sharded[d] != 0, "ModelCluster::sample: vocabulary sharding must be uniform across ranks" );
	}
	for ( int r = 0; r < count; ++r ) {
		int best = toks[0][r];
		float best_val = vals[0][r];
		for ( int d = 1; d < tp_; ++d ) {
			if ( vals[d][r] > best_val || ( vals[d][r] == best_val && toks[d][r] < best ) ) {
				best_val = vals[d][r];
				best = toks[d][r];
			}
		}
		out[r] = best;
	}
}

void ModelCluster::read_logits( int rows, backend::half_t* host_dst ) {
	backend::set_device( 0 );
	const backend::half_t* peer_shards[ENGINE_NUM_GPUS] = { nullptr };
	int peer_cols = 0;
	for ( int d = 0; d < tp_; ++d ) {
		peer_shards[d] = models_[d]->shard_logits_ptr();
		peer_cols = models_[d]->shard_logits_cols();
	}
	const pod::TensorView& full = models_[0]->gather_logits( rows, peer_shards, peer_cols, streams_[0] );
	const size_t bytes = static_cast<size_t>( rows ) * metadata().vocab_size * sizeof( backend::half_t );
	backend::memcpy_async( host_dst, full.data, bytes, backend::Copy::D2H, streams_[0] );
	backend::stream_sync( streams_[0] );
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
	// Rows spread over several slots can take a different graph variant than a causal block in one slot
	const int slots = MIN( metadata().max_slots, pod::MAX_BATCH );
	if ( slots >= 2 ) {
		for ( int m = 2; m <= max_m; ++m ) {
			int ids[pod::MAX_BATCH], positions[pod::MAX_BATCH];
			for ( int r = 0; r < m; ++r ) { ids[r] = r % slots; positions[r] = 0; }
			const pod::BatchRows rows = pod::BatchRows::batched( m, ids, positions );
			forward( tokens, rows );
			forward( tokens, rows );
		}
		for ( int s = 0; s < slots; ++s ) reset_slot( s );
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

bool ModelCluster::rebalance_experts() {
	int any[ENGINE_NUM_GPUS] = {};
	pool_.run( [&]( int d ) {
		any[d] = models_[d]->rebalance_experts( streams_[d] ) ? 1 : 0;
	} );
	return any[0] != 0;
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
