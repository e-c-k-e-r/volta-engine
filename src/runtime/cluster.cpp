#include "engine/runtime/cluster.h"
#include "engine/backend/tiers.h"
#include "engine/runtime/mtp_runner.h"
#include "engine/runtime/suffix_runner.h"
#include "engine/runtime/dspark_runner.h"
#include "engine/core/log.h"
#include "engine/runtime/options.h"
#include "engine/backend/runtime.h"
#if ENGINE_BACKEND_CUDA
#include "trellis_context.h"
#endif
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <cstring>
#include <thread>

namespace engine::runtime {

bool ModelCluster::init( const ClusterConfig& cfg ) {
	ENGINE_ASSERT( !initialized_, "ModelCluster::init: already initialized" );
	if ( !engine::models::is_known_model_type( cfg.model_type ) ) {
		LOG_ERROR( "ModelCluster: unsupported model_type '%s'\n", cfg.model_type.c_str() );
		return false;
	}
	const bool is_glm = ( cfg.model_type == "glm5_next" );
	const auto& opt = runtime::options();
	tp_ = opt.tp > 0 ? opt.tp : cfg.tp;
	if ( opt.expert_stream ) runtime::options().no_graphs = true;   // the layers' expert reads are host work between kernels
#if ENGINE_BACKEND_CPU
	// The CPU backend is single-device: one rank holds the whole model
	if ( tp_ != 1 ) {
		LOG_INFO( "ModelCluster: tp %d requested; the %s backend runs one rank\n", tp_, ENGINE_BACKEND_NAME );
		tp_ = 1;
	}
#endif
#if ENGINE_BACKEND_VULKAN
	// Several Vulkan devices share nothing but the host: the all-reduce goes through it (src/backend/vulkan/mod_comm.cpp), which a graph cannot contain
	if ( tp_ > backend::device_count() ) {
		LOG_ERROR( "ModelCluster: tp %d requested, %d Vulkan devices available\n", tp_, backend::device_count() );
		return false;
	}
	if ( tp_ > 1 ) runtime::options().no_graphs = true;
#endif
#if ENGINE_BACKEND_CUDA
	// The configs in configs/ are written for four GPUs: on a smaller machine say so here, not from inside the collective's setup
	if ( tp_ > backend::device_count() ) {
		LOG_ERROR( "ModelCluster: tp %d requested, %d GPU(s) visible: set \"tp_size\" in the config (or --tp-size) to %d or fewer\n", tp_,
			backend::device_count(), backend::device_count() );
		return false;
	}
#endif
	// The arena is a bump allocator over one up-front allocation per GPU. Unless a size is given (cfg.arena_bytes,
	// arena_gib) it takes the device's free memory less a reserve for what lives outside it (graphs, workspaces, the
	// CUDA context); vram_reserve_mib overrides the reserve. The model's own tables (KV pools, scratch) are sized by
	// max_seq / max_slots, not by the arena, so a larger context needs no arena tuning.
	const size_t reserve = ( opt.vram_reserve_mib >= 0 ? static_cast<size_t>( opt.vram_reserve_mib ) : ( is_glm ? 288ULL : 1536ULL ) ) * 1024 * 1024;
	ENGINE_ASSERT( tp_ == 1 || tp_ == 2 || tp_ == ENGINE_NUM_GPUS, "ModelCluster::init: the collective supports tp 1, 2 or 4 ranks" );
	created_ = true;
	if ( tp_ > 1 ) engine::comm::init_4gpu_mesh( rings_, staging_ptrs_, flag_ptrs_, engine::comm::kMeshStagingBytes, tp_ );
#if ENGINE_BACKEND_CUDA
	engine::modules::trellis_context_init_all();
#endif

	for ( int d = 0; d < tp_; ++d ) {
		backend::set_device( d );
		streams_[d] = backend::stream_create();
		size_t arena_bytes = cfg.arena_bytes ? cfg.arena_bytes : static_cast<size_t>( std::max( opt.arena_gib, 0 ) ) * 1024 * 1024 * 1024;
		if ( arena_bytes == 0 ) {
			const size_t free_bytes = backend::device_free_bytes( d );
			if ( free_bytes <= reserve ) {
				LOG_ERROR( "ModelCluster: GPU %d has %.2f GiB free, less than the %.2f GiB reserve\n", d, free_bytes / 1073741824.0, reserve / 1073741824.0 );
				return false;
			}
			arena_bytes = ( free_bytes - reserve ) & ~( ( size_t ) 0xFFFFF );
			if ( d == 0 ) {
				LOG_INFO( "Arena sized from free memory: %.2f GiB per GPU (%.2f GiB free, %.2f GiB reserve)\n",
					arena_bytes / 1073741824.0, free_bytes / 1073741824.0, reserve / 1073741824.0 );
			}
		}
		if ( !engine::core::arena_create( arenas_[d], d, arena_bytes ) ) return false;
		models_[d] = engine::models::create_model_pipeline( cfg.model_type );
		if ( !models_[d] ) return false;
		models_[d]->set_tp_size( tp_ );
		if ( !models_[d]->init( d, arenas_[d], ( tp_ > 1 ) ? &rings_[d] : nullptr ) ) return false;
	}

	constexpr double kGiB = 1073741824.0;
	const size_t mem_buffers = arenas_[0].offset_bytes;
	LOG_INFO( "[mem] GPU 0 arena: %.2f GiB of %.2f GiB: buffers, caches and workspaces %.2f GiB\n", mem_buffers / kGiB, arenas_[0].total_bytes / kGiB, mem_buffers / kGiB );
	LOG_INFO( "Loading %s weights across %d GPUs from %s...\n", cfg.model_type.c_str(), tp_, cfg.model_dir.c_str() );
	std::fflush( stdout );
	pool_.start( tp_ );
	// What a model sizing itself from the free arena (expert_cache=auto) has to leave for the drafter
	size_t draft_bytes = 0;
	if ( !cfg.draft_model_dir.empty() && !( opt.speculator == "mtp" && cfg.speculate ) ) {
		std::error_code ec;
		for ( const auto& f : std::filesystem::directory_iterator( cfg.draft_model_dir, ec ) ) {
			if ( f.path().extension() == ".safetensors" ) draft_bytes += static_cast<size_t>( f.file_size( ec ) );
		}
	}
	for ( int d = 0; d < tp_; ++d ) models_[d]->set_reserve_after_weights( draft_bytes );
	bool ok[ENGINE_NUM_GPUS] = {};
	pool_.run( [&]( int d ) {
		ok[d] = models_[d]->load_weights( cfg.model_dir.c_str(), streams_[d] );
		backend::stream_sync( streams_[d] );
	} );
	for ( int d = 0; d < tp_; ++d ) {
		if ( !ok[d] ) return false;
		LOG_INFO( "[GPU %d] arena: %.3f / %.3f GiB\n", d,
			arenas_[d].offset_bytes / 1073741824.0, arenas_[d].total_bytes / 1073741824.0 );
	}

	const size_t mem_weights = arenas_[0].offset_bytes;
	LOG_INFO( "[mem] GPU 0 arena: weights %.2f GiB (%.2f GiB free)\n", ( mem_weights - mem_buffers ) / kGiB, ( arenas_[0].total_bytes - mem_weights ) / kGiB );
	if ( opt.speculator == "mtp" && cfg.speculate ) {
		if ( !models_[0]->has_mtp() ) {
			LOG_ERROR( "ModelCluster: speculator=mtp, but %s has no MTP head\n", cfg.model_type.c_str() );
			return false;
		}
		auto mtp = std::make_unique<MtpRunner>();
		if ( !mtp->init( *this ) ) return false;
		runner_ = std::move( mtp );
	}
	else if ( opt.speculator == "dspark" && cfg.speculate ) {
		if ( cfg.draft_model_dir.empty() ) {
			LOG_ERROR( "ModelCluster: speculator=dspark needs draft_model_dir (the directory quantize_dspark.py wrote)\n" );
			return false;
		}
		bool loaded[ENGINE_NUM_GPUS] = {};
		pool_.run( [&]( int d ) {
			loaded[d] = models_[d]->load_dspark( cfg.draft_model_dir.c_str(), streams_[d] );
			backend::stream_sync( streams_[d] );
		} );
		for ( int d = 0; d < tp_; ++d ) if ( !loaded[d] ) { LOG_ERROR( "ModelCluster: GPU %d could not load the DSpark drafter from %s\n", d, cfg.draft_model_dir.c_str() ); return false; }
		auto dspark = std::make_unique<DSparkRunner>();
		if ( !dspark->init( *this ) ) return false;
		runner_ = std::move( dspark );
	}
	else if ( opt.speculator == "suffix" && cfg.speculate ) {
		if ( !models_[0]->free_rollback() ) {
			LOG_ERROR( "ModelCluster: speculator=suffix needs a model whose caches are indexed by position; %s keeps recurrent state\n", cfg.model_type.c_str() );
			return false;
		}
		auto suffix = std::make_unique<SuffixRunner>();
		if ( !suffix->init( *this ) ) return false;
		runner_ = std::move( suffix );
	}
	else if ( !cfg.draft_model_dir.empty() ) {
		auto dflash = std::make_unique<DFlash2Runner>();
		if ( !dflash->init( *this, cfg.draft_model_dir ) ) return false;
		dflash_ = dflash.get();
		dflash->set_profile( opt.spec_profile );
		runner_ = std::move( dflash );
		for ( int d = 0; d < tp_; ++d ) models_[d]->set_snapshot_taps( true );
	}

	LOG_INFO( "[mem] GPU 0 arena: speculator %.2f GiB (%.2f GiB free)\n", ( arenas_[0].offset_bytes - mem_weights ) / kGiB, ( arenas_[0].total_bytes - arenas_[0].offset_bytes ) / kGiB );
	// Image encoder last, into whatever the arena has left (rank 0)
	models_[0]->load_vision( cfg.model_dir.c_str(), streams_[0] );

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
	if ( snapshot_prewarm_.joinable() ) snapshot_prewarm_.join();
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		for ( auto& kv : snapshot_free_[d] ) {
			for ( void* p : kv.second ) backend::host_free_pinned( p );
		}
		snapshot_free_[d].clear();
	}
	created_ = false;
	initialized_ = false;
	pool_.stop();
	if ( runner_ ) {
		if ( runtime::options().spec_profile ) runner_->profile_dump();
		runner_->shutdown();
		runner_.reset();
		dflash_ = nullptr;
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

void ModelCluster::set_slot_mm( int slot, MmLayout layout ) {
	if ( slot_mm_.size() <= static_cast<size_t>( slot ) ) slot_mm_.resize( slot + 1 );
	mm_slots_ += ( layout.empty() ? 0 : 1 ) - ( slot_mm_[slot].empty() ? 0 : 1 );
	slot_mm_[slot] = std::move( layout );
}

bool ModelCluster::encode_image( const engine::loader::PreparedImage& image, std::vector<backend::half_t>* out ) {
	backend::set_device( 0 );
	return models_[0]->encode_image( image, out, streams_[0] );
}

// Fills the M-RoPE positions of rows whose slot has images (and sets rows.mrope); a no-op when no slot has any
void ModelCluster::apply_mrope( pod::BatchRows& rows ) const {
	if ( mm_slots_ == 0 ) return;
	bool any = false;
	for ( int m = 0; m < rows.count; ++m ) {
		const int slot = rows.slot[m];
		const MmLayout* layout = ( static_cast<size_t>( slot ) < slot_mm_.size() && !slot_mm_[slot].empty() ) ? &slot_mm_[slot] : nullptr;
		int c[3] = { rows.pos[m], rows.pos[m], rows.pos[m] };
		if ( layout ) {
			any = true;
			layout->rope_at( rows.pos[m], c );
		}
		rows.rope_pos[0][m] = c[0];
		rows.rope_pos[1][m] = c[1];
		rows.rope_pos[2][m] = c[2];
	}
	rows.mrope = any ? 1 : 0;
}

namespace {
// ENGINE_LOG_LEVEL=debug: rows and wall time of every forward (a slow backend shows where a request spends its time)
struct ForwardTimer {
	int rows;
	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	~ForwardTimer() { LOG_DEBUG( "[forward] %d rows %.1f ms\n", rows, std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t0 ).count() ); }
};
}

void ModelCluster::forward( const int* tokens, const pod::BatchRows& rows ) {
	ForwardTimer timer{ rows.count };
	if ( mm_slots_ == 0 ) {
		pool_.run( [&]( int d ) {
			models_[d]->embed_tokens( tokens, rows.count, streams_[d] );
			models_[d]->forward( rows, streams_[d] );
			backend::stream_sync( streams_[d] );
		} );
		return;
	}
	// Rows of slots with images: M-RoPE positions for every row of the batch, and runs of image rows to overwrite after the lookup
	pod::BatchRows r = rows;
	struct Run { int row; int rows; const backend::half_t* src; };
	std::vector<Run> runs;
	const int hidden = metadata().hidden_size;
	apply_mrope( r );
	for ( int m = 0; m < rows.count; ++m ) {
		const int slot = rows.slot[m];
		const MmLayout* layout = ( static_cast<size_t>( slot ) < slot_mm_.size() && !slot_mm_[slot].empty() ) ? &slot_mm_[slot] : nullptr;
		const MmImage* im = layout ? layout->image_at( rows.pos[m] ) : nullptr;
		if ( !im ) continue;
		ENGINE_ASSERT( im->embeds != nullptr, "ModelCluster::forward: image rows without embeddings" );
		const backend::half_t* src = im->embeds->data() + static_cast<size_t>( rows.pos[m] - im->start ) * hidden;
		if ( !runs.empty() && runs.back().src + static_cast<size_t>( runs.back().rows ) * hidden == src && runs.back().row + runs.back().rows == m ) ++runs.back().rows;
		else runs.push_back( { m, 1, src } );
	}
	pool_.run( [&]( int d ) {
		models_[d]->embed_tokens( tokens, r.count, streams_[d] );
		for ( const Run& run : runs ) {
			const bool injected = models_[d]->inject_embeddings( run.src, run.row, run.rows, streams_[d] );
			ENGINE_ASSERT( injected, "ModelCluster::forward: the model cannot take image embeddings" );
		}
		models_[d]->forward( r, streams_[d] );
		backend::stream_sync( streams_[d] );
	} );
}

void ModelCluster::forward( const int* tokens, int count, int pos, int slot ) {
	forward( tokens, pod::BatchRows::causal( count, pos, slot ) );
}

void ModelCluster::dspark_draft( int slot, int pos, int from, int win_start, int anchor, int* drafts, float* conf ) {
	pool_.run( [&]( int d ) {
		models_[d]->dspark_forward( slot, pos, from, win_start, anchor, streams_[d] );
		backend::stream_sync( streams_[d] );
	} );
	backend::set_device( 0 );
	const backend::half_t* peer_shards[ENGINE_NUM_GPUS] = { nullptr };
	int peer_cols = 0;
	for ( int d = 0; d < tp_; ++d ) {
		peer_shards[d] = models_[d]->shard_logits_ptr();
		peer_cols = models_[d]->shard_logits_cols();
	}
	models_[0]->dspark_chain( peer_shards, peer_cols, anchor, drafts, conf, streams_[0] );
}

void ModelCluster::mtp_step( int slot, int pos0, int count, const int* tokens, int hidden_mode, int hist_row0, bool logits_last ) {
	pod::BatchRows rows = pod::BatchRows::causal( count, pos0, slot );
	apply_mrope( rows );
	pool_.run( [&]( int d ) {
		models_[d]->mtp_step( rows, tokens, hidden_mode, hist_row0, logits_last, streams_[d] );
		backend::stream_sync( streams_[d] );
	} );
}

bool ModelCluster::mtp_joint() const {
	return models_[0] != nullptr && models_[0]->mtp_joint();
}

void ModelCluster::mtp_step_rows( const pod::BatchRows& rows_in, const int* tokens, int hidden_mode, int hist_row0, const int* out_rows, int n_out ) {
	pod::BatchRows rows = rows_in;
	apply_mrope( rows );
	pool_.run( [&]( int d ) {
		models_[d]->mtp_step_rows( rows, tokens, hidden_mode, hist_row0, out_rows, n_out, streams_[d] );
		backend::stream_sync( streams_[d] );
	} );
}

void ModelCluster::mtp_gather_hist( const int* src_rows, int count ) {
	pool_.run( [&]( int d ) {
		models_[d]->mtp_gather_hist( src_rows, count, streams_[d] );
		backend::stream_sync( streams_[d] );
	} );
}

void ModelCluster::mtp_save_hist( int src_row, int rows, int dst_row ) {
	pool_.run( [&]( int d ) {
		models_[d]->mtp_save_hist( src_row, rows, dst_row, streams_[d] );
		backend::stream_sync( streams_[d] );
	} );
}

int ModelCluster::prefill( const int* tokens, int count, int slot, int chunk, Candidates* last_cands, int pos0 ) {
	ENGINE_ASSERT( count >= 1, "ModelCluster::prefill: empty prompt" );
	const int max_rows = metadata().max_rows;
	chunk = ( chunk <= 0 ) ? max_rows : MAX( 1, MIN( chunk, max_rows ) );
	int last_rows = 1;
	// A model that prefills several chunks layer by layer (prefill_window) takes up to that many consecutive chunks at once; a chunk at the decode
	// sizes (the tail of a prompt) and prompts with images go the ordinary way
	const int window = ( mm_slots_ == 0 && chunk > pod::MAX_BATCH ) ? models_[0]->window_chunks() : 1;
	for ( int pos = 0; pos < count; ) {
		int n = 0;
		pod::BatchRows win[16];
		for ( int p = pos; window > 1 && n < MIN( window, 16 ) && p < count; ) {
			const int rows = MIN( chunk, count - p );
			if ( rows <= pod::MAX_BATCH ) break;
			win[n++] = pod::BatchRows::causal( rows, pos0 + p, slot, ( p + rows >= count ) ? pod::BatchRows::LOGITS_LAST : pod::BatchRows::LOGITS_NONE );
			p += rows;
		}
		if ( n >= 2 ) {
			// the tokens of the window after this one, for a model that fetches host-side lookups early
			int p_end = pos;
			for ( int c = 0; c < n; ++c ) p_end += win[c].count;
			if ( count - p_end > pod::MAX_BATCH )
				for ( int d = 0; d < tp_; ++d ) models_[d]->set_prefetch_hint( tokens + p_end, MIN( count - p_end, MIN( window, 16 ) * chunk ), slot, pos0 + p_end );
			ForwardTimer timer{ win[n - 1].count };
			pool_.run( [&]( int d ) {
				models_[d]->forward_window( tokens + pos, win, n, streams_[d] );
				backend::stream_sync( streams_[d] );
			} );
			int p = pos;
			for ( int c = 0; c < n; ++c ) {
				if ( runner_ ) runner_->on_prefill_chunk( slot, tokens + p, win[c].count, pos0 + p );
				p += win[c].count;
			}
			last_rows = win[n - 1].count;
			pos = p;
			continue;
		}
		last_rows = MIN( chunk, count - pos );
		const bool last = ( pos + last_rows >= count );
		forward( tokens + pos, pod::BatchRows::causal( last_rows, pos0 + pos, slot,
			last ? pod::BatchRows::LOGITS_LAST : pod::BatchRows::LOGITS_NONE ) );
		if ( runner_ ) runner_->on_prefill_chunk( slot, tokens + pos, last_rows, pos0 + pos );
		pos += last_rows;
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
#if !ENGINE_BACKEND_HAS_GRAPHS
	// No capture on this backend: nothing to warm (a dummy forward only costs time)
	( void )max_m;
	( void )slot;
	return;
#endif
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
	// A full prefill chunk, so every rank's lazily grown workspaces reach their steady size together. Growth frees and allocates
	// device memory, which waits for the device to go idle: done on one rank in the middle of a forward (an image encode on rank 0
	// had grown only its own) it blocks behind a collective that waits for the others, which wait for the allocation lock.
	{
		const int rows = metadata().max_rows;
		std::vector<int> ids( rows, 0 );
		forward( ids.data(), pod::BatchRows::causal( rows, 0, slot, pod::BatchRows::LOGITS_NONE ) );
		forward( ids.data(), pod::BatchRows::causal( rows, 0, slot, pod::BatchRows::LOGITS_LAST ) );
	}
	reset_slot( slot );
	LOG_INFO( "Decode graphs warmed for m=1..%d in %.0f ms\n", max_m,
		std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t0 ).count() );
}

bool ModelCluster::read_taps( int slot, int first_pos, int count, backend::half_t* host_dst ) {
	return models_[0]->read_taps( slot, first_pos, count, host_dst, streams_[0] );
}

size_t ModelCluster::snapshot_bytes( int from, int tokens, bool with_taps ) const {
	return models_[0]->seq_state_bytes( from, tokens, with_taps );
}

bool ModelCluster::save_snapshot( int slot, int from, int tokens, SeqSnapshot* snap, bool with_taps ) {
	constexpr size_t kRound = size_t( 8 ) << 20;
	with_taps = with_taps && runner_ != nullptr && runner_->needs_taps();
	const size_t need = snapshot_bytes( from, tokens, with_taps );
	if ( need == 0 ) return false;
	const size_t cls = ( need + kRound - 1 ) / kRound * kRound;
	snap->from = from;
	snap->with_taps = with_taps;
	snap->tokens = tokens;
	snap->bytes = cls;
	const auto t_alloc = std::chrono::steady_clock::now();
	for ( int d = 0; d < tp_; ++d ) {
		void* p = nullptr;
		{
			std::lock_guard<std::mutex> lock( snapshot_mtx_ );
			auto& list = snapshot_free_[d][cls];
			if ( !list.empty() ) {
				p = list.back();
				list.pop_back();
			}
		}
		if ( !p ) {
			backend::set_device( d );
			p = backend::host_alloc_pinned( cls, backend::Pin::Default );
		}
		if ( !p ) {
			free_snapshot( snap );
			return false;
		}
		snap->host[d] = p;
	}
	const auto t_copy = std::chrono::steady_clock::now();
	int ok[ENGINE_NUM_GPUS] = {};
	pool_.run( [&]( int d ) {
		ok[d] = models_[d]->save_seq_state( slot, from, tokens, with_taps, snap->host[d], streams_[d] ) ? 1 : 0;
		backend::stream_sync( streams_[d] );
	} );
	for ( int d = 0; d < tp_; ++d ) {
		if ( !ok[d] ) {
			free_snapshot( snap );
			return false;
		}
	}
	LOG_DEBUG( "[snapshot] %zu MiB per rank: host buffers %.0f ms, copy %.0f ms\n", cls >> 20,
		std::chrono::duration<double, std::milli>( t_copy - t_alloc ).count(), std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t_copy ).count() );
	return true;
}

void ModelCluster::prewarm_snapshots( const std::vector<SnapshotPlan>& plan ) {
	constexpr size_t kRound = size_t( 8 ) << 20;
	std::map<size_t, int> want;   // rounded size -> buffers needed per rank
	for ( const SnapshotPlan& sp : plan ) {
		const bool taps = sp.with_taps && runner_ != nullptr && runner_->needs_taps();
		const size_t need = snapshot_bytes( sp.from, sp.tokens, taps );
		if ( need > 0 ) ++want[( need + kRound - 1 ) / kRound * kRound];
	}
	if ( want.empty() ) return;
	if ( snapshot_prewarm_.joinable() ) snapshot_prewarm_.join();
	snapshot_prewarm_ = std::thread( [this, want]() {
		for ( int d = 0; d < tp_; ++d ) {
			backend::set_device( d );
			for ( const auto& w : want ) {
				for ( ;; ) {
					{
						std::lock_guard<std::mutex> lock( snapshot_mtx_ );
						if ( static_cast<int>( snapshot_free_[d][w.first].size() ) >= w.second ) break;
					}
					void* p = backend::host_alloc_pinned( w.first, backend::Pin::Default );
					if ( !p ) break;
					{
						std::lock_guard<std::mutex> lock( snapshot_mtx_ );
						snapshot_free_[d][w.first].push_back( p );
					}
					// each pin holds the driver for ~0.2 s: leave room for the scheduler's decode steps in between
					std::this_thread::sleep_for( std::chrono::milliseconds( 250 ) );
				}
			}
		}
	} );
}

bool ModelCluster::load_snapshot( int slot, const SeqSnapshot& snap, bool kv_only ) {
	if ( snap.tokens <= 0 || !snap.host[0] ) return false;
	int ok[ENGINE_NUM_GPUS] = {};
	pool_.run( [&]( int d ) {
		ok[d] = models_[d]->load_seq_state( slot, snap.from, snap.tokens, snap.with_taps, snap.host[d], kv_only, streams_[d] ) ? 1 : 0;
		backend::stream_sync( streams_[d] );
	} );
	for ( int d = 0; d < tp_; ++d ) {
		if ( !ok[d] ) return false;
	}
	// The drafter takes the slot over: the restored tap rows are its context for the prefix, and the tail prefill publishes the rest
	if ( runner_ && !kv_only ) {
		if ( snap.with_taps || !runner_->needs_taps() ) {
			runner_->claim( slot );
			if ( snap.with_taps ) runner_->on_restore( slot, snap.tokens );
		}
		else runner_->release( slot ); // no taps for this prefix: the slot decodes without the drafter
	}
	return true;
}

void ModelCluster::free_snapshot( SeqSnapshot* snap ) {
	std::lock_guard<std::mutex> lock( snapshot_mtx_ );
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		if ( snap->host[d] ) snapshot_free_[d][snap->bytes].push_back( snap->host[d] );
		snap->host[d] = nullptr;
	}
	snap->tokens = 0;
}

void ModelCluster::reset_slot( int slot ) {
	// The drafter's context (tap ring, projected KV, fill level) describes one slot at a time. The
	// slot being recycled takes it over, so its prefill publishes taps from position 0.
	if ( runner_ ) runner_->claim( slot );
	set_slot_mm( slot, MmLayout() );
	pool_.run( [&]( int d ) {
		models_[d]->reset_state( slot );
	} );
}

bool ModelCluster::reserve_tokens( int slot, int tokens ) {
	bool ok = true;
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		if ( models_[d] && !models_[d]->reserve_tokens( slot, tokens ) ) ok = false;
	}
	return ok;
}

void ModelCluster::release_tokens( int slot ) {
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		if ( models_[d] ) models_[d]->release_tokens( slot );
	}
}

int ModelCluster::pool_tokens() const {
	return models_[0] ? models_[0]->pool_tokens() : 0;
}

bool ModelCluster::rebalance_experts() {
	int any[ENGINE_NUM_GPUS] = {};
	pool_.run( [&]( int d ) {
		any[d] = models_[d]->rebalance_experts( streams_[d] ) ? 1 : 0;
	} );
	return any[0] != 0;
}

bool ModelCluster::expert_heat( std::vector<std::vector<uint32_t>>& per_rank, int& layers, int& experts ) {
	per_rank.assign( tp_, {} );
	int ok[ENGINE_NUM_GPUS] = {}, l[ENGINE_NUM_GPUS] = {}, e[ENGINE_NUM_GPUS] = {};
	pool_.run( [&]( int d ) {
		ok[d] = models_[d]->expert_heat( per_rank[d], l[d], e[d], true ) ? 1 : 0;
	} );
	layers = l[0];
	experts = e[0];
	return ok[0] != 0;
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

bool ModelCluster::spec_commit_always() const {
	return models_[0] && models_[0]->spec_commit_always();
}

void ModelCluster::commit_spec_rows( int slot, int first_row, int last_row ) {
	for ( int d = 0; d < tp_; ++d ) models_[d]->commit_spec_rows( slot, first_row, last_row, streams_[d] );
}

void ModelCluster::commit_spec_row( int slot, int row ) {
	pool_.run( [&]( int d ) {
		models_[d]->commit_spec_row( slot, row, streams_[d] );
	} );
}

} // namespace engine::runtime
