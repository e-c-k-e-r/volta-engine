#pragma once

// Slang host-target kernels (build/slang/<entry>.cpp) as called from the CPU backend. The Params structs mirror
// src/kernels/slang/*.slang (tests/2_parity/test_slang_cpu.cpp keeps its own copy). Kernels run serially: one call per launch.
#include <slang-cpp-prelude.h>
#include <cstdint>
#include <algorithm>
#include <functional>
#include <cstdlib>
#include <cstdio>
#include <sched.h>
#include <thread>
#include <atomic>
#include <mutex>
#include "engine/kernels/cpu/pool.h"
#include <vector>

extern "C" {
void rms_norm( ComputeVaryingInput*, void*, void* );
void rms_norm_heads( ComputeVaryingInput*, void*, void* );
void rms_norm_f32( ComputeVaryingInput*, void*, void* );
void add_normed( ComputeVaryingInput*, void*, void* );
void scale_to_f32( ComputeVaryingInput*, void*, void* );
void softcap( ComputeVaryingInput*, void*, void* );
void add_residual( ComputeVaryingInput*, void*, void* );
void swiglu( ComputeVaryingInput*, void*, void* );
void swiglu_fused( ComputeVaryingInput*, void*, void* );
void embed_tokens( ComputeVaryingInput*, void*, void* );
void rope( ComputeVaryingInput*, void*, void* );
void moe_sum_planes( ComputeVaryingInput*, void*, void* );
void moe_accumulate( ComputeVaryingInput*, void*, void* );
void batched_argmax( ComputeVaryingInput*, void*, void* );
void router_select( ComputeVaryingInput*, void*, void* );
void l2_norm_heads( ComputeVaryingInput*, void*, void* );
void gated_rmsnorm( ComputeVaryingInput*, void*, void* );
void project_gates( ComputeVaryingInput*, void*, void* );
void causal_conv1d_silu_rows( ComputeVaryingInput*, void*, void* );
void causal_conv1d_silu_tiled( ComputeVaryingInput*, void*, void* );
void causal_conv1d_history( ComputeVaryingInput*, void*, void* );
void recurrent_step_rows( ComputeVaryingInput*, void*, void* );
void kv_store_f16( ComputeVaryingInput*, void*, void* );
void kv_store_q( ComputeVaryingInput*, void*, void* );
void init_streams( ComputeVaryingInput*, void*, void* );
void hc_apply( ComputeVaryingInput*, void*, void* );
void hc_mix_partials( ComputeVaryingInput*, void*, void* );
void hc_mix_finalize( ComputeVaryingInput*, void*, void* );
void hyperhead_mean( ComputeVaryingInput*, void*, void* );
void dsa_prep( ComputeVaryingInput*, void*, void* );
void dsa_pool_update( ComputeVaryingInput*, void*, void* );
void dflash_store_taps( ComputeVaryingInput*, void*, void* );
void reduce_streams( ComputeVaryingInput*, void*, void* );
}

struct NormParams { half* src; half* dst; half* weight; int hidden_size; int has_weight; float eps; float bias; };
struct HeadNormParams { half* qk; half* weight; int num_heads; int head_dim; int has_weight; float eps; float bias; };
struct AddParams { half* hidden; half* residual; int n; };
struct SwigluParams { half* gate; half* up; half* out_; int n; float limit; int gelu; };
struct SwigluFusedParams { half* gate_up; half* out_; int inter_dim; int total; float limit; };
struct EmbedParams { uint16_t* table; int* token_ids; half* out_; int hidden_size; };
struct RopeParams { half* q; half* k; int* pos; int num_q_heads; int num_kv_heads; int head_dim; int rot_dim; int interleaved; float freq_base; int rot_pairs; };
struct NormF32Params { float* src; half* dst; half* weight; int hidden_size; int has_weight; float eps; float bias; };
struct AddNormedParams { float* resid; half* y; half* weight; int hidden_size; int has_weight; float eps; float bias; float scale; };
struct ScaleF32Params { half* x; float* out_; int n; float scale; };
struct SoftcapParams { half* x; int n; float cap; };

struct SumPlanesParams { half* planes; half* out_; int n; int groups; };
struct AccumParams { half* accum; half* expert_out; float weight; int n; };
struct ArgmaxParams { half* logits; int* out_idx; float* out_val; int numel; int batch_size; int index_offset; int row_stride; int has_val; };
struct RouterParams { float* partials; half* bias; int* topk_idx; half* topk_w; int num_experts; int top_k; int k_slices; int max_slices; float scale; int has_bias; };

struct L2Params { half* qkv; int num_k_heads; int head_dim; int scale_q; int64_t row_stride; };
struct GatesParams { half* norm_hidden; half* in_proj_a; half* in_proj_b; half* A_log; half* dt_bias; float* gate_out; float* beta_out; int num_v_heads; int hidden_size; int has_a_log; int has_dt_bias; };
struct ConvRowsParams { half* qkv; float* history; half* weights; int* slots; float* snap; int* snap_on; int channels; int rows; int snap_rows; int has_weights; int64_t slot_stride; int64_t snap_stride; };
struct ConvTiledParams { half* src; half* dst; float* history; half* weights; int* slots; int channels; int rows; int tile_rows; int has_weights; int64_t slot_stride; };
struct RecurrentParams { float* state; half* qkv; float* gate; float* beta; half* output; int* slots; float* snap; int* snap_on; int h_k; int h_v; int rows; int snap_rows; float scale; int64_t slot_stride; int64_t snap_stride; };
struct KvStoreParams { half* data; half* src; int* slots; int* positions; int64_t slot_stride; int layer_idx; int count; int total_elems; int max_seq; int ring; };
struct KvStoreQParams { uint8_t* data; uint8_t* scales; half* src; int* slots; int* positions; int64_t slot_stride; int layer_idx; int count; int total_elems; int max_seq; int q; int ring; };
struct InitStreamsParams { half* in_; float* streams; int hidden_size; };
struct ApplyParams { float* x_in; float* x_out; half* y; float* post; float* comb; int hidden_size; int has_comb; };
struct PartialsParams { float* streams; float* fn; float* partials; int row_len; int chunk_cols; int num_chunks; };
struct FinalizeParams { float* streams; float* partials; float* base; float* scale; float* post; float* comb; half* collapsed; float* inv_rms_buf; float* mix_logits_buf; float* pre_buf; int hidden_size; int num_chunks; int chunk_cols_c; int sinkhorn_iters; float rms_eps; float hc_eps; };
struct HeadMeanParams { float* streams; half* final_hidden; int hidden_size; };
struct PrepParams { half* proj; half* ln_w; half* ln_b; half* plane; half* w_out; int* slots; int* positions; int layer; int64_t plane_slot_stride; };
struct PoolParams { half* plane; half* pool_plane; half* ape; int* slots; int* positions; int layer; int max_seq; int64_t plane_slot_stride; int64_t pool_slot_stride; };
struct TapStoreParams { half* taps; half* hidden; int* slots; int* positions; int* owner; int m; int tap; int H; int cap; };
struct ReduceStreamsParams { float* streams; half* out_; int H; int mode; };
struct GatedNormParams { half* attn_out; half* z; half* norm_weight; int num_v_heads; int head_dim; int use_sigmoid; int has_z; int has_weight; };

namespace impl {

// FP16 <-> float for the host-side reference arithmetic (the compiler conversion rounds to nearest even)
inline float h2f( half v ) { return static_cast<float>( v ); }
inline half f2h( float v ) { return static_cast<half>( v ); }

inline void cpu_parallel_for( int64_t total, int64_t min_chunk, const std::function<void( int64_t, int64_t )>& fn );

// Runs a kernel over a gx * gy * gz grid of thread groups, the flattened grid split across the worker threads. The kernels' host entry
// points keep no state across groups (no groupshared, no cross-group atomics on the host target), so the groups run in any order
inline void cpu_dispatch( void ( *fn )( ComputeVaryingInput*, void*, void* ), void* params, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1 ) {
	if ( gx == 0 || gy == 0 || gz == 0 ) return;
	const int64_t total = static_cast<int64_t>( gx ) * gy * gz;
	cpu_parallel_for( total, 1, [&]( int64_t b, int64_t e ) {
		// one call per run of x groups at a fixed (y, z)
		while ( b < e ) {
			const uint32_t x = static_cast<uint32_t>( b % gx ), yz = static_cast<uint32_t>( b / gx ), y = yz % gy, z = yz / gy;
			const uint32_t x1 = static_cast<uint32_t>( std::min<int64_t>( gx, x + ( e - b ) ) );
			ComputeVaryingInput vi = {};
			vi.startGroupID = { x, y, z };
			vi.endGroupID = { x1, y + 1, z + 1 };
			fn( &vi, params, nullptr );
			b += x1 - x;
		}
	} );
}

// Worker count for the parallel helpers: ENGINE_CPU_THREADS, else the physical cores of the affinity mask
inline int cpu_threads() {
	static const int n = [] {
		const char* e = std::getenv( "ENGINE_CPU_THREADS" );
		cpu_set_t mask;
		const int allowed = ( sched_getaffinity( 0, sizeof( mask ), &mask ) == 0 ) ? CPU_COUNT( &mask ) : static_cast<int>( std::thread::hardware_concurrency() );
		// default: one thread per physical core the process may run on (hyperthread siblings share the cores' units: measured slower)
		int physical = 0;
		{
			std::vector<std::pair<int, int>> seen;
			for ( int c = 0; c < CPU_SETSIZE; ++c ) {
				if ( !CPU_ISSET( c, &mask ) ) continue;
				int pkg = -1, core = -1;
				char path[128];
				std::snprintf( path, sizeof( path ), "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", c );
				if ( FILE* f = std::fopen( path, "r" ) ) { if ( std::fscanf( f, "%d", &pkg ) != 1 ) pkg = -1; std::fclose( f ); }
				std::snprintf( path, sizeof( path ), "/sys/devices/system/cpu/cpu%d/topology/core_id", c );
				if ( FILE* f = std::fopen( path, "r" ) ) { if ( std::fscanf( f, "%d", &core ) != 1 ) core = -1; std::fclose( f ); }
				const std::pair<int, int> id{ pkg, core < 0 ? c : core };
				if ( std::find( seen.begin(), seen.end(), id ) == seen.end() ) seen.push_back( id );
			}
			physical = static_cast<int>( seen.size() );
		}
		const int v = e ? std::atoi( e ) : ( physical > 0 ? physical : allowed );
		return std::max( 1, v );
	}();
	return n;
}

// Splits [0, total) into pieces of at least min_chunk and runs fn( begin, end ) on each, on the process's persistent worker pool (the workers
// spin between the back-to-back calls of a forward instead of being created per call). fn must be re-entrant
inline thread_local bool g_in_pool_task = false;   // a piece of a pool task runs on this thread: nested loops run inline

struct RangeTask : engine::kernels::cpu::Task {
	const std::function<void( int64_t, int64_t )>* fn = nullptr;
	int64_t total = 0, chunk = 1;
	std::atomic<int64_t> next{ 0 }, done{ 0 };
	bool work() override {
		const int64_t b = next.fetch_add( chunk );
		if ( b >= total ) return false;
		const int64_t e = std::min( total, b + chunk );
		const bool outer = g_in_pool_task;
		g_in_pool_task = true;
		( *fn )( b, e );
		g_in_pool_task = outer;
		done.fetch_add( e - b );
		return true;
	}
	bool finished() const override { return done.load() >= total; }
};

inline engine::kernels::cpu::Pool& cpu_pool() {
	static engine::kernels::cpu::Pool pool( cpu_threads(), {} );
	return pool;
}

inline void cpu_parallel_for( int64_t total, int64_t min_chunk, const std::function<void( int64_t, int64_t )>& fn ) {
	const int64_t t = std::min<int64_t>( cpu_threads(), std::max<int64_t>( 1, total / std::max<int64_t>( 1, min_chunk ) ) );
	if ( t <= 1 ) {
		fn( 0, total );
		return;
	}
	if ( g_in_pool_task ) {   // inside a piece of another loop: the pool's threads are already busy
		fn( 0, total );
		return;
	}
	// one caller at a time owns the pool; another thread's loop (the loader's, a second rank's) gets threads of its own
	static std::mutex pool_mu;
	std::unique_lock<std::mutex> own( pool_mu, std::try_to_lock );
	if ( !own.owns_lock() ) {
		const int64_t step = ( total + t - 1 ) / t;
		std::vector<std::thread> pool;
		for ( int64_t i = 1; i < t; ++i ) {
			if ( i * step >= total ) break;
			pool.emplace_back( [&fn, i, step, total] { fn( i * step, std::min( total, ( i + 1 ) * step ) ); } );
		}
		fn( 0, std::min( total, step ) );
		for ( auto& th : pool ) th.join();
		return;
	}
	// The pool's workers may still hold a task after run() returns (Pool's contract): the tasks are a ring of long-lived objects, each reused
	// once no worker refers to it any more. A few pieces per worker for balance, none below min_chunk
	static RangeTask ring[64];
	static unsigned at = 0;
	RangeTask& task = ring[at++ % 64];
	while ( task.refs.load( std::memory_order_acquire ) != 0 ) std::this_thread::yield();
	task.fn = &fn;
	task.total = total;
	task.chunk = std::max<int64_t>( min_chunk, ( total + 4 * t - 1 ) / ( 4 * t ) );
	task.done.store( 0, std::memory_order_relaxed );
	task.next.store( 0, std::memory_order_release );
	cpu_pool().run( task );
}

// cpu_dispatch with the x grid split across threads. Only for kernels whose groups are independent and keep no shared
// state (no groupshared on the host target); the Slang host entry points take a group range, so each thread gets its own
inline void cpu_dispatch_mt( void ( *fn )( ComputeVaryingInput*, void*, void* ), void* params, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1 ) {
	if ( gx == 0 || gy == 0 || gz == 0 ) return;
	cpu_parallel_for( gx, 16, [&]( int64_t b, int64_t e ) {
		ComputeVaryingInput vi = {};
		vi.startGroupID = { static_cast<uint32_t>( b ), 0, 0 };
		vi.endGroupID = { static_cast<uint32_t>( e ), gy, gz };
		fn( &vi, params, nullptr );
	} );
}

} // namespace impl
