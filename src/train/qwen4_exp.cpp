// The qwen4_exp training model (engine/train/qwen4_exp.h): loading a range of layers of an EXL3 checkpoint (tensor parallel: a rank's share of
// them), each layer's forward and its backward (the forward recomputed), the KL head and Adam on the scales.
#include "engine/train/qwen4_exp.h"
#include "engine/train/adam8.h"
#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/core/log.h"
#include "engine/loader/safetensors.h"
#include "reconstruct.cuh"
#include "exl3_vl.cuh"
#include "engine/quant/exl3_quantize.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#if defined( ENGINE_HAVE_NCCL )
#include <nccl.h>
#endif
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <chrono>
#include <map>

namespace engine::train::qwen4_exp {

bool Config::read( const std::string& model_dir ) {
	std::ifstream f( model_dir + "/config.json" );
	if ( !f ) return false;
	std::stringstream ss;
	ss << f.rdbuf();
	const auto j = nlohmann::json::parse( ss.str(), nullptr, false );
	if ( j.is_discarded() ) return false;
	const auto& t = j.contains( "text_config" ) ? j["text_config"] : j;
	hidden = t.value( "hidden_size", hidden );
	hc = t.value( "hc_count", hc );
	rank = t.value( "hc_lowrank", rank );
	layers = t.value( "num_hidden_layers", layers );
	full_interval = t.value( "full_attention_interval", full_interval );
	heads = t.value( "num_attention_heads", heads );
	kv_heads = t.value( "num_key_value_heads", kv_heads );
	head_dim = t.value( "head_dim", head_dim );
	if ( t.contains( "rope_parameters" ) ) {
		const auto& rp = t["rope_parameters"];
		rope_theta = rp.value( "rope_theta", rope_theta );
		rot_dim = static_cast<int>( head_dim * rp.value( "partial_rotary_factor", 0.25 ) );
	}
	lin_k = t.value( "linear_num_key_heads", lin_k );
	lin_v = t.value( "linear_num_value_heads", lin_v );
	lin_d = t.value( "linear_key_head_dim", lin_d );
	experts = t.value( "num_experts", experts );
	top_k = t.value( "num_experts_per_tok", top_k );
	inter = t.value( "moe_intermediate_size", inter );
	shared_inter = t.value( "shared_expert_intermediate_size", shared_inter );
	vocab = t.value( "vocab_size", vocab );
	eps = t.value( "rms_norm_eps", eps );
	return true;
}

namespace {

cudaStream_t cu( backend::stream_t s ) { return static_cast<cudaStream_t>( s ); }

#if defined( ENGINE_HAVE_NCCL )
#define Q4E_NCCL( x ) \
	do { \
		const ncclResult_t r_ = ( x ); \
		ENGINE_ASSERT( r_ == ncclSuccess, "qwen4_exp training model: an NCCL call failed" ); \
	} while ( 0 )
#endif

// A bump allocator over one device allocation
struct Pool {
	char* base = nullptr;
	size_t cap = 0, off = 0, peak = 0;
	void init( size_t bytes ) {
		CUDA_CHECK( cudaMalloc( &base, bytes ) );
		cap = bytes;
		off = peak = 0;
	}
	void release() {
		if ( base ) cudaFree( base );
		base = nullptr;
		cap = off = peak = 0;
	}
	void* raw( size_t bytes ) {
		const size_t a = ( off + 255 ) / 256 * 256;
		ENGINE_ASSERT( a + bytes <= cap, "qwen4_exp training model: device pool exhausted" );
		off = a + bytes;
		peak = std::max( peak, off );
		return base + a;
	}
	float* f( size_t n ) { return static_cast<float*>( raw( n * sizeof( float ) ) ); }
	float* fz( size_t n, cudaStream_t st ) {
		float* p = f( n );
		CUDA_CHECK( cudaMemsetAsync( p, 0, n * sizeof( float ), st ) );
		return p;
	}
	int* i( size_t n ) { return static_cast<int*>( raw( n * sizeof( int ) ) ); }
};

// The per-call workspace: a bump allocator that grows. A call needing more than it holds gets another device block (the pointers handed
// out stay valid); at the next call's start, nothing live, several blocks become one of the peak's size, so it settles at what the calls
// need (the routed experts' share varies with the routing: a later call may grow it once more)
struct Workspace {
	struct Block {
		char* base;
		size_t start, size;   // its range of the offsets
	};
	std::vector<Block> blocks;
	size_t off = 0, peak = 0;
	void add( size_t start, size_t bytes ) {
		Block b{ nullptr, start, ( bytes + 255 ) / 256 * 256 };
		CUDA_CHECK( cudaMalloc( &b.base, b.size ) );
		blocks.push_back( b );
	}
	void init( size_t bytes ) { add( 0, bytes ); }
	void release() {
		for ( Block& b : blocks ) cudaFree( b.base );
		blocks.clear();
		off = peak = 0;
	}
	size_t cap() const { return blocks.empty() ? 0 : blocks.back().start + blocks.back().size; }
	void reset( cudaStream_t st ) {
		off = 0;
		if ( blocks.size() <= 1 ) return;
		CUDA_CHECK( cudaStreamSynchronize( st ) );
		for ( Block& b : blocks ) CUDA_CHECK( cudaFree( b.base ) );
		blocks.clear();
		add( 0, peak );
	}
	void* raw( size_t bytes ) {
		for ( size_t i = 0;; ) {
			const size_t a = ( off + 255 ) / 256 * 256;
			while ( i < blocks.size() && a >= blocks[i].start + blocks[i].size ) ++i;
			if ( i < blocks.size() && a + bytes <= blocks[i].start + blocks[i].size ) {
				off = a + bytes;
				peak = std::max( peak, off );
				return blocks[i].base + ( a - blocks[i].start );
			}
			if ( i + 1 < blocks.size() ) {
				off = blocks[i + 1].start;   // the rest of block i too small: on to the next
				continue;
			}
			const size_t start = cap();
			add( start, std::max( bytes, std::max<size_t>( 64ull << 20, start / 4 ) ) );
			off = start;
		}
	}
	float* f( size_t n ) { return static_cast<float*>( raw( n * sizeof( float ) ) ); }
	float* fz( size_t n, cudaStream_t st ) {
		float* p = f( n );
		CUDA_CHECK( cudaMemsetAsync( p, 0, n * sizeof( float ), st ) );
		return p;
	}
	int* i( size_t n ) { return static_cast<int*>( raw( n * sizeof( int ) ) ); }
};

// Parallel arrays of trainable scale vectors, a tensor's vectors at the same offset in each: on the device the current values and their
// gradients; the optimizer state (the checkpoint's values, the deltas, Adam's moments, the best values kept) on the device too or, with
// host, in pinned host memory, streamed through the device in parts for each step (the same kernels: the same result); its moments FP32
// or 8-bit block-wise (engine/train/adam8.h) either way
struct ParamPool {
	float *s = nullptr, *g = nullptr;                                                 // device
	float *s0 = nullptr, *a = nullptr, *m = nullptr, *v = nullptr, *best = nullptr;   // the state (m, v: FP32 moments)
	uint8_t *mq = nullptr, *vq = nullptr;                                             // 8-bit moments and their blocks' absmax
	float *mmax = nullptr, *vmax = nullptr;
	// host state: a part's device copy of it (s0, a, the moments and their absmax)
	static constexpr size_t kPart = size_t( 1 ) << 22;
	float *p_s0 = nullptr, *p_a = nullptr, *p_m = nullptr, *p_v = nullptr, *p_mmax = nullptr, *p_vmax = nullptr;
	uint8_t *p_mq = nullptr, *p_vq = nullptr;
	size_t cap = 0, used = 0;
	bool host = false;
	int bits = 32;
	void init( size_t n, bool host_state, int moment_bits ) {
		cap = n;
		used = 0;
		host = host_state;
		bits = moment_bits;
		const size_t n4 = std::max<size_t>( n, 4 ), nb = ( n4 + adam8::kBlock - 1 ) / adam8::kBlock;
		auto dev = [&]( void* p, size_t bytes ) {
			CUDA_CHECK( cudaMalloc( static_cast<void**>( p ), bytes ) );
			CUDA_CHECK( cudaMemset( *static_cast<void**>( p ), 0, bytes ) );
		};
		auto state = [&]( void* p, size_t bytes ) {
			if ( !host ) return dev( p, bytes );
			CUDA_CHECK( cudaMallocHost( static_cast<void**>( p ), bytes ) );
			std::memset( *static_cast<void**>( p ), 0, bytes );
		};
		dev( &s, n4 * sizeof( float ) );
		dev( &g, n4 * sizeof( float ) );
		for ( float** p : { &s0, &a, &best } ) state( p, n4 * sizeof( float ) );
		if ( bits == 32 ) {
			state( &m, n4 * sizeof( float ) );
			state( &v, n4 * sizeof( float ) );
		}
		else {
			state( &mq, n4 );
			state( &vq, n4 );
			state( &mmax, nb * sizeof( float ) );
			state( &vmax, nb * sizeof( float ) );
		}
		if ( !host ) return;
		const size_t P = std::min( n4, kPart );
		for ( float** p : { &p_s0, &p_a } ) dev( p, P * sizeof( float ) );
		if ( bits == 32 ) {
			dev( &p_m, P * sizeof( float ) );
			dev( &p_v, P * sizeof( float ) );
		}
		else {
			dev( &p_mq, P );
			dev( &p_vq, P );
			dev( &p_mmax, ( P / adam8::kBlock + 1 ) * sizeof( float ) );
			dev( &p_vmax, ( P / adam8::kBlock + 1 ) * sizeof( float ) );
		}
	}
	void release() {
		void* d[] = { s, g, p_s0, p_a, p_m, p_v, p_mmax, p_vmax, p_mq, p_vq };
		for ( void* p : d ) if ( p ) cudaFree( p );
		void* st[] = { s0, a, m, v, best, mq, vq, mmax, vmax };
		for ( void* p : st ) if ( p ) host ? static_cast<void>( cudaFreeHost( p ) ) : static_cast<void>( cudaFree( p ) );
		s = g = s0 = a = m = v = best = mmax = vmax = p_s0 = p_a = p_m = p_v = p_mmax = p_vmax = nullptr;
		mq = vq = p_mq = p_vq = nullptr;
		cap = used = 0;
	}
	size_t take( size_t n ) {
		const size_t o = used;
		used += ( n + 3 ) / 4 * 4;   // float4 aligned
		ENGINE_ASSERT( used <= cap, "qwen4_exp training model: scale pool exhausted" );
		return o;
	}
	bool holds( const float* p ) const { return p >= s && p < s + cap; }
	// After the load (s: the checkpoint's values): the reference values s0
	void start( cudaStream_t st ) {
		if ( used == 0 ) return;
		CUDA_CHECK( cudaMemcpyAsync( s0, s, used * sizeof( float ), host ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice, st ) );
	}
	// One Adam step from the gradients g (log: s = s0 e^a, else s = s0 + a), over [o0, o0 + n) of the state at s0 ... (device pointers)
	void step_part( bool log, const engine::train::AdamOptions& o, int k, float gs, cudaStream_t st, size_t o0, size_t n, float* ps0, float* pa, float* pm,
		float* pv, uint8_t* pmq, uint8_t* pvq, float* pmm, float* pvm ) {
		if ( bits == 32 && log ) adam_log_scales( ps0, pa, s + o0, g + o0, pm, pv, n, k, gs, o, st );
		else if ( bits == 32 ) adam_deltas( ps0, pa, s + o0, g + o0, pm, pv, n, k, gs, o, st );
		else if ( log ) adam_log_scales8( ps0, pa, s + o0, g + o0, pmq, pvq, pmm, pvm, n, k, gs, o, st, o0 );
		else adam_deltas8( ps0, pa, s + o0, g + o0, pmq, pvq, pmm, pvm, n, k, gs, o, st, o0 );
	}
	void step( bool log, const engine::train::AdamOptions& o, int k, float gs, cudaStream_t st ) {
		if ( used == 0 ) return;
		if ( !host ) return step_part( log, o, k, gs, st, 0, used, s0, a, m, v, mq, vq, mmax, vmax );
		// the host state part by part through the device: in, the step, out
		const auto h2d = cudaMemcpyHostToDevice, d2h = cudaMemcpyDeviceToHost;
		for ( size_t o0 = 0; o0 < used; o0 += kPart ) {
			const size_t n = std::min( kPart, used - o0 ), b0 = o0 / adam8::kBlock, nb = ( n + adam8::kBlock - 1 ) / adam8::kBlock;
			CUDA_CHECK( cudaMemcpyAsync( p_s0, s0 + o0, n * sizeof( float ), h2d, st ) );
			CUDA_CHECK( cudaMemcpyAsync( p_a, a + o0, n * sizeof( float ), h2d, st ) );
			if ( bits == 32 ) {
				CUDA_CHECK( cudaMemcpyAsync( p_m, m + o0, n * sizeof( float ), h2d, st ) );
				CUDA_CHECK( cudaMemcpyAsync( p_v, v + o0, n * sizeof( float ), h2d, st ) );
			}
			else {
				CUDA_CHECK( cudaMemcpyAsync( p_mq, mq + o0, n, h2d, st ) );
				CUDA_CHECK( cudaMemcpyAsync( p_vq, vq + o0, n, h2d, st ) );
				CUDA_CHECK( cudaMemcpyAsync( p_mmax, mmax + b0, nb * sizeof( float ), h2d, st ) );
				CUDA_CHECK( cudaMemcpyAsync( p_vmax, vmax + b0, nb * sizeof( float ), h2d, st ) );
			}
			step_part( log, o, k, gs, st, o0, n, p_s0, p_a, p_m, p_v, p_mq, p_vq, p_mmax, p_vmax );
			CUDA_CHECK( cudaMemcpyAsync( a + o0, p_a, n * sizeof( float ), d2h, st ) );
			if ( bits == 32 ) {
				CUDA_CHECK( cudaMemcpyAsync( m + o0, p_m, n * sizeof( float ), d2h, st ) );
				CUDA_CHECK( cudaMemcpyAsync( v + o0, p_v, n * sizeof( float ), d2h, st ) );
			}
			else {
				CUDA_CHECK( cudaMemcpyAsync( mq + o0, p_mq, n, d2h, st ) );
				CUDA_CHECK( cudaMemcpyAsync( vq + o0, p_vq, n, d2h, st ) );
				CUDA_CHECK( cudaMemcpyAsync( mmax + b0, p_mmax, nb * sizeof( float ), d2h, st ) );
				CUDA_CHECK( cudaMemcpyAsync( vmax + b0, p_vmax, nb * sizeof( float ), d2h, st ) );
			}
		}
	}
	void keep_best( cudaStream_t st ) {
		if ( used == 0 ) return;
		CUDA_CHECK( cudaMemcpyAsync( best, s, used * sizeof( float ), host ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice, st ) );
	}
	// The kept values on the host
	std::vector<float> best_values() const {
		std::vector<float> h( used );
		if ( used == 0 ) return h;
		if ( host ) std::memcpy( h.data(), best, used * sizeof( float ) );
		else CUDA_CHECK( cudaMemcpy( h.data(), best, used * sizeof( float ), cudaMemcpyDeviceToHost ) );
		return h;
	}
	size_t state_bytes( size_t n ) const {
		const size_t n4 = std::max<size_t>( n, 4 ), nb = ( n4 + adam8::kBlock - 1 ) / adam8::kBlock;
		return n4 * 3 * sizeof( float ) + ( bits == 32 ? n4 * 2 * sizeof( float ) : n4 * 2 + nb * 2 * sizeof( float ) );
	}
	size_t device_bytes() const {
		const size_t P = std::min( std::max<size_t>( cap, 4 ), kPart );
		const size_t part = P * 2 * sizeof( float ) + ( bits == 32 ? P * 2 * sizeof( float ) : P * 2 + ( P / adam8::kBlock + 1 ) * 2 * sizeof( float ) );
		return std::max<size_t>( cap, 4 ) * 2 * sizeof( float ) + ( host ? part : state_bytes( cap ) );
	}
	size_t host_bytes() const { return host ? state_bytes( cap ) : 0; }
};

// Rows r0 .. r0 + k and columns c0 .. c0 + n of a tensor (k, n < 0: to the end)
struct View {
	int r0 = 0, k = -1, c0 = 0, n = -1;
};

// A dense tensor kept as FP16 (its checkpoint values, exact), converted to FP32 in the workspace for each call
struct Half {
	const uint16_t* p = nullptr;
	size_t n = 0;
};

// The slots of a layer's dense tensors
enum DenseSlot { kAhcDown, kAhcUp, kAhcInject, kMhcDown, kMhcUp, kMhcInject, kGdnA, kGdnB, kAlog, kDtBias, kConv, kRouter, kShGate, kPleConv, kDenseSlots };

struct Layer {
	bool full = false, ple = false;
	Half d16[kDenseSlots];   // the dense tensors (the FP32 pointers below point into the workspace during a call)
	HcWeights ahc, mhc;
	// GDN (this rank's key heads from k0h, value heads from v0h; conv_l: the convolution of their channels)
	Tuned qkv, z, out;
	float *A = nullptr, *B = nullptr, *A_log = nullptr, *dt_bias = nullptr, *conv_l = nullptr, *norm = nullptr, *g_norm = nullptr;
	int kh_l = 0, vh_l = 0, k0h = 0, v0h = 0;
	// trained small dense tensors (train_dense): the whole tensors' gradients (this rank adds its heads' rows), the whole convolution [C, 4]
	float *g_A = nullptr, *g_B = nullptr, *g_A_log = nullptr, *g_dt_bias = nullptr, *conv_full = nullptr, *g_conv = nullptr;
	float *g_sh_gate = nullptr, *g_ple_conv = nullptr;
	float* g_router = nullptr;   // train_router: the router's gradient (this rank's routes' share)
	// attention (this rank's query heads from q0h, key / value heads from kv0)
	Tuned q, k, v, o;
	float *q_norm = nullptr, *k_norm = nullptr, *g_q_norm = nullptr, *g_k_norm = nullptr;
	int hq_l = 0, hk_l = 0, q0h = 0, kv0 = 0;
	Tuned idx;               // the QSA indexer's projection (frozen: it only selects)
	Half idx_qn, idx_kn;     // its query and key norms
	// MoE
	float* router = nullptr;
	std::vector<Tuned> eg, eu, ed;   // this rank's experts loaded, the others empty
	float** etab = nullptr;          // device: per expert, (kind * 4 + q) * experts + e for kind gate / up / down and q su / sv / gu / gv
	Tuned sg, su, sd;
	float* sh_gate = nullptr;
	// PLE
	float *ple_nk = nullptr, *ple_nq = nullptr, *ple_nc = nullptr, *ple_conv = nullptr;
	float *g_ple_nk = nullptr, *g_ple_nq = nullptr, *g_ple_nc = nullptr;   // their gradients (rank 0: the PLE layer runs whole on every rank)
	// the sublayer outputs of a kept forward (the rank's rows), for the next backward
	float *cyA = nullptr, *cyM = nullptr;
	bool cached = false;
};

} // namespace

struct Model::Impl {
	Config cfg;
	cudaStream_t st = nullptr;
	std::unique_ptr<Blas> blas;
	engine::loader::SafeTensorsRepo repo;
	pod::MemoryArena arena{};   // packed codes
	Pool state;                 // FP32 dense copies, the expert tables, the kept sublayer outputs
	size_t dense_bytes = 0, kept_bytes = 0;   // of state: the dense copies, the kept outputs
	Workspace ws;               // per-call workspace
	ParamPool shared_p;         // the scales of the tensors every rank holds
	ParamPool expert_p;         // the scales of this rank's experts
	ParamPool norm_p;           // the norm weights (s0: the checkpoint's, a: additive deltas, s: current)
	ParamPool dense_p;          // with train_dense, the small dense tensors (as norm_p)
	ParamPool hc_p;             // with train_hc, the hyper-connection sites' down / up mixes (as norm_p)
	std::vector<Dense> dense_t; // per trainable dense tensor its name and buffers in norm_p, dense_p or hc_p
	bool train_norms = true;
	bool train_dense = false;   // the small dense tensors trained too (set before the load): the hyper-connection sites' inject mixes, GDN's
	                            // in_proj_a / b, A_log, dt_bias and convolution, the shared expert's gate, PLE's convolution
	float dense_lr = 0.1f;      // their learning rate, a fraction of the step's
	bool train_router = false;  // the routers' weights trained too (set before the load; in dense_p at dense_lr)
	bool keep_outputs = true;   // room for every layer's kept sublayer outputs (set before the load; off: a kept forward recomputes them)
	int route_cand = 0;         // per row the next-best experts given the gradient of being swapped in (moe)
	bool train_hc = false;      // the hyper-connection sites' down / up mixes trained too (set before the load; in hc_p at hc_lr)
	float hc_lr = 0.1f;
	bool train_gains = false;   // the tile gains trained too (set before the load; at gain_lr, additive like the norms)
	float gain_lr = 1.0f;
	ParamPool gain_sp, gain_ep; // their values: of the shared tensors, of this rank's experts
	uint8_t *gcodes_s = nullptr, *gcodes_e = nullptr;   // and their codes, at the pools' offsets (the tensors' w4 point here)
	// code updates (set_accumulate): per expert of this rank, the sums of layer acc_layer's backward (dR: gate, up, down; Hx: the gate / up
	// input's Hessian, Ha: down's), D in place of dR after reencode_prepare; the loaded codes and starts kept for restore_codes
	struct ExpertAccum {
		float* dR[3] = { nullptr, nullptr, nullptr };
		float *Hx = nullptr, *Ha = nullptr;
		int64_t rows = 0;
	};
	int acc_layer = -1;
	bool acc_prepared = false;
	std::vector<ExpertAccum> acc;
	float* acc_buf = nullptr;
	std::vector<std::vector<uint16_t>> saved_codes, saved_starts;   // per expert and kind (3 e + kind)
	bool opt_host = false;      // the optimizer state in host memory
	int opt_bits = 32;          // Adam's moments: 32 (FP32) or 8 (block-wise)
	std::vector<Layer> layers;
	int first = 0;
	int rank = 0, ranks = 1;
	void* comm = nullptr;
	int e0 = 0, e1 = 0;         // this rank's experts
	HcWeights fhc;              // the final mixer
	Half fhc16[3];              // its down / up mixes
	Tuned lm;
	bool head = false;
	float* ones_f = nullptr;
	int max_rows = 0;
	int seqs = 1;               // the current layer call's sequences (its rows: seqs of T / seqs)
	bool frozen = false;
	bool fp16 = true;           // the tensor-core products (decoded cores in FP16)
	bool timing = false;
	std::map<std::string, double> times;
	std::chrono::steady_clock::time_point t_mark;
	// the time since the last mark into phase name (stream synchronized)
	void phase( const char* name ) {
		if ( !timing ) return;
		CUDA_CHECK( cudaStreamSynchronize( st ) );
		const auto now = std::chrono::steady_clock::now();
		times[name] += std::chrono::duration<double>( now - t_mark ).count();
		t_mark = now;
	}
	void mark_start() {
		if ( !timing ) return;
		CUDA_CHECK( cudaStreamSynchronize( st ) );
		t_mark = std::chrono::steady_clock::now();
	}
	std::vector<std::vector<int>> last_ids;   // per layer of the range: the last call's selection
	static constexpr int kLmChunk = 8192;
	static constexpr int kGdnChunk = 64;
	static constexpr int kAttnBlock = 1024;   // attention query rows per block of scores (longer sequences)
	static constexpr int kQsaBlock = 1024;    // indexer query rows per block of pool scores
	bool qsa_on = true;                       // the attention's QSA selection past its budget (off: dense)
	// The selection's words per row for sequences of SL rows (0: the attention stays dense: no row past the budget, no indexer, or off)
	int qsa_words( const Layer& lw, int SL ) const {
		if ( !qsa_on || lw.idx.k == 0 || SL / kQsaPool <= kQsaBudget ) return 0;
		return ( SL / kQsaPool + 31 ) / 32;
	}
	// The selection of every row (the indexer over all T rows, sequences of SL) in the workspace; the same on every rank
	uint32_t* qsa_selection( Layer& lw, const float* mix, int T, int SL ) {
		const int ns = T / SL, np = SL / kQsaPool, words = ( np + 31 ) / 32;
		const size_t TR = static_cast<size_t>( T );
		auto* bits = reinterpret_cast<uint32_t*>( ws.i( TR * words ) );
		const size_t mark = ws.off;
		float* Ti = ws.f( static_cast<size_t>( lw.idx.k ) * lw.idx.n );
		float *Zi = ws.f( TR * kQsaProj ), *qk = ws.f( TR * kQsaProj );
		lin_fwd( lw.idx, Ti, mix, T, Zi, qk );
		float *qi = ws.f( TR * kQsaHeads * kQsaDim ), *rk = ws.f( TR * kQsaDim ), *pools = ws.f( static_cast<size_t>( ns ) * np * kQsaDim );
		qsa_queries( qk, T, SL, lw.idx_qn.p, cfg.rot_dim, cfg.rope_theta, cfg.eps, qi, rk, st );
		qsa_pools( rk, ns, SL, lw.idx_kn.p, cfg.rot_dim, cfg.rope_theta, cfg.eps, pools, st );
		const int qb = std::min( kQsaBlock, SL );
		float* scr = ws.f( qsa_scratch_floats( SL, qb ) );
		for ( int s = 0; s < ns; ++s ) {
			qsa_select( qi + static_cast<size_t>( s ) * SL * kQsaHeads * kQsaDim, pools + static_cast<size_t>( s ) * np * kQsaDim, SL, kQsaBudget, qb, scr,
				bits + static_cast<size_t>( s ) * SL * words, st );
		}
		ws.off = mark;   // the selection stays
		return bits;
	}
	// routed experts per group, and padded routes per group (or one expert's routes when more)
	static constexpr int kGroupExperts = 32;
	static constexpr int kGroupRows = 4096;

	// ---- collectives (ranks > 1), the time of each counted as comm rather than the phase around it ----
	template <class F>
	void collective( F&& f ) {
		if ( !timing ) {
			f();
			return;
		}
		CUDA_CHECK( cudaStreamSynchronize( st ) );
		const auto t0 = std::chrono::steady_clock::now();
		f();
		CUDA_CHECK( cudaStreamSynchronize( st ) );
		const auto t1 = std::chrono::steady_clock::now();
		times["comm (and waits for the other ranks)"] += std::chrono::duration<double>( t1 - t0 ).count();
		t_mark += t1 - t0;
	}
#if defined( ENGINE_HAVE_NCCL )
	ncclComm_t nc() const { return static_cast<ncclComm_t>( comm ); }
#endif
	// All ranks' blocks of per_rank values (a row-sharded [rows / ranks, cols]) as one [ranks * per_rank] in rank order
	const float* gather( const float* mine, size_t per_rank ) {
		if ( ranks == 1 ) return mine;
		float* all = ws.f( per_rank * ranks );
#if defined( ENGINE_HAVE_NCCL )
		collective( [&] { Q4E_NCCL( ncclAllGather( mine, all, per_rank, ncclFloat, nc(), st ) ); } );
#endif
		return all;
	}
	// The rank's block of the sum over the ranks of a [ranks * per_rank] partial
	float* reduce_rows( float* part, size_t per_rank ) {
		if ( ranks == 1 ) return part;
		float* mine = ws.f( per_rank );
#if defined( ENGINE_HAVE_NCCL )
		collective( [&] { Q4E_NCCL( ncclReduceScatter( part, mine, per_rank, ncclFloat, ncclSum, nc(), st ) ); } );
#endif
		return mine;
	}
	void sum_ranks( float* buf, size_t n ) {
		if ( ranks == 1 || n == 0 ) return;
#if defined( ENGINE_HAVE_NCCL )
		collective( [&] { Q4E_NCCL( ncclAllReduce( buf, buf, n, ncclFloat, ncclSum, nc(), st ) ); } );
#endif
	}
	void bcast_ints( int* buf, size_t n ) {
		if ( ranks == 1 ) return;
#if defined( ENGINE_HAVE_NCCL )
		collective( [&] { Q4E_NCCL( ncclBroadcast( buf, buf, n, ncclInt32, 0, nc(), st ) ); } );
#endif
	}

	// A dense tensor as an FP16 copy in the state pool (its load into the arena only the copy's source: the arena is rewound after)
	Half dense( const std::string& name ) {
		const size_t mark = arena.offset_bytes;
		pod::TensorView t = repo.load( name.c_str(), arena, st, pod::DType::FP16 );
		ENGINE_ASSERT( t.data != nullptr, "qwen4_exp training model: a dense tensor is missing" );
		Half h;
		h.n = t.numel();
		auto* p = static_cast<uint16_t*>( state.raw( h.n * 2 ) );
		dense_bytes += h.n * 2;
		CUDA_CHECK( cudaMemcpyAsync( p, t.data, h.n * 2, cudaMemcpyDeviceToDevice, st ) );
		CUDA_CHECK( cudaStreamSynchronize( st ) );
		arena.offset_bytes = mark;
		h.p = p;
		return h;
	}
	// A dense tensor as FP32 in the workspace (null: none)
	float* unpack( const Half& h ) {
		if ( h.p == nullptr ) return nullptr;
		float* f = ws.f( h.n );
		half_to_float( h.p, f, h.n, st );
		return f;
	}
	// A layer's dense tensors as FP32 for one call (after the workspace's reset); trained ones are FP32 in their pool already (the convolution:
	// this rank's channels gathered from the whole one)
	void unpack( Layer& lw ) {
		auto set = [&]( const float*& field, DenseSlot k ) {
			if ( lw.d16[k].p ) field = unpack( lw.d16[k] );
		};
		auto set_w = [&]( float*& field, DenseSlot k ) {
			if ( lw.d16[k].p ) field = unpack( lw.d16[k] );
		};
		set( lw.ahc.down, kAhcDown );
		set( lw.ahc.up, kAhcUp );
		set( lw.ahc.inject, kAhcInject );
		set( lw.mhc.down, kMhcDown );
		set( lw.mhc.up, kMhcUp );
		set( lw.mhc.inject, kMhcInject );
		set_w( lw.A, kGdnA );
		set_w( lw.B, kGdnB );
		set_w( lw.A_log, kAlog );
		set_w( lw.dt_bias, kDtBias );
		set_w( lw.conv_l, kConv );
		set_w( lw.router, kRouter );
		set_w( lw.sh_gate, kShGate );
		set_w( lw.ple_conv, kPleConv );
		if ( lw.conv_full ) {
			const size_t D = cfg.lin_d, CQK = static_cast<size_t>( lw.kh_l ) * D, CV = static_cast<size_t>( lw.vh_l ) * D;
			lw.conv_l = ws.f( ( 2 * CQK + CV ) * 4 );
			for ( int j = 0; j < 3; ++j ) {
				const ConvBlock b = conv_block( lw, j );
				CUDA_CHECK( cudaMemcpyAsync( lw.conv_l + b.to * 4, lw.conv_full + b.from * 4, b.n * 4 * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
			}
		}
	}
	// This rank's convolution channels in three column blocks (its q, k and v heads): block j from channel `from` of the whole convolution to
	// `to` of the rank's, n channels
	struct ConvBlock {
		size_t from, to, n;
	};
	ConvBlock conv_block( const Layer& lw, int j ) const {
		const size_t D = cfg.lin_d, GQK = static_cast<size_t>( cfg.lin_k ) * D, CQK = static_cast<size_t>( lw.kh_l ) * D, CV = static_cast<size_t>( lw.vh_l ) * D;
		if ( j == 0 ) return { lw.k0h * D, 0, CQK };
		if ( j == 1 ) return { GQK + lw.k0h * D, CQK, CQK };
		return { 2 * GQK + lw.v0h * D, 2 * CQK, CV };
	}

	// A trainable dense tensor in pool P (additive deltas): its current values (returned; the checkpoint's become the pool's reference after the
	// load), its gradient into *grad (null where this rank adds none: grad itself null)
	float* pooled( ParamPool& P, const std::string& name, float** grad ) {
		const size_t mark = arena.offset_bytes;
		pod::TensorView t = repo.load( name.c_str(), arena, st, pod::DType::FP16 );
		ENGINE_ASSERT( t.data != nullptr, "qwen4_exp training model: a trainable dense tensor is missing" );
		const size_t n = t.numel(), o = P.take( n );
		half_to_float( t.data, P.s + o, n, st );
		CUDA_CHECK( cudaStreamSynchronize( st ) );
		arena.offset_bytes = mark;
		Dense d;
		d.name = name;
		d.w = P.s + o;
		d.g = P.g + o;
		d.n = static_cast<int>( n );
		dense_t.push_back( d );
		if ( grad ) *grad = P.g + o;
		return P.s + o;
	}
	float* norm_weights( const std::string& name, float** grad ) { return pooled( norm_p, name, grad ); }

	Tuned tuned( const std::string& name, bool expert ) {
		Tuned t;
		t.name = name;
		t.w = repo.load_linear( name.c_str(), arena, 4, st );
		ENGINE_ASSERT( t.w.w0.data != nullptr && t.w.type == pod::QuantType::TRELLIS, "qwen4_exp training model: an EXL3 tensor is missing" );
		t.k = t.w.w1.shape[0];
		t.n = t.w.w2.shape[0];
		ParamPool& P = expert ? expert_p : shared_p;
		const size_t ou = P.take( t.k ), ov = P.take( t.n );
		t.su = P.s + ou;
		t.gu = P.g + ou;
		t.sv = P.s + ov;
		t.gv = P.g + ov;
		half_to_float( t.w.w1.data, t.su, t.k, st );   // the checkpoint's values (the pool's reference copy taken after the load)
		half_to_float( t.w.w2.data, t.sv, t.n, st );
		if ( train_gains && t.w.w4.data ) {
			// the tile gains: values from the codes, the codes moved next to the others (set from the values after each step)
			ParamPool& G = expert ? gain_ep : gain_sp;
			uint8_t* codes = ( expert ? gcodes_e : gcodes_s ) + G.used;
			const size_t nt = static_cast<size_t>( t.k / 16 ) * ( t.n / 16 ), o = G.take( nt );
			CUDA_CHECK( cudaMemcpyAsync( codes, t.w.w4.data, nt, cudaMemcpyDeviceToDevice, st ) );
			t.w.w4.data = codes;
			gain_values( codes, G.s + o, nt, st );
			t.tg = G.s + o;
			t.gtg = G.g + o;
		}
		return t;
	}

	// The tile gains' gradient of t over the view from its product's input X [rows, v.k] and output gradient dY [rows, v.n]: the gradient with
	// respect to the tiles R (the core H R H) is ( X diag( su ) H )^T ( dY diag( sv ) H )
	void gain_bwd( const Tuned& t, const float* X, const float* dY, int rows, const View& v ) {
		if ( t.gtg == nullptr || rows <= 0 ) return;
		const size_t mark = ws.off;
		float* Xr = ws.f( static_cast<size_t>( rows ) * v.k );
		float* dYr = ws.f( static_cast<size_t>( rows ) * v.n );
		float* dR = ws.f( static_cast<size_t>( v.k ) * v.n );
		had_rows( X, t.su + v.r0, rows, v.k, Xr, st );
		had_rows( dY, t.sv + v.c0, rows, v.n, dYr, st );
		blas->gemm( true, false, v.k, v.n, rows, 1.0f, Xr, dYr, 0.0f, dR );
		EXL3_TILES_CALL( t.w.volta_tiles, tile_gain_grad, dR, t.gtg, static_cast<const uint16_t*>( t.w.w0.data ), static_cast<const uint16_t*>( t.w.w3.data ),
			static_cast<const uint8_t*>( t.w.w4.data ), t.w.bits, t.w.mcg, t.w.mul1, v.r0, v.k, v.c0, v.n, t.n, st );
		ws.off = mark;
	}

	// An EXL3 tensor used but not trained: its scales in the state pool, no gradients
	Tuned untrained( const std::string& name ) {
		Tuned t;
		t.name = name;
		t.w = repo.load_linear( name.c_str(), arena, 4, st );
		ENGINE_ASSERT( t.w.w0.data != nullptr && t.w.type == pod::QuantType::TRELLIS, "qwen4_exp training model: an EXL3 tensor is missing" );
		t.k = t.w.w1.shape[0];
		t.n = t.w.w2.shape[0];
		t.su = static_cast<float*>( state.raw( static_cast<size_t>( t.k ) * sizeof( float ) ) );
		t.sv = static_cast<float*>( state.raw( static_cast<size_t>( t.n ) * sizeof( float ) ) );
		half_to_float( t.w.w1.data, t.su, t.k, st );
		half_to_float( t.w.w2.data, t.sv, t.n, st );
		return t;
	}

	// A hyper-connection site: its norm, its down / up / inject mixes into d16[0..2] (set as FP32 for each call; with train_dense the inject mix
	// trained in dense_p instead)
	HcWeights hcw( const std::string& pre, bool inject, Half* d16 ) {
		HcWeights w;
		w.hidden = cfg.hidden;
		w.hc = cfg.hc;
		w.rank = cfg.rank;
		w.eps = cfg.eps;
		w.norm = norm_weights( pre + ".hc_norm.weight", &w.gnorm );
		if ( train_hc ) {
			w.down = pooled( hc_p, pre + ".input_mix_weight_down.weight", &w.gdown );
			w.up = pooled( hc_p, pre + ".input_mix_weight_up.weight", &w.gup );
		}
		else {
			d16[0] = dense( pre + ".input_mix_weight_down.weight" );
			d16[1] = dense( pre + ".input_mix_weight_up.weight" );
		}
		if ( inject && train_dense ) w.inject = pooled( dense_p, pre + ".block_inject_weight.weight", &w.ginject );
		else if ( inject ) d16[2] = dense( pre + ".block_inject_weight.weight" );
		return w;
	}

	View whole( const Tuned& t, View v ) const {
		if ( v.k < 0 ) v.k = t.k - v.r0;
		if ( v.n < 0 ) v.n = t.n - v.c0;
		return v;
	}

	// The core of t (unit scales) over the view into T [k, n], FP16 with fp16 (the buffer sized for floats either way), else FP32
	void decode( const Tuned& t, float* T, const View& v ) {
		const auto* codes = static_cast<const uint16_t*>( t.w.w0.data );
		const auto* starts = static_cast<const uint16_t*>( t.w.w3.data );
		const auto* gains = static_cast<const uint8_t*>( t.w.w4.data );
		std::chrono::steady_clock::time_point t0;
		if ( timing ) {
			CUDA_CHECK( cudaStreamSynchronize( st ) );
			t0 = std::chrono::steady_clock::now();
		}
		EXL3_TILES_CALL( t.w.volta_tiles, reconstruct_had_block, T, codes, starts, gains, t.w.bits, t.w.mcg, t.w.mul1, v.r0, v.k, v.c0, v.n, t.n, fp16, st );
		if ( timing ) {
			CUDA_CHECK( cudaStreamSynchronize( st ) );
			times["(decode, inside the phases)"] += std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
		}
	}

	// t's tiles R (gains applied, unit scales, before the rotations: the core is H R H) into R [k, n] FP32, through R16 (room for them in FP16)
	void decode_tiles( const Tuned& t, float* R, uint16_t* R16 ) {
		EXL3_TILES_CALL( t.w.volta_tiles, reconstruct_slice_ext, reinterpret_cast<half*>( R16 ), static_cast<const uint16_t*>( t.w.w0.data ),
			static_cast<const uint16_t*>( t.w.w3.data ), static_cast<const uint8_t*>( t.w.w4.data ), t.w.bits, t.w.mcg, t.w.mul1, t.k / 16, t.n, 0, st );
		half_to_float( R16, R, static_cast<size_t>( t.k ) * t.n, st );
	}

	ScaledLinear sl( Tuned& t, const float* T, const View& v ) {
		ScaledLinear l;
		l.k = v.k;
		l.n = v.n;
		if ( fp16 ) l.T16 = T;
		else l.T = T;
		l.su = t.su + v.r0;
		l.sv = t.sv + v.c0;
		l.dsu = t.gu + v.r0;
		l.dsv = t.gv + v.c0;
		return l;
	}

	// Y = X W_t over the view (X [rows, k], Y [rows, n]), Z kept for the backward; T [k, n] decoded here
	void lin_fwd( Tuned& t, float* T, const float* X, int rows, float* Z, float* Y, View v = {} ) {
		v = whole( t, v );
		decode( t, T, v );
		float* scr = ws.f( static_cast<size_t>( rows ) * v.k );
		scaled_linear_forward( *blas, sl( t, T, v ), X, rows, Z, Y, scr );
	}
	void lin_bwd( Tuned& t, const float* T, const float* X, const float* Z, const float* dY, int rows, float* dX, bool add, View v = {} ) {
		v = whole( t, v );
		float* sn = ws.f( static_cast<size_t>( rows ) * v.n );
		float* sk = ws.f( static_cast<size_t>( rows ) * v.k );
		scaled_linear_backward( *blas, sl( t, T, v ), X, Z, dY, rows, dX, sn, sk, add );
		gain_bwd( t, X, dY, rows, v );
	}

	void run( int L, const float* x_in, float* x_out, int T, const float* ple_k, const float* ple_v, const float* dx_out, float* dx_in, bool keep,
		int n_seqs );
	// Sublayers on mixed [T, hidden] (all rows) -> y [T, hidden], with ranks > 1 the rank's partial sum; with dy (all rows), the backward
	// into dmix (+=, partial) after the forward
	void gdn( Layer& lw, const float* mix, int T, float* y, const float* dy, float* dmix );
	void attention( Layer& lw, const float* mix, int T, float* y, const float* dy, float* dmix );
	// The router and this rank's routed experts (y: overwritten, null: not wanted)
	void moe( Layer& lw, const float* mix, int T, float* y, const float* dy, float* dmix );
	// The shared expert times its gate on rows of this rank: y += (null: not wanted), with dy the backward into dmix (+=)
	void shared( Layer& lw, const float* mix, int T, float* y, const float* dy, float* dmix );
	void expert_mlp( Tuned& g, Tuned& u, Tuned& d, const float* X, int rows, float* Y, const float* dY, float* dX, int inter );
};

Model::Model() : p_( std::make_unique<Impl>() ) {}

Model::~Model() {
	if ( !p_ ) return;
	cudaSetDevice( device_ );
	p_->state.release();
	p_->ws.release();
	p_->shared_p.release();
	p_->expert_p.release();
	p_->norm_p.release();
	p_->dense_p.release();
	p_->hc_p.release();
	p_->gain_sp.release();
	p_->gain_ep.release();
	for ( uint8_t* c : { p_->gcodes_s, p_->gcodes_e } ) if ( c ) cudaFree( c );
	if ( p_->acc_buf ) cudaFree( p_->acc_buf );
	if ( p_->ones_f ) cudaFree( p_->ones_f );
	engine::core::arena_destroy( p_->arena );
	if ( p_->st ) cudaStreamDestroy( p_->st );
}

backend::stream_t Model::stream() const { return p_->st; }

bool Model::load( const std::string& dir, int first, int last, int device, int max_rows, int rank, int ranks, void* comm ) {
	Impl& m = *p_;
	if ( !cfg_.read( dir ) ) {
		LOG_ERROR( "[train] %s: no readable config.json", dir.c_str() );
		return false;
	}
	m.cfg = cfg_;
	first_ = first;
	last_ = last;
	device_ = device;
	rank_ = rank;
	ranks_ = ranks;
	m.first = first;
	m.max_rows = max_rows;
	m.rank = rank;
	m.ranks = ranks;
	m.comm = comm;
	const Config& c = cfg_;
#if !defined( ENGINE_HAVE_NCCL )
	if ( ranks > 1 ) {
		LOG_ERROR( "[train] tensor-parallel training needs NCCL" );
		return false;
	}
#endif
	if ( ranks > 1 && ( comm == nullptr || c.lin_k % ranks || c.lin_v % ranks || c.heads % ranks || c.experts % ranks || max_rows % ranks ||
		c.vocab % 128 || c.vocab / 128 < ranks ) ) {
		LOG_ERROR( "[train] %d ranks do not divide the model's heads, experts, vocabulary or %d rows", ranks, max_rows );
		return false;
	}
	// attention: a rank's query heads read whole key / value heads
	const int hq_l = c.heads / ranks, ratio = c.heads / c.kv_heads;
	const int kv_first = rank * hq_l / ratio, hk_l = ( ( rank + 1 ) * hq_l - 1 ) / ratio - kv_first + 1;
	if ( hq_l % hk_l != 0 || hq_l / hk_l != std::min( hq_l, ratio ) ) {
		LOG_ERROR( "[train] %d ranks split the attention's key / value head groups", ranks );
		return false;
	}
	CUDA_CHECK( cudaSetDevice( device ) );
	CUDA_CHECK( cudaStreamCreateWithFlags( &m.st, cudaStreamNonBlocking ) );
	m.blas = std::make_unique<Blas>( m.st );
	if ( !m.repo.open( dir.c_str() ) ) return false;
	m.head = last == c.layers;
	m.e0 = c.experts * rank / ranks;
	m.e1 = c.experts * ( rank + 1 ) / ranks;
	const int E = c.experts, owned = m.e1 - m.e0, H = c.hidden, I = c.inter, nl = last - first;
	// Exact reservations from the checkpoint's tensors. The arena: the range's EXL3 tensors (codes, FP16 scales, tile starts and gains; of the
	// experts this rank's share) and room for the largest dense tensor's FP16 load (rewound after its conversion). The state pool: the dense
	// copies, the convolution's channels, the kept sublayer outputs and the expert tables
	const size_t Rl = static_cast<size_t>( max_rows ) / ranks;
	size_t trained = 0;   // values of the trained small dense tensors (train_dense)
	size_t hc_trained = 0;   // and of the trained hyper-connection mixes (train_hc)
	size_t gains_s = 0, gains_e = 0;   // the trained tile gains (train_gains): of the shared tensors, of this rank's experts
	{
		auto al = []( size_t b ) { return ( b + 255 ) / 256 * 256; };
		auto numel = [&]( const std::string& name ) -> size_t {
			const pod::SafeTensorMeta* t = m.repo.find( name.c_str() );
			if ( t == nullptr ) return 0;
			size_t n = 1;
			for ( int d = 0; d < t->ndim; ++d ) n *= static_cast<size_t>( t->shape[d] );
			return n;
		};
		size_t codes = 0, dense_b = 0, dense_max = 0;
		// an EXL3 tensor; gains: its trained tile gains' count (null: not trained)
		auto linear = [&]( const std::string& base, size_t* gains ) {
			codes += al( m.repo.trellis_bytes( base.c_str() ) ) + al( numel( base + ".suh" ) * 2 ) + al( numel( base + ".svh" ) * 2 );
			for ( const char* s : { ".tstart", ".tgain" } )
				if ( const pod::SafeTensorMeta* t = m.repo.find( ( base + s ).c_str() ) ) codes += al( t->size_bytes );
			if ( gains && m.train_gains ) *gains += ( numel( base + ".tgain" ) + 3 ) / 4 * 4;
		};
		auto dense_w = [&]( const std::string& name, bool copy ) {
			const size_t n = numel( name );
			dense_max = std::max( dense_max, n );
			if ( copy ) dense_b += al( n * 2 );   // FP16
		};
		// a small dense tensor: trained in dense_p (train_dense), else an FP16 copy
		auto small = [&]( const std::string& name ) {
			if ( !m.train_dense ) return dense_w( name, true );
			dense_w( name, false );
			trained += ( numel( name ) + 3 ) / 4 * 4;
		};
		auto hc_site = [&]( const std::string& pre, bool inject ) {
			dense_w( pre + ".hc_norm.weight", false );
			for ( const char* mix : { ".input_mix_weight_down.weight", ".input_mix_weight_up.weight" } ) {
				if ( !m.train_hc ) dense_w( pre + mix, true );
				else {
					dense_w( pre + mix, false );
					hc_trained += ( numel( pre + mix ) + 3 ) / 4 * 4;
				}
			}
			if ( inject ) small( pre + ".block_inject_weight.weight" );
		};
		size_t extra = 0;
		for ( int L = first; L < last; ++L ) {
			const std::string pre = "model.language_model.layers." + std::to_string( L );
			hc_site( pre + ".attn_hyper_connection", true );
			hc_site( pre + ".mlp_hyper_connection", true );
			if ( c.full_attention( L ) ) {
				for ( const char* p : { "q_proj", "k_proj", "v_proj", "o_proj" } ) linear( pre + ".self_attn." + p, &gains_s );
				dense_w( pre + ".self_attn.q_norm.weight", false );
				dense_w( pre + ".self_attn.k_norm.weight", false );
				if ( m.repo.find( ( pre + ".self_attn.indexer.index_qk_proj.suh" ).c_str() ) ) {
					linear( pre + ".self_attn.indexer.index_qk_proj", nullptr );
					dense_w( pre + ".self_attn.indexer.q_layernorm.weight", true );
					dense_w( pre + ".self_attn.indexer.k_layernorm.weight", true );
					extra += al( static_cast<size_t>( H + kQsaProj ) * sizeof( float ) );
				}
			}
			else {
				for ( const char* p : { "in_proj_qkv", "in_proj_z", "out_proj" } ) linear( pre + ".linear_attn." + p, &gains_s );
				for ( const char* p : { "in_proj_a.weight", "in_proj_b.weight", "A_log", "dt_bias" } ) small( pre + ".linear_attn." + p );
				dense_w( pre + ".linear_attn.norm.weight", false );
				if ( m.train_dense ) small( pre + ".linear_attn.conv1d.weight" );
				else {
					dense_w( pre + ".linear_attn.conv1d.weight", false );
					extra += al( static_cast<size_t>( 2 * c.lin_k + c.lin_v ) / ranks * c.lin_d * 4 * 2 );   // the rank's channels of it
				}
			}
			if ( m.train_router ) {
				dense_w( pre + ".mlp.gate.weight", false );
				trained += ( numel( pre + ".mlp.gate.weight" ) + 3 ) / 4 * 4;
			}
			else dense_w( pre + ".mlp.gate.weight", true );
			for ( int e = m.e0; e < m.e1; ++e )
				for ( const char* p : { "gate_proj", "up_proj", "down_proj" } ) linear( pre + ".mlp.experts." + std::to_string( e ) + "." + p, &gains_e );
			for ( const char* p : { "gate_proj", "up_proj", "down_proj" } ) linear( pre + ".mlp.shared_expert." + p, &gains_s );
			small( pre + ".mlp.shared_expert_gate.weight" );
			if ( L == c.ple_layer ) {
				for ( const char* p : { "norm_key", "norm_query", "norm_conv" } ) dense_w( pre + ".ple." + p + ".weight", false );
				small( pre + ".ple.conv1d.weight" );
			}
			extra += ( m.keep_outputs ? 2 * al( Rl * H * sizeof( float ) ) : 0 ) + al( static_cast<size_t>( 12 ) * E * sizeof( float* ) );
		}
		if ( m.head ) {
			hc_site( "model.language_model.hyper_connection_mixer", false );
			linear( "lm_head", &gains_s );
		}
		const size_t arena_b = codes + al( dense_max * 2 ) + ( 16ull << 20 );
		if ( !engine::core::arena_create( m.arena, device, arena_b ) ) {
			LOG_ERROR( "[train] device %d: no arena of %.2f GiB", device, arena_b / 1073741824.0 );
			return false;
		}
		m.state.init( dense_b + extra + ( 16ull << 20 ) );
	}
	{
		const size_t C = static_cast<size_t>( 2 * c.lin_k + c.lin_v ) * c.lin_d, CV = static_cast<size_t>( c.lin_v ) * c.lin_d;
		const size_t QD = static_cast<size_t>( c.heads ) * c.head_dim, KD = static_cast<size_t>( c.kv_heads ) * c.head_dim;
		const size_t gdn = ( H + C ) + ( H + CV ) + ( CV + H ), att = ( H + 2 * QD ) + 2 * ( H + KD ) + ( QD + H ), sh = 3 * ( H + c.shared_inter );
		m.shared_p.init( nl * ( std::max( gdn, att ) + sh + 64 ) + ( m.head ? H + c.vocab + 64 : 0 ), m.opt_host, m.opt_bits );
		m.expert_p.init( static_cast<size_t>( nl ) * owned * 3 * ( H + I + 8 ) + 64, m.opt_host, m.opt_bits );
		const size_t flat = static_cast<size_t>( c.hc ) * H;
		m.norm_p.init( nl * ( 2 * flat + 2 * c.head_dim + c.lin_d + 64 ) + 4 * flat + 256, m.opt_host, m.opt_bits );
		if ( m.train_dense || m.train_router ) m.dense_p.init( trained + 64, m.opt_host, m.opt_bits );
		if ( m.train_hc ) m.hc_p.init( hc_trained + 64, m.opt_host, m.opt_bits );
		if ( m.train_gains ) {
			m.gain_sp.init( gains_s + 64, m.opt_host, m.opt_bits );
			m.gain_ep.init( gains_e + 64, m.opt_host, m.opt_bits );
			CUDA_CHECK( cudaMalloc( &m.gcodes_s, m.gain_sp.cap ) );
			CUDA_CHECK( cudaMalloc( &m.gcodes_e, m.gain_ep.cap ) );
		}
		m.dense_t.reserve( static_cast<size_t>( nl ) * 16 + 10 );
	}
	const size_t ones = std::max<size_t>( c.vocab, 65536 );
	CUDA_CHECK( cudaMalloc( &m.ones_f, ones * sizeof( float ) ) );
	{
		std::vector<float> f( ones, 1.0f );
		CUDA_CHECK( cudaMemcpy( m.ones_f, f.data(), ones * sizeof( float ), cudaMemcpyHostToDevice ) );
	}
	char nm[256];
	for ( int L = first; L < last; ++L ) {
		Layer lw;
		const std::string pre = "model.language_model.layers." + std::to_string( L );
		lw.full = c.full_attention( L );
		lw.ple = L == c.ple_layer;
		lw.ahc = m.hcw( pre + ".attn_hyper_connection", true, lw.d16 + kAhcDown );
		lw.mhc = m.hcw( pre + ".mlp_hyper_connection", true, lw.d16 + kMhcDown );
		if ( lw.full ) {
			lw.hq_l = hq_l;
			lw.hk_l = hk_l;
			lw.q0h = rank * hq_l;
			lw.kv0 = kv_first;
			lw.q = m.tuned( pre + ".self_attn.q_proj", false );
			lw.k = m.tuned( pre + ".self_attn.k_proj", false );
			lw.v = m.tuned( pre + ".self_attn.v_proj", false );
			lw.o = m.tuned( pre + ".self_attn.o_proj", false );
			lw.q_norm = m.norm_weights( pre + ".self_attn.q_norm.weight", &lw.g_q_norm );
			lw.k_norm = m.norm_weights( pre + ".self_attn.k_norm.weight", &lw.g_k_norm );
			if ( m.repo.find( ( pre + ".self_attn.indexer.index_qk_proj.suh" ).c_str() ) ) {
				lw.idx = m.untrained( pre + ".self_attn.indexer.index_qk_proj" );
				lw.idx_qn = m.dense( pre + ".self_attn.indexer.q_layernorm.weight" );
				lw.idx_kn = m.dense( pre + ".self_attn.indexer.k_layernorm.weight" );
			}
		}
		else {
			lw.kh_l = c.lin_k / ranks;
			lw.vh_l = c.lin_v / ranks;
			lw.k0h = rank * lw.kh_l;
			lw.v0h = rank * lw.vh_l;
			lw.qkv = m.tuned( pre + ".linear_attn.in_proj_qkv", false );
			lw.z = m.tuned( pre + ".linear_attn.in_proj_z", false );
			lw.out = m.tuned( pre + ".linear_attn.out_proj", false );
			lw.norm = m.norm_weights( pre + ".linear_attn.norm.weight", &lw.g_norm );
			if ( m.train_dense ) {
				// whole tensors, each rank using (and adding the gradient of) its heads' rows; the convolution's channels gathered per call
				lw.A = m.pooled( m.dense_p, pre + ".linear_attn.in_proj_a.weight", &lw.g_A );
				lw.B = m.pooled( m.dense_p, pre + ".linear_attn.in_proj_b.weight", &lw.g_B );
				lw.A_log = m.pooled( m.dense_p, pre + ".linear_attn.A_log", &lw.g_A_log );
				lw.dt_bias = m.pooled( m.dense_p, pre + ".linear_attn.dt_bias", &lw.g_dt_bias );
				lw.conv_full = m.pooled( m.dense_p, pre + ".linear_attn.conv1d.weight", &lw.g_conv );
			}
			else {
				lw.d16[kGdnA] = m.dense( pre + ".linear_attn.in_proj_a.weight" );
				lw.d16[kGdnB] = m.dense( pre + ".linear_attn.in_proj_b.weight" );
				lw.d16[kAlog] = m.dense( pre + ".linear_attn.A_log" );
				lw.d16[kDtBias] = m.dense( pre + ".linear_attn.dt_bias" );
			}
			// the convolution's channels of this rank's heads: its q, k and v column blocks, [C_l, 4] (the whole one loaded into the arena only)
			const size_t mark = m.arena.offset_bytes;
			pod::TensorView conv = m.repo.load( ( pre + ".linear_attn.conv1d.weight" ).c_str(), m.arena, m.st, pod::DType::FP16 );
			ENGINE_ASSERT( conv.data != nullptr, "qwen4_exp training model: a dense tensor is missing" );
			const size_t D = c.lin_d, CQK = lw.kh_l * D, CV = lw.vh_l * D;
			if ( !m.train_dense ) {
				Half& cl = lw.d16[kConv];
				cl.n = ( 2 * CQK + CV ) * 4;
				auto* conv_l = static_cast<uint16_t*>( m.state.raw( cl.n * 2 ) );
				m.dense_bytes += cl.n * 2;
				for ( int j = 0; j < 3; ++j ) {
					const auto b = m.conv_block( lw, j );
					CUDA_CHECK( cudaMemcpyAsync( conv_l + b.to * 4, static_cast<const uint16_t*>( conv.data ) + b.from * 4, b.n * 4 * 2, cudaMemcpyDeviceToDevice, m.st ) );
				}
				CUDA_CHECK( cudaStreamSynchronize( m.st ) );
				cl.p = conv_l;
			}
			m.arena.offset_bytes = mark;
		}
		if ( m.train_router ) lw.router = m.pooled( m.dense_p, pre + ".mlp.gate.weight", &lw.g_router );
		else lw.d16[kRouter] = m.dense( pre + ".mlp.gate.weight" );
		lw.eg.resize( E );
		lw.eu.resize( E );
		lw.ed.resize( E );
		for ( int e = m.e0; e < m.e1; ++e ) {
			std::snprintf( nm, sizeof( nm ), "%s.mlp.experts.%d", pre.c_str(), e );
			lw.eg[e] = m.tuned( std::string( nm ) + ".gate_proj", true );
			lw.eu[e] = m.tuned( std::string( nm ) + ".up_proj", true );
			lw.ed[e] = m.tuned( std::string( nm ) + ".down_proj", true );
		}
		{
			std::vector<float*> tab( static_cast<size_t>( 12 ) * E, nullptr );
			const std::vector<Tuned>* kinds[] = { &lw.eg, &lw.eu, &lw.ed };
			for ( int kd = 0; kd < 3; ++kd ) for ( int e = m.e0; e < m.e1; ++e ) {
				const Tuned& t = ( *kinds[kd] )[e];
				float* q[] = { t.su, t.sv, t.gu, t.gv };
				for ( int j = 0; j < 4; ++j ) tab[static_cast<size_t>( kd * 4 + j ) * E + e] = q[j];
			}
			lw.etab = static_cast<float**>( m.state.raw( tab.size() * sizeof( float* ) ) );
			CUDA_CHECK( cudaMemcpyAsync( lw.etab, tab.data(), tab.size() * sizeof( float* ), cudaMemcpyHostToDevice, m.st ) );
		}
		lw.sg = m.tuned( pre + ".mlp.shared_expert.gate_proj", false );
		lw.su = m.tuned( pre + ".mlp.shared_expert.up_proj", false );
		lw.sd = m.tuned( pre + ".mlp.shared_expert.down_proj", false );
		if ( m.train_dense ) lw.sh_gate = m.pooled( m.dense_p, pre + ".mlp.shared_expert_gate.weight", &lw.g_sh_gate );
		else lw.d16[kShGate] = m.dense( pre + ".mlp.shared_expert_gate.weight" );
		if ( lw.ple ) {
			// every rank runs the PLE layer whole: rank 0 alone adds its norms' gradients
			lw.ple_nk = m.norm_weights( pre + ".ple.norm_key.weight", rank == 0 ? &lw.g_ple_nk : nullptr );
			lw.ple_nq = m.norm_weights( pre + ".ple.norm_query.weight", rank == 0 ? &lw.g_ple_nq : nullptr );
			lw.ple_nc = m.norm_weights( pre + ".ple.norm_conv.weight", rank == 0 ? &lw.g_ple_nc : nullptr );
			if ( m.train_dense ) lw.ple_conv = m.pooled( m.dense_p, pre + ".ple.conv1d.weight", rank == 0 ? &lw.g_ple_conv : nullptr );
			else lw.d16[kPleConv] = m.dense( pre + ".ple.conv1d.weight" );
		}
		if ( m.keep_outputs ) {
			lw.cyA = m.state.f( Rl * H );
			lw.cyM = m.state.f( Rl * H );
			m.kept_bytes += 2 * Rl * H * sizeof( float );
		}
		m.layers.push_back( std::move( lw ) );
		m.last_ids.emplace_back();
		CUDA_CHECK( cudaStreamSynchronize( m.st ) );
		LOG_INFO( "[train] device %d: layer %d loaded (%s), arena %.2f GiB, state %.2f GiB", device, L, c.full_attention( L ) ? "attention" : "GDN",
			m.arena.offset_bytes / 1073741824.0, m.state.off / 1073741824.0 );
	}
	if ( m.head ) {
		m.fhc = m.hcw( "model.language_model.hyper_connection_mixer", false, m.fhc16 );
		m.lm = m.tuned( "lm_head", false );
	}
	CUDA_CHECK( cudaStreamSynchronize( m.st ) );
	for ( ParamPool* P : { &m.shared_p, &m.expert_p, &m.norm_p, &m.dense_p, &m.hc_p, &m.gain_sp, &m.gain_ep } ) P->start( m.st );
	CUDA_CHECK( cudaStreamSynchronize( m.st ) );
	// The workspace starts at a layer's streams and grows to what the calls need
	m.ws.init( static_cast<size_t>( max_rows ) * c.hc * c.hidden * sizeof( float ) * 8 );
	return true;
}

void Model::Impl::expert_mlp( Tuned& g, Tuned& u, Tuned& d, const float* X, int rows, float* Y, const float* dY, float* dX, int I ) {
	// Y = down( silu( X Wg ) * ( X Wu ) ); with dY: dX (+=) and the scale gradients
	const int H = cfg.hidden;
	const size_t mark = ws.off;
	float* Tg = ws.f( static_cast<size_t>( H ) * I );
	float* Tu = ws.f( static_cast<size_t>( H ) * I );
	float* Td = ws.f( static_cast<size_t>( I ) * H );
	float *Zg = ws.f( static_cast<size_t>( rows ) * I ), *G = ws.f( static_cast<size_t>( rows ) * I ), *Zu = ws.f( static_cast<size_t>( rows ) * I ),
		*U = ws.f( static_cast<size_t>( rows ) * I ), *A = ws.f( static_cast<size_t>( rows ) * I ), *Zd = ws.f( static_cast<size_t>( rows ) * H );
	lin_fwd( g, Tg, X, rows, Zg, G );
	lin_fwd( u, Tu, X, rows, Zu, U );
	swiglu_forward( G, U, A, static_cast<size_t>( rows ) * I, 0.0f, st );
	lin_fwd( d, Td, A, rows, Zd, Y );
	if ( dY ) {
		float *dA = ws.f( static_cast<size_t>( rows ) * I ), *dG = ws.f( static_cast<size_t>( rows ) * I ), *dU = ws.f( static_cast<size_t>( rows ) * I );
		lin_bwd( d, Td, A, Zd, dY, rows, dA, false );
		swiglu_backward( G, U, dA, dG, dU, static_cast<size_t>( rows ) * I, 0.0f, st );
		lin_bwd( g, Tg, X, Zg, dG, rows, dX, true );
		lin_bwd( u, Tu, X, Zu, dU, rows, dX, true );
	}
	ws.off = mark;
}

void Model::Impl::shared( Layer& lw, const float* mix, int T, float* y, const float* dy, float* dmix ) {
	// y += sigmoid( mix . w_sg ) Sh
	const int H = cfg.hidden;
	const size_t mark = ws.off;
	float* Sh = ws.f( static_cast<size_t>( T ) * H );
	float* zs = ws.f( T );
	float* s = ws.f( T );
	blas->gemm( false, false, T, 1, H, 1.0f, mix, lw.sh_gate, 0.0f, zs );
	sigmoid_gate_forward( ones_f, zs, s, T, st );
	if ( dy ) {
		// dSh = s dy, ds = dy . Sh (after Sh), dz = ds sigmoid'( zs )
		float* dSh = ws.f( static_cast<size_t>( T ) * H );
		float* ds = ws.f( T );
		float* dz = ws.fz( T, st );
		apply_backward( dy, s, Sh /* not yet computed: only dSh is used */, T, H, 1, dSh, ds, st );
		expert_mlp( lw.sg, lw.su, lw.sd, mix, T, Sh, dSh, dmix, cfg.shared_inter );
		apply_backward( dy, s, Sh, T, H, 1, dSh, ds, st );
		sigmoid_gate_backward( ones_f, zs, ds, nullptr, dz, T, st );
		blas->gemm( false, false, T, H, 1, 1.0f, dz, lw.sh_gate, 1.0f, dmix );
		if ( lw.g_sh_gate ) blas->gemm( true, false, H, 1, T, 1.0f, mix, dz, 1.0f, lw.g_sh_gate );
	}
	else {
		expert_mlp( lw.sg, lw.su, lw.sd, mix, T, Sh, nullptr, nullptr, cfg.shared_inter );
	}
	if ( y ) apply_forward( y, s, Sh, T, H, 1, st );
	phase( dy ? "moe shared expert fwd+bwd" : "moe shared expert fwd" );
	ws.off = mark;
}

void Model::Impl::moe( Layer& lw, const float* mix, int T, float* y, const float* dy, float* dmix ) {
	const int H = cfg.hidden, E = cfg.experts, K = cfg.top_k;
	const size_t mark = ws.off;
	if ( y ) CUDA_CHECK( cudaMemsetAsync( y, 0, static_cast<size_t>( T ) * H * sizeof( float ), st ) );
	// Routing (every rank on all rows; rank 0's selection broadcast, so that the ranks' routes partition the routes)
	float* logits = ws.f( static_cast<size_t>( T ) * E );
	blas->gemm( false, true, T, E, H, 1.0f, mix, lw.router, 0.0f, logits );
	int* ids = ws.i( static_cast<size_t>( T ) * K );
	float* w = ws.f( static_cast<size_t>( T ) * K );
	std::vector<int>& kept = last_ids[&lw - layers.data()];
	std::vector<int> h_ids( static_cast<size_t>( T ) * K );
	if ( frozen && kept.size() == h_ids.size() ) {
		h_ids = kept;
		CUDA_CHECK( cudaMemcpyAsync( ids, h_ids.data(), h_ids.size() * sizeof( int ), cudaMemcpyHostToDevice, st ) );
		router_given( logits, ids, T, E, K, w, st );
	}
	else {
		router_topk( logits, T, E, K, ids, w, st );
		if ( ranks > 1 ) {
			bcast_ints( ids, h_ids.size() );
			router_given( logits, ids, T, E, K, w, st );
		}
		CUDA_CHECK( cudaMemcpyAsync( h_ids.data(), ids, h_ids.size() * sizeof( int ), cudaMemcpyDeviceToHost, st ) );
		CUDA_CHECK( cudaStreamSynchronize( st ) );
		kept = h_ids;
	}
	// The routes by slot: the K selected and, with route_cand (backward only), the next-best route_cand experts as zero-weight slots: their
	// outputs computed (the forward unchanged) so that their logits get the gradient of being swapped in
	const int C = dy && route_cand > 0 ? route_cand : 0, KS = K + C;
	int* sids = ids;
	float* sw = w;
	float* wp = nullptr;
	std::vector<int> h_cand;
	if ( C ) {
		sids = ws.i( static_cast<size_t>( T ) * KS );
		sw = ws.f( static_cast<size_t>( T ) * KS );
		wp = ws.f( static_cast<size_t>( T ) * C );
		route_candidates( logits, ids, w, T, E, K, C, sids, sw, wp, st );
		h_cand.resize( static_cast<size_t>( T ) * KS );
		CUDA_CHECK( cudaMemcpyAsync( h_cand.data(), sids, h_cand.size() * sizeof( int ), cudaMemcpyDeviceToHost, st ) );
		CUDA_CHECK( cudaStreamSynchronize( st ) );
	}
	const std::vector<int>& h_slots = C ? h_cand : h_ids;
	// This rank's routed experts in groups of similar route counts (largest first): a group's routes are padded to its largest count (zero
	// rows), so that each of its products is one strided batch over its experts' cores, decoded together (FP16 with fp16, else FP32)
	const int I = cfg.inter;
	std::vector<int> cnt( E, 0 ), order;
	for ( int id : h_slots ) cnt[id]++;
	for ( int e = e0; e < e1; ++e ) if ( cnt[e] > 0 ) order.push_back( e );
	std::stable_sort( order.begin(), order.end(), [&]( int a, int b ) { return cnt[a] > cnt[b]; } );
	struct Group { int first, count, m, p0; };   // experts order[first, first + count), padded rows from p0, m per expert
	std::vector<Group> groups;
	int P = 0, Pmax = 0;
	for ( size_t i = 0; i < order.size(); ) {
		Group g{ static_cast<int>( i ), 0, cnt[order[i]], P };
		while ( i < order.size() && g.count < kGroupExperts && ( g.count == 0 || ( ( g.count + 1 ) * g.m <= kGroupRows &&
			( cnt[order[i]] * 4 >= g.m * 3 || g.m - cnt[order[i]] <= 8 ) ) ) ) {
			++g.count;
			++i;
		}
		P += g.count * g.m;
		Pmax = std::max( Pmax, g.count * g.m );
		groups.push_back( g );
	}
	// per padded row its source row and slot, per slot its padded row (-1: another rank's expert); per group its experts and decode jobs
	std::vector<int> ints( static_cast<size_t>( 2 ) * P + h_slots.size() + order.size(), -1 );
	int *prow_h = ints.data(), *pslot_h = prow_h + P, *inv_h = pslot_h + P, *gexp_h = inv_h + h_slots.size();
	{
		std::vector<int> base( E, -1 ), fill( E, 0 );
		for ( const Group& g : groups ) for ( int i = 0; i < g.count; ++i ) {
			base[order[g.first + i]] = g.p0 + i * g.m;
			gexp_h[g.first + i] = order[g.first + i];
		}
		for ( int r = 0; r < T; ++r ) for ( int j = 0; j < KS; ++j ) {
			const int s = r * KS + j, e = h_slots[s];
			if ( base[e] < 0 ) continue;
			const int p = base[e] + fill[e]++;
			prow_h[p] = r;
			pslot_h[p] = s;
			inv_h[s] = p;
		}
	}
	float* Wk[3];   // the group's cores per kind: [count, H, I] gate and up, [count, I, H] down (FP16 or FP32 elements)
	const size_t esz = fp16 ? 2 : 4;
	for ( float*& p : Wk ) p = static_cast<float*>( ws.raw( static_cast<size_t>( kGroupExperts ) * H * I * esz ) );
	// with trained tile gains (backward), per expert of a group the gradient with respect to its tiles, one kind at a time
	const bool gains = dy && train_gains && gain_ep.used > 0;
	float* dRk = gains ? ws.f( static_cast<size_t>( kGroupExperts ) * H * I ) : nullptr;
	// with set_accumulate on this layer, per expert the gradient with respect to its tiles and its inputs' Hessians added up (code updates)
	const bool accum = dy && acc_layer == static_cast<int>( &lw - layers.data() ) && !acc.empty();
	struct Job { int off, count, bits, kind; bool mcg, mul1, volta; int goff; };
	std::vector<std::vector<Job>> jobs( groups.size() );
	std::vector<const void*> tab_h, gtab_h;   // gtab_h: per job the tile gains' table (codes, starts, gains, dR, gradients)
	{
		const std::vector<Tuned>* kinds[] = { &lw.eg, &lw.eu, &lw.ed };
		for ( size_t gi = 0; gi < groups.size(); ++gi ) for ( int kd = 0; kd < 3; ++kd ) {
			const Group& g = groups[gi];
			std::vector<bool> done( g.count, false );
			for ( int i = 0; i < g.count; ++i ) {
				if ( done[i] ) continue;
				const pod::LinearWeight& w0 = ( *kinds[kd] )[order[g.first + i]].w;
				std::vector<int> same;
				for ( int j = i; j < g.count; ++j ) {
					const pod::LinearWeight& wj = ( *kinds[kd] )[order[g.first + j]].w;
					if ( !done[j] && wj.bits == w0.bits && wj.mcg == w0.mcg && wj.mul1 == w0.mul1 && wj.volta_tiles == w0.volta_tiles ) {
						same.push_back( j );
						done[j] = true;
					}
				}
				Job jb{ static_cast<int>( tab_h.size() ), static_cast<int>( same.size() ), w0.bits, kd, w0.mcg, w0.mul1, w0.volta_tiles,
					static_cast<int>( gtab_h.size() ) };
				for ( int q = 0; q < 4; ++q ) for ( int j : same ) {
					const pod::LinearWeight& wj = ( *kinds[kd] )[order[g.first + j]].w;
					tab_h.push_back( q == 0 ? wj.w0.data : q == 1 ? wj.w3.data : q == 2 ? wj.w4.data :
						static_cast<const void*>( reinterpret_cast<char*>( Wk[kd] ) + static_cast<size_t>( j ) * H * I * esz ) );
				}
				if ( gains ) for ( int q = 0; q < 5; ++q ) for ( int j : same ) {
					const Tuned& tj = ( *kinds[kd] )[order[g.first + j]];
					gtab_h.push_back( q == 0 ? tj.w.w0.data : q == 1 ? tj.w.w3.data : q == 2 ? tj.w.w4.data :
						q == 3 ? static_cast<const void*>( dRk + static_cast<size_t>( j ) * H * I ) : static_cast<const void*>( tj.gtg ) );
				}
				jobs[gi].push_back( jb );
			}
		}
	}
	int* d_ints = ws.i( ints.size() );
	const void** d_tab = static_cast<const void**>( ws.raw( std::max<size_t>( tab_h.size(), 1 ) * sizeof( void* ) ) );
	CUDA_CHECK( cudaMemcpyAsync( d_ints, ints.data(), ints.size() * sizeof( int ), cudaMemcpyHostToDevice, st ) );
	if ( !tab_h.empty() ) CUDA_CHECK( cudaMemcpyAsync( d_tab, tab_h.data(), tab_h.size() * sizeof( void* ), cudaMemcpyHostToDevice, st ) );
	const void** d_gtab = nullptr;
	if ( !gtab_h.empty() ) {
		d_gtab = static_cast<const void**>( ws.raw( gtab_h.size() * sizeof( void* ) ) );
		CUDA_CHECK( cudaMemcpyAsync( d_gtab, gtab_h.data(), gtab_h.size() * sizeof( void* ), cudaMemcpyHostToDevice, st ) );
	}
	const int *d_prow = d_ints, *d_pslot = d_prow + P, *d_inv = d_pslot + P, *d_gexp = d_inv + h_slots.size();
	float* dw = dy ? ws.fz( static_cast<size_t>( T ) * KS, st ) : nullptr;   // dy against each slot's expert output
	phase( "moe routing" );
	// Routed experts, group by group (tables: su, sv, gu, gv per kind)
	{
		auto tab = [&]( int kind, int q ) { return lw.etab + static_cast<size_t>( kind * 4 + q ) * E; };
		const size_t PH = static_cast<size_t>( Pmax ) * H, PI = static_cast<size_t>( Pmax ) * I;
		float *Xg = ws.f( PH ), *Zd = ws.f( PH ), *Yr = ws.f( PH );
		float *Zg = ws.f( PI ), *Zu = ws.f( PI ), *Gm = ws.f( PI ), *Um = ws.f( PI ), *Am = ws.f( PI );
		float *dYr = nullptr, *dXg = nullptr, *dXu = nullptr, *dAk = nullptr, *dG = nullptr, *dU = nullptr;
		if ( dy ) {
			dYr = ws.f( PH ); dXg = ws.f( PH ); dXu = ws.f( PH );
			dAk = ws.f( PI ); dG = ws.f( PI ); dU = ws.f( PI );
		}
		// a product's input times its experts' column scales: FP16 rows at their own scales (A16, rinv), or FP32 (Xs)
		void* A16 = fp16 ? ws.raw( PH * 2 ) : nullptr;
		float* rinv = fp16 ? ws.f( Pmax ) : nullptr;
		float* Xs = fp16 ? nullptr : ws.f( PH );
		float *Xh = gains || accum ? ws.f( PH ) : nullptr, *Yh = gains || accum ? ws.f( PH ) : nullptr;
		for ( size_t gi = 0; gi < groups.size(); ++gi ) {
			const Group& g = groups[gi];
			const int Pg = g.count * g.m, m = g.m, n = g.count;
			const int *gx = d_gexp + g.first, *pr = d_prow + g.p0, *ps = d_pslot + g.p0;
			const long long sW = static_cast<long long>( H ) * I;
			{
				std::chrono::steady_clock::time_point t0;
				if ( timing ) {
					CUDA_CHECK( cudaStreamSynchronize( st ) );
					t0 = std::chrono::steady_clock::now();
				}
				for ( const Job& j : jobs[gi] ) {
					const bool down = j.kind == 2;   // down cores are [I, H], the others [H, I]
					EXL3_TILES_CALL( j.volta, reconstruct_had_batch, d_tab + j.off, j.count, j.bits, j.mcg, j.mul1, down ? I : H, down ? H : I, fp16, st );
				}
				if ( timing ) {
					CUDA_CHECK( cudaStreamSynchronize( st ) );
					times["(decode, inside the phases)"] += std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
				}
			}
			// C [Pg, nn] = ( A [Pg, kk] times the experts' column scales cs ) W_e per expert slab (tb: W_e stored transposed, [nn, kk])
			auto prod = [&]( const float* A, const float* const* cs, int kk, int nn, bool tb, const float* W, float* C ) {
				const long long sa = static_cast<long long>( m ) * kk, sc = static_cast<long long>( m ) * nn;
				if ( fp16 ) {
					route_scale_half( A, gx, m, cs, Pg, kk, A16, rinv, st );
					blas->gemm_h_strided( false, tb, m, nn, kk, A16, sa, W, sW, 0.0f, C, sc, n );
					scale_rows( C, rinv, Pg, nn, st );
				}
				else {
					route_scale( A, gx, m, cs, Pg, kk, Xs, false, st );
					blas->gemm_strided( false, tb, m, nn, kk, 1.0f, Xs, sa, W, sW, 0.0f, C, sc, n );
				}
			};
			route_gather( mix, pr, ps, nullptr, Pg, H, Xg, st );
			prod( Xg, tab( 0, 0 ), H, I, false, Wk[0], Zg );
			prod( Xg, tab( 1, 0 ), H, I, false, Wk[1], Zu );
			route_swiglu( Zg, Zu, gx, m, tab( 0, 1 ), tab( 1, 1 ), Pg, I, Gm, Um, Am, st );
			prod( Am, tab( 2, 0 ), I, H, false, Wk[2], Zd );
			route_scale( Zd, gx, m, tab( 2, 1 ), Pg, H, Yr, false, st );
			if ( y ) route_combine( Yr, d_inv, g.p0, Pg, sw, T, KS, H, y, st );
			if ( !dy ) continue;
			// kind kd's tile gains: per expert dR = ( X su H )^T ( dY sv H ) over its slab of rows (kk inputs, nn outputs), then its tiles' sums
			// (with accum, also each expert's dR summed, and its input's Hessian: X^T X over its rows, the padding's zero)
			auto gain_kind = [&]( int kd, const float* X, int kk, const float* dYk, int nn ) {
				route_scale( X, gx, m, tab( kd, 0 ), Pg, kk, Xh, false, st );
				had_rows( Xh, nullptr, Pg, kk, Xh, st );
				route_scale( dYk, gx, m, tab( kd, 1 ), Pg, nn, Yh, false, st );
				had_rows( Yh, nullptr, Pg, nn, Yh, st );
				if ( gains ) {
					blas->gemm_strided( true, false, kk, nn, m, 1.0f, Xh, static_cast<long long>( m ) * kk, Yh, static_cast<long long>( m ) * nn, 0.0f,
						dRk, static_cast<long long>( kk ) * nn, n );
					for ( const Job& j : jobs[gi] )
						if ( j.kind == kd ) EXL3_TILES_CALL( j.volta, tile_gain_grad_batch, d_gtab + j.goff, j.count, j.bits, j.mcg, j.mul1, kk, nn, st );
				}
				if ( !accum ) return;
				for ( int j = 0; j < n; ++j ) {
					const int e = order[g.first + j];
					ExpertAccum& A = acc[e - e0];
					const size_t o = static_cast<size_t>( j ) * m;
					blas->gemm( true, false, kk, nn, m, 1.0f, Xh + o * kk, Yh + o * nn, 1.0f, A.dR[kd] );
					if ( kd == 0 ) {
						blas->gemm( true, false, kk, kk, m, 1.0f, X + o * kk, X + o * kk, 1.0f, A.Hx );
						A.rows += cnt[e];
					}
					if ( kd == 2 ) blas->gemm( true, false, kk, kk, m, 1.0f, X + o * kk, X + o * kk, 1.0f, A.Ha );
				}
			};
			// down: dsv += colsum( dY Zd ), dAk = ( dY sv ) Td^T, dsu += colsum( dAk A ) (a candidate slot's dY: zero, its weight)
			route_gather( dy, pr, ps, sw, Pg, H, dYr, st );
			route_dots_pad( dy, pr, ps, Yr, Pg, H, dw, st );
			if ( gains || accum ) gain_kind( 2, Am, I, dYr, H );
			seg_colsum_prod( dYr, Zd, m, H, n, gx, tab( 2, 3 ), st );
			prod( dYr, tab( 2, 1 ), H, I, true, Wk[2], dAk );
			seg_colsum_prod( dAk, Am, m, I, n, gx, tab( 2, 2 ), st );
			route_swiglu_backward( dAk, Gm, Um, gx, m, tab( 2, 0 ), Pg, I, dG, dU, st );
			// gate and up: dsv += colsum( dG Zg ), dXk = ( dG sv ) T^T, dsu += colsum( dXk X ), dX = dXk su
			seg_colsum_prod( dG, Zg, m, I, n, gx, tab( 0, 3 ), st );
			seg_colsum_prod( dU, Zu, m, I, n, gx, tab( 1, 3 ), st );
			if ( gains || accum ) {
				gain_kind( 0, Xg, H, dG, I );
				gain_kind( 1, Xg, H, dU, I );
			}
			prod( dG, tab( 0, 1 ), I, H, true, Wk[0], dXg );
			prod( dU, tab( 1, 1 ), I, H, true, Wk[1], dXu );
			seg_colsum_prod( dXg, Xg, m, H, n, gx, tab( 0, 2 ), st );
			seg_colsum_prod( dXu, Xg, m, H, n, gx, tab( 1, 2 ), st );
			route_scale( dXg, gx, m, tab( 0, 0 ), Pg, H, dYr, false, st );
			route_scale( dXu, gx, m, tab( 1, 0 ), Pg, H, dYr, true, st );
			route_combine( dYr, d_inv, g.p0, Pg, nullptr, T, KS, H, dmix, st );
		}
	}
	phase( dy ? "moe routed experts fwd+bwd" : "moe routed experts fwd" );
	if ( dy ) {
		// The routing weights' gradient (this rank's routes) into the router input
		float* dl = ws.f( static_cast<size_t>( T ) * KS );
		if ( C ) route_candidates_backward( sw, dw, wp, T, K, C, dl, st );
		else topk_softmax_backward( w, dw, T, K, dl, st );
		router_input_backward( dl, sids, lw.router, T, KS, H, dmix, st );
		if ( lw.g_router ) {
			// the router's weights: dR = dL^T mix with dL the slots' logit gradients as dense rows (this rank's routes' share)
			float* dL = ws.fz( static_cast<size_t>( T ) * E, st );
			slots_to_dense( dl, sids, T, KS, E, dL, st );
			blas->gemm( true, false, E, H, T, 1.0f, dL, mix, 1.0f, lw.g_router );
		}
	}
	phase( "moe router bwd" );
	CUDA_CHECK( cudaStreamSynchronize( st ) );   // the host route lists above are released with the stack
	ws.off = mark;
}

void Model::Impl::gdn( Layer& lw, const float* mix, int T, float* y, const float* dy, float* dmix ) {
	// this rank's heads: kh key heads, vh value heads; in_proj_qkv's q, k and v column blocks of them side by side as [T, C]
	const int H = cfg.hidden, kh = lw.kh_l, vh = lw.vh_l, D = cfg.lin_d, GQK = cfg.lin_k * D;
	const int CQK = kh * D, CV = vh * D, C = 2 * CQK + CV;
	const View vq{ 0, H, lw.k0h * D, CQK }, vk{ 0, H, GQK + lw.k0h * D, CQK }, vv{ 0, H, 2 * GQK + lw.v0h * D, CV }, vz{ 0, H, lw.v0h * D, CV },
		vo{ lw.v0h * D, CV, 0, H };
	const size_t mark = ws.off;
	const size_t TR = static_cast<size_t>( T );
	float *Tq = ws.f( static_cast<size_t>( H ) * CQK ), *Tk = ws.f( static_cast<size_t>( H ) * CQK ), *Tv = ws.f( static_cast<size_t>( H ) * CV );
	float* Tz = ws.f( static_cast<size_t>( H ) * CV );
	float* Tout = ws.f( static_cast<size_t>( CV ) * H );
	float *Zq = ws.f( TR * CQK ), *Zk = ws.f( TR * CQK ), *Zv = ws.f( TR * CV ), *Yp = ws.f( TR * CV ), *QKV = ws.f( TR * C ), *Zz = ws.f( TR * CV ),
		*Zo = ws.f( TR * CV );
	lin_fwd( lw.qkv, Tq, mix, T, Zq, Yp, vq );
	copy_cols( Yp, CQK, 0, QKV, C, 0, T, CQK, false, st );
	lin_fwd( lw.qkv, Tk, mix, T, Zk, Yp, vk );
	copy_cols( Yp, CQK, 0, QKV, C, CQK, T, CQK, false, st );
	lin_fwd( lw.qkv, Tv, mix, T, Zv, Yp, vv );
	copy_cols( Yp, CV, 0, QKV, C, 2 * CQK, T, CV, false, st );
	lin_fwd( lw.z, Tz, mix, T, Zz, Zo, vz );
	float *a = ws.f( TR * vh ), *b = ws.f( TR * vh ), *gate = ws.f( TR * vh ), *beta = ws.f( TR * vh );
	const float *Al = lw.A + static_cast<size_t>( lw.v0h ) * H, *Bl = lw.B + static_cast<size_t>( lw.v0h ) * H;
	blas->gemm( false, true, T, vh, H, 1.0f, mix, Al, 0.0f, a );
	blas->gemm( false, true, T, vh, H, 1.0f, mix, Bl, 0.0f, b );
	gdn_gates_forward( a, b, lw.A_log + lw.v0h, lw.dt_bias + lw.v0h, T, vh, gate, beta, st );
	const int S = T / seqs;   // a sequence's rows
	float *sconv = ws.f( TR * C ), *Cv = ws.f( TR * C );
	conv4_silu_forward( QKV, lw.conv_l, T, C, 1, sconv, Cv, st, S );
	float *qr = ws.f( TR * CQK ), *kr = ws.f( TR * CQK ), *v = ws.f( TR * CV ), *qn = ws.f( TR * CQK ), *kn = ws.f( TR * CQK );
	float *iq = ws.f( TR * kh ), *ik = ws.f( TR * kh );
	copy_cols( Cv, C, 0, qr, CQK, 0, T, CQK, false, st );
	copy_cols( Cv, C, CQK, kr, CQK, 0, T, CQK, false, st );
	copy_cols( Cv, C, 2 * CQK, v, CV, 0, T, CV, false, st );
	rms_forward( qr, nullptr, 1, false, false, T * kh, D, 1e-6f, iq, qn, st );
	rms_forward( kr, nullptr, 1, false, false, T * kh, D, 1e-6f, ik, kn, st );
	float* ck = ws.f( gdn_ckpt_floats( T, vh, kGdnChunk, seqs ) );
	float* oraw = ws.f( TR * CV );
	const float scale = 1.0f / std::sqrt( static_cast<float>( D ) );
	gdn_forward( qn, kn, v, gate, beta, T, kh, vh, scale, kGdnChunk, ck, oraw, st, seqs );
	float *io = ws.f( TR * vh ), *o = ws.f( TR * CV ), *Zout = ws.f( TR * H );
	gated_rms_forward( oraw, Zo, lw.norm, T * vh, D, 1e-6f, io, o, st );
	lin_fwd( lw.out, Tout, o, T, Zout, y, vo );
	if ( dy ) {
		float* dO = ws.f( TR * CV );
		lin_bwd( lw.out, Tout, o, Zout, dy, T, dO, false, vo );
		float *doraw = ws.fz( TR * CV, st ), *dZo = ws.fz( TR * CV, st );
		gated_rms_backward( oraw, Zo, lw.norm, T * vh, D, io, dO, doraw, dZo, st, lw.g_norm );
		float *dqn = ws.fz( TR * CQK, st ), *dkn = ws.fz( TR * CQK, st ), *dv = ws.fz( TR * CV, st ), *dgate = ws.fz( TR * vh, st ), *dbeta = ws.fz( TR * vh, st );
		float* gscr = ws.f( gdn_scratch_floats( vh, kGdnChunk, seqs ) );
		gdn_backward( qn, kn, v, gate, beta, T, kh, vh, scale, kGdnChunk, ck, doraw, dqn, dkn, dv, dgate, dbeta, gscr, st, seqs );
		// dC [T, C]: the q, k and v column blocks (the norms' gradients through blocks of their own)
		float* dC = ws.f( TR * C );
		float* dkr = ws.fz( TR * CQK, st );
		float* dqr = ws.fz( TR * CQK, st );
		rms_backward( qr, nullptr, 1, false, false, T * kh, D, iq, dqn, dqr, st );
		rms_backward( kr, nullptr, 1, false, false, T * kh, D, ik, dkn, dkr, st );
		copy_cols( dqr, CQK, 0, dC, C, 0, T, CQK, false, st );
		copy_cols( dkr, CQK, 0, dC, C, CQK, T, CQK, false, st );
		copy_cols( dv, CV, 0, dC, C, 2 * CQK, T, CV, false, st );
		float* dQKV = ws.fz( TR * C, st );
		conv4_silu_backward( sconv, lw.conv_l, T, C, 1, dC, dQKV, st, S );
		if ( lw.g_conv ) {
			for ( int j = 0; j < 3; ++j ) {
				const ConvBlock bl = conv_block( lw, j );
				conv4_silu_dw( QKV + bl.to, sconv + bl.to, dC + bl.to, T, static_cast<int>( bl.n ), C, 1, lw.g_conv + bl.from * 4, st, S );
			}
		}
		float *da = ws.f( TR * vh ), *db = ws.f( TR * vh );
		gdn_gates_backward( a, b, lw.A_log + lw.v0h, lw.dt_bias + lw.v0h, T, vh, dgate, dbeta, da, db, st );
		if ( lw.g_A ) {
			// in_proj_a / b's rows of this rank's heads; gate = -e^A_log softplus( a + dt_bias ): d / dA_log = gate, d / ddt_bias = d / da
			blas->gemm( true, false, vh, H, T, 1.0f, da, mix, 1.0f, lw.g_A + static_cast<size_t>( lw.v0h ) * H );
			blas->gemm( true, false, vh, H, T, 1.0f, db, mix, 1.0f, lw.g_B + static_cast<size_t>( lw.v0h ) * H );
			blas->gemm( true, false, vh, 1, T, 1.0f, da, ones_f, 1.0f, lw.g_dt_bias + lw.v0h );
			float* gg = ws.f( TR * vh );
			mul( dgate, gate, gg, TR * vh, st );
			blas->gemm( true, false, vh, 1, T, 1.0f, gg, ones_f, 1.0f, lw.g_A_log + lw.v0h );
		}
		blas->gemm( false, false, T, H, vh, 1.0f, da, Al, 1.0f, dmix );
		blas->gemm( false, false, T, H, vh, 1.0f, db, Bl, 1.0f, dmix );
		// in_proj_qkv per column block: its slice of dQKV
		copy_cols( dQKV, C, 0, Yp, CQK, 0, T, CQK, false, st );
		lin_bwd( lw.qkv, Tq, mix, Zq, Yp, T, dmix, true, vq );
		copy_cols( dQKV, C, CQK, Yp, CQK, 0, T, CQK, false, st );
		lin_bwd( lw.qkv, Tk, mix, Zk, Yp, T, dmix, true, vk );
		copy_cols( dQKV, C, 2 * CQK, Yp, CV, 0, T, CV, false, st );
		lin_bwd( lw.qkv, Tv, mix, Zv, Yp, T, dmix, true, vv );
		lin_bwd( lw.z, Tz, mix, Zz, dZo, T, dmix, true, vz );
	}
	ws.off = mark;
}

void Model::Impl::attention( Layer& lw, const float* mix, int T, float* y, const float* dy, float* dmix ) {
	// this rank's hq query heads and their hk key / value heads
	const int H = cfg.hidden, hq = lw.hq_l, hk = lw.hk_l, d = cfg.head_dim;
	const size_t TR = static_cast<size_t>( T ), QD = static_cast<size_t>( hq ) * d, KD = static_cast<size_t>( hk ) * d;
	const View vq{ 0, H, 2 * lw.q0h * d, static_cast<int>( 2 * QD ) }, vk{ 0, H, lw.kv0 * d, static_cast<int>( KD ) }, vo{ lw.q0h * d, static_cast<int>( QD ), 0, H };
	const size_t mark = ws.off;
	float* Tq = ws.f( static_cast<size_t>( H ) * 2 * QD );
	float* Tk = ws.f( static_cast<size_t>( H ) * KD );
	float* Tv = ws.f( static_cast<size_t>( H ) * KD );
	float* To = ws.f( QD * H );
	float *Zq = ws.f( TR * 2 * QD ), *QG = ws.f( TR * 2 * QD ), *Zk = ws.f( TR * KD ), *K = ws.f( TR * KD ), *Zv = ws.f( TR * KD ), *V = ws.f( TR * KD );
	lin_fwd( lw.q, Tq, mix, T, Zq, QG, vq );
	lin_fwd( lw.k, Tk, mix, T, Zk, K, vk );
	lin_fwd( lw.v, Tv, mix, T, Zv, V, vk );
	float *q = ws.f( TR * QD ), *gate = ws.f( TR * QD ), *qn = ws.f( TR * QD ), *kn = ws.f( TR * KD ), *iq = ws.f( TR * hq ), *ik = ws.f( TR * hk );
	unpack_q_gate( QG, T, hq, d, q, gate, st );
	rms_forward( q, lw.q_norm, 1, true, true, T * hq, d, cfg.eps, iq, qn, st );
	rms_forward( K, lw.k_norm, 1, true, true, T * hk, d, cfg.eps, ik, kn, st );
	// per sequence of SL rows: RoPE positions from its start, the causal attention within it, past the QSA budget over the selected keys only
	const int SL = T / seqs;
	const size_t SR = static_cast<size_t>( SL );
	rope_forward( qn, T, hq, d, cfg.rot_dim, cfg.rope_theta, st, SL );
	rope_forward( kn, T, hk, d, cfg.rot_dim, cfg.rope_theta, st, SL );
	const int words = qsa_words( lw, SL );
	uint32_t* bits = words > 0 ? qsa_selection( lw, mix, T, SL ) : nullptr;
	const int QB = SL > kAttnBlock ? kAttnBlock : SL;   // query rows per block of scores
	float *S = ws.f( static_cast<size_t>( QB ) * SR ), *O = ws.f( TR * QD ), *Gd = ws.f( TR * QD ), *Zo = ws.f( TR * H );
	const float scale = 1.0f / std::sqrt( static_cast<float>( d ) );
	auto sel_of = [&]( int s ) {
		QsaSelection q;
		if ( bits ) {
			q.bits = bits + static_cast<size_t>( s ) * SR * words;
			q.words = words;
		}
		return q;
	};
	for ( int s = 0; s < seqs; ++s ) {
		const size_t r0 = static_cast<size_t>( s ) * SR;
		attention_forward( *blas, qn + r0 * QD, kn + r0 * KD, V + r0 * KD, SL, hq, hk, d, scale, S, O + r0 * QD, sel_of( s ), QB );
	}
	sigmoid_gate_forward( O, gate, Gd, TR * QD, st );
	lin_fwd( lw.o, To, Gd, T, Zo, y, vo );
	if ( dy ) {
		float* dGd = ws.f( TR * QD );
		lin_bwd( lw.o, To, Gd, Zo, dy, T, dGd, false, vo );
		float *dO = ws.fz( TR * QD, st ), *dgate = ws.fz( TR * QD, st );
		sigmoid_gate_backward( O, gate, dGd, dO, dgate, TR * QD, st );
		float *dqn = ws.fz( TR * QD, st ), *dkn = ws.fz( TR * KD, st ), *dV = ws.fz( TR * KD, st ), *dP = ws.f( static_cast<size_t>( QB ) * SR );
		for ( int s = 0; s < seqs; ++s ) {
			const size_t r0 = static_cast<size_t>( s ) * SR;
			attention_backward( *blas, qn + r0 * QD, kn + r0 * KD, V + r0 * KD, SL, hq, hk, d, scale, dO + r0 * QD, dqn + r0 * QD, dkn + r0 * KD,
				dV + r0 * KD, S, dP, sel_of( s ), QB );
		}
		rope_backward( dqn, T, hq, d, cfg.rot_dim, cfg.rope_theta, st, SL );
		rope_backward( dkn, T, hk, d, cfg.rot_dim, cfg.rope_theta, st, SL );
		float *dq = ws.fz( TR * QD, st ), *dK = ws.fz( TR * KD, st );
		rms_backward( q, lw.q_norm, 1, true, true, T * hq, d, iq, dqn, dq, st, lw.g_q_norm );
		rms_backward( K, lw.k_norm, 1, true, true, T * hk, d, ik, dkn, dK, st, lw.g_k_norm );
		float* dQG = ws.f( TR * 2 * QD );
		pack_q_gate( dq, dgate, T, hq, d, dQG, st );
		lin_bwd( lw.q, Tq, mix, Zq, dQG, T, dmix, true, vq );
		lin_bwd( lw.k, Tk, mix, Zk, dK, T, dmix, true, vk );
		lin_bwd( lw.v, Tv, mix, Zv, dV, T, dmix, true, vk );
	}
	ws.off = mark;
}

void Model::Impl::run( int L, const float* x_in, float* x_out, int T, const float* ple_k, const float* ple_v, const float* dx_out, float* dx_in, bool keep,
	int n_seqs ) {
	Layer& lw = layers[L - first];
	const bool bwd = dx_out != nullptr;
	ENGINE_ASSERT( T % ranks == 0 && T <= max_rows, "qwen4_exp training model: the rows exceed the workspace or do not split over the ranks" );
	ENGINE_ASSERT( n_seqs > 0 && T % n_seqs == 0, "qwen4_exp training model: the rows do not hold whole sequences" );
	seqs = n_seqs;
	keep = keep && keep_outputs;
	ws.reset( st );
	unpack( lw );
	const int H = cfg.hidden, hc = cfg.hc, Tl = T / ranks;
	const size_t TR = static_cast<size_t>( T ), TL = static_cast<size_t>( Tl ), flat = static_cast<size_t>( hc ) * H, row0 = static_cast<size_t>( rank ) * Tl;
	const bool cached = bwd && lw.cached;
	const float* mixA = nullptr;   // the attention site's sublayer input, all rows
	auto sublayer = [&]( float* y, const float* dy, float* dmix ) {
		if ( lw.full ) attention( lw, mixA, T, y, dy, dmix );
		else gdn( lw, mixA, T, y, dy, dmix );
	};
	mark_start();
	// x0: the input streams after PLE, the rank's rows
	float* x0 = ws.f( TL * flat );
	CUDA_CHECK( cudaMemcpyAsync( x0, x_in, TL * flat * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
	// PLE (before the attention site of its layer, all rows on every rank): x += gated + silu( conv( norm( gated ) ) ), gated =
	// sigmoid-gate( <norm(key), norm(x)> ) value
	const float* xf = x_in;
	float *kn = nullptr, *qn = nullptr, *ik = nullptr, *iq = nullptr, *g = nullptr, *sg = nullptr, *gated = nullptr, *pn = nullptr, *ip = nullptr, *ps = nullptr;
	const float ple_scale = 1.0f / std::sqrt( static_cast<float>( H ) );
	if ( lw.ple ) {
		ENGINE_ASSERT( ple_k && ple_v, "qwen4_exp training model: the PLE layer needs the n-gram key and value rows" );
		xf = gather( x_in, TL * flat );
		float* x0f = ranks == 1 ? x0 : ws.f( TR * flat );
		if ( ranks > 1 ) CUDA_CHECK( cudaMemcpyAsync( x0f, xf, TR * flat * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
		kn = ws.f( TR * flat ); qn = ws.f( TR * flat ); ik = ws.f( TR * hc ); iq = ws.f( TR * hc ); g = ws.f( TR * hc ); sg = ws.f( TR * hc );
		gated = ws.fz( TR * flat, st ); pn = ws.f( TR * flat ); ip = ws.f( TR * hc ); ps = ws.f( TR * flat );
		float* act = ws.f( TR * flat );
		rms_forward( ple_k, lw.ple_nk, hc, true, true, T * hc, H, cfg.eps, ik, kn, st );
		rms_forward( xf, lw.ple_nq, hc, true, true, T * hc, H, cfg.eps, iq, qn, st );
		seg_dot( kn, qn, T * hc, H, ple_scale, g, st );
		ple_gate_forward( g, sg, TR * hc, st );
		apply_forward( gated, sg, ple_v, T, H, hc, st );
		rms_forward( gated, lw.ple_nc, hc, true, true, T * hc, H, cfg.eps, ip, pn, st );
		conv4_silu_forward( pn, lw.ple_conv, T, static_cast<int>( flat ), 3, ps, act, st, T / seqs );
		copy_cols( gated, static_cast<int>( flat ), 0, x0f, static_cast<int>( flat ), 0, T, static_cast<int>( flat ), true, st );
		copy_cols( act, static_cast<int>( flat ), 0, x0f, static_cast<int>( flat ), 0, T, static_cast<int>( flat ), true, st );
		if ( ranks > 1 ) CUDA_CHECK( cudaMemcpyAsync( x0, x0f + row0 * flat, TL * flat * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
		phase( "ple fwd" );
	}
	auto hc_cache = [&]() {
		HcCache c;
		c.inv = ws.f( TL * hc ); c.normed = ws.f( TL * flat ); c.zt = ws.f( TL * cfg.rank ); c.t = ws.f( TL * cfg.rank ); c.u = ws.f( TL * flat );
		c.zi = ws.f( TL * hc );
		return c;
	};
	// Attention site
	const HcCache ca = hc_cache();
	float *mixA_l = ws.f( TL * H ), *postA = ws.f( TL * hc );
	hc_forward( *blas, lw.ahc, x0, Tl, ca, mixA_l, postA );
	phase( "hc sites" );
	mixA = gather( mixA_l, TL * H );
	float* yA = lw.cyA;
	if ( !cached ) {
		// the MLP site's inputs need yA first: the sublayer's forward alone here (in the backward its forward and backward run after the MLP site's)
		float* part = ws.f( TR * H );
		sublayer( part, nullptr, nullptr );
		phase( lw.full ? "attention fwd" : "gdn fwd" );
		yA = reduce_rows( part, TL * H );
		if ( keep ) CUDA_CHECK( cudaMemcpyAsync( lw.cyA, yA, TL * H * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
	}
	float* x1 = ws.f( TL * flat );
	CUDA_CHECK( cudaMemcpyAsync( x1, x0, TL * flat * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
	apply_forward( x1, postA, yA, Tl, H, hc, st );
	// MLP site
	const HcCache cm = hc_cache();
	float *mixM_l = ws.f( TL * H ), *postM = ws.f( TL * hc );
	hc_forward( *blas, lw.mhc, x1, Tl, cm, mixM_l, postM );
	phase( "hc sites" );
	const float* mixM = gather( mixM_l, TL * H );
	if ( !bwd ) {
		float* part = ws.f( TR * H );
		moe( lw, mixM, T, part, nullptr, nullptr );
		float* yM = reduce_rows( part, TL * H );
		shared( lw, mixM_l, Tl, yM, nullptr, nullptr );
		if ( keep ) {
			CUDA_CHECK( cudaMemcpyAsync( lw.cyM, yM, TL * H * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
			lw.cached = true;
		}
		CUDA_CHECK( cudaMemcpyAsync( x_out, x1, TL * flat * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
		apply_forward( x_out, postM, yM, Tl, H, hc, st );
		phase( "streams" );
		return;
	}
	lw.cached = false;
	// Backward. MLP site: dyM = sum_s postM dx_out (needs no yM), the routed experts' forward and backward over all rows and the shared
	// expert's over the rank's, then dpostM = dx_out . yM
	float *dyM_l = ws.f( TL * H ), *dpostM = ws.f( TL * hc ), *dmixM_part = ws.fz( TR * H, st );
	apply_backward( dx_out, postM, yA /* only dyM is used */, Tl, H, hc, dyM_l, dpostM, st );
	const float* dyM = gather( dyM_l, TL * H );
	float* yM = lw.cyM;
	{
		const size_t moe_mark = ws.off;
		float* part = cached ? nullptr : ws.f( TR * H );
		moe( lw, mixM, T, part, dyM, dmixM_part );
		if ( !cached ) yM = reduce_rows( part, TL * H );
		else ws.off = moe_mark;
	}
	float* dmixM = reduce_rows( dmixM_part, TL * H );
	shared( lw, mixM_l, Tl, cached ? nullptr : yM, dyM_l, dmixM );
	apply_backward( dx_out, postM, yM, Tl, H, hc, dyM_l, dpostM, st );
	float* dx1 = ws.f( TL * flat );
	CUDA_CHECK( cudaMemcpyAsync( dx1, dx_out, TL * flat * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
	float* hscr = ws.f( hc_scratch_floats( lw.mhc, Tl ) );
	phase( "streams" );
	hc_backward( *blas, lw.mhc, x1, Tl, cm, dmixM, dpostM, dx1, hscr );
	phase( "hc sites bwd" );
	// Attention site
	float *dyA_l = ws.f( TL * H ), *dpostA = ws.f( TL * hc ), *dmixA_part = ws.fz( TR * H, st );
	apply_backward( dx1, postA, yA, Tl, H, hc, dyA_l, dpostA, st );
	const float* dyA = gather( dyA_l, TL * H );
	{
		float* part = ws.f( TR * H );   // the sublayer's output again (not needed)
		phase( "streams" );
		sublayer( part, dyA, dmixA_part );
		phase( lw.full ? "attention fwd+bwd" : "gdn fwd+bwd" );
	}
	float* dmixA = reduce_rows( dmixA_part, TL * H );
	float* dx0 = dx1;   // the residual passes through
	hc_backward( *blas, lw.ahc, x0, Tl, ca, dmixA, dpostA, dx0, hscr );
	phase( "hc sites bwd" );
	if ( !lw.ple ) {
		CUDA_CHECK( cudaMemcpyAsync( dx_in, dx0, TL * flat * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
		return;
	}
	// PLE over all rows: dx_in = dx0 + through the PLE terms (act: the conv of the normed gated, gated directly and through the norm, the
	// gate, the query norm)
	const float* dx0f = gather( dx0, TL * flat );
	float* dxf = ranks == 1 ? dx_in : ws.f( TR * flat );
	CUDA_CHECK( cudaMemcpyAsync( dxf, dx0f, TR * flat * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
	float* dpn = ws.fz( TR * flat, st );
	conv4_silu_backward( ps, lw.ple_conv, T, static_cast<int>( flat ), 3, dx0f, dpn, st, T / seqs );
	if ( lw.g_ple_conv ) conv4_silu_dw( pn, ps, dx0f, T, static_cast<int>( flat ), static_cast<int>( flat ), 3, lw.g_ple_conv, st, T / seqs );
	float* dgated = ws.f( TR * flat );
	CUDA_CHECK( cudaMemcpyAsync( dgated, dx0f, TR * flat * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
	rms_backward( gated, lw.ple_nc, hc, true, true, T * hc, H, ip, dpn, dgated, st, lw.g_ple_nc );
	float *dv_unused = ws.f( TR * H ), *dsg = ws.f( TR * hc ), *dg = ws.f( TR * hc ), *dqn = ws.fz( TR * flat, st );
	apply_backward( dgated, sg, ple_v, T, H, hc, dv_unused, dsg, st );
	ple_gate_backward( g, dsg, dg, TR * hc, st );
	seg_dot_backward_b( kn, dg, T * hc, H, ple_scale, dqn, st );
	rms_backward( xf, lw.ple_nq, hc, true, true, T * hc, H, iq, dqn, dxf, st, lw.g_ple_nq );
	if ( lw.g_ple_nk ) {
		// the key norm's weights through dkn (the key rows themselves are inputs: their gradient is dropped)
		float *dkn = ws.fz( TR * flat, st ), *dk_unused = ws.fz( TR * flat, st );
		seg_dot_backward_b( qn, dg, T * hc, H, ple_scale, dkn, st );
		rms_backward( ple_k, lw.ple_nk, hc, true, true, T * hc, H, ik, dkn, dk_unused, st, lw.g_ple_nk );
	}
	if ( ranks > 1 ) CUDA_CHECK( cudaMemcpyAsync( dx_in, dxf + row0 * flat, TL * flat * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
	phase( "ple bwd" );
}

void Model::forward( int L, const float* x_in, float* x_out, int rows, const float* ple_k, const float* ple_v, bool keep, int seqs ) {
	CUDA_CHECK( cudaSetDevice( device_ ) );
	p_->run( L, x_in, x_out, rows, ple_k, ple_v, nullptr, nullptr, keep, seqs );
}

void Model::backward( int L, const float* x_in, const float* dx_out, float* dx_in, int rows, const float* ple_k, const float* ple_v, int seqs ) {
	CUDA_CHECK( cudaSetDevice( device_ ) );
	p_->run( L, x_in, nullptr, rows, ple_k, ple_v, dx_out, dx_in, false, seqs );
}

double Model::head( const float* x, int rows, const Teacher& t, float* dx ) {
	Impl& m = *p_;
	CUDA_CHECK( cudaSetDevice( device_ ) );
	ENGINE_ASSERT( m.head, "qwen4_exp training model: this range has no head" );
	ENGINE_ASSERT( rows % m.ranks == 0 && rows <= m.max_rows, "qwen4_exp training model: the rows exceed the workspace or do not split over the ranks" );
	// vocabulary chunks of up to kLmChunk columns, narrower for many rows (a chunk's buffers hold at most kLmChunk^2 values each)
	const int NC = std::max( 1024, std::min( Impl::kLmChunk, Impl::kLmChunk / 128 * Impl::kLmChunk / std::max( rows, 1 ) * 128 ) );
	const int H = cfg_.hidden, V = cfg_.vocab, Tl = rows / m.ranks;
	const size_t TR = static_cast<size_t>( rows ), TL = static_cast<size_t>( Tl ), flat = static_cast<size_t>( cfg_.hc ) * H;
	// this rank's vocabulary columns (whole 128-column blocks)
	const int vb = V / 128, v0 = vb * m.rank / m.ranks * 128, v1 = vb * ( m.rank + 1 ) / m.ranks * 128;
	m.ws.reset( m.st );
	if ( m.fhc16[0].p ) m.fhc.down = m.unpack( m.fhc16[0] );   // else trained (train_hc): in dense_p
	if ( m.fhc16[1].p ) m.fhc.up = m.unpack( m.fhc16[1] );
	HcCache c;
	c.inv = m.ws.f( TL * cfg_.hc ); c.normed = m.ws.f( TL * flat ); c.zt = m.ws.f( TL * cfg_.rank ); c.t = m.ws.f( TL * cfg_.rank ); c.u = m.ws.f( TL * flat );
	c.zi = m.ws.f( TL * cfg_.hc );
	float* mixF_l = m.ws.f( TL * H );
	m.mark_start();
	hc_forward( *m.blas, m.fhc, x, Tl, c, mixF_l, nullptr );
	const float* mixF = m.gather( mixF_l, TL * H );
	float *Tc = m.ws.f( static_cast<size_t>( H ) * NC ), *Zc = m.ws.f( TR * NC ), *Lc = m.ws.f( TR * NC ), *scr = m.ws.f( TR * H );
	float *mx = m.ws.f( TR ), *sx = m.ws.fz( TR, m.st ), *sl = m.ws.fz( TR * t.tk, m.st );
	{
		std::vector<float> ninf( TR, -INFINITY );
		CUDA_CHECK( cudaMemcpyAsync( mx, ninf.data(), TR * sizeof( float ), cudaMemcpyHostToDevice, m.st ) );
		CUDA_CHECK( cudaStreamSynchronize( m.st ) );
	}
	for ( int c0 = v0; c0 < v1; c0 += NC ) {
		const View v{ 0, H, c0, std::min( NC, v1 - c0 ) };
		m.decode( m.lm, Tc, v );
		scaled_linear_forward( *m.blas, m.sl( m.lm, Tc, v ), mixF, rows, Zc, Lc, scr );
		kl_gather( Lc, rows, v.n, c0, t.ids, t.tk, sl, m.st );   // the teacher's top masked out of Lc after
		kl_lse_update( Lc, rows, v.n, mx, sx, m.st );
	}
	if ( m.ranks > 1 ) {
		// the ranks' slices combined: the running max and sum, the teacher's top logits (each from the rank holding its column)
		const float* m_all = m.gather( mx, TR );
		const float* s_all = m.gather( sx, TR );
		kl_lse_combine( m_all, s_all, m.ranks, rows, mx, sx, m.st );
		m.sum_ranks( sl, TR * t.tk );
	}
	m.phase( "head fwd" );
	float *lt = m.ws.f( TR ), *gq = m.ws.f( TR * t.tk ), *kl = m.ws.f( TR ), *ut = m.ws.f( TR );
	kl_finish( mx, sx, sl, t.p, t.tail, rows, t.tk, lt, gq, ut, kl, m.st );
	std::vector<float> h_kl( TR );
	CUDA_CHECK( cudaMemcpyAsync( h_kl.data(), kl, TR * sizeof( float ), cudaMemcpyDeviceToHost, m.st ) );
	CUDA_CHECK( cudaStreamSynchronize( m.st ) );
	double total = 0.0;
	for ( float v : h_kl ) total += v;
	if ( dx == nullptr ) return total;
	float *dmixF_part = m.ws.fz( TR * H, m.st ), *dl = m.ws.f( TR * NC ), *sn = m.ws.f( TR * NC ), *sk = m.ws.f( TR * H );
	for ( int c0 = v0; c0 < v1; c0 += NC ) {
		const View v{ 0, H, c0, std::min( NC, v1 - c0 ) };
		m.decode( m.lm, Tc, v );
		scaled_linear_forward( *m.blas, m.sl( m.lm, Tc, v ), mixF, rows, Zc, Lc, scr );
		kl_grad( Lc, rows, v.n, c0, lt, ut, t.ids, gq, t.tk, dl, m.st );
		scaled_linear_backward( *m.blas, m.sl( m.lm, Tc, v ), mixF, Zc, dl, rows, dmixF_part, sn, sk, true );
		m.gain_bwd( m.lm, mixF, dl, rows, v );
	}
	float* dmixF = m.reduce_rows( dmixF_part, TL * H );
	CUDA_CHECK( cudaMemsetAsync( dx, 0, TL * flat * sizeof( float ), m.st ) );
	float* hscr = m.ws.f( hc_scratch_floats( m.fhc, Tl ) );
	hc_backward( *m.blas, m.fhc, x, Tl, c, dmixF, nullptr, dx, hscr );
	m.phase( "head bwd" );
	return total;
}

void Model::freeze_routing( bool on ) { p_->frozen = on; }

void Model::set_timing( bool on ) { p_->timing = on; }

void Model::set_fp16( bool on ) { p_->fp16 = on; }

void Model::set_qsa( bool on ) { p_->qsa_on = on; }

std::string Model::timing_report( bool reset ) {
	std::vector<std::pair<double, std::string>> v;
	double total = 0.0;
	for ( const auto& kv : p_->times ) {
		v.push_back( { kv.second, kv.first } );
		if ( kv.first[0] != '(' ) total += kv.second;
	}
	std::sort( v.rbegin(), v.rend() );
	std::string out;
	char line[160];
	for ( const auto& e : v ) {
		std::snprintf( line, sizeof( line ), "  %-40s %8.2f s  %5.1f%%\n", e.second.c_str(), e.first, total > 0 ? 100.0 * e.first / total : 0.0 );
		out += line;
	}
	if ( reset ) p_->times.clear();
	return out;
}

std::string Model::memory_report() const {
	const Impl& m = *p_;
	const double G = 1073741824.0;
	size_t fr = 0, tot = 0, pd = 0, ph = 0;
	for ( const ParamPool* P : { &m.shared_p, &m.expert_p, &m.norm_p, &m.dense_p, &m.hc_p, &m.gain_sp, &m.gain_ep } ) {
		pd += P->device_bytes();
		ph += P->host_bytes();
	}
	pd += m.gain_sp.cap + m.gain_ep.cap;   // the gains' codes
	CUDA_CHECK( cudaSetDevice( device_ ) );
	CUDA_CHECK( cudaMemGetInfo( &fr, &tot ) );
	char out[900];
	std::snprintf( out, sizeof( out ),
		"  codes and FP16 scales (arena)   %6.2f GiB of %.2f\n"
		"  dense copies (FP16)             %6.2f GiB\n"
		"  kept sublayer outputs           %6.2f GiB\n"
		"  state pool                      %6.2f GiB of %.2f\n"
		"  workspace (peak)                %6.2f GiB of %.2f\n"
		"  scale pools                     %6.2f GiB (shared %zu, experts %zu, norms %zu, small dense %zu, hyper-connection mixes %zu, tile gains %zu + %zu "
		"values; Adam state %s, %d-bit moments; %.2f GiB on the host)\n"
		"  device in use                   %6.2f GiB of %.2f\n",
		m.arena.offset_bytes / G, m.arena.total_bytes / G, m.dense_bytes / G, m.kept_bytes / G, m.state.off / G, m.state.cap / G, m.ws.peak / G,
		m.ws.cap() / G, pd / G, m.shared_p.used, m.expert_p.used, m.norm_p.used, m.dense_p.used, m.hc_p.used, m.gain_sp.used, m.gain_ep.used,
		m.opt_host ? "on the host" : "on the device", m.opt_bits, ph / G, ( tot - fr ) / G, tot / G );
	return out;
}

std::vector<Tuned*> Model::tensors() {
	std::vector<Tuned*> v;
	for ( Layer& lw : p_->layers ) {
		for ( Tuned* t : { &lw.qkv, &lw.z, &lw.out, &lw.q, &lw.k, &lw.v, &lw.o, &lw.sg, &lw.su, &lw.sd } ) if ( t->k > 0 ) v.push_back( t );
		for ( auto* list : { &lw.eg, &lw.eu, &lw.ed } ) for ( Tuned& t : *list ) if ( t.k > 0 ) v.push_back( &t );
	}
	if ( p_->head ) v.push_back( &p_->lm );
	return v;
}

std::vector<Dense*> Model::dense_tensors() {
	std::vector<Dense*> v;
	for ( Dense& d : p_->dense_t ) v.push_back( &d );
	return v;
}

void Model::set_train_norms( bool on ) { p_->train_norms = on; }

void Model::zero_grads() {
	CUDA_CHECK( cudaSetDevice( device_ ) );
	for ( ParamPool* P : { &p_->shared_p, &p_->expert_p, &p_->norm_p, &p_->dense_p, &p_->hc_p, &p_->gain_sp, &p_->gain_ep } )
		if ( P->used ) CUDA_CHECK( cudaMemsetAsync( P->g, 0, P->used * sizeof( float ), p_->st ) );
}

void Model::adam( const AdamOptions& o, int step, float grad_scale ) {
	Impl& m = *p_;
	CUDA_CHECK( cudaSetDevice( device_ ) );
	m.sum_ranks( m.shared_p.g, m.shared_p.used );
	for ( ParamPool* P : { &m.shared_p, &m.expert_p } ) P->step( true, o, step, grad_scale, m.st );
	if ( m.train_norms ) {
		m.sum_ranks( m.norm_p.g, m.norm_p.used );
		m.norm_p.step( false, o, step, grad_scale, m.st );
	}
	if ( m.train_dense || m.train_router ) {
		// a dense row feeds a dot product: Adam's step of about lr on every entry moves it by up to lr times its width, so these take a
		// fraction of the rate
		AdamOptions od = o;
		od.lr *= m.dense_lr;
		m.sum_ranks( m.dense_p.g, m.dense_p.used );
		m.dense_p.step( false, od, step, grad_scale, m.st );
	}
	if ( m.train_hc ) {
		AdamOptions oh = o;
		oh.lr *= m.hc_lr;
		m.sum_ranks( m.hc_p.g, m.hc_p.used );
		m.hc_p.step( false, oh, step, grad_scale, m.st );
	}
	if ( m.train_gains ) {
		AdamOptions og = o;
		og.lr *= m.gain_lr;
		m.sum_ranks( m.gain_sp.g, m.gain_sp.used );
		for ( ParamPool* P : { &m.gain_sp, &m.gain_ep } ) P->step( false, og, step, grad_scale, m.st );
		gain_codes( m.gain_sp.s, m.gcodes_s, m.gain_sp.used, m.st );
		gain_codes( m.gain_ep.s, m.gcodes_e, m.gain_ep.used, m.st );
	}
	CUDA_CHECK( cudaStreamSynchronize( m.st ) );
}

void Model::set_train_gains( bool on, float lr_scale ) {
	p_->train_gains = on;
	p_->gain_lr = lr_scale;
}

void Model::set_accumulate( int L ) {
	Impl& m = *p_;
	CUDA_CHECK( cudaSetDevice( device_ ) );
	const int li = L < 0 ? -1 : L - first_;
	ENGINE_ASSERT( li < static_cast<int>( m.layers.size() ), "qwen4_exp training model: set_accumulate outside the loaded layers" );
	if ( li == m.acc_layer ) return;
	if ( m.acc_buf ) CUDA_CHECK( cudaFree( m.acc_buf ) );
	m.acc_buf = nullptr;
	m.acc.clear();
	m.acc_prepared = false;
	m.acc_layer = li;
	if ( li < 0 ) return;
	const size_t H = cfg_.hidden, I = cfg_.inter, per = 3 * H * I + H * H + I * I, ne = static_cast<size_t>( m.e1 - m.e0 );
	CUDA_CHECK( cudaMalloc( &m.acc_buf, ne * per * sizeof( float ) ) );
	CUDA_CHECK( cudaMemsetAsync( m.acc_buf, 0, ne * per * sizeof( float ), m.st ) );
	m.acc.resize( ne );
	for ( size_t e = 0; e < ne; ++e ) {
		float* b = m.acc_buf + e * per;
		for ( int kd = 0; kd < 3; ++kd ) m.acc[e].dR[kd] = b + kd * H * I;
		m.acc[e].Hx = b + 3 * H * I;
		m.acc[e].Ha = b + 3 * H * I + H * H;
	}
	CUDA_CHECK( cudaStreamSynchronize( m.st ) );
}

namespace {

// A routed expert projection's quantization options as it was encoded
engine::quant::Exl3QuantOptions reencode_options( const pod::LinearWeight& w ) {
	engine::quant::Exl3QuantOptions o;
	o.code = w.bits;
	o.cb = w.mul1 ? 2 : w.mcg ? 1 : 0;
	o.volta = w.volta_tiles;
	o.open_trellis = w.w3.data != nullptr;
	o.tile_gain = false;   // held: each tile encoded at its gain
	return o;
}

} // namespace

double Model::reencode_prepare( bool feedback ) {
	Impl& m = *p_;
	CUDA_CHECK( cudaSetDevice( device_ ) );
	ENGINE_ASSERT( m.acc_layer >= 0 && !m.acc_prepared, "qwen4_exp training model: reencode_prepare without accumulated sums" );
	Layer& lw = m.layers[m.acc_layer];
	std::vector<Tuned>* kinds[] = { &lw.eg, &lw.eu, &lw.ed };
	const size_t HI = static_cast<size_t>( cfg_.hidden ) * cfg_.inter;
	float *R = m.ws.f( HI ), *D = m.ws.f( HI );
	auto* R16 = static_cast<uint16_t*>( m.ws.raw( HI * 2 ) );
	double sums[2] = { 0.0, 0.0 };
	m.saved_codes.assign( m.acc.size() * 3, {} );
	m.saved_starts.assign( m.acc.size() * 3, {} );
	for ( size_t e = 0; e < m.acc.size(); ++e ) for ( int kd = 0; kd < 3; ++kd ) {
		Tuned& t = ( *kinds[kd] )[m.e0 + e];
		std::vector<uint16_t>& sc = m.saved_codes[3 * e + kd];
		sc.resize( t.w.w0.numel() );
		CUDA_CHECK( cudaMemcpy( sc.data(), t.w.w0.data, sc.size() * 2, cudaMemcpyDeviceToHost ) );
		if ( t.w.w3.data ) {
			std::vector<uint16_t>& ss = m.saved_starts[3 * e + kd];
			ss.resize( t.w.w3.numel() );
			CUDA_CHECK( cudaMemcpy( ss.data(), t.w.w3.data, ss.size() * 2, cudaMemcpyDeviceToHost ) );
		}
		Impl::ExpertAccum& A = m.acc[e];
		if ( A.rows == 0 ) continue;
		m.decode_tiles( t, R, R16 );
		double dn = 0.0, tn = 0.0;
		if ( feedback ) {
			// the step D = M^-1 G and the norms in the metric
			CUDA_CHECK( cudaMemcpyAsync( D, A.dR[kd], HI * sizeof( float ), cudaMemcpyDeviceToDevice, m.st ) );
			ENGINE_ASSERT( engine::quant::exl3_metric_solve( D, R, kd < 2 ? A.Hx : A.Ha, A.rows, t.su, t.k, t.n, 0.025f, &dn, &tn, m.st ),
				"qwen4_exp training model: the re-encode metric's factorization failed" );
		}
		else {
			// the step G itself and the plain norms
			dn = engine::train::weighted_sq_norm( A.dR[kd], nullptr, t.k, t.n, m.st );
			tn = engine::train::weighted_sq_norm( R, nullptr, t.k, t.n, m.st );
		}
		sums[0] += dn;
		sums[1] += tn;
	}
	m.acc_prepared = true;
	float s2[2] = { static_cast<float>( sums[0] ), static_cast<float>( sums[1] ) };
	float* d2 = m.ws.f( 2 );
	CUDA_CHECK( cudaMemcpyAsync( d2, s2, sizeof( s2 ), cudaMemcpyHostToDevice, m.st ) );
	m.sum_ranks( d2, 2 );
	CUDA_CHECK( cudaMemcpyAsync( s2, d2, sizeof( s2 ), cudaMemcpyDeviceToHost, m.st ) );
	CUDA_CHECK( cudaStreamSynchronize( m.st ) );
	m.ws.reset( m.st );
	return s2[1] > 0.0f ? std::sqrt( static_cast<double>( s2[0] ) / s2[1] ) : 0.0;
}

double Model::reencode( float eta, bool feedback, double* lin, double* quad ) {
	Impl& m = *p_;
	CUDA_CHECK( cudaSetDevice( device_ ) );
	ENGINE_ASSERT( m.acc_prepared, "qwen4_exp training model: reencode before reencode_prepare" );
	Layer& lw = m.layers[m.acc_layer];
	std::vector<Tuned>* kinds[] = { &lw.eg, &lw.eu, &lw.ed };
	const size_t HI = static_cast<size_t>( cfg_.hidden ) * cfg_.inter;
	float *T = m.ws.f( HI ), *D = m.ws.f( HI );
	auto* R16 = static_cast<uint16_t*>( m.ws.raw( HI * 2 ) );
	double err = 0.0, sums[2] = { 0.0, 0.0 };
	int count = 0;
	for ( size_t e = 0; e < m.acc.size(); ++e ) for ( int kd = 0; kd < 3; ++kd ) {
		Impl::ExpertAccum& A = m.acc[e];
		if ( A.rows == 0 ) continue;
		Tuned& t = ( *kinds[kd] )[m.e0 + e];
		const float* H = kd < 2 ? A.Hx : A.Ha;
		m.decode_tiles( t, T, R16 );
		// the step: with the feedback D = M^-1 G (the encode's metric M), else G itself (each tile encoded on its own, in the plain metric)
		const float* dir = A.dR[kd];
		if ( feedback ) {
			CUDA_CHECK( cudaMemcpyAsync( D, A.dR[kd], HI * sizeof( float ), cudaMemcpyDeviceToDevice, m.st ) );
			ENGINE_ASSERT( engine::quant::exl3_metric_solve( D, T, H, A.rows, t.su, t.k, t.n, 0.025f, nullptr, nullptr, m.st ),
				"qwen4_exp training model: the re-encode metric's factorization failed" );
			dir = D;
		}
		add_scaled( T, dir, -eta, HI, m.st );
		engine::quant::Exl3Tensor out;
		engine::quant::Exl3ReencodeStep step;
		step.feedback = feedback;
		step.D = dir;
		step.G = A.dR[kd];
		step.eta = eta;
		ENGINE_ASSERT( engine::quant::exl3_reencode( T, H, A.rows, t.su, static_cast<const uint8_t*>( t.w.w4.data ), t.k, t.n, reencode_options( t.w ),
			out, m.st, &step ), "qwen4_exp training model: a re-encode failed" );
		sums[0] += step.lin;
		sums[1] += step.quad;
		ENGINE_ASSERT( out.trellis.size() == static_cast<size_t>( t.w.w0.numel() ), "qwen4_exp training model: a re-encode's codes differ in size" );
		CUDA_CHECK( cudaMemcpyAsync( const_cast<void*>( t.w.w0.data ), out.trellis.data(), out.trellis.size() * 2, cudaMemcpyHostToDevice, m.st ) );
		if ( t.w.w3.data ) CUDA_CHECK( cudaMemcpyAsync( const_cast<void*>( t.w.w3.data ), out.starts.data(), out.starts.size() * 2, cudaMemcpyHostToDevice, m.st ) );
		CUDA_CHECK( cudaStreamSynchronize( m.st ) );
		err += out.proxy_err;
		++count;
	}
	float s2[2] = { static_cast<float>( sums[0] ), static_cast<float>( sums[1] ) };
	float* d2 = m.ws.f( 2 );
	CUDA_CHECK( cudaMemcpyAsync( d2, s2, sizeof( s2 ), cudaMemcpyHostToDevice, m.st ) );
	m.sum_ranks( d2, 2 );
	CUDA_CHECK( cudaMemcpyAsync( s2, d2, sizeof( s2 ), cudaMemcpyDeviceToHost, m.st ) );
	CUDA_CHECK( cudaStreamSynchronize( m.st ) );
	if ( lin ) *lin = s2[0];
	if ( quad ) *quad = s2[1];
	m.ws.reset( m.st );
	return count ? err / count : 0.0;
}

void Model::restore_codes() {
	Impl& m = *p_;
	CUDA_CHECK( cudaSetDevice( device_ ) );
	if ( m.acc_layer < 0 || m.saved_codes.empty() ) return;
	Layer& lw = m.layers[m.acc_layer];
	std::vector<Tuned>* kinds[] = { &lw.eg, &lw.eu, &lw.ed };
	for ( size_t e = 0; e < m.acc.size(); ++e ) for ( int kd = 0; kd < 3; ++kd ) {
		Tuned& t = ( *kinds[kd] )[m.e0 + e];
		const std::vector<uint16_t>& sc = m.saved_codes[3 * e + kd];
		CUDA_CHECK( cudaMemcpy( const_cast<void*>( t.w.w0.data ), sc.data(), sc.size() * 2, cudaMemcpyHostToDevice ) );
		const std::vector<uint16_t>& ss = m.saved_starts[3 * e + kd];
		if ( !ss.empty() ) CUDA_CHECK( cudaMemcpy( const_cast<void*>( t.w.w3.data ), ss.data(), ss.size() * 2, cudaMemcpyHostToDevice ) );
	}
}

void Model::set_train_dense( bool on, float lr_scale ) {
	p_->train_dense = on;
	p_->dense_lr = lr_scale;
}

void Model::set_train_router( bool on ) { p_->train_router = on; }

void Model::set_keep_outputs( bool on ) { p_->keep_outputs = on; }

void Model::set_route_candidates( int n ) {
	ENGINE_ASSERT( n >= 0 && cfg_.top_k + n <= 16, "qwen4_exp training model: at most 16 routes and candidates a row" );
	p_->route_cand = n;
}

void Model::set_train_hc( bool on, float lr_scale ) {
	p_->train_hc = on;
	p_->hc_lr = lr_scale;
}

void Model::set_optimizer( bool host, int moment_bits ) {
	ENGINE_ASSERT( moment_bits == 32 || moment_bits == 8, "qwen4_exp training model: Adam's moments are 32 or 8 bits" );
	p_->opt_host = host;
	p_->opt_bits = moment_bits;
}

void Model::keep_best() {
	CUDA_CHECK( cudaSetDevice( device_ ) );
	for ( ParamPool* P : { &p_->shared_p, &p_->expert_p, &p_->norm_p, &p_->dense_p, &p_->hc_p, &p_->gain_sp, &p_->gain_ep } ) P->keep_best( p_->st );
	CUDA_CHECK( cudaStreamSynchronize( p_->st ) );
}

namespace {

std::vector<uint16_t> to_fp16( const float* src, int n ) {
	std::vector<uint16_t> h( n );
	for ( int j = 0; j < n; ++j ) {
		const __half v = __float2half( src[j] );
		std::memcpy( &h[j], &v, 2 );
	}
	return h;
}

} // namespace

void Model::best_scales( std::vector<std::pair<std::string, std::pair<std::vector<uint16_t>, std::vector<uint16_t>>>>& out ) {
	CUDA_CHECK( cudaSetDevice( device_ ) );
	std::vector<float> hb[2];
	ParamPool* pools[] = { &p_->shared_p, &p_->expert_p };
	for ( int i = 0; i < 2; ++i ) hb[i] = pools[i]->best_values();
	auto fp16 = [&]( const float* dev, int n ) {
		const int i = pools[0]->holds( dev ) ? 0 : 1;
		return to_fp16( hb[i].data() + ( dev - pools[i]->s ), n );
	};
	for ( Tuned* t : tensors() ) out.push_back( { t->name, { fp16( t->su, t->k ), fp16( t->sv, t->n ) } } );
}

void Model::best_dense( std::vector<std::pair<std::string, std::vector<uint16_t>>>& out ) {
	CUDA_CHECK( cudaSetDevice( device_ ) );
	const ParamPool* pools[] = { &p_->norm_p, &p_->dense_p, &p_->hc_p };
	const std::vector<float> hb[] = { pools[0]->best_values(), pools[1]->best_values(), pools[2]->best_values() };
	for ( const Dense& d : p_->dense_t ) {
		const int i = pools[0]->holds( d.w ) ? 0 : pools[1]->holds( d.w ) ? 1 : 2;
		out.push_back( { d.name, to_fp16( hb[i].data() + ( d.w - pools[i]->s ), d.n ) } );
	}
}

void Model::best_gains( std::vector<std::pair<std::string, std::vector<uint8_t>>>& out ) {
	CUDA_CHECK( cudaSetDevice( device_ ) );
	if ( !p_->train_gains ) return;
	const ParamPool* pools[] = { &p_->gain_sp, &p_->gain_ep };
	const std::vector<float> hb[] = { pools[0]->best_values(), pools[1]->best_values() };
	for ( Tuned* t : tensors() ) {
		if ( t->tg == nullptr ) continue;
		const int i = pools[0]->holds( t->tg ) ? 0 : 1;
		const float* g = hb[i].data() + ( t->tg - pools[i]->s );
		std::vector<uint8_t> c( static_cast<size_t>( t->k / 16 ) * ( t->n / 16 ) );
		for ( size_t j = 0; j < c.size(); ++j ) c[j] = static_cast<uint8_t>( std::clamp( std::nearbyint( g[j] * 1024.0f - 896.0f ), 0.0f, 255.0f ) );
		out.push_back( { t->name, std::move( c ) } );
	}
}

} // namespace engine::train::qwen4_exp
