// The qwen4_exp training model (engine/train/qwen4_exp.h): loading a range of layers of an EXL3 checkpoint (tensor parallel: a rank's share of
// them), each layer's forward and its backward (the forward recomputed), the KL head and Adam on the scales.
#include "engine/train/qwen4_exp.h"
#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/core/log.h"
#include "engine/loader/safetensors.h"
#include "reconstruct.cuh"
#include "exl3_vl.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#if defined( ENGINE_HAVE_NCCL )
#include <nccl.h>
#endif
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
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
	size_t cap = 0, off = 0;
	void init( size_t bytes ) {
		CUDA_CHECK( cudaMalloc( &base, bytes ) );
		cap = bytes;
		off = 0;
	}
	void release() {
		if ( base ) cudaFree( base );
		base = nullptr;
		cap = off = 0;
	}
	void* raw( size_t bytes ) {
		const size_t a = ( off + 255 ) / 256 * 256;
		ENGINE_ASSERT( a + bytes <= cap, "qwen4_exp training model: device pool exhausted" );
		off = a + bytes;
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

// Parallel arrays of trainable scale vectors, a tensor's vectors at the same offset in each: the checkpoint's scales, the log deltas, the
// current scales, their gradients, Adam's moments and the best scales kept
struct ParamPool {
	float *s0 = nullptr, *a = nullptr, *s = nullptr, *g = nullptr, *m = nullptr, *v = nullptr, *best = nullptr;
	size_t cap = 0, used = 0;
	void init( size_t n ) {
		cap = n;
		used = 0;
		for ( float** p : { &s0, &a, &s, &g, &m, &v, &best } ) {
			CUDA_CHECK( cudaMalloc( p, std::max<size_t>( n, 4 ) * sizeof( float ) ) );
			CUDA_CHECK( cudaMemset( *p, 0, std::max<size_t>( n, 4 ) * sizeof( float ) ) );
		}
	}
	void release() {
		for ( float** p : { &s0, &a, &s, &g, &m, &v, &best } ) {
			if ( *p ) cudaFree( *p );
			*p = nullptr;
		}
		cap = used = 0;
	}
	size_t take( size_t n ) {
		const size_t o = used;
		used += ( n + 3 ) / 4 * 4;   // float4 aligned
		ENGINE_ASSERT( used <= cap, "qwen4_exp training model: scale pool exhausted" );
		return o;
	}
	bool holds( const float* p ) const { return p >= s && p < s + cap; }
};

// Rows r0 .. r0 + k and columns c0 .. c0 + n of a tensor (k, n < 0: to the end)
struct View {
	int r0 = 0, k = -1, c0 = 0, n = -1;
};

struct Layer {
	bool full = false, ple = false;
	HcWeights ahc, mhc;
	// GDN (this rank's key heads from k0h, value heads from v0h; conv_l: the convolution of their channels)
	Tuned qkv, z, out;
	float *A = nullptr, *B = nullptr, *A_log = nullptr, *dt_bias = nullptr, *conv_l = nullptr, *norm = nullptr, *g_norm = nullptr;
	int kh_l = 0, vh_l = 0, k0h = 0, v0h = 0;
	// attention (this rank's query heads from q0h, key / value heads from kv0)
	Tuned q, k, v, o;
	float *q_norm = nullptr, *k_norm = nullptr, *g_q_norm = nullptr, *g_k_norm = nullptr;
	int hq_l = 0, hk_l = 0, q0h = 0, kv0 = 0;
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
	Pool ws;                    // per-call workspace
	ParamPool shared_p;         // the scales of the tensors every rank holds
	ParamPool expert_p;         // the scales of this rank's experts
	ParamPool norm_p;           // the norm weights (s0: the checkpoint's, a: additive deltas, s: current)
	std::vector<Dense> dense_t; // per norm weight its name and buffers in norm_p
	bool train_norms = true;
	std::vector<Layer> layers;
	int first = 0;
	int rank = 0, ranks = 1;
	void* comm = nullptr;
	int e0 = 0, e1 = 0;         // this rank's experts
	HcWeights fhc;              // the final mixer
	Tuned lm;
	bool head = false;
	float* ones_f = nullptr;
	int max_rows = 0;
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
	// routed experts per group, and padded routes per group (or one expert's routes when more)
	static constexpr int kGroupExperts = 32;
	static constexpr int kGroupRows = 4096;
	size_t moe_ws_bytes() const {
		const size_t H = cfg.hidden, I = cfg.inter, rows = std::max<size_t>( kGroupRows, max_rows );
		return ( static_cast<size_t>( kGroupExperts ) * 3 * H * I + rows * ( 9 * H + 12 * I ) ) * sizeof( float ) + ( 64ull << 20 );
	}

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

	// A dense tensor as an FP32 copy (its FP16 load only the conversion's source: the arena is rewound after)
	float* dense( const std::string& name, size_t* n_out = nullptr ) {
		const size_t mark = arena.offset_bytes;
		pod::TensorView t = repo.load( name.c_str(), arena, st, pod::DType::FP16 );
		ENGINE_ASSERT( t.data != nullptr, "qwen4_exp training model: a dense tensor is missing" );
		const size_t n = t.numel();
		float* f = state.f( n );
		half_to_float( t.data, f, n, st );
		CUDA_CHECK( cudaStreamSynchronize( st ) );
		arena.offset_bytes = mark;
		if ( n_out ) *n_out = n;
		return f;
	}

	// A norm's weights, trainable: the checkpoint's values (s0) and the current ones (returned) in norm_p, its gradient into *grad (null where this
	// rank adds none: grad itself null)
	float* norm_weights( const std::string& name, float** grad ) {
		const size_t mark = arena.offset_bytes;
		pod::TensorView t = repo.load( name.c_str(), arena, st, pod::DType::FP16 );
		ENGINE_ASSERT( t.data != nullptr, "qwen4_exp training model: a norm's weights are missing" );
		const size_t n = t.numel(), o = norm_p.take( n );
		half_to_float( t.data, norm_p.s0 + o, n, st );
		CUDA_CHECK( cudaMemcpyAsync( norm_p.s + o, norm_p.s0 + o, n * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
		CUDA_CHECK( cudaStreamSynchronize( st ) );
		arena.offset_bytes = mark;
		Dense d;
		d.name = name;
		d.w = norm_p.s + o;
		d.g = norm_p.g + o;
		d.n = static_cast<int>( n );
		dense_t.push_back( d );
		if ( grad ) *grad = norm_p.g + o;
		return norm_p.s + o;
	}

	Tuned tuned( const std::string& name, bool expert ) {
		Tuned t;
		t.name = name;
		t.w = repo.load_linear( name.c_str(), arena, 4, st );
		ENGINE_ASSERT( t.w.w0.data != nullptr && t.w.type == pod::QuantType::TRELLIS, "qwen4_exp training model: an EXL3 tensor is missing" );
		t.k = t.w.w1.shape[0];
		t.n = t.w.w2.shape[0];
		ParamPool& P = expert ? expert_p : shared_p;
		const size_t ou = P.take( t.k ), ov = P.take( t.n );
		t.s0u = P.s0 + ou; t.au = P.a + ou; t.su = P.s + ou; t.gu = P.g + ou; t.mu = P.m + ou; t.vu = P.v + ou;
		t.s0v = P.s0 + ov; t.av = P.a + ov; t.sv = P.s + ov; t.gv = P.g + ov; t.mv = P.m + ov; t.vv = P.v + ov;
		half_to_float( t.w.w1.data, t.s0u, t.k, st );
		half_to_float( t.w.w2.data, t.s0v, t.n, st );
		CUDA_CHECK( cudaMemcpyAsync( t.su, t.s0u, t.k * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
		CUDA_CHECK( cudaMemcpyAsync( t.sv, t.s0v, t.n * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
		return t;
	}

	HcWeights hcw( const std::string& pre, bool inject ) {
		HcWeights w;
		w.hidden = cfg.hidden;
		w.hc = cfg.hc;
		w.rank = cfg.rank;
		w.eps = cfg.eps;
		w.norm = norm_weights( pre + ".hc_norm.weight", &w.gnorm );
		w.down = dense( pre + ".input_mix_weight_down.weight" );
		w.up = dense( pre + ".input_mix_weight_up.weight" );
		w.inject = inject ? dense( pre + ".block_inject_weight.weight" ) : nullptr;
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
	}

	void run( int L, const float* x_in, float* x_out, int T, const float* ple_k, const float* ple_v, const float* dx_out, float* dx_in, bool keep );
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
	// Arena: the packed codes of the range (of the experts this rank's share) and the FP16 scales
	size_t total = 0;
	for ( int L = first; L < last; ++L ) {
		const std::string pre = "model.language_model.layers." + std::to_string( L ) + ".";
		const size_t all = m.repo.bytes_of( pre, { "ple_embedding", "ngram_table" } ), ex = m.repo.bytes_of( pre + "mlp.experts." );
		total += all - ex + ex / E * ( owned + 1 );
	}
	if ( m.head ) total += m.repo.bytes_of( "lm_head" ) + m.repo.bytes_of( "model.language_model.hyper_connection_mixer" );
	if ( !engine::core::arena_create( m.arena, device, total + total / 8 + ( 256u << 20 ) ) ) {
		LOG_ERROR( "[train] device %d: no arena of %.2f GiB", device, ( total + total / 8 ) / 1073741824.0 );
		return false;
	}
	const size_t Rl = static_cast<size_t>( max_rows ) / ranks;
	m.state.init( static_cast<size_t>( nl ) * ( ( 80ull << 20 ) + 2 * Rl * H * sizeof( float ) ) + ( 512ull << 20 ) );
	{
		const size_t C = static_cast<size_t>( 2 * c.lin_k + c.lin_v ) * c.lin_d, CV = static_cast<size_t>( c.lin_v ) * c.lin_d;
		const size_t QD = static_cast<size_t>( c.heads ) * c.head_dim, KD = static_cast<size_t>( c.kv_heads ) * c.head_dim;
		const size_t gdn = ( H + C ) + ( H + CV ) + ( CV + H ), att = ( H + 2 * QD ) + 2 * ( H + KD ) + ( QD + H ), sh = 3 * ( H + c.shared_inter );
		m.shared_p.init( nl * ( std::max( gdn, att ) + sh + 64 ) + ( m.head ? H + c.vocab + 64 : 0 ) );
		m.expert_p.init( static_cast<size_t>( nl ) * owned * 3 * ( H + I + 8 ) + 64 );
		const size_t flat = static_cast<size_t>( c.hc ) * H;
		m.norm_p.init( nl * ( 2 * flat + 2 * c.head_dim + c.lin_d + 64 ) + 4 * flat + 256 );
		m.dense_t.reserve( static_cast<size_t>( nl ) * 4 + 8 );
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
		lw.ahc = m.hcw( pre + ".attn_hyper_connection", true );
		lw.mhc = m.hcw( pre + ".mlp_hyper_connection", true );
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
		}
		else {
			lw.kh_l = c.lin_k / ranks;
			lw.vh_l = c.lin_v / ranks;
			lw.k0h = rank * lw.kh_l;
			lw.v0h = rank * lw.vh_l;
			lw.qkv = m.tuned( pre + ".linear_attn.in_proj_qkv", false );
			lw.z = m.tuned( pre + ".linear_attn.in_proj_z", false );
			lw.out = m.tuned( pre + ".linear_attn.out_proj", false );
			lw.A = m.dense( pre + ".linear_attn.in_proj_a.weight" );
			lw.B = m.dense( pre + ".linear_attn.in_proj_b.weight" );
			lw.A_log = m.dense( pre + ".linear_attn.A_log" );
			lw.dt_bias = m.dense( pre + ".linear_attn.dt_bias" );
			lw.norm = m.norm_weights( pre + ".linear_attn.norm.weight", &lw.g_norm );
			// the convolution's channels of this rank's heads: its q, k and v column blocks, [C_l, 4]
			const float* conv = m.dense( pre + ".linear_attn.conv1d.weight" );
			const size_t D = c.lin_d, GQK = c.lin_k * D, CQK = lw.kh_l * D, CV = lw.vh_l * D;
			lw.conv_l = m.state.f( ( 2 * CQK + CV ) * 4 );
			const size_t from[] = { lw.k0h * D, GQK + lw.k0h * D, 2 * GQK + lw.v0h * D }, to[] = { 0, CQK, 2 * CQK }, n[] = { CQK, CQK, CV };
			for ( int j = 0; j < 3; ++j )
				CUDA_CHECK( cudaMemcpyAsync( lw.conv_l + to[j] * 4, conv + from[j] * 4, n[j] * 4 * sizeof( float ), cudaMemcpyDeviceToDevice, m.st ) );
		}
		lw.router = m.dense( pre + ".mlp.gate.weight" );
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
		lw.sh_gate = m.dense( pre + ".mlp.shared_expert_gate.weight" );
		if ( lw.ple ) {
			// every rank runs the PLE layer whole: rank 0 alone adds its norms' gradients
			lw.ple_nk = m.norm_weights( pre + ".ple.norm_key.weight", rank == 0 ? &lw.g_ple_nk : nullptr );
			lw.ple_nq = m.norm_weights( pre + ".ple.norm_query.weight", rank == 0 ? &lw.g_ple_nq : nullptr );
			lw.ple_nc = m.norm_weights( pre + ".ple.norm_conv.weight", rank == 0 ? &lw.g_ple_nc : nullptr );
			lw.ple_conv = m.dense( pre + ".ple.conv1d.weight" );
		}
		lw.cyA = m.state.f( Rl * H );
		lw.cyM = m.state.f( Rl * H );
		m.layers.push_back( std::move( lw ) );
		m.last_ids.emplace_back();
		CUDA_CHECK( cudaStreamSynchronize( m.st ) );
		LOG_INFO( "[train] device %d: layer %d loaded (%s), arena %.2f GiB, state %.2f GiB", device, L, c.full_attention( L ) ? "attention" : "GDN",
			m.arena.offset_bytes / 1073741824.0, m.state.off / 1073741824.0 );
	}
	if ( m.head ) {
		m.fhc = m.hcw( "model.language_model.hyper_connection_mixer", false );
		m.lm = m.tuned( "lm_head", false );
	}
	CUDA_CHECK( cudaStreamSynchronize( m.st ) );
	// Workspace for one layer call at max_rows (the largest sublayer plus the streams, see run)
	const size_t R = static_cast<size_t>( max_rows ), flat = static_cast<size_t>( c.hc ) * c.hidden;
	m.ws.init( R * flat * 4 * 40 + ( 1024ull << 20 ) + m.moe_ws_bytes() );
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
	// This rank's routed experts in groups of similar route counts (largest first): a group's routes are padded to its largest count (zero
	// rows), so that each of its products is one strided batch over its experts' cores, decoded together (FP16 with fp16, else FP32)
	const int I = cfg.inter;
	std::vector<int> cnt( E, 0 ), order;
	for ( int id : h_ids ) cnt[id]++;
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
	std::vector<int> ints( static_cast<size_t>( 2 ) * P + h_ids.size() + order.size(), -1 );
	int *prow_h = ints.data(), *pslot_h = prow_h + P, *inv_h = pslot_h + P, *gexp_h = inv_h + h_ids.size();
	{
		std::vector<int> base( E, -1 ), fill( E, 0 );
		for ( const Group& g : groups ) for ( int i = 0; i < g.count; ++i ) {
			base[order[g.first + i]] = g.p0 + i * g.m;
			gexp_h[g.first + i] = order[g.first + i];
		}
		for ( int r = 0; r < T; ++r ) for ( int j = 0; j < K; ++j ) {
			const int s = r * K + j, e = h_ids[s];
			if ( base[e] < 0 ) continue;
			const int p = base[e] + fill[e]++;
			prow_h[p] = r;
			pslot_h[p] = s;
			inv_h[s] = p;
		}
	}
	float* Wk[3];   // the group's cores per kind: [count, H, I] gate and up, [count, I, H] down (FP16 or FP32 elements)
	for ( float*& p : Wk ) p = ws.f( static_cast<size_t>( kGroupExperts ) * H * I );
	const size_t esz = fp16 ? 2 : 4;
	struct Job { int off, count, bits, kind; bool mcg, mul1, volta; };
	std::vector<std::vector<Job>> jobs( groups.size() );
	std::vector<const void*> tab_h;
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
				Job jb{ static_cast<int>( tab_h.size() ), static_cast<int>( same.size() ), w0.bits, kd, w0.mcg, w0.mul1, w0.volta_tiles };
				for ( int q = 0; q < 4; ++q ) for ( int j : same ) {
					const pod::LinearWeight& wj = ( *kinds[kd] )[order[g.first + j]].w;
					tab_h.push_back( q == 0 ? wj.w0.data : q == 1 ? wj.w3.data : q == 2 ? wj.w4.data :
						static_cast<const void*>( reinterpret_cast<char*>( Wk[kd] ) + static_cast<size_t>( j ) * H * I * esz ) );
				}
				jobs[gi].push_back( jb );
			}
		}
	}
	int* d_ints = ws.i( ints.size() );
	const void** d_tab = static_cast<const void**>( ws.raw( std::max<size_t>( tab_h.size(), 1 ) * sizeof( void* ) ) );
	CUDA_CHECK( cudaMemcpyAsync( d_ints, ints.data(), ints.size() * sizeof( int ), cudaMemcpyHostToDevice, st ) );
	if ( !tab_h.empty() ) CUDA_CHECK( cudaMemcpyAsync( d_tab, tab_h.data(), tab_h.size() * sizeof( void* ), cudaMemcpyHostToDevice, st ) );
	const int *d_prow = d_ints, *d_pslot = d_prow + P, *d_inv = d_pslot + P, *d_gexp = d_inv + h_ids.size();
	float* dw = dy ? ws.fz( static_cast<size_t>( T ) * K, st ) : nullptr;   // the routing weights' gradient by slot
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
			if ( y ) route_combine( Yr, d_inv, g.p0, Pg, w, T, K, H, y, st );
			if ( !dy ) continue;
			// down: dsv += colsum( dY Zd ), dAk = ( dY sv ) Td^T, dsu += colsum( dAk A )
			route_gather( dy, pr, ps, w, Pg, H, dYr, st );
			route_dots_pad( dy, pr, ps, Yr, Pg, H, dw, st );
			seg_colsum_prod( dYr, Zd, m, H, n, gx, tab( 2, 3 ), st );
			prod( dYr, tab( 2, 1 ), H, I, true, Wk[2], dAk );
			seg_colsum_prod( dAk, Am, m, I, n, gx, tab( 2, 2 ), st );
			route_swiglu_backward( dAk, Gm, Um, gx, m, tab( 2, 0 ), Pg, I, dG, dU, st );
			// gate and up: dsv += colsum( dG Zg ), dXk = ( dG sv ) T^T, dsu += colsum( dXk X ), dX = dXk su
			seg_colsum_prod( dG, Zg, m, I, n, gx, tab( 0, 3 ), st );
			seg_colsum_prod( dU, Zu, m, I, n, gx, tab( 1, 3 ), st );
			prod( dG, tab( 0, 1 ), I, H, true, Wk[0], dXg );
			prod( dU, tab( 1, 1 ), I, H, true, Wk[1], dXu );
			seg_colsum_prod( dXg, Xg, m, H, n, gx, tab( 0, 2 ), st );
			seg_colsum_prod( dXu, Xg, m, H, n, gx, tab( 1, 2 ), st );
			route_scale( dXg, gx, m, tab( 0, 0 ), Pg, H, dYr, false, st );
			route_scale( dXu, gx, m, tab( 1, 0 ), Pg, H, dYr, true, st );
			route_combine( dYr, d_inv, g.p0, Pg, nullptr, T, K, H, dmix, st );
		}
	}
	phase( dy ? "moe routed experts fwd+bwd" : "moe routed experts fwd" );
	if ( dy ) {
		// The routing weights' gradient (this rank's routes) into the router input
		float* dl = ws.f( static_cast<size_t>( T ) * K );
		topk_softmax_backward( w, dw, T, K, dl, st );
		router_input_backward( dl, ids, lw.router, T, K, H, dmix, st );
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
	float *sconv = ws.f( TR * C ), *Cv = ws.f( TR * C );
	conv4_silu_forward( QKV, lw.conv_l, T, C, 1, sconv, Cv, st );
	float *qr = ws.f( TR * CQK ), *kr = ws.f( TR * CQK ), *v = ws.f( TR * CV ), *qn = ws.f( TR * CQK ), *kn = ws.f( TR * CQK );
	float *iq = ws.f( TR * kh ), *ik = ws.f( TR * kh );
	copy_cols( Cv, C, 0, qr, CQK, 0, T, CQK, false, st );
	copy_cols( Cv, C, CQK, kr, CQK, 0, T, CQK, false, st );
	copy_cols( Cv, C, 2 * CQK, v, CV, 0, T, CV, false, st );
	rms_forward( qr, nullptr, 1, false, false, T * kh, D, 1e-6f, iq, qn, st );
	rms_forward( kr, nullptr, 1, false, false, T * kh, D, 1e-6f, ik, kn, st );
	float* ck = ws.f( gdn_ckpt_floats( T, vh, kGdnChunk ) );
	float* oraw = ws.f( TR * CV );
	const float scale = 1.0f / std::sqrt( static_cast<float>( D ) );
	gdn_forward( qn, kn, v, gate, beta, T, kh, vh, scale, kGdnChunk, ck, oraw, st );
	float *io = ws.f( TR * vh ), *o = ws.f( TR * CV ), *Zout = ws.f( TR * H );
	gated_rms_forward( oraw, Zo, lw.norm, T * vh, D, 1e-6f, io, o, st );
	lin_fwd( lw.out, Tout, o, T, Zout, y, vo );
	if ( dy ) {
		float* dO = ws.f( TR * CV );
		lin_bwd( lw.out, Tout, o, Zout, dy, T, dO, false, vo );
		float *doraw = ws.fz( TR * CV, st ), *dZo = ws.fz( TR * CV, st );
		gated_rms_backward( oraw, Zo, lw.norm, T * vh, D, io, dO, doraw, dZo, st, lw.g_norm );
		float *dqn = ws.fz( TR * CQK, st ), *dkn = ws.fz( TR * CQK, st ), *dv = ws.fz( TR * CV, st ), *dgate = ws.fz( TR * vh, st ), *dbeta = ws.fz( TR * vh, st );
		float* gscr = ws.f( gdn_scratch_floats( vh, kGdnChunk ) );
		gdn_backward( qn, kn, v, gate, beta, T, kh, vh, scale, kGdnChunk, ck, doraw, dqn, dkn, dv, dgate, dbeta, gscr, st );
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
		conv4_silu_backward( sconv, lw.conv_l, T, C, 1, dC, dQKV, st );
		float *da = ws.f( TR * vh ), *db = ws.f( TR * vh );
		gdn_gates_backward( a, b, lw.A_log + lw.v0h, lw.dt_bias + lw.v0h, T, vh, dgate, dbeta, da, db, st );
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
	rope_forward( qn, T, hq, d, cfg.rot_dim, cfg.rope_theta, st );
	rope_forward( kn, T, hk, d, cfg.rot_dim, cfg.rope_theta, st );
	float *S = ws.f( TR * TR ), *O = ws.f( TR * QD ), *Gd = ws.f( TR * QD ), *Zo = ws.f( TR * H );
	const float scale = 1.0f / std::sqrt( static_cast<float>( d ) );
	attention_forward( *blas, qn, kn, V, T, hq, hk, d, scale, S, O );
	sigmoid_gate_forward( O, gate, Gd, TR * QD, st );
	lin_fwd( lw.o, To, Gd, T, Zo, y, vo );
	if ( dy ) {
		float* dGd = ws.f( TR * QD );
		lin_bwd( lw.o, To, Gd, Zo, dy, T, dGd, false, vo );
		float *dO = ws.fz( TR * QD, st ), *dgate = ws.fz( TR * QD, st );
		sigmoid_gate_backward( O, gate, dGd, dO, dgate, TR * QD, st );
		float *dqn = ws.fz( TR * QD, st ), *dkn = ws.fz( TR * KD, st ), *dV = ws.fz( TR * KD, st ), *dP = ws.f( TR * TR );
		attention_backward( *blas, qn, kn, V, T, hq, hk, d, scale, dO, dqn, dkn, dV, S, dP );
		rope_backward( dqn, T, hq, d, cfg.rot_dim, cfg.rope_theta, st );
		rope_backward( dkn, T, hk, d, cfg.rot_dim, cfg.rope_theta, st );
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

void Model::Impl::run( int L, const float* x_in, float* x_out, int T, const float* ple_k, const float* ple_v, const float* dx_out, float* dx_in, bool keep ) {
	Layer& lw = layers[L - first];
	const bool bwd = dx_out != nullptr;
	ENGINE_ASSERT( T % ranks == 0 && T <= max_rows, "qwen4_exp training model: the rows exceed the workspace or do not split over the ranks" );
	ws.off = 0;
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
		conv4_silu_forward( pn, lw.ple_conv, T, static_cast<int>( flat ), 3, ps, act, st );
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
	conv4_silu_backward( ps, lw.ple_conv, T, static_cast<int>( flat ), 3, dx0f, dpn, st );
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

void Model::forward( int L, const float* x_in, float* x_out, int rows, const float* ple_k, const float* ple_v, bool keep ) {
	CUDA_CHECK( cudaSetDevice( device_ ) );
	p_->run( L, x_in, x_out, rows, ple_k, ple_v, nullptr, nullptr, keep );
}

void Model::backward( int L, const float* x_in, const float* dx_out, float* dx_in, int rows, const float* ple_k, const float* ple_v ) {
	CUDA_CHECK( cudaSetDevice( device_ ) );
	p_->run( L, x_in, nullptr, rows, ple_k, ple_v, dx_out, dx_in, false );
}

double Model::head( const float* x, int rows, const Teacher& t, float* dx ) {
	Impl& m = *p_;
	CUDA_CHECK( cudaSetDevice( device_ ) );
	ENGINE_ASSERT( m.head, "qwen4_exp training model: this range has no head" );
	ENGINE_ASSERT( rows % m.ranks == 0 && rows <= m.max_rows, "qwen4_exp training model: the rows exceed the workspace or do not split over the ranks" );
	const int H = cfg_.hidden, V = cfg_.vocab, NC = Impl::kLmChunk, Tl = rows / m.ranks;
	const size_t TR = static_cast<size_t>( rows ), TL = static_cast<size_t>( Tl ), flat = static_cast<size_t>( cfg_.hc ) * H;
	// this rank's vocabulary columns (whole 128-column blocks)
	const int vb = V / 128, v0 = vb * m.rank / m.ranks * 128, v1 = vb * ( m.rank + 1 ) / m.ranks * 128;
	m.ws.off = 0;
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
	for ( ParamPool* P : { &p_->shared_p, &p_->expert_p, &p_->norm_p } ) CUDA_CHECK( cudaMemsetAsync( P->g, 0, P->used * sizeof( float ), p_->st ) );
}

void Model::adam( const AdamOptions& o, int step, float grad_scale ) {
	Impl& m = *p_;
	CUDA_CHECK( cudaSetDevice( device_ ) );
	m.sum_ranks( m.shared_p.g, m.shared_p.used );
	for ( ParamPool* P : { &m.shared_p, &m.expert_p } ) adam_log_scales( P->s0, P->a, P->s, P->g, P->m, P->v, P->used, step, grad_scale, o, m.st );
	if ( m.train_norms ) {
		m.sum_ranks( m.norm_p.g, m.norm_p.used );
		ParamPool& P = m.norm_p;
		adam_deltas( P.s0, P.a, P.s, P.g, P.m, P.v, P.used, step, grad_scale, o, m.st );
	}
	CUDA_CHECK( cudaStreamSynchronize( m.st ) );
}

void Model::keep_best() {
	CUDA_CHECK( cudaSetDevice( device_ ) );
	for ( ParamPool* P : { &p_->shared_p, &p_->expert_p, &p_->norm_p } )
		CUDA_CHECK( cudaMemcpyAsync( P->best, P->s, P->used * sizeof( float ), cudaMemcpyDeviceToDevice, p_->st ) );
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
	for ( int i = 0; i < 2; ++i ) {
		hb[i].resize( pools[i]->used );
		CUDA_CHECK( cudaMemcpy( hb[i].data(), pools[i]->best, hb[i].size() * sizeof( float ), cudaMemcpyDeviceToHost ) );
	}
	auto fp16 = [&]( const float* dev, int n ) {
		const int i = pools[0]->holds( dev ) ? 0 : 1;
		return to_fp16( hb[i].data() + ( dev - pools[i]->s ), n );
	};
	for ( Tuned* t : tensors() ) out.push_back( { t->name, { fp16( t->su, t->k ), fp16( t->sv, t->n ) } } );
}

void Model::best_dense( std::vector<std::pair<std::string, std::vector<uint16_t>>>& out ) {
	CUDA_CHECK( cudaSetDevice( device_ ) );
	const ParamPool& P = p_->norm_p;
	std::vector<float> hb( P.used );
	CUDA_CHECK( cudaMemcpy( hb.data(), P.best, hb.size() * sizeof( float ), cudaMemcpyDeviceToHost ) );
	for ( const Dense& d : p_->dense_t ) out.push_back( { d.name, to_fp16( hb.data() + ( d.w - P.s ), d.n ) } );
}

} // namespace engine::train::qwen4_exp
