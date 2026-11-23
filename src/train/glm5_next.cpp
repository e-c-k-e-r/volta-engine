// The glm5_next training model (engine/train/glm5_next.h): loading a range of layers of a GLM-5.3-Flash EXL3 checkpoint (tensor parallel: a
// rank's share of them), each layer's forward and its backward (the forward recomputed), the KL head and Adam on the scales.
#include "engine/train/glm5_next.h"
#include "pools.h"
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
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>

namespace engine::train::glm5_next {

namespace q4 = engine::train::qwen4_exp;

bool Config::read( const std::string& model_dir ) {
	std::ifstream f( model_dir + "/config.json" );
	if ( !f ) return false;
	std::stringstream ss;
	ss << f.rdbuf();
	const auto j = nlohmann::json::parse( ss.str(), nullptr, false );
	if ( j.is_discarded() ) return false;
	const auto& t = j.contains( "text_config" ) ? j["text_config"] : j;
	hidden = t.value( "hidden_size", hidden );
	hc = t.value( "hc_mult", hc );
	layers = t.value( "num_hidden_layers", layers );
	dense_layers = t.value( "first_k_dense_replace", dense_layers );
	int kda_dim = kKdaDim;
	if ( t.contains( "linear_attn_config" ) ) {
		const auto& la = t["linear_attn_config"];
		kda_heads = la.value( "num_heads", kda_heads );
		kda_dim = la.value( "head_dim", kda_dim );
		gate_lower = la.value( "gate_lower_bound", gate_lower );
		if ( la.contains( "full_attn_layers" ) ) full_layers = la["full_attn_layers"].get<std::vector<int>>();
	}
	heads = t.value( "num_attention_heads", heads );
	head_dim = t.value( "qk_nope_head_dim", head_dim );
	const int rope = t.value( "qk_rope_head_dim", 0 ), v_dim = t.value( "v_head_dim", head_dim );
	q_rank = t.value( "q_lora_rank", q_rank );
	kv_rank = t.value( "kv_lora_rank", kv_rank );
	experts = t.value( "n_routed_experts", experts );
	top_k = t.value( "num_experts_per_tok", top_k );
	inter = t.value( "moe_intermediate_size", inter );
	dense_inter = t.value( "intermediate_size", dense_inter );
	shared_inter = inter * t.value( "n_shared_experts", 1 );
	route_scale = t.value( "routed_scaling_factor", route_scale );
	swiglu_limit = t.value( "swiglu_limit", swiglu_limit );
	vocab = t.value( "vocab_size", vocab );
	eps = t.value( "rms_norm_eps", eps );
	hc_eps = t.value( "hc_eps", hc_eps );
	sinkhorn = t.value( "hc_sinkhorn_iters", sinkhorn );
	// the training forms' fixed shapes: four streams, KDA heads of 128, MLA without a rotary part and value heads as wide as the keys
	return hc == kHc && kda_dim == kKdaDim && rope == 0 && v_dim == head_dim;
}

bool Config::full_attention( int layer ) const { return std::find( full_layers.begin(), full_layers.end(), layer ) != full_layers.end(); }

namespace {

#if defined( ENGINE_HAVE_NCCL )
#define G5_NCCL( x ) \
	do { \
		const ncclResult_t r_ = ( x ); \
		ENGINE_ASSERT( r_ == ncclSuccess, "glm5_next training model: an NCCL call failed" ); \
	} while ( 0 )
#endif

// Rows r0 .. r0 + k and columns c0 .. c0 + n of a tensor (k, n < 0: to the end)
struct View {
	int r0 = 0, k = -1, c0 = 0, n = -1;
};

struct Layer {
	int L = 0;
	bool mla = false, moe = false;
	MhcSite hcA, hcF;                  // the attention and FFN sites
	int h0 = 0, hl = 0;                // this rank's attention heads (KDA or MLA)
	Tuned o;                           // the attention's o_proj (this rank's input rows)
	// KDA: qkv_proj ([q | k | v] columns), the gates' dense projections (b_proj, f_b, g_b and the convolution: this rank's rows; f_a, g_a whole)
	Tuned qkv;
	const float *b_l = nullptr, *fa = nullptr, *fb_l = nullptr, *ga = nullptr, *gb_l = nullptr, *conv_l = nullptr, *alog_l = nullptr, *dt_l = nullptr;
	int fr = 0, gr = 0;                // f_a's and g_a's rows
	float *onorm = nullptr, *g_onorm = nullptr;
	// MLA: kv_b's rows of this rank's heads, their K rows and then their V rows ([2 hl head_dim, kv_rank])
	Tuned qa, qb, kva;
	const float* kvb_l = nullptr;
	float *qa_norm = nullptr, *g_qa_norm = nullptr, *kva_norm = nullptr, *g_kva_norm = nullptr;
	// the dense MLP
	Tuned mg, mu, md;
	// MoE: the router [experts, hidden] and its selection bias, this rank's experts (indexed by expert, the others empty), the shared expert
	const float *router = nullptr, *bias = nullptr;
	std::vector<Tuned> eg, eu, ed;
	float** etab = nullptr;            // device: per expert, (kind * 4 + q) * experts + e for kind gate / up / down and q su / sv / gu / gv
	Tuned sg, su, sd;
	size_t code_lo = 0, code_hi = 0;   // with host codes: the experts' codes' range of the host arena
	// the sublayer outputs of a kept forward (the rank's rows), for the next backward
	float *cyA = nullptr, *cyF = nullptr;
	bool cached = false;
};

} // namespace

struct Model::Impl {
	Config cfg;
	cudaStream_t st = nullptr;
	std::unique_ptr<Blas> blas;
	engine::loader::SafeTensorsRepo repo;
	pod::MemoryArena arena{};        // codes, FP16 scales, the dense loads' room (with expert_host: the routed experts' codes elsewhere)
	pod::MemoryArena host_codes{};   // with expert_host: the routed experts' codes, pinned host memory
	bool expert_host = true;
	double expert_device_gib = 0.0;  // with expert_host: whole layers' codes kept on the device up to this size
	std::vector<bool> host_layer;    // per layer of the range: its experts' codes in host memory
	// Two device stages of one MoE layer's expert codes each (expert_host), copied on their own stream; ready: the copy landed, done: the last
	// layer call reading it finished
	struct Stage {
		char* base = nullptr;
		int layer = -1;
		cudaEvent_t ready = nullptr, done = nullptr;
	};
	Stage stage[2];
	cudaStream_t cst = nullptr;
	size_t stage_bytes = 0;
	int stage_last = 0;
	// A layer call's small tables (the routes, the decode jobs) go to the device through mapped pinned memory and a kernel (upload_words): a
	// copy would wait on the copy engine behind the stage copy of the next layer's codes
	char* up_host = nullptr;
	size_t up_cap = 0, up_off = 0;
	void upload( const void* src, size_t bytes, void* dst ) {
		const size_t at = ( up_off + 15 ) / 16 * 16;
		if ( at + bytes > up_cap ) {
			// grown before anything of this call is written past the old room (the uploads queued so far read it: wait for them)
			CUDA_CHECK( cudaStreamSynchronize( st ) );
			if ( up_host ) CUDA_CHECK( cudaFreeHost( up_host ) );
			up_cap = std::max<size_t>( 2 * ( at + bytes ), 1 << 20 );
			CUDA_CHECK( cudaHostAlloc( reinterpret_cast<void**>( &up_host ), up_cap, cudaHostAllocMapped | cudaHostAllocPortable ) );
		}
		std::memcpy( up_host + at, src, bytes );
		void* mapped = nullptr;
		CUDA_CHECK( cudaHostGetDevicePointer( &mapped, up_host + at, 0 ) );
		upload_words( mapped, dst, ( bytes + 3 ) / 4 * 4, st );
		up_off = at + bytes;
	}
	Pool state;                      // the FP32 dense copies, the expert tables, the kept sublayer outputs
	size_t dense_bytes = 0, kept_bytes = 0;
	Workspace ws;
	ParamPool shared_p;              // the scales of the tensors every rank holds
	ParamPool expert_p;              // the scales of this rank's experts
	ParamPool norm_p;                // the norm weights (s0: the checkpoint's, a: log deltas, s: current)
	std::vector<Dense> dense_t;
	bool train_norms = true;
	bool keep_outputs = true;
	bool opt_host = false;
	int opt_bits = 32;
	std::vector<Layer> layers;
	int first = 0;
	int rank = 0, ranks = 1;
	void* comm = nullptr;
	int e0 = 0, e1 = 0;              // this rank's experts
	float *fnorm = nullptr, *g_fnorm = nullptr;
	Tuned lm;
	bool head = false;
	int max_rows = 0;
	int seqs = 1;                    // the current layer call's sequences
	bool frozen = false;
	bool fp16 = true;
	bool timing = false;
	std::map<std::string, double> times;
	std::chrono::steady_clock::time_point t_mark;
	std::vector<std::vector<int>> last_ids;
	static constexpr int kLmChunk = 8192;
	static constexpr int kKdaChunk = 64;
	static constexpr int kAttnBlock = 1024;
	static constexpr int kGroupExperts = 8;   // routed experts per group (their decoded cores: 8 x 3 x hidden x inter)
	static constexpr int kGroupRows = 4096;

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
		collective( [&] { G5_NCCL( ncclAllGather( mine, all, per_rank, ncclFloat, nc(), st ) ); } );
#endif
		return all;
	}
	// The rank's block of the sum over the ranks of a [ranks * per_rank] partial
	float* reduce_rows( float* part, size_t per_rank ) {
		if ( ranks == 1 ) return part;
		float* mine = ws.f( per_rank );
#if defined( ENGINE_HAVE_NCCL )
		collective( [&] { G5_NCCL( ncclReduceScatter( part, mine, per_rank, ncclFloat, ncclSum, nc(), st ) ); } );
#endif
		return mine;
	}
	void sum_ranks( float* buf, size_t n ) {
		if ( ranks == 1 || n == 0 ) return;
#if defined( ENGINE_HAVE_NCCL )
		collective( [&] { G5_NCCL( ncclAllReduce( buf, buf, n, ncclFloat, ncclSum, nc(), st ) ); } );
#endif
	}
	void bcast_ints( int* buf, size_t n ) {
		if ( ranks == 1 ) return;
#if defined( ENGINE_HAVE_NCCL )
		collective( [&] { G5_NCCL( ncclBroadcast( buf, buf, n, ncclInt32, 0, nc(), st ) ); } );
#endif
	}

	// ---- loading ----
	// A dense tensor as FP32 in the state pool: blocks of its rows (first row, rows; a 1-D tensor's rows are its values) end to end, or
	// with none the whole tensor (its load into the arena only the copy's source: the arena is rewound after)
	const float* dense( const std::string& name, const std::vector<std::pair<size_t, size_t>>& blocks = {} ) {
		const size_t mark = arena.offset_bytes;
		pod::TensorView t = repo.load( name.c_str(), arena, st, pod::DType::FP32 );
		ENGINE_ASSERT( t.data != nullptr, "glm5_next training model: a dense tensor is missing" );
		const size_t n = t.numel(), rows = static_cast<size_t>( t.shape[0] ), row_len = n / rows;
		size_t total = 0;
		for ( const auto& b : blocks ) {
			ENGINE_ASSERT( b.first + b.second <= rows, "glm5_next training model: a dense tensor's rows out of range" );
			total += b.second;
		}
		if ( blocks.empty() ) total = rows;
		float* p = state.f( total * row_len );
		dense_bytes += total * row_len * sizeof( float );
		const float* src = static_cast<const float*>( t.data );
		if ( blocks.empty() ) CUDA_CHECK( cudaMemcpyAsync( p, src, n * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
		size_t o = 0;
		for ( const auto& b : blocks ) {
			CUDA_CHECK( cudaMemcpyAsync( p + o * row_len, src + b.first * row_len, b.second * row_len * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
			o += b.second;
		}
		CUDA_CHECK( cudaStreamSynchronize( st ) );
		arena.offset_bytes = mark;
		return p;
	}
	int rows_of( const std::string& name ) const {
		const pod::SafeTensorMeta* t = repo.find( name.c_str() );
		ENGINE_ASSERT( t != nullptr, "glm5_next training model: a tensor is missing" );
		return t->shape[0];
	}
	// A trainable dense tensor in pool P: its current values (returned; the checkpoint's become the pool's reference after the
	// load), its gradient into *grad
	float* pooled( ParamPool& P, const std::string& name, float** grad ) {
		const size_t mark = arena.offset_bytes;
		pod::TensorView t = repo.load( name.c_str(), arena, st, pod::DType::FP32 );
		ENGINE_ASSERT( t.data != nullptr, "glm5_next training model: a trainable dense tensor is missing" );
		const size_t n = t.numel(), o = P.take( n );
		CUDA_CHECK( cudaMemcpyAsync( P.s + o, t.data, n * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
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

	// host: a routed expert's codes go to the host arena
	Tuned tuned( const std::string& name, bool expert, bool host = false ) {
		Tuned t;
		t.name = name;
		const size_t mark = arena.offset_bytes;
		t.w = repo.load_linear( name.c_str(), arena, 4, st, host ? &host_codes : nullptr );
		ENGINE_ASSERT( t.w.w0.data != nullptr && t.w.type == pod::QuantType::TRELLIS, "glm5_next training model: an EXL3 tensor is missing" );
		t.k = t.w.w1.shape[0];
		t.n = t.w.w2.shape[0];
		ParamPool& P = expert ? expert_p : shared_p;
		const size_t ou = P.take( t.k ), ov = P.take( t.n );
		t.su = P.s + ou;
		t.gu = P.g + ou;
		t.sv = P.s + ov;
		t.gv = P.g + ov;
		q4::half_to_float( t.w.w1.data, t.su, t.k, st );   // the checkpoint's values (the pool's reference copy taken after the load)
		q4::half_to_float( t.w.w2.data, t.sv, t.n, st );
		if ( host && t.w.w3.data == nullptr && t.w.w4.data == nullptr ) {
			// the codes in host memory and the scales copied: the FP16 scales' room back
			CUDA_CHECK( cudaStreamSynchronize( st ) );
			arena.offset_bytes = mark;
			t.w.w1.data = t.w.w2.data = nullptr;
		}
		return t;
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

	// ---- the expert stages ----
	// The stage holding layer li's codes; when none does, the copy into one started (the stage other than avoid's, else the one used less
	// recently)
	int stage_load( int li, int avoid ) {
		for ( int b = 0; b < 2; ++b ) if ( stage[b].layer == li ) return b;
		const int b = avoid >= 0 ? 1 - avoid : 1 - stage_last;
		const Layer& lw = layers[li];
		CUDA_CHECK( cudaStreamWaitEvent( cst, stage[b].done, 0 ) );
		CUDA_CHECK( cudaMemcpyAsync( stage[b].base, static_cast<const char*>( host_codes.base_ptr ) + lw.code_lo, lw.code_hi - lw.code_lo,
			cudaMemcpyHostToDevice, cst ) );
		CUDA_CHECK( cudaEventRecord( stage[b].ready, cst ) );
		stage[b].layer = li;
		return b;
	}
	// At a layer call's start: the stage of layer li's codes (-1: not a MoE layer) and the copy of the MoE layer expected next (next, -1:
	// none) started
	int stage_start( int li, int next ) {
		const int b = layers[li].moe && host_layer[li] ? stage_load( li, -1 ) : -1;
		if ( b >= 0 ) stage_last = b;
		if ( next >= 0 && next != li ) stage_load( next, b );
		return b;
	}
	// Before the routed experts: the stream waits for the stage's copy
	void stage_wait( int b ) {
		if ( b < 0 ) return;
		CUDA_CHECK( cudaStreamWaitEvent( st, stage[b].ready, 0 ) );
		phase( "expert stage waits" );
	}
	// The next MoE layer of the range with its codes in host memory from li in direction dir (-1: none)
	int next_moe( int li, int dir ) const {
		for ( int j = li + dir; j >= 0 && j < static_cast<int>( layers.size() ); j += dir ) if ( layers[j].moe && host_layer[j] ) return j;
		return -1;
	}
	// A routed expert projection's codes as the decode reads them: in the stage sb (host codes), else where they were loaded
	const void* codes( const Layer& lw, const Tuned& t, int sb ) const {
		if ( sb < 0 ) return t.w.w0.data;
		const size_t off = static_cast<size_t>( static_cast<const char*>( t.w.w0.data ) - static_cast<const char*>( host_codes.base_ptr ) );
		return stage[sb].base + ( off - lw.code_lo );
	}

	void run( int L, const float* x_in, float* x_out, int T, const float* dx_out, float* dx_in, bool keep, int n_seqs );
	// Sublayers on a [T, hidden] (all rows) -> y [T, hidden], with ranks > 1 the rank's partial sum; with dy (all rows), the backward into da
	// (+=, partial) after the forward
	void kda( Layer& lw, const float* a, int T, float* y, const float* dy, float* da );
	void mla( Layer& lw, const float* a, int T, float* y, const float* dy, float* da );
	// The router and this rank's routed experts (y: overwritten, null: not wanted; sb: the stage of their codes, -1: none)
	void moe( Layer& lw, int sb, const float* a, int T, float* y, const float* dy, float* da );
	// Y = down( act( X gate ) ( X up ) ) on rows of this rank (Y overwritten), with dY its backward into dX (+=): the dense MLP and the shared
	// expert (the clamp before the SiLU)
	void mlp( Tuned& g, Tuned& u, Tuned& d, const float* X, int rows, float* Y, const float* dY, float* dX, int inter );
};

Model::Model() : p_( std::make_unique<Impl>() ) {}

Model::~Model() {
	if ( !p_ ) return;
	cudaSetDevice( device_ );
	if ( p_->st ) cudaStreamSynchronize( p_->st );
	if ( p_->cst ) cudaStreamSynchronize( p_->cst );
	p_->state.release();
	p_->ws.release();
	p_->shared_p.release();
	p_->expert_p.release();
	p_->norm_p.release();
	for ( Impl::Stage& s : p_->stage ) {
		if ( s.base ) cudaFree( s.base );
		if ( s.ready ) cudaEventDestroy( s.ready );
		if ( s.done ) cudaEventDestroy( s.done );
	}
	if ( p_->up_host ) cudaFreeHost( p_->up_host );
	engine::core::arena_destroy( p_->arena );
	engine::core::arena_destroy( p_->host_codes );
	if ( p_->cst ) cudaStreamDestroy( p_->cst );
	if ( p_->st ) cudaStreamDestroy( p_->st );
}

backend::stream_t Model::stream() const { return p_->st; }

bool Model::load( const std::string& dir, int first, int last, int device, int max_rows, int rank, int ranks, void* comm ) {
	Impl& m = *p_;
	if ( !cfg_.read( dir ) ) {
		LOG_ERROR( "[train] %s: no readable GLM-5.3-Flash config.json (or a shape the training forms do not cover)", dir.c_str() );
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
	if ( ranks > 1 && ( comm == nullptr || c.kda_heads % ranks || c.heads % ranks || c.experts % ranks || max_rows % ranks || c.vocab % 128 ||
		c.vocab / 128 < ranks ) ) {
		LOG_ERROR( "[train] %d ranks do not divide the model's heads, experts, vocabulary or %d rows", ranks, max_rows );
		return false;
	}
	CUDA_CHECK( cudaSetDevice( device ) );
	CUDA_CHECK( cudaStreamCreateWithFlags( &m.st, cudaStreamNonBlocking ) );
	m.blas = std::make_unique<Blas>( m.st );
	if ( !m.repo.open( dir.c_str() ) ) return false;
	m.head = last == c.layers;
	m.e0 = c.experts * rank / ranks;
	m.e1 = c.experts * ( rank + 1 ) / ranks;
	const int E = c.experts, H = c.hidden;
	const size_t Rl = static_cast<size_t>( max_rows ) / ranks;
	const std::string P0 = "model.language_model.layers.";
	// With expert_host, the MoE layers whose codes stay on the device: up to expert_device_gib of them, spread evenly (MoE layer i of M of them
	// is resident when floor( ( i + 1 ) R / M ) passes an integer)
	m.host_layer.assign( static_cast<size_t>( last - first ), false );
	if ( m.expert_host ) {
		std::vector<int> moe_l;
		for ( int L = first; L < last; ++L ) if ( c.moe( L ) ) moe_l.push_back( L );
		size_t per = 0;   // a layer's codes of this rank's experts
		if ( !moe_l.empty() )
			for ( int e = m.e0; e < m.e1; ++e )
				for ( const char* p : { "gate_proj", "up_proj", "down_proj" } )
					per += ( m.repo.trellis_bytes( ( P0 + std::to_string( moe_l[0] ) + ".mlp.experts." + std::to_string( e ) + "." + p ).c_str() ) + 255 ) / 256 * 256;
		const int M = static_cast<int>( moe_l.size() );
		const int R = per > 0 ? std::min( M, static_cast<int>( m.expert_device_gib * 1073741824.0 / static_cast<double>( per ) ) ) : 0;
		for ( int i = 0; i < M; ++i ) m.host_layer[moe_l[i] - first] = ( i + 1 ) * R / M == i * R / M;
		if ( M > 0 )
			LOG_INFO( "[train] device %d: routed experts' codes of %d of %d MoE layers on the device (%.2f GiB each), the others staged from host memory", device, R, M,
				per / 1073741824.0 );
	}
	// Exact reservations from the checkpoint's tensors. The arena: the range's EXL3 tensors (codes, FP16 scales, tile starts and gains; of the
	// experts this rank's share, their codes in the host arena with expert_host) and room for the largest dense tensor's FP32 load (rewound
	// after its copy). The state pool: the dense copies (this rank's rows), the kept sublayer outputs and the expert tables. The scale pools:
	// every tuned tensor's two vectors, the norm weights
	{
		auto al = []( size_t b ) { return ( b + 255 ) / 256 * 256; };
		auto r4 = []( size_t n ) { return ( n + 3 ) / 4 * 4; };
		auto numel = [&]( const std::string& name ) -> size_t {
			const pod::SafeTensorMeta* t = m.repo.find( name.c_str() );
			if ( t == nullptr ) return 0;
			size_t n = 1;
			for ( int d = 0; d < t->ndim; ++d ) n *= static_cast<size_t>( t->shape[d] );
			return n;
		};
		size_t codes = 0, host_b = 0, dense_b = 0, dense_max = 0, n_shared = 0, n_expert = 0, n_norm = 0, extra = 0;
		auto linear = [&]( const std::string& base, bool expert, bool host = false ) {
			bool companions = false;
			for ( const char* s : { ".tstart", ".tgain" } )
				if ( const pod::SafeTensorMeta* t = m.repo.find( ( base + s ).c_str() ) ) {
					codes += al( t->size_bytes );
					companions = true;
				}
			( host ? host_b : codes ) += al( m.repo.trellis_bytes( base.c_str() ) );
			const size_t k = numel( base + ".suh" ), n = numel( base + ".svh" );
			if ( !host || companions ) codes += al( k * 2 ) + al( n * 2 );   // (else rewound after each load)
			( expert ? n_expert : n_shared ) += r4( k ) + r4( n );
		};
		auto dense_w = [&]( const std::string& name, size_t kept ) {
			dense_max = std::max( dense_max, numel( name ) );
			dense_b += al( kept * sizeof( float ) );
		};
		auto norm = [&]( const std::string& name ) {
			dense_max = std::max( dense_max, numel( name ) );
			n_norm += r4( numel( name ) );
		};
		for ( int L = first; L < last; ++L ) {
			const std::string pre = P0 + std::to_string( L );
			for ( const char* s : { "attn", "ffn" } )
				for ( const char* p : { "_fn", "_base", "_scale" } ) dense_w( pre + ".hc_" + s + p, numel( pre + ".hc_" + s + p ) );
			norm( pre + ".input_layernorm.weight" );
			norm( pre + ".post_attention_layernorm.weight" );
			if ( !c.full_attention( L ) ) {
				for ( const char* p : { "qkv_proj", "o_proj" } ) linear( pre + ".self_attn." + p, false );
				const std::string sa = pre + ".self_attn.";
				dense_w( sa + "b_proj.weight", numel( sa + "b_proj.weight" ) / ranks );
				dense_w( sa + "f_a_proj.weight", numel( sa + "f_a_proj.weight" ) );
				dense_w( sa + "g_a_proj.weight", numel( sa + "g_a_proj.weight" ) );
				dense_w( sa + "f_b_proj.weight", numel( sa + "f_b_proj.weight" ) / ranks );
				dense_w( sa + "g_b_proj.weight", numel( sa + "g_b_proj.weight" ) / ranks );
				dense_w( sa + "conv1d.weight", numel( sa + "conv1d.weight" ) / ranks );
				dense_w( sa + "A_log", numel( sa + "A_log" ) / ranks );
				dense_w( sa + "dt_bias", numel( sa + "dt_bias" ) / ranks );
				norm( sa + "o_norm.weight" );
			}
			else {
				for ( const char* p : { "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj" } ) linear( pre + ".self_attn." + p, false );
				norm( pre + ".self_attn.q_a_layernorm.weight" );
				norm( pre + ".self_attn.kv_a_layernorm.weight" );
				dense_w( pre + ".self_attn.kv_b_proj.weight", numel( pre + ".self_attn.kv_b_proj.weight" ) / ranks );
			}
			if ( !c.moe( L ) ) {
				for ( const char* p : { "gate_proj", "up_proj", "down_proj" } ) linear( pre + ".mlp." + p, false );
			}
			else {
				dense_w( pre + ".mlp.gate.weight", numel( pre + ".mlp.gate.weight" ) );
				dense_w( pre + ".mlp.gate.e_score_correction_bias", numel( pre + ".mlp.gate.e_score_correction_bias" ) );
				for ( int e = m.e0; e < m.e1; ++e )
					for ( const char* p : { "gate_proj", "up_proj", "down_proj" } )
						linear( pre + ".mlp.experts." + std::to_string( e ) + "." + p, true, m.host_layer[L - first] );
				for ( const char* p : { "gate_proj", "up_proj", "down_proj" } ) linear( pre + ".mlp.shared_experts." + p, false );
				extra += al( static_cast<size_t>( 12 ) * E * sizeof( float* ) );
			}
			extra += m.keep_outputs ? 2 * al( Rl * H * sizeof( float ) ) : 0;
		}
		if ( m.head ) {
			norm( "model.language_model.norm.weight" );
			linear( "lm_head", false );
		}
		const size_t arena_b = codes + al( dense_max * sizeof( float ) ) + ( 16ull << 20 );
		if ( !engine::core::arena_create( m.arena, device, arena_b ) ) {
			LOG_ERROR( "[train] device %d: no arena of %.2f GiB", device, arena_b / 1073741824.0 );
			return false;
		}
		if ( host_b > 0 && !engine::core::arena_create_pinned( m.host_codes, host_b + ( 1ull << 20 ) ) ) {
			LOG_ERROR( "[train] no pinned host memory of %.2f GiB for the routed experts' codes", host_b / 1073741824.0 );
			return false;
		}
		m.state.init( dense_b + extra + ( 16ull << 20 ) );
		m.shared_p.init( n_shared + 64, m.opt_host, m.opt_bits );
		m.expert_p.init( n_expert + 64, m.opt_host, m.opt_bits );
		m.norm_p.init( n_norm + 64, m.opt_host, m.opt_bits );
		m.dense_t.reserve( static_cast<size_t>( last - first ) * 8 + 4 );
	}
	char nm[256];
	for ( int L = first; L < last; ++L ) {
		Layer lw;
		const std::string pre = P0 + std::to_string( L ), sa = pre + ".self_attn.";
		lw.L = L;
		lw.mla = c.full_attention( L );
		lw.moe = c.moe( L );
		auto site = [&]( const char* s, const char* norm ) {
			MhcSite w;
			w.hidden = H;
			w.fn = m.dense( pre + ".hc_" + s + "_fn" );
			w.base = m.dense( pre + ".hc_" + s + "_base" );
			w.scale = m.dense( pre + ".hc_" + s + "_scale" );
			w.norm_w = m.norm_weights( pre + "." + norm + ".weight", &w.gnorm );
			w.iters = c.sinkhorn;
			w.hc_eps = c.hc_eps;
			w.norm_eps = c.eps;
			return w;
		};
		lw.hcA = site( "attn", "input_layernorm" );
		lw.hcF = site( "ffn", "post_attention_layernorm" );
		if ( !lw.mla ) {
			lw.hl = c.kda_heads / ranks;
			lw.h0 = rank * lw.hl;
			const size_t D = kKdaDim, C = lw.hl * D, G = static_cast<size_t>( c.kda_heads ) * D, c0 = lw.h0 * D;
			lw.qkv = m.tuned( sa + "qkv_proj", false );
			lw.o = m.tuned( sa + "o_proj", false );
			lw.b_l = m.dense( sa + "b_proj.weight", { { lw.h0, lw.hl } } );
			lw.fa = m.dense( sa + "f_a_proj.weight" );
			lw.ga = m.dense( sa + "g_a_proj.weight" );
			lw.fr = m.rows_of( sa + "f_a_proj.weight" );
			lw.gr = m.rows_of( sa + "g_a_proj.weight" );
			lw.fb_l = m.dense( sa + "f_b_proj.weight", { { c0, C } } );
			lw.gb_l = m.dense( sa + "g_b_proj.weight", { { c0, C } } );
			// the convolution's channels of this rank's heads: their q, k and v blocks, [3 C, 4]
			lw.conv_l = m.dense( sa + "conv1d.weight", { { c0, C }, { G + c0, C }, { 2 * G + c0, C } } );
			lw.alog_l = m.dense( sa + "A_log", { { lw.h0, lw.hl } } );
			lw.dt_l = m.dense( sa + "dt_bias", { { c0, C } } );
			lw.onorm = m.norm_weights( sa + "o_norm.weight", &lw.g_onorm );
		}
		else {
			lw.hl = c.heads / ranks;
			lw.h0 = rank * lw.hl;
			lw.qa = m.tuned( sa + "q_a_proj", false );
			lw.qb = m.tuned( sa + "q_b_proj", false );
			lw.kva = m.tuned( sa + "kv_a_proj_with_mqa", false );
			lw.o = m.tuned( sa + "o_proj", false );
			lw.qa_norm = m.norm_weights( sa + "q_a_layernorm.weight", &lw.g_qa_norm );
			lw.kva_norm = m.norm_weights( sa + "kv_a_layernorm.weight", &lw.g_kva_norm );
			// kv_b holds per head its K rows then its V rows (head_dim each): this rank's heads' K rows, then their V rows
			const size_t hd = c.head_dim;
			std::vector<std::pair<size_t, size_t>> blocks;
			for ( int part = 0; part < 2; ++part )
				for ( int h = lw.h0; h < lw.h0 + lw.hl; ++h ) blocks.push_back( { static_cast<size_t>( h ) * 2 * hd + part * hd, hd } );
			lw.kvb_l = m.dense( sa + "kv_b_proj.weight", blocks );
		}
		if ( !lw.moe ) {
			lw.mg = m.tuned( pre + ".mlp.gate_proj", false );
			lw.mu = m.tuned( pre + ".mlp.up_proj", false );
			lw.md = m.tuned( pre + ".mlp.down_proj", false );
		}
		else {
			lw.router = m.dense( pre + ".mlp.gate.weight" );
			lw.bias = m.dense( pre + ".mlp.gate.e_score_correction_bias" );
			lw.eg.resize( E );
			lw.eu.resize( E );
			lw.ed.resize( E );
			size_t lo = SIZE_MAX, hi = 0;
			const bool host = m.host_layer[L - first];
			for ( int e = m.e0; e < m.e1; ++e ) {
				std::snprintf( nm, sizeof( nm ), "%s.mlp.experts.%d", pre.c_str(), e );
				lw.eg[e] = m.tuned( std::string( nm ) + ".gate_proj", true, host );
				lw.eu[e] = m.tuned( std::string( nm ) + ".up_proj", true, host );
				lw.ed[e] = m.tuned( std::string( nm ) + ".down_proj", true, host );
				if ( !host ) continue;
				for ( const Tuned* t : { &lw.eg[e], &lw.eu[e], &lw.ed[e] } ) {
					const size_t off = static_cast<size_t>( static_cast<const char*>( t->w.w0.data ) - static_cast<const char*>( m.host_codes.base_ptr ) );
					lo = std::min( lo, off );
					hi = std::max( hi, off + t->w.w0.numel() * 2 );
				}
			}
			if ( host && hi > lo ) {
				lw.code_lo = lo;
				lw.code_hi = hi;
				m.stage_bytes = std::max( m.stage_bytes, hi - lo );
			}
			std::vector<float*> tab( static_cast<size_t>( 12 ) * E, nullptr );
			const std::vector<Tuned>* kinds[] = { &lw.eg, &lw.eu, &lw.ed };
			for ( int kd = 0; kd < 3; ++kd ) for ( int e = m.e0; e < m.e1; ++e ) {
				const Tuned& t = ( *kinds[kd] )[e];
				float* q[] = { t.su, t.sv, t.gu, t.gv };
				for ( int j = 0; j < 4; ++j ) tab[static_cast<size_t>( kd * 4 + j ) * E + e] = q[j];
			}
			lw.etab = static_cast<float**>( m.state.raw( tab.size() * sizeof( float* ) ) );
			CUDA_CHECK( cudaMemcpyAsync( lw.etab, tab.data(), tab.size() * sizeof( float* ), cudaMemcpyHostToDevice, m.st ) );
			lw.sg = m.tuned( pre + ".mlp.shared_experts.gate_proj", false );
			lw.su = m.tuned( pre + ".mlp.shared_experts.up_proj", false );
			lw.sd = m.tuned( pre + ".mlp.shared_experts.down_proj", false );
		}
		if ( m.keep_outputs ) {
			lw.cyA = m.state.f( Rl * H );
			lw.cyF = m.state.f( Rl * H );
			m.kept_bytes += 2 * Rl * H * sizeof( float );
		}
		m.layers.push_back( std::move( lw ) );
		m.last_ids.emplace_back();
		CUDA_CHECK( cudaStreamSynchronize( m.st ) );
		LOG_INFO( "[train] device %d: layer %d loaded (%s, %s), arena %.2f GiB, host codes %.2f GiB, state %.2f GiB", device, L, c.full_attention( L ) ? "MLA" : "KDA",
			c.moe( L ) ? "MoE" : "dense MLP", m.arena.offset_bytes / 1073741824.0, m.host_codes.offset_bytes / 1073741824.0, m.state.off / 1073741824.0 );
	}
	if ( m.head ) {
		m.fnorm = m.norm_weights( "model.language_model.norm.weight", &m.g_fnorm );
		m.lm = m.tuned( "lm_head", false );
	}
	if ( m.stage_bytes > 0 ) {
		CUDA_CHECK( cudaStreamCreateWithFlags( &m.cst, cudaStreamNonBlocking ) );
		for ( Impl::Stage& s : m.stage ) {
			CUDA_CHECK( cudaMalloc( &s.base, m.stage_bytes ) );
			CUDA_CHECK( cudaEventCreateWithFlags( &s.ready, cudaEventDisableTiming ) );
			CUDA_CHECK( cudaEventCreateWithFlags( &s.done, cudaEventDisableTiming ) );
			CUDA_CHECK( cudaEventRecord( s.done, m.st ) );
		}
	}
	CUDA_CHECK( cudaStreamSynchronize( m.st ) );
	for ( ParamPool* P : { &m.shared_p, &m.expert_p, &m.norm_p } ) P->start( m.st );
	CUDA_CHECK( cudaStreamSynchronize( m.st ) );
	// The workspace starts at a layer's streams and grows to what the calls need
	m.ws.init( static_cast<size_t>( max_rows ) * c.hc * c.hidden * sizeof( float ) * 4 );
	return true;
}

void Model::Impl::mlp( Tuned& g, Tuned& u, Tuned& d, const float* X, int rows, float* Y, const float* dY, float* dX, int I ) {
	const int H = cfg.hidden;
	const size_t mark = ws.off, RI = static_cast<size_t>( rows ) * I;
	float* Tg = ws.f( static_cast<size_t>( H ) * I );
	float* Tu = ws.f( static_cast<size_t>( H ) * I );
	float* Td = ws.f( static_cast<size_t>( I ) * H );
	float *Zg = ws.f( RI ), *G = ws.f( RI ), *Zu = ws.f( RI ), *U = ws.f( RI ), *A = ws.f( RI ), *Zd = ws.f( static_cast<size_t>( rows ) * H );
	lin_fwd( g, Tg, X, rows, Zg, G );
	lin_fwd( u, Tu, X, rows, Zu, U );
	swiglu_forward( G, U, A, RI, cfg.swiglu_limit, st );
	lin_fwd( d, Td, A, rows, Zd, Y );
	if ( dY ) {
		float *dA = ws.f( RI ), *dG = ws.f( RI ), *dU = ws.f( RI );
		lin_bwd( d, Td, A, Zd, dY, rows, dA, false );
		swiglu_backward( G, U, dA, dG, dU, RI, cfg.swiglu_limit, st );
		lin_bwd( g, Tg, X, Zg, dG, rows, dX, true );
		lin_bwd( u, Tu, X, Zu, dU, rows, dX, true );
	}
	ws.off = mark;
}

void Model::Impl::kda( Layer& lw, const float* a, int T, float* y, const float* dy, float* da ) {
	// this rank's hl heads: qkv_proj's q, k and v column blocks of them side by side as [T, 3 C]
	const int H = cfg.hidden, D = kKdaDim, hl = lw.hl, C = hl * D, G = cfg.kda_heads * D, C3 = 3 * C, FR = lw.fr, GR = lw.gr;
	const View vq{ 0, H, lw.h0 * D, C }, vk{ 0, H, G + lw.h0 * D, C }, vv{ 0, H, 2 * G + lw.h0 * D, C }, vo{ lw.h0 * D, C, 0, H };
	const size_t mark = ws.off, TR = static_cast<size_t>( T ), HC = static_cast<size_t>( H ) * C;
	float *Tq = ws.f( HC ), *Tk = ws.f( HC ), *Tv = ws.f( HC ), *To = ws.f( HC );
	float *Zq = ws.f( TR * C ), *Zk = ws.f( TR * C ), *Zv = ws.f( TR * C ), *Yp = ws.f( TR * C ), *QKV = ws.f( TR * C3 );
	lin_fwd( lw.qkv, Tq, a, T, Zq, Yp, vq );
	q4::copy_cols( Yp, C, 0, QKV, C3, 0, T, C, false, st );
	lin_fwd( lw.qkv, Tk, a, T, Zk, Yp, vk );
	q4::copy_cols( Yp, C, 0, QKV, C3, C, T, C, false, st );
	lin_fwd( lw.qkv, Tv, a, T, Zv, Yp, vv );
	q4::copy_cols( Yp, C, 0, QKV, C3, 2 * C, T, C, false, st );
	const int S = T / seqs;   // a sequence's rows
	float *sconv = ws.f( TR * C3 ), *Cv = ws.f( TR * C3 );
	q4::conv4_silu_forward( QKV, lw.conv_l, T, C3, 1, sconv, Cv, st, S );
	float *qr = ws.f( TR * C ), *kr = ws.f( TR * C ), *v = ws.f( TR * C ), *qn = ws.f( TR * C ), *kn = ws.f( TR * C );
	float *iq = ws.f( TR * hl ), *ik = ws.f( TR * hl );
	q4::copy_cols( Cv, C3, 0, qr, C, 0, T, C, false, st );
	q4::copy_cols( Cv, C3, C, kr, C, 0, T, C, false, st );
	q4::copy_cols( Cv, C3, 2 * C, v, C, 0, T, C, false, st );
	const float qscale = 1.0f / std::sqrt( static_cast<float>( D ) );
	l2norm_forward( qr, T * hl, D, 1e-6f, qscale, iq, qn, st );
	l2norm_forward( kr, T * hl, D, 1e-6f, 1.0f, ik, kn, st );
	// the gates: beta = sigmoid( a b^T ), the decay from f_b( f_a a ); the output gate z = g_b( g_a a )
	float *b = ws.f( TR * hl ), *fa = ws.f( TR * FR ), *fpre = ws.f( TR * C ), *beta = ws.f( TR * hl ), *g = ws.f( TR * C );
	blas->gemm( false, true, T, hl, H, 1.0f, a, lw.b_l, 0.0f, b );
	blas->gemm( false, true, T, FR, H, 1.0f, a, lw.fa, 0.0f, fa );
	blas->gemm( false, true, T, C, FR, 1.0f, fa, lw.fb_l, 0.0f, fpre );
	kda_gates_forward( b, fpre, lw.dt_l, lw.alog_l, T, hl, D, cfg.gate_lower, beta, g, st );
	float *gaz = ws.f( TR * GR ), *z = ws.f( TR * C );
	blas->gemm( false, true, T, GR, H, 1.0f, a, lw.ga, 0.0f, gaz );
	blas->gemm( false, true, T, C, GR, 1.0f, gaz, lw.gb_l, 0.0f, z );
	phase( "kda projections" );
	float* ck = ws.f( kda_ckpt_floats( T, hl, kKdaChunk, seqs ) );
	float* oraw = ws.f( TR * C );
	kda_forward( qn, kn, v, g, beta, T, hl, kKdaChunk, ck, oraw, st, seqs );
	phase( "kda recurrence" );
	float *io = ws.f( TR * hl ), *o = ws.f( TR * C ), *Zo = ws.f( TR * H );
	q4::gated_rms_forward( oraw, z, lw.onorm, T * hl, D, cfg.eps, io, o, st );
	lin_fwd( lw.o, To, o, T, Zo, y, vo );
	if ( dy ) {
		float* dO = ws.f( TR * C );
		lin_bwd( lw.o, To, o, Zo, dy, T, dO, false, vo );
		float *doraw = ws.fz( TR * C, st ), *dz = ws.fz( TR * C, st );
		q4::gated_rms_backward( oraw, z, lw.onorm, T * hl, D, io, dO, doraw, dz, st, lw.g_onorm );
		phase( "kda projections" );
		float *dqn = ws.fz( TR * C, st ), *dkn = ws.fz( TR * C, st ), *dv = ws.fz( TR * C, st ), *dg = ws.fz( TR * C, st ), *dbeta = ws.fz( TR * hl, st );
		float* scr = ws.f( kda_scratch_floats( hl, kKdaChunk, seqs ) );
		kda_backward( qn, kn, v, g, beta, T, hl, kKdaChunk, ck, doraw, dqn, dkn, dv, dg, dbeta, scr, st, seqs );
		phase( "kda recurrence" );
		// dC [T, 3 C]: the q, k and v column blocks (the norms' gradients through blocks of their own)
		float *dqr = ws.fz( TR * C, st ), *dkr = ws.fz( TR * C, st ), *dC = ws.f( TR * C3 );
		l2norm_backward( qr, T * hl, D, iq, qscale, dqn, dqr, st );
		l2norm_backward( kr, T * hl, D, ik, 1.0f, dkn, dkr, st );
		q4::copy_cols( dqr, C, 0, dC, C3, 0, T, C, false, st );
		q4::copy_cols( dkr, C, 0, dC, C3, C, T, C, false, st );
		q4::copy_cols( dv, C, 0, dC, C3, 2 * C, T, C, false, st );
		float* dQKV = ws.fz( TR * C3, st );
		q4::conv4_silu_backward( sconv, lw.conv_l, T, C3, 1, dC, dQKV, st, S );
		float *db = ws.f( TR * hl ), *dfpre = ws.f( TR * C ), *dfa = ws.f( TR * FR ), *dgaz = ws.f( TR * GR );
		kda_gates_backward( b, fpre, lw.dt_l, lw.alog_l, T, hl, D, cfg.gate_lower, dbeta, dg, db, dfpre, nullptr, nullptr, st );
		blas->gemm( false, false, T, H, hl, 1.0f, db, lw.b_l, 1.0f, da );
		blas->gemm( false, false, T, FR, C, 1.0f, dfpre, lw.fb_l, 0.0f, dfa );
		blas->gemm( false, false, T, H, FR, 1.0f, dfa, lw.fa, 1.0f, da );
		blas->gemm( false, false, T, GR, C, 1.0f, dz, lw.gb_l, 0.0f, dgaz );
		blas->gemm( false, false, T, H, GR, 1.0f, dgaz, lw.ga, 1.0f, da );
		// qkv_proj per column block: its slice of dQKV
		q4::copy_cols( dQKV, C3, 0, Yp, C, 0, T, C, false, st );
		lin_bwd( lw.qkv, Tq, a, Zq, Yp, T, da, true, vq );
		q4::copy_cols( dQKV, C3, C, Yp, C, 0, T, C, false, st );
		lin_bwd( lw.qkv, Tk, a, Zk, Yp, T, da, true, vk );
		q4::copy_cols( dQKV, C3, 2 * C, Yp, C, 0, T, C, false, st );
		lin_bwd( lw.qkv, Tv, a, Zv, Yp, T, da, true, vv );
		phase( "kda projections" );
	}
	ws.off = mark;
}

void Model::Impl::mla( Layer& lw, const float* a, int T, float* y, const float* dy, float* da ) {
	// this rank's hl heads of head_dim: q from the low-rank query, k and v from the latent through kv_b's rows of those heads
	const int H = cfg.hidden, hd = cfg.head_dim, hl = lw.hl, QD = hl * hd, QR = cfg.q_rank, KR = cfg.kv_rank;
	const View vqb{ 0, QR, lw.h0 * hd, QD }, vo{ lw.h0 * hd, QD, 0, H };
	const size_t mark = ws.off, TR = static_cast<size_t>( T );
	float *Tqa = ws.f( static_cast<size_t>( H ) * QR ), *Tqb = ws.f( static_cast<size_t>( QR ) * QD ), *Tkv = ws.f( static_cast<size_t>( H ) * KR ),
		*To = ws.f( static_cast<size_t>( QD ) * H );
	float *Zqa = ws.f( TR * QR ), *cqr = ws.f( TR * QR ), *icq = ws.f( TR ), *cq = ws.f( TR * QR );
	lin_fwd( lw.qa, Tqa, a, T, Zqa, cqr );
	q4::rms_forward( cqr, lw.qa_norm, 1, false, true, T, QR, cfg.eps, icq, cq, st );
	float *Zqb = ws.f( TR * QD ), *q = ws.f( TR * QD );
	lin_fwd( lw.qb, Tqb, cq, T, Zqb, q, vqb );
	float *Zkv = ws.f( TR * KR ), *cr = ws.f( TR * KR ), *ic = ws.f( TR ), *cl = ws.f( TR * KR );
	lin_fwd( lw.kva, Tkv, a, T, Zkv, cr );
	q4::rms_forward( cr, lw.kva_norm, 1, false, true, T, KR, cfg.eps, ic, cl, st );
	float *KV = ws.f( TR * 2 * QD ), *K = ws.f( TR * QD ), *V = ws.f( TR * QD );
	blas->gemm( false, true, T, 2 * QD, KR, 1.0f, cl, lw.kvb_l, 0.0f, KV );
	q4::copy_cols( KV, 2 * QD, 0, K, QD, 0, T, QD, false, st );
	q4::copy_cols( KV, 2 * QD, QD, V, QD, 0, T, QD, false, st );
	phase( "mla projections" );
	// per sequence of SL rows the causal attention within it (dense: the DSA selection keeps every key up to its budget)
	const int SL = T / seqs;
	const size_t SR = static_cast<size_t>( SL );
	const int QB = SL > kAttnBlock ? kAttnBlock : SL;
	float *S = ws.f( static_cast<size_t>( QB ) * SR ), *O = ws.f( TR * QD ), *Zo = ws.f( TR * H );
	const float scale = 1.0f / std::sqrt( static_cast<float>( hd ) );
	for ( int s = 0; s < seqs; ++s ) {
		const size_t r0 = static_cast<size_t>( s ) * SR * QD;
		q4::attention_forward( *blas, q + r0, K + r0, V + r0, SL, hl, hl, hd, scale, S, O + r0, {}, QB );
	}
	phase( "mla attention" );
	lin_fwd( lw.o, To, O, T, Zo, y, vo );
	if ( dy ) {
		float* dO = ws.f( TR * QD );
		lin_bwd( lw.o, To, O, Zo, dy, T, dO, false, vo );
		float *dq = ws.fz( TR * QD, st ), *dK = ws.fz( TR * QD, st ), *dV = ws.fz( TR * QD, st ), *dP = ws.f( static_cast<size_t>( QB ) * SR );
		phase( "mla projections" );
		for ( int s = 0; s < seqs; ++s ) {
			const size_t r0 = static_cast<size_t>( s ) * SR * QD;
			q4::attention_backward( *blas, q + r0, K + r0, V + r0, SL, hl, hl, hd, scale, dO + r0, dq + r0, dK + r0, dV + r0, S, dP, {}, QB );
		}
		phase( "mla attention" );
		float *dKV = ws.f( TR * 2 * QD ), *dc = ws.f( TR * KR ), *dcr = ws.fz( TR * KR, st );
		q4::copy_cols( dK, QD, 0, dKV, 2 * QD, 0, T, QD, false, st );
		q4::copy_cols( dV, QD, 0, dKV, 2 * QD, QD, T, QD, false, st );
		blas->gemm( false, false, T, KR, 2 * QD, 1.0f, dKV, lw.kvb_l, 0.0f, dc );
		q4::rms_backward( cr, lw.kva_norm, 1, false, true, T, KR, ic, dc, dcr, st, lw.g_kva_norm );
		lin_bwd( lw.kva, Tkv, a, Zkv, dcr, T, da, true );
		float *dcq = ws.f( TR * QR ), *dcqr = ws.fz( TR * QR, st );
		lin_bwd( lw.qb, Tqb, cq, Zqb, dq, T, dcq, false, vqb );
		q4::rms_backward( cqr, lw.qa_norm, 1, false, true, T, QR, icq, dcq, dcqr, st, lw.g_qa_norm );
		lin_bwd( lw.qa, Tqa, a, Zqa, dcqr, T, da, true );
		phase( "mla projections" );
	}
	ws.off = mark;
}

void Model::Impl::moe( Layer& lw, int sb, const float* a, int T, float* y, const float* dy, float* da ) {
	const int H = cfg.hidden, E = cfg.experts, K = cfg.top_k, I = cfg.inter;
	const size_t mark = ws.off, TR = static_cast<size_t>( T );
	up_off = 0;   // (the previous call's uploads were read: it synchronized at its end)
	if ( y ) CUDA_CHECK( cudaMemsetAsync( y, 0, TR * H * sizeof( float ), st ) );
	// Routing (every rank on all rows; rank 0's selection broadcast, so that the ranks' routes partition the routes)
	float* logits = ws.f( TR * E );
	blas->gemm( false, true, T, E, H, 1.0f, a, lw.router, 0.0f, logits );
	int* ids = ws.i( TR * K );
	float* w = ws.f( TR * K );
	std::vector<int>& kept = last_ids[&lw - layers.data()];
	std::vector<int> h_ids( TR * K );
	if ( frozen && kept.size() == h_ids.size() ) {
		h_ids = kept;
		upload( h_ids.data(), h_ids.size() * sizeof( int ), ids );
		glm_route_given( logits, ids, T, E, K, cfg.route_scale, w, st );
	}
	else {
		glm_route_forward( logits, lw.bias, T, E, K, cfg.route_scale, ids, w, st );
		if ( ranks > 1 ) {
			bcast_ints( ids, h_ids.size() );
			glm_route_given( logits, ids, T, E, K, cfg.route_scale, w, st );
		}
		CUDA_CHECK( cudaMemcpyAsync( h_ids.data(), ids, h_ids.size() * sizeof( int ), cudaMemcpyDeviceToHost, st ) );
		CUDA_CHECK( cudaStreamSynchronize( st ) );
		kept = h_ids;
	}
	// This rank's routed experts in groups of similar route counts (largest first): a group's routes are padded to its largest count (zero
	// rows), so that each of its products is one strided batch over its experts' cores, decoded together (FP16 with fp16, else FP32)
	std::vector<int> cnt( E, 0 ), order;
	for ( int id : h_ids ) cnt[id]++;
	for ( int e = e0; e < e1; ++e ) if ( cnt[e] > 0 ) order.push_back( e );
	std::stable_sort( order.begin(), order.end(), [&]( int x, int z ) { return cnt[x] > cnt[z]; } );
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
	const size_t esz = fp16 ? 2 : 4;
	for ( float*& p : Wk ) p = static_cast<float*>( ws.raw( static_cast<size_t>( kGroupExperts ) * H * I * esz ) );
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
					const Tuned& tj = ( *kinds[kd] )[order[g.first + j]];
					tab_h.push_back( q == 0 ? codes( lw, tj, sb ) : q == 1 ? tj.w.w3.data : q == 2 ? tj.w.w4.data :
						static_cast<const void*>( reinterpret_cast<char*>( Wk[kd] ) + static_cast<size_t>( j ) * H * I * esz ) );
				}
				jobs[gi].push_back( jb );
			}
		}
	}
	int* d_ints = ws.i( ints.size() );
	const void** d_tab = static_cast<const void**>( ws.raw( std::max<size_t>( tab_h.size(), 1 ) * sizeof( void* ) ) );
	upload( ints.data(), ints.size() * sizeof( int ), d_ints );
	if ( !tab_h.empty() ) upload( tab_h.data(), tab_h.size() * sizeof( void* ), d_tab );
	const int *d_prow = d_ints, *d_pslot = d_prow + P, *d_inv = d_pslot + P, *d_gexp = d_inv + h_ids.size();
	float* dw = dy ? ws.fz( TR * K, st ) : nullptr;   // dy against each slot's expert output
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
					q4::route_scale_half( A, gx, m, cs, Pg, kk, A16, rinv, st );
					blas->gemm_h_strided( false, tb, m, nn, kk, A16, sa, W, sW, 0.0f, C, sc, n );
					scale_rows( C, rinv, Pg, nn, st );
				}
				else {
					q4::route_scale( A, gx, m, cs, Pg, kk, Xs, false, st );
					blas->gemm_strided( false, tb, m, nn, kk, 1.0f, Xs, sa, W, sW, 0.0f, C, sc, n );
				}
			};
			q4::route_gather( a, pr, ps, nullptr, Pg, H, Xg, st );
			prod( Xg, tab( 0, 0 ), H, I, false, Wk[0], Zg );
			prod( Xg, tab( 1, 0 ), H, I, false, Wk[1], Zu );
			glm_route_swiglu( Zg, Zu, gx, m, tab( 0, 1 ), tab( 1, 1 ), Pg, I, cfg.swiglu_limit, Gm, Um, Am, st );
			prod( Am, tab( 2, 0 ), I, H, false, Wk[2], Zd );
			q4::route_scale( Zd, gx, m, tab( 2, 1 ), Pg, H, Yr, false, st );
			if ( y ) q4::route_combine( Yr, d_inv, g.p0, Pg, w, T, K, H, y, st );
			if ( !dy ) continue;
			// down: dsv += colsum( dY Zd ), dAk = ( dY sv ) Td^T, dsu += colsum( dAk A )
			q4::route_gather( dy, pr, ps, w, Pg, H, dYr, st );
			q4::route_dots_pad( dy, pr, ps, Yr, Pg, H, dw, st );
			q4::seg_colsum_prod( dYr, Zd, m, H, n, gx, tab( 2, 3 ), st );
			prod( dYr, tab( 2, 1 ), H, I, true, Wk[2], dAk );
			q4::seg_colsum_prod( dAk, Am, m, I, n, gx, tab( 2, 2 ), st );
			glm_route_swiglu_backward( dAk, Gm, Um, gx, m, tab( 2, 0 ), Pg, I, cfg.swiglu_limit, dG, dU, st );
			// gate and up: dsv += colsum( dG Zg ), dXk = ( dG sv ) T^T, dsu += colsum( dXk X ), dX = dXk su
			q4::seg_colsum_prod( dG, Zg, m, I, n, gx, tab( 0, 3 ), st );
			q4::seg_colsum_prod( dU, Zu, m, I, n, gx, tab( 1, 3 ), st );
			prod( dG, tab( 0, 1 ), I, H, true, Wk[0], dXg );
			prod( dU, tab( 1, 1 ), I, H, true, Wk[1], dXu );
			q4::seg_colsum_prod( dXg, Xg, m, H, n, gx, tab( 0, 2 ), st );
			q4::seg_colsum_prod( dXu, Xg, m, H, n, gx, tab( 1, 2 ), st );
			q4::route_scale( dXg, gx, m, tab( 0, 0 ), Pg, H, dYr, false, st );
			q4::route_scale( dXu, gx, m, tab( 1, 0 ), Pg, H, dYr, true, st );
			q4::route_combine( dYr, d_inv, g.p0, Pg, nullptr, T, K, H, da, st );
		}
	}
	phase( dy ? "moe routed experts fwd+bwd" : "moe routed experts fwd" );
	if ( dy ) {
		// The routing weights' gradient (this rank's routes) into the router input
		float* dl = ws.f( TR * K );
		glm_route_backward( logits, ids, T, E, K, cfg.route_scale, dw, dl, st );
		q4::router_input_backward( dl, ids, lw.router, T, K, H, da, st );
		phase( "moe router bwd" );
	}
	CUDA_CHECK( cudaStreamSynchronize( st ) );   // the host route lists above are released with the stack
	ws.off = mark;
}

void Model::Impl::run( int L, const float* x_in, float* x_out, int T, const float* dx_out, float* dx_in, bool keep, int n_seqs ) {
	const int li = L - first;
	Layer& lw = layers[li];
	const bool bwd = dx_out != nullptr;
	ENGINE_ASSERT( T % ranks == 0 && T <= max_rows, "glm5_next training model: the rows exceed the workspace or do not split over the ranks" );
	ENGINE_ASSERT( n_seqs > 0 && T % n_seqs == 0, "glm5_next training model: the rows do not hold whole sequences" );
	seqs = n_seqs;
	keep = keep && keep_outputs;
	ws.reset( st );
	const int H = cfg.hidden, Tl = T / ranks;
	const size_t TR = static_cast<size_t>( T ), TL = static_cast<size_t>( Tl ), flat = static_cast<size_t>( kHc ) * H;
	const bool cached = bwd && lw.cached;
	mark_start();
	// the stage of this layer's expert codes, and the next one's copy: the forward goes on to the next MoE layer (after the range's last, the
	// backward of the same layer when kept, else the next pass's first), the backward back to the one before
	int sb = -1;
	if ( stage_bytes > 0 ) {
		int next = next_moe( li, bwd ? -1 : 1 );
		if ( next < 0 ) next = !bwd && keep && lw.moe && host_layer[li] ? li : next_moe( -1, 1 );
		sb = stage_start( li, next );
	}
	auto sublayer = [&]( const float* a, float* y, const float* dy, float* da ) {
		if ( lw.mla ) mla( lw, a, T, y, dy, da );
		else kda( lw, a, T, y, dy, da );
	};
	// Attention site
	const MhcCache ca = mhc_cache( ws.f( TL * kMhcCacheFloats ), Tl );
	float *aA_l = ws.f( TL * H ), *postA = ws.f( TL * kHc ), *combA = ws.f( TL * kHcComb );
	mhc_pre_forward( *blas, lw.hcA, x_in, Tl, ca, aA_l, postA, combA );
	phase( "mhc sites" );
	const float* aA = gather( aA_l, TL * H );
	float* yA = lw.cyA;
	if ( !cached ) {
		// the FFN site's inputs need yA first: the sublayer's forward alone here (in the backward its forward and backward run after the FFN site's)
		float* part = ws.f( TR * H );
		sublayer( aA, part, nullptr, nullptr );
		yA = reduce_rows( part, TL * H );
		if ( keep ) CUDA_CHECK( cudaMemcpyAsync( lw.cyA, yA, TL * H * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
	}
	float* X1 = ws.f( TL * flat );
	mhc_post_forward( x_in, postA, combA, yA, Tl, H, X1, st );
	// FFN site
	const MhcCache cf = mhc_cache( ws.f( TL * kMhcCacheFloats ), Tl );
	float *aF_l = ws.f( TL * H ), *postF = ws.f( TL * kHc ), *combF = ws.f( TL * kHcComb );
	mhc_pre_forward( *blas, lw.hcF, X1, Tl, cf, aF_l, postF, combF );
	phase( "mhc sites" );
	const float* aF = lw.moe ? gather( aF_l, TL * H ) : nullptr;
	if ( !bwd ) {
		float* yF = nullptr;
		if ( lw.moe ) {
			float* part = ws.f( TR * H );
			stage_wait( sb );
			moe( lw, sb, aF, T, part, nullptr, nullptr );
			yF = reduce_rows( part, TL * H );
			float* Sh = ws.f( TL * H );
			mlp( lw.sg, lw.su, lw.sd, aF_l, Tl, Sh, nullptr, nullptr, cfg.shared_inter );
			q4::add_scaled( yF, Sh, 1.0f, TL * H, st );
			phase( "shared expert" );
		}
		else {
			yF = ws.f( TL * H );
			mlp( lw.mg, lw.mu, lw.md, aF_l, Tl, yF, nullptr, nullptr, cfg.dense_inter );
			phase( "dense mlp" );
		}
		if ( sb >= 0 ) CUDA_CHECK( cudaEventRecord( stage[sb].done, st ) );
		if ( keep ) {
			CUDA_CHECK( cudaMemcpyAsync( lw.cyF, yF, TL * H * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
			lw.cached = true;
		}
		mhc_post_forward( X1, postF, combF, yF, Tl, H, x_out, st );
		phase( "mhc sites" );
		return;
	}
	lw.cached = false;
	// Backward. FFN site: dX1 = comb^T dx_out, dyF = sum_h post_h dx_out_h and dcombF need no yF (yA stands in); dpostF once yF is there
	float *dX1 = ws.fz( TL * flat, st ), *dyF_l = ws.f( TL * H ), *dpostF = ws.f( TL * kHc ), *dcombF = ws.f( TL * kHcComb );
	float* yF = cached ? lw.cyF : nullptr;
	mhc_post_backward( X1, postF, combF, cached ? yF : yA, dx_out, Tl, H, dX1, dyF_l, dpostF, dcombF, st );
	float* daF_l = nullptr;
	if ( lw.moe ) {
		const float* dyF = gather( dyF_l, TL * H );
		float* part = cached ? nullptr : ws.f( TR * H );
		float* dpart = ws.fz( TR * H, st );
		stage_wait( sb );
		moe( lw, sb, aF, T, part, dyF, dpart );
		if ( sb >= 0 ) CUDA_CHECK( cudaEventRecord( stage[sb].done, st ) );
		if ( !cached ) yF = reduce_rows( part, TL * H );
		daF_l = reduce_rows( dpart, TL * H );
		float* Sh = ws.f( TL * H );
		mlp( lw.sg, lw.su, lw.sd, aF_l, Tl, Sh, dyF_l, daF_l, cfg.shared_inter );
		if ( !cached ) q4::add_scaled( yF, Sh, 1.0f, TL * H, st );
		phase( "shared expert" );
	}
	else {
		daF_l = ws.fz( TL * H, st );
		float* Y = ws.f( TL * H );
		mlp( lw.mg, lw.mu, lw.md, aF_l, Tl, Y, dyF_l, daF_l, cfg.dense_inter );
		if ( !cached ) yF = Y;
		phase( "dense mlp" );
	}
	if ( !cached ) {
		float* dy_unused = ws.f( TL * H );
		q4::apply_backward( dx_out, postF, yF, Tl, H, kHc, dy_unused, dpostF, st );
	}
	float* hscr = ws.f( mhc_pre_scratch_floats( lw.hcF, Tl ) );
	mhc_pre_backward( *blas, lw.hcF, X1, Tl, cf, daF_l, dpostF, dcombF, dX1, hscr );
	phase( "mhc sites bwd" );
	// Attention site
	CUDA_CHECK( cudaMemsetAsync( dx_in, 0, TL * flat * sizeof( float ), st ) );
	float *dyA_l = ws.f( TL * H ), *dpostA = ws.f( TL * kHc ), *dcombA = ws.f( TL * kHcComb );
	mhc_post_backward( x_in, postA, combA, yA, dX1, Tl, H, dx_in, dyA_l, dpostA, dcombA, st );
	const float* dyA = gather( dyA_l, TL * H );
	{
		float* part = ws.f( TR * H );   // the sublayer's output again (not needed)
		float* dpart = ws.fz( TR * H, st );
		phase( "mhc sites bwd" );
		sublayer( aA, part, dyA, dpart );
		float* daA_l = reduce_rows( dpart, TL * H );
		mhc_pre_backward( *blas, lw.hcA, x_in, Tl, ca, daA_l, dpostA, dcombA, dx_in, hscr );
	}
	phase( "mhc sites bwd" );
}

void Model::forward( int L, const float* x_in, float* x_out, int rows, bool keep, int seqs ) {
	CUDA_CHECK( cudaSetDevice( device_ ) );
	p_->run( L, x_in, x_out, rows, nullptr, nullptr, keep, seqs );
}

void Model::backward( int L, const float* x_in, const float* dx_out, float* dx_in, int rows, int seqs ) {
	CUDA_CHECK( cudaSetDevice( device_ ) );
	p_->run( L, x_in, nullptr, rows, dx_out, dx_in, false, seqs );
}

double Model::head( const float* x, int rows, const Teacher& t, float* dx ) {
	Impl& m = *p_;
	CUDA_CHECK( cudaSetDevice( device_ ) );
	ENGINE_ASSERT( m.head, "glm5_next training model: this range has no head" );
	ENGINE_ASSERT( rows % m.ranks == 0 && rows <= m.max_rows, "glm5_next training model: the rows exceed the workspace or do not split over the ranks" );
	// vocabulary chunks of up to kLmChunk columns, narrower for many rows (a chunk's buffers hold at most kLmChunk^2 values each)
	const int NC = std::max( 1024, std::min( Impl::kLmChunk, Impl::kLmChunk / 128 * Impl::kLmChunk / std::max( rows, 1 ) * 128 ) );
	const int H = cfg_.hidden, V = cfg_.vocab, Tl = rows / m.ranks;
	const size_t TR = static_cast<size_t>( rows ), TL = static_cast<size_t>( Tl );
	// this rank's vocabulary columns (whole 128-column blocks)
	const int vb = V / 128, v0 = vb * m.rank / m.ranks * 128, v1 = vb * ( m.rank + 1 ) / m.ranks * 128;
	m.ws.reset( m.st );
	m.mark_start();
	// h = RMSNorm( mean of the streams ) on the rank's rows, then all rows for the vocabulary slice
	float *hm = m.ws.f( TL * H ), *inv = m.ws.f( TL ), *hn_l = m.ws.f( TL * H );
	hc_mean_forward( x, Tl, H, hm, m.st );
	q4::rms_forward( hm, m.fnorm, 1, false, true, Tl, H, cfg_.eps, inv, hn_l, m.st );
	const float* hn = m.gather( hn_l, TL * H );
	float *Tc = m.ws.f( static_cast<size_t>( H ) * NC ), *Zc = m.ws.f( TR * NC ), *Lc = m.ws.f( TR * NC ), *scr = m.ws.f( TR * H );
	float *mx = m.ws.f( TR ), *sx = m.ws.fz( TR, m.st ), *sl = m.ws.fz( TR * t.tk, m.st );
	{
		std::vector<float> ninf( TR, -INFINITY );
		m.up_off = 0;   // (the layer calls' uploads were read: their routed experts synchronized after them)
		m.upload( ninf.data(), TR * sizeof( float ), mx );
	}
	for ( int c0 = v0; c0 < v1; c0 += NC ) {
		const View v{ 0, H, c0, std::min( NC, v1 - c0 ) };
		m.decode( m.lm, Tc, v );
		scaled_linear_forward( *m.blas, m.sl( m.lm, Tc, v ), hn, rows, Zc, Lc, scr );
		q4::kl_gather( Lc, rows, v.n, c0, t.ids, t.tk, sl, m.st );   // the teacher's top masked out of Lc after
		q4::kl_lse_update( Lc, rows, v.n, mx, sx, m.st );
	}
	if ( m.ranks > 1 ) {
		// the ranks' slices combined: the running max and sum, the teacher's top logits (each from the rank holding its column)
		const float* m_all = m.gather( mx, TR );
		const float* s_all = m.gather( sx, TR );
		q4::kl_lse_combine( m_all, s_all, m.ranks, rows, mx, sx, m.st );
		m.sum_ranks( sl, TR * t.tk );
	}
	m.phase( "head fwd" );
	float *lt = m.ws.f( TR ), *gq = m.ws.f( TR * t.tk ), *kl = m.ws.f( TR ), *ut = m.ws.f( TR );
	q4::kl_finish( mx, sx, sl, t.p, t.tail, rows, t.tk, lt, gq, ut, kl, m.st );
	std::vector<float> h_kl( TR );
	CUDA_CHECK( cudaMemcpyAsync( h_kl.data(), kl, TR * sizeof( float ), cudaMemcpyDeviceToHost, m.st ) );
	CUDA_CHECK( cudaStreamSynchronize( m.st ) );
	double total = 0.0;
	for ( float v : h_kl ) total += v;
	if ( dx == nullptr ) return total;
	float *dhn_part = m.ws.fz( TR * H, m.st ), *dl = m.ws.f( TR * NC ), *sn = m.ws.f( TR * NC ), *sk = m.ws.f( TR * H );
	for ( int c0 = v0; c0 < v1; c0 += NC ) {
		const View v{ 0, H, c0, std::min( NC, v1 - c0 ) };
		m.decode( m.lm, Tc, v );
		scaled_linear_forward( *m.blas, m.sl( m.lm, Tc, v ), hn, rows, Zc, Lc, scr );
		q4::kl_grad( Lc, rows, v.n, c0, lt, ut, t.ids, gq, t.tk, dl, m.st );
		scaled_linear_backward( *m.blas, m.sl( m.lm, Tc, v ), hn, Zc, dl, rows, dhn_part, sn, sk, true );
	}
	float* dhn = m.reduce_rows( dhn_part, TL * H );
	float* dhm = m.ws.fz( TL * H, m.st );
	q4::rms_backward( hm, m.fnorm, 1, false, true, Tl, H, inv, dhn, dhm, m.st, m.g_fnorm );
	CUDA_CHECK( cudaMemsetAsync( dx, 0, TL * kHc * H * sizeof( float ), m.st ) );
	hc_mean_backward( dhm, Tl, H, dx, m.st );
	m.phase( "head bwd" );
	return total;
}

void Model::freeze_routing( bool on ) { p_->frozen = on; }

const std::vector<int>& Model::selection( int L ) const { return p_->last_ids.at( static_cast<size_t>( L - first_ ) ); }

void Model::set_fp16( bool on ) { p_->fp16 = on; }

void Model::set_expert_host( bool on, double device_gib ) {
	p_->expert_host = on;
	p_->expert_device_gib = device_gib;
}

void Model::set_timing( bool on ) { p_->timing = on; }

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
	for ( const ParamPool* P : { &m.shared_p, &m.expert_p, &m.norm_p } ) {
		pd += P->device_bytes();
		ph += P->host_bytes();
	}
	CUDA_CHECK( cudaSetDevice( device_ ) );
	CUDA_CHECK( cudaMemGetInfo( &fr, &tot ) );
	char out[900];
	std::snprintf( out, sizeof( out ),
		"  codes and FP16 scales (arena)   %6.2f GiB of %.2f\n"
		"  routed experts' codes (host)    %6.2f GiB, device stages 2 x %.2f GiB\n"
		"  dense copies (FP32)             %6.2f GiB\n"
		"  kept sublayer outputs           %6.2f GiB\n"
		"  state pool                      %6.2f GiB of %.2f\n"
		"  workspace (peak)                %6.2f GiB of %.2f\n"
		"  scale pools                     %6.2f GiB (shared %zu, experts %zu, norms %zu values; Adam state %s, %d-bit moments; %.2f GiB on the host)\n"
		"  device in use                   %6.2f GiB of %.2f\n",
		m.arena.offset_bytes / G, m.arena.total_bytes / G, m.host_codes.offset_bytes / G, m.stage_bytes / G, m.dense_bytes / G, m.kept_bytes / G,
		m.state.off / G, m.state.cap / G, m.ws.peak / G, m.ws.cap() / G, pd / G, m.shared_p.used, m.expert_p.used, m.norm_p.used,
		m.opt_host ? "on the host" : "on the device", m.opt_bits, ph / G, ( tot - fr ) / G, tot / G );
	return out;
}

std::vector<Tuned*> Model::tensors() {
	std::vector<Tuned*> v;
	for ( Layer& lw : p_->layers ) {
		for ( Tuned* t : { &lw.qkv, &lw.o, &lw.qa, &lw.qb, &lw.kva, &lw.mg, &lw.mu, &lw.md, &lw.sg, &lw.su, &lw.sd } ) if ( t->k > 0 ) v.push_back( t );
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

void Model::set_keep_outputs( bool on ) { p_->keep_outputs = on; }

void Model::set_optimizer( bool host, int moment_bits ) {
	ENGINE_ASSERT( moment_bits == 32 || moment_bits == 8, "glm5_next training model: Adam's moments are 32 or 8 bits" );
	p_->opt_host = host;
	p_->opt_bits = moment_bits;
}

void Model::zero_grads() {
	CUDA_CHECK( cudaSetDevice( device_ ) );
	for ( ParamPool* P : { &p_->shared_p, &p_->expert_p, &p_->norm_p } )
		if ( P->used ) CUDA_CHECK( cudaMemsetAsync( P->g, 0, P->used * sizeof( float ), p_->st ) );
}

void Model::adam( const AdamOptions& o, int step, float grad_scale ) {
	Impl& m = *p_;
	CUDA_CHECK( cudaSetDevice( device_ ) );
	m.sum_ranks( m.shared_p.g, m.shared_p.used );
	for ( ParamPool* P : { &m.shared_p, &m.expert_p } ) P->step( true, o, step, grad_scale, m.st );
	if ( m.train_norms ) {
		// log deltas like the scales: GLM's norm weights are plain (no 1 + w), from about 0.02 to 1.5, so an additive step would be a
		// different relative change on each
		m.sum_ranks( m.norm_p.g, m.norm_p.used );
		m.norm_p.step( true, o, step, grad_scale, m.st );
	}
	CUDA_CHECK( cudaStreamSynchronize( m.st ) );
}

void Model::keep_best() {
	CUDA_CHECK( cudaSetDevice( device_ ) );
	for ( ParamPool* P : { &p_->shared_p, &p_->expert_p, &p_->norm_p } ) P->keep_best( p_->st );
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
	const std::vector<float> hb = p_->norm_p.best_values();
	for ( const Dense& d : p_->dense_t ) out.push_back( { d.name, to_fp16( hb.data() + ( d.w - p_->norm_p.s ), d.n ) } );
}

// ---- the token embeddings ----

struct Embedding::Impl {
	std::vector<uint16_t> table;   // BF16 [vocab, hidden]
	int vocab = 0, hidden = 0;
};

Embedding::Embedding() : p_( std::make_unique<Impl>() ) {}

Embedding::~Embedding() = default;

bool Embedding::open( const std::string& model_dir ) {
	engine::loader::SafeTensorsRepo repo;
	if ( !repo.open( model_dir.c_str() ) ) return false;
	const char* name = "model.language_model.embed_tokens.weight";
	const pod::SafeTensorMeta* t = repo.find( name );
	if ( t == nullptr || t->ndim != 2 ) return false;
	p_->vocab = t->shape[0];
	p_->hidden = t->shape[1];
	p_->table.resize( static_cast<size_t>( p_->vocab ) * p_->hidden );
	return repo.copy_host_bf16( name, p_->table.data(), p_->table.size() );
}

int Embedding::hidden() const { return p_->hidden; }

void Embedding::rows( const int* ids, int n, float* out ) const {
	const size_t H = static_cast<size_t>( p_->hidden );
	for ( int i = 0; i < n; ++i ) {
		ENGINE_ASSERT( ids[i] >= 0 && ids[i] < p_->vocab, "glm5_next embedding: a token id outside the vocabulary" );
		const uint16_t* r = p_->table.data() + static_cast<size_t>( ids[i] ) * H;
		for ( size_t j = 0; j < H; ++j ) {
			const uint32_t b = static_cast<uint32_t>( r[j] ) << 16;
			std::memcpy( out + i * H + j, &b, 4 );
		}
	}
}

} // namespace engine::train::glm5_next
