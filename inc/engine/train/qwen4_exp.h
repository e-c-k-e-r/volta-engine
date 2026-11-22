#pragma once

// Training forms of the qwen4_exp (Flash-Next) layer operations, FP32 on the caller's stream: each forward keeps what its backward needs, each
// backward adds the input gradients into the caller's buffers (+=) unless it says otherwise. They re-implement the inference kernels'
// arithmetic in FP32 (the inference path rounds to FP16 between some steps), for the end-to-end tuning of quantized checkpoints, where a
// layer is recomputed from its input streams when its backward runs.
//
// The model at the end (Config, Tuned, Model) puts them together over an EXL3 checkpoint, for the end-to-end tuning of its quantized tensors'
// scales: a contiguous range of decoder layers on one device (and the head when the range ends the model), each layer's forward and its
// backward from the layer's input streams, the forward recomputed. Every EXL3 tensor's input and output scales are trainable as log deltas
// on the checkpoint's own (su = s0u e^au, sv = s0v e^av); everything else is frozen. Streams are [rows, hc, hidden] FP32 for one sequence
// from position 0; the PLE layer takes the n-gram key and value rows ([rows, hc * hidden], [rows, hidden]) from the caller (they depend on
// the tokens only).
//
// Tensor parallel (ranks > 1, one Model per device and host thread, an NCCL communicator between them, the ranks calling in step): every
// rank holds its block of rows of the streams (rows / ranks of them, in rank order) and runs the per-position work on them (the
// hyper-connection sites, the shared expert, the final mixer); the sublayers see all rows through all-gathers and give partial sums back
// through reduce-scatters: GDN and attention by heads, the routed experts by expert (a contiguous block per rank), the head by vocabulary.
// The PLE layer runs whole on every rank. The scales of the tensors every rank holds stay identical (their gradients are summed before
// Adam); a rank's experts are its own.
#include "engine/train/grad.h"
#include "engine/core/tensor.h"
#include "engine/modules/linear.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace engine::train::qwen4_exp {

// RMS norm over segments of n values: y = x * inv * ( one_plus ? 1 + w : w ) with inv = rsqrt( sum( x^2 ) / ( mean ? n : 1 ) + eps ); the weight
// row of segment i is i % wrows (w null: weights of 1); inv[segments] is kept
void rms_forward( const float* x, const float* w, int wrows, bool one_plus, bool mean, int segments, int n, float eps, float* inv, float* y,
	backend::stream_t stream );
// dx += the gradient through the norm at the same x and inv; with dw, the weights' gradient as well ([wrows, n], +=)
void rms_backward( const float* x, const float* w, int wrows, bool one_plus, bool mean, int segments, int n, const float* inv, const float* dy, float* dx,
	backend::stream_t stream, float* dw = nullptr );

// The hyper-connection site before each sublayer (dense weights, FP32 copies): streams x [rows, hc, hidden] in, the sublayer input mixed
// [rows, hidden] and the post gates [rows, hc] out
struct HcWeights {
	int hidden = 0, hc = 0, rank = 0;
	float eps = 1e-6f;
	const float* norm = nullptr;    // [hc, hidden], used as 1 + w
	const float* down = nullptr;    // [rank, hc * hidden]
	const float* up = nullptr;      // [hc * hidden, rank]
	const float* inject = nullptr;  // [hc, hc * hidden], null on the final mixer (no post gates)
	float* gnorm = nullptr;         // the norm weights' gradient (+=, null: not wanted)
	float* ginject = nullptr;       // the inject mix's gradient (+=, null: not wanted)
	float *gdown = nullptr, *gup = nullptr;   // the down / up mixes' gradients (+=, null: not wanted)
};
// What the backward reads (rows x the size given): inv rows * hc, normed rows * flat, zt rows * rank, t rows * rank, u rows * flat (sigmoid
// inputs of the up mix), zi rows * hc
struct HcCache {
	float *inv = nullptr, *normed = nullptr, *zt = nullptr, *t = nullptr, *u = nullptr, *zi = nullptr;
};
void hc_forward( Blas& b, const HcWeights& w, const float* x, int rows, const HcCache& c, float* mixed, float* post );
// dx += from dmixed [rows, hidden] and dpost [rows, hc] (null on the final mixer), and the norm weights' gradient into w.gnorm when given;
// scratch: 2 * rows * flat + rows * rank + rows * hc floats
void hc_backward( Blas& b, const HcWeights& w, const float* x, int rows, const HcCache& c, const float* dmixed, const float* dpost, float* dx, float* scratch );
size_t hc_scratch_floats( const HcWeights& w, int rows );

// The gated add after a sublayer, x[r, s] += post[r, s] y[r]: forward adds into x; backward: dy[r] = sum_s post[r, s] dx[r, s] (overwrites),
// dpost[r, s] = dx[r, s] . y[r] (overwrites); dx passes through unchanged
void apply_forward( float* x, const float* post, const float* y, int rows, int hidden, int hc, backend::stream_t stream );
void apply_backward( const float* dx, const float* post, const float* y, int rows, int hidden, int hc, float* dy, float* dpost, backend::stream_t stream );

// Causal convolution over rows per channel with SiLU: s[t, c] = sum_j w[c, j] x[t - ( 3 - j ) dil, c] (rows before their sequence's start
// are 0; the rows hold sequences of seq_len, 0 meaning one), y = silu( s ); s kept
void conv4_silu_forward( const float* x, const float* w, int rows, int channels, int dil, float* s, float* y, backend::stream_t stream,
	int seq_len = 0 );
void conv4_silu_backward( const float* s, const float* w, int rows, int channels, int dil, const float* dy, float* dx, backend::stream_t stream,
	int seq_len = 0 );
// Its weight gradient: dw[c, j] += sum_t dy[t, c] silu'( s[t, c] ) x[t - ( 3 - j ) dil, c] over the first `channels` columns of x, s and dy
// (rows of stride ld)
void conv4_silu_dw( const float* x, const float* s, const float* dy, int rows, int channels, int ld, int dil, float* dw, backend::stream_t stream,
	int seq_len = 0 );

// o = a * sigmoid( g ), elementwise; backward adds into da and dg
void sigmoid_gate_forward( const float* a, const float* g, float* o, size_t n, backend::stream_t stream );
void sigmoid_gate_backward( const float* a, const float* g, const float* dout, float* da, float* dg, size_t n, backend::stream_t stream );

// GDN's output norm per head of n: y = x * inv * gamma * sigmoid( z ), inv = rsqrt( mean( x^2 ) + eps ); backward adds into dx and dz, and
// with dgamma into gamma's gradient [n]
void gated_rms_forward( const float* x, const float* z, const float* gamma, int segments, int n, float eps, float* inv, float* y, backend::stream_t stream );
void gated_rms_backward( const float* x, const float* z, const float* gamma, int segments, int n, const float* inv, const float* dy, float* dx, float* dz,
	backend::stream_t stream, float* dgamma = nullptr );

// Softmax routing over the top k logits of each row (selection fixed): w[r, j] = softmax_j( logit[r, ids[r, j]] ); backward from dw [rows, k] to
// dlogit [rows, k] (overwrites)
void topk_softmax_backward( const float* w, const float* dw, int rows, int k, float* dlogit, backend::stream_t stream );
// Routing candidates: per row its slots sids [rows, k + c], the k selected (ids) then the c best of the other experts by logit, with weights
// sw (w, then 0) and wp [rows, c], the weight a candidate would take in the softmax over the selected ones and itself. The backward from the
// slots' dw (dy against each slot's expert output) to dl [rows, k + c]: the selected logits' gradients through their softmax, and a
// candidate's wp ( dw - sum_j w_j dw_j ) (its logit's gradient were it in the softmax at a vanishing weight: the swap-in direction)
void route_candidates( const float* logits, const int* ids, const float* w, int rows, int experts, int k, int c, int* sids, float* sw, float* wp,
	backend::stream_t stream );
void route_candidates_backward( const float* sw, const float* dw, const float* wp, int rows, int k, int c, float* dl, backend::stream_t stream );
// The selected slots' values as dense rows: out[r, ids[r, j]] += v[r, j] (out [rows, experts]; a row's ids distinct)
void slots_to_dense( const float* v, const int* ids, int rows, int k, int experts, float* out, backend::stream_t stream );

// GDN's gated delta rule per value head h (its key head h / ( vh / kh )), head dims 128: for each value column c,
//   u = S_c . k_t, d = beta_t ( v_t[c] - e^gate_t u ), S_c = e^gate_t S_c + d k_t, o_t[c] = scale S_c . q_t
// from a zero state at the start of each of the seqs sequences of rows / seqs. q, k [rows, kh, 128], v and o [rows, vh, 128], gate and beta
// [rows, vh]. The forward keeps the state before every chunk-th row of a sequence in ckpt; the backward recomputes each chunk from its
// checkpoint, last chunk first, and adds into dq, dk, dv, dgate, dbeta
size_t gdn_ckpt_floats( int rows, int vh, int chunk, int seqs = 1 );
size_t gdn_scratch_floats( int vh, int chunk, int seqs = 1 );
void gdn_forward( const float* q, const float* k, const float* v, const float* gate, const float* beta, int rows, int kh, int vh, float scale, int chunk,
	float* ckpt, float* o, backend::stream_t stream, int seqs = 1 );
void gdn_backward( const float* q, const float* k, const float* v, const float* gate, const float* beta, int rows, int kh, int vh, float scale, int chunk,
	const float* ckpt, const float* dout, float* dq, float* dk, float* dv, float* dgate, float* dbeta, float* scratch, backend::stream_t stream,
	int seqs = 1 );

// GDN's gates from the dense projections a = X A^T, b = X B^T [rows, vh]: gate = -exp( A_log ) softplus( a + dt_bias ), beta = sigmoid( b );
// backward to da, db (overwrites)
void gdn_gates_forward( const float* a, const float* b, const float* A_log, const float* dt_bias, int rows, int vh, float* gate, float* beta,
	backend::stream_t stream );
void gdn_gates_backward( const float* a, const float* b, const float* A_log, const float* dt_bias, int rows, int vh, const float* dgate, const float* dbeta,
	float* da, float* db, backend::stream_t stream );

// Attention pieces. RoPE on the first rot_dim channels of every head (pairs i, i + rot_dim / 2, the inference kernel's frequencies and fast
// sin / cos), the position of row r being r % seq_len (0: rows); in place, the backward rotating gradients back
void rope_forward( float* x, int rows, int heads, int head_dim, int rot_dim, float base, backend::stream_t stream, int seq_len = 0 );
void rope_backward( float* dx, int rows, int heads, int head_dim, int rot_dim, float base, backend::stream_t stream, int seq_len = 0 );
// q_proj's interleaved [rows, heads, 2 * head_dim] output into q and gate [rows, heads, head_dim], and the gradients back
void unpack_q_gate( const float* qg, int rows, int heads, int head_dim, float* q, float* gate, backend::stream_t stream );
void pack_q_gate( const float* dq, const float* dgate, int rows, int heads, int head_dim, float* dqg, backend::stream_t stream );
// QSA, the attention's key selection past its budget (engine/modules/qsa.h): a query at position p sees every earlier key while it has at
// most budget complete pools of kQsaPool tokens (p + 1 <= budget * kQsaPool + 3), else only the budget pools the indexer scores best (ties in
// pool order) and its incomplete tail. The selection is held fixed (no gradient through it), like the routing
constexpr int kQsaHeads = 4, kQsaDim = 128, kQsaPool = 4, kQsaProj = ( kQsaHeads + 1 ) * kQsaDim, kQsaBudget = 512;
constexpr int kQsaMaxWords = 1024;   // a row's selection bits: up to 32768 pools (131072 tokens)
// A sequence's selection: per position words of pool bits (bits null: every row sees every earlier key)
struct QsaSelection {
	const uint32_t* bits = nullptr;
	int words = 0;
	int budget = kQsaBudget;
};
// The indexer on a sequence's rows (rows a multiple of seq_len; positions restart at each): qk [rows, kQsaProj] = X index_qk_proj; q
// [rows, kQsaHeads, kQsaDim] normed (q_norm: the FP16 weight) and roped at the row's position, raw_k [rows, kQsaDim]; the pools' keys
// [seqs, seq_len / kQsaPool, kQsaDim] from raw_k (k_norm); the inference kernels' FP16 roundings throughout
void qsa_queries( const float* qk, int rows, int seq_len, const void* q_norm, int rot_dim, float theta, float eps, float* q, float* raw_k,
	backend::stream_t stream );
void qsa_pools( const float* raw_k, int seqs, int seq_len, const void* k_norm, int rot_dim, float theta, float eps, float* pools, backend::stream_t stream );
// One sequence's selection (q, pools: its rows' and its pools'), qblock rows of scores at a time (scratch: qsa_scratch_floats); bits [seq_len,
// words] with words = ( seq_len / kQsaPool + 31 ) / 32
size_t qsa_scratch_floats( int seq_len, int qblock );
void qsa_select( const float* q, const float* pools, int seq_len, int budget, int qblock, float* scratch, uint32_t* bits, backend::stream_t stream );

// Causal softmax attention with grouped key / value heads over one sequence (q [rows, hq, d], k and v [rows, hk, d], q head h reads kv head
// h / ( hq / hk )): o = softmax( scale q k^T, causal, the keys sel lets each row see ) v, qblock query rows at a time (0: all). One
// qblock x rows score buffer (s) is reused per head; the backward recomputes the scores and adds into dq, dk, dv (dp: a second such buffer)
void attention_forward( Blas& b, const float* q, const float* k, const float* v, int rows, int hq, int hk, int d, float scale, float* s, float* o,
	const QsaSelection& sel = {}, int qblock = 0 );
void attention_backward( Blas& b, const float* q, const float* k, const float* v, int rows, int hq, int hk, int d, float scale, const float* dout, float* dq,
	float* dk, float* dv, float* s, float* dp, const QsaSelection& sel = {}, int qblock = 0 );

// Softmax routing: the k largest logits of each row [rows, experts] (ties to the lowest expert index), ids [rows, k] best first and weights
// w [rows, k] = softmax over those k logits (experts <= 1024)
void router_topk( const float* logits, int rows, int experts, int k, int* ids, float* w, backend::stream_t stream );

// PLE's gate per (row, stream) from g = < norm( key ), norm( x ) > / sqrt( hidden ): sg = sigmoid( sign( g ) sqrt( max( |g|, 1e-6 ) ) ); the
// backward turns dsg into dg (overwrites)
void ple_gate_forward( const float* g, float* sg, size_t n, backend::stream_t stream );
void ple_gate_backward( const float* g, const float* dsg, float* dg, size_t n, backend::stream_t stream );
// out[i] = scale * a_i . b_i over segments of n (overwrites); backward db_i += scale * dout[i] * a_i (a frozen)
void seg_dot( const float* a, const float* b, int segments, int n, float scale, float* out, backend::stream_t stream );
void seg_dot_backward_b( const float* a, const float* dout, int segments, int n, float scale, float* db, backend::stream_t stream );

// Column blocks of row-major matrices: dst[r, dst_col + c] (=, or += with add) src[r, src_col + c] for c < cols
void copy_cols( const float* src, int src_ld, int src_col, float* dst, int dst_ld, int dst_col, int rows, int cols, bool add, backend::stream_t stream );
// FP16 to FP32
void half_to_float( const void* h, float* f, size_t n, backend::stream_t stream );

// The router's input gradient: dX[r] += sum_j dlogit[r, j] R[ids[r, j]] (R [experts, hidden])
void router_input_backward( const float* dlogit, const int* ids, const float* R, int rows, int k, int hidden, float* dX, backend::stream_t stream );

// Routed experts in groups: a group's routes are padded rows, m per expert (the expert of row p is gexp[p / m], its padding rows zero),
// prow / pslot giving each padded row's source row and route slot (r * k + j; -1 on padding). Per-expert vectors come from tables of
// pointers indexed by expert. cols are multiples of 4.
// dst[p] = src[prow[p]] (times w[pslot[p]] when w is given), padding rows zero
void route_gather( const float* src, const int* prow, const int* pslot, const float* w, int P, int cols, float* dst, backend::stream_t stream );
// out[p] (= or +=) in[p] * tab[gexp[p / m]]
void route_scale( const float* in, const int* gexp, int m, const float* const* tab, int P, int cols, float* out, bool add, backend::stream_t stream );
// The same as FP16 rows at a power-of-two scale each (as engine::train::rows_to_half: rinv[p] the inverse), for the tensor-core products
void route_scale_half( const float* in, const int* gexp, int m, const float* const* tab, int P, int cols, void* out, float* rinv, backend::stream_t stream );
// G = Zg sv_gate, U = Zu sv_up, A = silu( G ) U (per expert)
void route_swiglu( const float* Zg, const float* Zu, const int* gexp, int m, const float* const* svg, const float* const* svu, int P, int cols, float* G, float* U,
	float* A, backend::stream_t stream );
// From dAk (the down projection's input gradient before su_down): dG, dU through the SwiGLU (overwrite)
void route_swiglu_backward( const float* dAk, const float* G, const float* U, const int* gexp, int m, const float* const* sud, int P, int cols, float* dG,
	float* dU, backend::stream_t stream );
// out[gexp[i]][c] += sum over expert i's m rows of A[., c] B[., c], for the group's experts
void seg_colsum_prod( const float* A, const float* B, int m, int cols, int experts, const int* gexp, float* const* out, backend::stream_t stream );
// out[r] += sum_j w[r k + j] Y[inv[r k + j] - p0] over the slots whose padded row (inv) lies in the group's [p0, p0 + P) (w null: 1)
void route_combine( const float* Y, const int* inv, int p0, int P, const float* w, int rows, int k, int cols, float* out, backend::stream_t stream );
// out[pslot[p]] = dY[prow[p]] . Y[p] for the group's routes (padding skipped)
void route_dots_pad( const float* dY, const int* prow, const int* pslot, const float* Y, int P, int cols, float* out, backend::stream_t stream );

// The KL head over vocabulary chunks (logits [rows, nc] of columns c0 .. c0 + nc), KL( teacher || model ) over the teacher's top tk tokens and
// the mass outside them, computed without the cancellations of 1 - sum( q ) and q - p (arXiv 2610.31642): kl_gather takes the logits at the
// teacher's ids that fall in the chunk (into sl [rows, tk]) and then masks them to -inf in place, so that kl_lse_update, run after it, keeps the
// running max m and sum s of the model's logits outside the teacher's top (m starts at -inf, s at 0)
void kl_lse_update( const float* logits, int rows, int nc, float* m, float* s, backend::stream_t stream );
void kl_gather( float* logits, int rows, int nc, int c0, const int* ids, int tk, float* sl, backend::stream_t stream );
// After the last chunk: lt = log of the model's mass outside the top (unnormalized: m + log s), kl [rows], the top entries' gradients gq
// [rows, tk] = q - p (as p expm1( log q - log p )) and the tail's ut = q_tail - p_tail, made to sum to zero
void kl_finish( const float* m, const float* s, const float* sl, const float* p, const float* tail, int rows, int tk, float* lt, float* gq, float* ut, float* kl,
	backend::stream_t stream );
// The running max and sum of ranks vocabulary slices ([ranks, rows] each, as gathered) into one (m, s [rows])
void kl_lse_combine( const float* m_all, const float* s_all, int ranks, int rows, float* m, float* s, backend::stream_t stream );
// dlogit of a chunk (overwrites dl): exp( logit - lt ) ut for the columns outside the top, gq at the teacher's ids
void kl_grad( const float* logits, int rows, int nc, int c0, const float* lt, const float* ut, const int* ids, const float* gq, int tk, float* dl,
	backend::stream_t stream );
// Routing with the selection given (ids [rows, k]): w = softmax over the logits at those ids
void router_given( const float* logits, const int* ids, int rows, int experts, int k, float* w, backend::stream_t stream );
// out = x * y * s, elementwise
void mul_scaled( const float* x, const float* y, float s, float* out, size_t n, backend::stream_t stream );

// x = FP16( x ), elementwise
void round_to_half( float* x, size_t n, backend::stream_t stream );

// out = ( in diag( col_scale ) ) H per 128 columns of each row (col_scale null: ones; H the EXL3 cores' 128-point Hadamard, unit norm; in
// place allowed): rotates a product's input or output gradient into the frame of the trellis tiles
void had_rows( const float* in, const float* col_scale, int rows, int cols, float* out, backend::stream_t stream );
// The tile gains' 8-bit codes from their values, g = ( 896 + c ) / 1024 (nearest, clamped), and back
void gain_codes( const float* g, uint8_t* codes, size_t n, backend::stream_t stream );
void gain_values( const uint8_t* codes, float* g, size_t n, backend::stream_t stream );
// x += a y, elementwise
void add_scaled( float* x, const float* y, float a, size_t n, backend::stream_t stream );

// ---- the model ----

struct Config {
	int hidden = 2560, hc = 4, rank = 320, layers = 48, ple_layer = 1, full_interval = 4;
	int heads = 24, kv_heads = 2, head_dim = 256, rot_dim = 64;
	float rope_theta = 10000000.0f;
	int lin_k = 16, lin_v = 48, lin_d = 128;
	int experts = 512, top_k = 10, inter = 640, shared_inter = 640, vocab = 248320;
	float eps = 1e-6f;
	bool read( const std::string& model_dir );   // config.json (text_config)
	bool full_attention( int layer ) const { return ( layer + 1 ) % full_interval == 0; }
};

// One EXL3 tensor with trainable scales (device buffers of k or n floats; the optimizer's state of them is the model's)
struct Tuned {
	std::string name;
	pod::LinearWeight w;                       // the packed codes, as the engine loads them
	int k = 0, n = 0;
	float *su = nullptr, *sv = nullptr;        // current scales
	float *gu = nullptr, *gv = nullptr;        // gradients with respect to su / sv, accumulated
	float *tg = nullptr, *gtg = nullptr;       // with trained tile gains: their values (k / 16 x n / 16; w.w4 holds their codes) and gradients
};

// A trainable dense tensor (the weights of a norm: the hyper-connection sites', attention's q / k norms, GDN's gated norm, PLE's), as an
// additive delta on the checkpoint's values: its name, current values and gradient (device buffers of n floats)
struct Dense {
	std::string name;
	float* w = nullptr;
	float* g = nullptr;
	int n = 0;
};

// The KL head's teacher per row: its top tk token ids and probabilities and the mass outside them
struct Teacher {
	const int* ids = nullptr;     // [rows, tk] (device)
	const float* p = nullptr;     // [rows, tk]
	const float* tail = nullptr;  // [rows]
	int tk = 0;
};

class Model {
public:
	Model();
	~Model();
	Model( const Model& ) = delete;
	Model& operator=( const Model& ) = delete;

	// Layers [first, last) of the checkpoint in dir on device, with workspace for up to max_rows rows; the head when last is the layer count.
	// ranks > 1: rank of a tensor-parallel group (comm: its ncclComm_t)
	bool load( const std::string& dir, int first, int last, int device, int max_rows, int rank = 0, int ranks = 1, void* comm = nullptr );
	const Config& config() const { return cfg_; }
	int first() const { return first_; }
	int last() const { return last_; }
	bool has_head() const { return last_ == cfg_.layers; }
	int device() const { return device_; }
	int rank() const { return rank_; }
	int ranks() const { return ranks_; }
	backend::stream_t stream() const;

	// Layer L forward: x_out = layer( x_in ) over rows of seqs sequences of rows / seqs, each from position 0 (x_in, x_out: the rank's rows;
	// ple_k, ple_v: all rows). keep: the sublayer outputs are kept for the layer's next backward, which then skips their recompute
	void forward( int L, const float* x_in, float* x_out, int rows, const float* ple_k, const float* ple_v, bool keep = false, int seqs = 1 );
	// Its backward: dx_in (overwritten) from dx_out, the forward recomputed from x_in; scale gradients accumulate in every tensor's gu / gv
	void backward( int L, const float* x_in, const float* dx_out, float* dx_in, int rows, const float* ple_k, const float* ple_v, int seqs = 1 );
	// The head on the final streams x (the rank's rows): KL( teacher || model ) summed over all rows (returned; the teacher's rows: all);
	// with dx, its gradient (overwritten) and the lm_head scale gradients
	double head( const float* x, int rows, const Teacher& t, float* dx );

	// Routing held fixed at the selection of each layer's last call (for gradient checks across a perturbation; the weights still follow
	// the logits); off: every call selects
	void freeze_routing( bool on );
	// Layer L's routed experts of its last call: rows x top_k ids (host)
	const std::vector<int>& selection( int L ) const;

	// The trainable tensors' products on the tensor cores (the default: FP16 cores, activations and gradients rounded to FP16 per row at a
	// power-of-two scale, FP32 accumulation; everything else FP32), or off: FP32 products
	void set_fp16( bool on );
	// The attention's QSA selection past its budget (on by default: as the engine attends), or off: dense attention at every length
	void set_qsa( bool on );

	// Phase timing (synchronizes the stream at every phase boundary): on, the seconds per phase accumulate until reported and reset
	void set_timing( bool on );
	std::string timing_report( bool reset = true );
	// The device memory by component: the arena's codes, the dense copies, the kept outputs, the workspace's peak so far, the scale pools
	std::string memory_report() const;

	// The tensors this rank holds (with ranks > 1: the shared ones and its own experts), and the trainable dense tensors: the norm weights and,
	// with set_train_dense, the small dense ones (every rank holds them all)
	std::vector<Tuned*> tensors();
	std::vector<Dense*> dense_tensors();
	// The norm weights trained along with the scales (on by default); off: their gradients are computed but not applied
	void set_train_norms( bool on );
	// The small dense tensors trained too (before load; off by default): the hyper-connection sites' inject mixes, GDN's in_proj_a / b, A_log,
	// dt_bias and convolution, the shared expert's gate and PLE's convolution, as additive deltas like the norms (dense_tensors lists them), at
	// lr_scale times the learning rate (a row of them feeds a dot product: Adam's per-entry step adds up over its width)
	void set_train_dense( bool on, float lr_scale = 0.1f );
	// The routers' weights trained too (before load; off by default), with the small dense tensors' learning rate (dense_tensors lists them)
	void set_train_router( bool on );
	// Room for every layer's kept sublayer outputs (before load; on by default): off, a kept forward keeps nothing and the backward
	// recomputes the sublayers (the room is two rows of hidden per row and layer: long sequences)
	void set_keep_outputs( bool on );
	// Routing candidates (0: off): in the backward each row's next n experts by logit after its selected ones run too as zero-weight routes (the
	// output unchanged), their logits given the gradient of being swapped in (route_candidates_backward), so that the router and its input
	// learn selections the KL prefers; ~n / top_k more routed-expert work in the backward
	void set_route_candidates( int n );
	// The hyper-connection sites' down and up mixes trained too (before load; off by default), as additive deltas at lr_scale times the
	// learning rate (dense_tensors lists them; every rank holds them all: rank rows of hc * hidden each way, ~636M values on Flash-Next)
	void set_train_hc( bool on, float lr_scale = 0.1f );
	// The tile gains of every trained tensor that has them trained too (before load; off by default): FP32 values in the scale pools at
	// lr_scale times the learning rate, the codes the forward decodes set from them after each step (nearest, straight through: the gradient
	// is the value's), their gradient from the product's input and output gradient rotated into the tiles' frame (one more product per
	// tensor in the backward)
	void set_train_gains( bool on, float lr_scale = 1.0f );
	// Code updates of layer L's routed experts on this rank (the V step of PV-tuning). set_accumulate( L ) (after load; -1 off) has each backward
	// add, per expert, the gradient G with respect to its projections' tiles R (in their frame: ( X diag( su ) H )^T ( dY diag( sv ) H ), the
	// weight diag( su ) H R H diag( sv )) and its inputs' Hessian sums over the rows routed to it, on the device (~48 MB an expert). The step
	// D is M^-1 G in the projection's quantization metric M (its input Hessian in the tiles' frame with its input scales folded in) with the LDL
	// feedback, or G itself without (each tile encoded on its own, in the plain metric). reencode_prepare( feedback ) keeps the loaded codes and
	// returns sqrt( sum |D|^2 / sum |R|^2 ) over every rank's experts (in M with the feedback, else plain). reencode( eta, feedback ) then
	// encodes every projection anew toward R - eta D, its scales and tile gains held (engine/quant/exl3_quantize.h, exl3_reencode), in place of
	// its codes, and returns the mean proxy error of those encodes; lin and quad (summed over the ranks) receive sum < G, Q - R > (the summed
	// loss's change to first order) and sum ( Q - R )^T M ( Q - R ). restore_codes() puts the loaded codes back
	void set_accumulate( int L );
	double reencode_prepare( bool feedback = false );
	double reencode( float eta, bool feedback = false, double* lin = nullptr, double* quad = nullptr );
	void restore_codes();
	// Where Adam's state lives (before load): on the device (the default) or with host, in pinned host memory, streamed through the device in
	// parts for each step (the same kernels, the same result; the device keeps the current values, their gradients and one part); its moments
	// FP32 (32) or 8-bit block-wise (8, engine/train/adam8.h)
	void set_optimizer( bool host, int moment_bits );
	void zero_grads();
	// Adam on every tensor's log deltas and the norm weights' deltas from the accumulated gradients (times grad_scale; the shared tensors' and
	// the norms' summed over the ranks first), then the current values
	void adam( const AdamOptions& o, int step, float grad_scale );
	// The current values kept as the best (on the device), and the kept ones at the end as FP16: per tensor name its (input, output) scales,
	// per norm weight name its values
	void keep_best();
	void best_scales( std::vector<std::pair<std::string, std::pair<std::vector<uint16_t>, std::vector<uint16_t>>>>& out );
	void best_dense( std::vector<std::pair<std::string, std::vector<uint16_t>>>& out );
	// with trained tile gains, per tensor name the kept gains' codes
	void best_gains( std::vector<std::pair<std::string, std::vector<uint8_t>>>& out );

private:
	struct Impl;
	std::unique_ptr<Impl> p_;
	Config cfg_;
	int first_ = 0, last_ = 0, device_ = 0, rank_ = 0, ranks_ = 1;
};

// A sequence's inputs from its token ids (positions from 0), as the engine makes them: the token embeddings (FP16 rows) and the PLE layer's
// n-gram key and value rows (FP16 rows of the n-gram table's lookups through the layer's key / value projections). The n-gram lookups run on
// the host (ple_rows, rows part, part + parts, .. of a shared FP16 buffer [n, ple_dim()], so that ranks split them), the rest on the device
class Inputs {
public:
	Inputs();
	~Inputs();
	bool open( const std::string& model_dir, int device, backend::stream_t stream );
	int ple_dim() const;
	// out [n, hidden] (FP32 of the FP16 rows)
	void embed( const int* ids, int n, float* out );
	void ple_rows( const int* ids, int n, int part, int parts, uint16_t* emb ) const;
	// key [n, hc hidden], value [n, hidden] from emb [n, ple_dim()] (host, FP16)
	void ple_project( const uint16_t* emb, int n, float* key, float* value );

private:
	struct Impl;
	std::unique_ptr<Impl> p_;
};

} // namespace engine::train::qwen4_exp
