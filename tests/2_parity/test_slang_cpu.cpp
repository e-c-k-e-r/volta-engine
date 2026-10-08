#include <slang-cpp-prelude.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

// Tier 2 parity of the Slang CPU (host) target: the generated C++ kernels run on the host against FP64 references.
// Struct layouts mirror the Params structs in src/shaders/*.slang.

extern "C" {
void rms_norm( ComputeVaryingInput*, void*, void* );
void rms_norm_heads( ComputeVaryingInput*, void*, void* );
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
struct SwigluParams { half* gate; half* up; half* out_; int n; float limit; };
struct SwigluFusedParams { half* gate_up; half* out_; int inter_dim; int total; float limit; };
struct EmbedParams { uint16_t* table; int* token_ids; half* out_; int hidden_size; };
struct RopeParams { half* q; half* k; int* pos; int num_q_heads; int num_kv_heads; int head_dim; int rot_dim; int interleaved; float freq_base; };

struct SumPlanesParams { half* planes; half* out_; int n; int groups; };
struct AccumParams { half* accum; half* expert_out; float weight; int n; };
struct ArgmaxParams { half* logits; int* out_idx; float* out_val; int numel; int batch_size; int index_offset; int row_stride; int has_val; };
struct RouterParams { float* partials; half* bias; int* topk_idx; half* topk_w; int num_experts; int top_k; int k_slices; int max_slices; float scale; int has_bias; };

struct L2Params { half* qkv; int num_k_heads; int head_dim; int scale_q; int64_t row_stride; };
struct GatesParams { half* norm_hidden; half* in_proj_a; half* in_proj_b; half* A_log; half* dt_bias; float* gate_out; float* beta_out; int num_v_heads; int hidden_size; int has_a_log; int has_dt_bias; };
struct ConvRowsParams { half* qkv; float* history; half* weights; int* slots; float* snap; int* snap_on; int channels; int rows; int snap_rows; int has_weights; int64_t slot_stride; int64_t snap_stride; };
struct ConvTiledParams { half* src; half* dst; float* history; half* weights; int* slots; int channels; int rows; int tile_rows; int has_weights; int64_t slot_stride; };
struct RecurrentParams { float* state; half* qkv; float* gate; float* beta; half* output; int* slots; float* snap; int* snap_on; int h_k; int h_v; int rows; int snap_rows; float scale; int64_t slot_stride; int64_t snap_stride; };
struct KvStoreParams { half* data; half* src; int* slots; int* positions; int64_t slot_stride; int layer_idx; int count; int total_elems; int max_seq; };
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

static uint32_t g_state = 777u;
static float frand( float scale ) {
	g_state = g_state * 1664525u + 1013904223u;
	return ( static_cast<int>( g_state >> 16 ) % 2001 - 1000 ) / 1000.0f * scale;
}
static std::vector<half> make( size_t n, float scale ) {
	std::vector<half> v( n );
	for ( auto& x : v ) x = static_cast<half>( frand( scale ) );
	return v;
}
static void dispatch( void ( *fn )( ComputeVaryingInput*, void*, void* ), void* params, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1 ) {
	ComputeVaryingInput vi = {};
	vi.startGroupID = { 0, 0, 0 };
	vi.endGroupID = { gx, gy, gz };
	fn( &vi, params, nullptr );
}
static bool check( const char* name, const std::vector<double>& ref, const std::vector<half>& got ) {
	double dot = 0, na = 0, nb = 0, maxd = 0, maxabs = 0;
	for ( size_t i = 0; i < ref.size(); ++i ) {
		const double b = static_cast<double>( static_cast<float>( got[i] ) );
		dot += ref[i] * b; na += ref[i] * ref[i]; nb += b * b;
		maxd = std::fmax( maxd, std::fabs( ref[i] - b ) );
		maxabs = std::fmax( maxabs, std::fabs( ref[i] ) );
	}
	const double cosv = dot / ( std::sqrt( na ) * std::sqrt( nb ) + 1e-30 );
	const bool ok = cosv >= 0.9999 && maxd <= 0.005 * maxabs + 1e-3;
	std::printf( "  %-22s cos %.7f max|d| %.3g (max %.3g)  %s\n", name, cosv, maxd, maxabs, ok ? "PASS" : "FAIL" );
	return ok;
}
static double h2d( half h ) { return static_cast<double>( static_cast<float>( h ) ); }
static std::vector<double> tod( const std::vector<float>& v ) { return std::vector<double>( v.begin(), v.end() ); }
static bool checkd( const char* name, const std::vector<double>& ref, const std::vector<double>& got ) {
	double dot = 0, na = 0, nb = 0, maxd = 0, maxabs = 0;
	for ( size_t i = 0; i < ref.size(); ++i ) {
		dot += ref[i] * got[i]; na += ref[i] * ref[i]; nb += got[i] * got[i];
		maxd = std::fmax( maxd, std::fabs( ref[i] - got[i] ) );
		maxabs = std::fmax( maxabs, std::fabs( ref[i] ) );
	}
	const double cosv = dot / ( std::sqrt( na ) * std::sqrt( nb ) + 1e-30 );
	const bool ok = ref.size() == got.size() && cosv >= 0.9999 && maxd <= 0.005 * maxabs + 1e-3;
	std::printf( "  %-22s cos %.7f max|d| %.3g (max %.3g)  %s\n", name, cosv, maxd, maxabs, ok ? "PASS" : "FAIL" );
	return ok;
}

int main() {
	bool pass = true;
	std::printf( "=== Slang CPU target vs FP64 references ===\n" );
	{ // rms_norm
		const int rows = 4, hs = 1000;
		auto x = make( rows * hs, 1.0f ), w = make( hs, 0.5f );
		std::vector<half> out( rows * hs );
		NormParams p{ x.data(), out.data(), w.data(), hs, 1, 1e-6f, 1.0f };
		dispatch( rms_norm, &p, rows );
		std::vector<double> ref( rows * hs );
		for ( int r = 0; r < rows; ++r ) {
			double sq = 0; for ( int i = 0; i < hs; ++i ) sq += h2d( x[r * hs + i] ) * h2d( x[r * hs + i] );
			const double inv = 1.0 / std::sqrt( sq / hs + 1e-6 );
			for ( int i = 0; i < hs; ++i ) ref[r * hs + i] = h2d( x[r * hs + i] ) * ( h2d( w[i] ) + 1.0 ) * inv;
		}
		pass = check( "rms_norm", ref, out ) && pass;
	}
	{ // rms_norm_heads (no weight)
		const int rows = 2, heads = 6, hd = 128;
		auto qk = make( rows * heads * hd, 1.0f );
		auto in = qk;
		HeadNormParams p{ qk.data(), nullptr, heads, hd, 0, 1e-6f, 0.0f };
		dispatch( rms_norm_heads, &p, heads, rows );
		std::vector<double> ref( rows * heads * hd );
		for ( int g = 0; g < rows * heads; ++g ) {
			double sq = 0; for ( int i = 0; i < hd; ++i ) sq += h2d( in[g * hd + i] ) * h2d( in[g * hd + i] );
			const double inv = 1.0 / std::sqrt( sq / hd + 1e-6 );
			for ( int i = 0; i < hd; ++i ) ref[g * hd + i] = h2d( in[g * hd + i] ) * inv;
		}
		pass = check( "rms_norm_heads", ref, qk ) && pass;
	}
	{ // add_residual
		const int n = 3000;
		auto h = make( n, 1.0f ), r = make( n, 1.0f );
		auto h0 = h;
		AddParams p{ h.data(), r.data(), n };
		dispatch( add_residual, &p, ( n + 255 ) / 256 );
		std::vector<double> ref( n );
		for ( int i = 0; i < n; ++i ) ref[i] = h2d( h0[i] ) + h2d( r[i] );
		pass = check( "add_residual", ref, h ) && pass;
	}
	for ( float limit : { 0.0f, 3.0f } ) { // swiglu
		const int n = 5000;
		auto g = make( n, 4.0f ), u = make( n, 4.0f );
		std::vector<half> out( n );
		SwigluParams p{ g.data(), u.data(), out.data(), n, limit };
		dispatch( swiglu, &p, ( n + 255 ) / 256 );
		std::vector<double> ref( n );
		for ( int i = 0; i < n; ++i ) {
			double gv = h2d( g[i] ), uv = h2d( u[i] );
			if ( limit > 0 ) { gv = std::fmin( gv, limit ); uv = std::fmax( -limit, std::fmin( uv, limit ) ); }
			ref[i] = gv / ( 1.0 + std::exp( -gv ) ) * uv;
		}
		pass = check( limit == 0.0f ? "swiglu" : "swiglu limit 3", ref, out ) && pass;
	}
	{ // swiglu_fused
		const int rows = 3, inter = 700;
		auto gu = make( rows * 2 * inter, 4.0f );
		std::vector<half> out( rows * inter );
		SwigluFusedParams p{ gu.data(), out.data(), inter, rows * inter, 0.0f };
		dispatch( swiglu_fused, &p, ( rows * inter + 255 ) / 256 );
		std::vector<double> ref( rows * inter );
		for ( int m = 0; m < rows; ++m ) for ( int c = 0; c < inter; ++c ) {
			const double gv = h2d( gu[m * 2 * inter + c] ), uv = h2d( gu[m * 2 * inter + inter + c] );
			ref[m * inter + c] = gv / ( 1.0 + std::exp( -gv ) ) * uv;
		}
		pass = check( "swiglu_fused", ref, out ) && pass;
	}
	{ // embed_tokens
		const int vocab = 64, hs = 300, count = 4;
		std::vector<uint16_t> table( vocab * hs );
		std::vector<float> tf( vocab * hs );
		for ( size_t i = 0; i < table.size(); ++i ) { tf[i] = frand( 2.0f ); uint32_t b; std::memcpy( &b, &tf[i], 4 ); table[i] = b >> 16; }
		int ids[count] = { 3, 63, 0, 17 };
		std::vector<half> out( count * hs );
		EmbedParams p{ table.data(), ids, out.data(), hs };
		dispatch( embed_tokens, &p, ( hs + 255 ) / 256, count );
		std::vector<double> ref( count * hs );
		for ( int r = 0; r < count; ++r ) for ( int c = 0; c < hs; ++c ) {
			uint32_t b = static_cast<uint32_t>( table[ids[r] * hs + c] ) << 16; float f; std::memcpy( &f, &b, 4 ); ref[r * hs + c] = f;
		}
		pass = check( "embed_tokens", ref, out ) && pass;
	}
	for ( int il = 0; il < 2; ++il ) { // rope
		const int rows = 3, qh = 4, kh = 2, hd = 64, rot = 32;
		auto q = make( rows * qh * hd, 1.0f ), k = make( rows * kh * hd, 1.0f );
		auto q0 = q;
		int pos[rows] = { 1000, 1001, 1002 };
		RopeParams p{ q.data(), k.data(), pos, qh, kh, hd, rot, il, 1000000.0f };
		dispatch( rope, &p, rows );
		std::vector<double> ref( q0.size() );
		for ( int r = 0; r < rows; ++r ) for ( int h = 0; h < qh; ++h ) for ( int c = 0; c < hd; ++c ) {
			const size_t o = ( static_cast<size_t>( r ) * qh + h ) * hd;
			double v = h2d( q0[o + c] );
			if ( c < rot ) {
				const int i = il ? c >> 1 : ( c < rot / 2 ? c : c - rot / 2 );
				const double th = pos[r] * std::pow( 1000000.0, -2.0 * i / rot );
				const int p0 = il ? 2 * i : i, p1 = il ? 2 * i + 1 : i + rot / 2;
				const double a = h2d( q0[o + p0] ), b = h2d( q0[o + p1] );
				v = ( c == p0 ) ? a * std::cos( th ) - b * std::sin( th ) : a * std::sin( th ) + b * std::cos( th );
			}
			ref[o + c] = v;
		}
		pass = check( il ? "rope interleaved q" : "rope contiguous q", ref, q ) && pass;
	}
	{ // moe_sum_planes + moe_accumulate
		const int n = 1000, groups = 4;
		auto planes = make( n * groups, 1.0f );
		std::vector<half> out( n );
		SumPlanesParams p{ planes.data(), out.data(), n, groups };
		dispatch( moe_sum_planes, &p, ( n + 255 ) / 256 );
		std::vector<double> ref( n );
		for ( int i = 0; i < n; ++i ) { double a = 0; for ( int g = 0; g < groups; ++g ) a += h2d( planes[g * n + i] ); ref[i] = a; }
		pass = check( "moe_sum_planes", ref, out ) && pass;
		auto acc = make( n, 1.0f ), eo = make( n, 1.0f );
		auto acc0 = acc;
		AccumParams q{ acc.data(), eo.data(), 0.37f, n };
		dispatch( moe_accumulate, &q, ( n + 255 ) / 256 );
		for ( int i = 0; i < n; ++i ) ref[i] = h2d( acc0[i] ) + h2d( eo[i] ) * 0.37;
		pass = check( "moe_accumulate", ref, acc ) && pass;
	}
	{ // batched_argmax
		const int batch = 3, numel = 900, stride = 1024, offset = 50;
		auto logits = make( batch * stride, 3.0f );
		int idx[batch]; float val[batch];
		ArgmaxParams p{ logits.data(), idx, val, numel, batch, offset, stride, 1 };
		dispatch( batched_argmax, &p, batch );
		bool ok = true;
		for ( int r = 0; r < batch; ++r ) {
			int bi = -1; float bv = -1e30f;
			for ( int i = 0; i < numel; ++i ) { const float v = static_cast<float>( logits[r * stride + i] ); if ( v > bv ) { bv = v; bi = i; } }
			ok = ok && idx[r] == bi + offset && val[r] == bv;
		}
		std::printf( "  %-22s %s\n", "batched_argmax", ok ? "PASS" : "FAIL" );
		pass = ok && pass;
	}
	{ // router_select
		const int rows = 2, E = 288, K = 8, slices = 32, maxs = 32;
		std::vector<float> partials( static_cast<size_t>( rows ) * maxs * 288 );
		for ( auto& v : partials ) v = frand( 0.25f );
		auto bias = make( E, 0.1f );
		int idx[rows * K];
		std::vector<half> w( rows * K );
		RouterParams p{ partials.data(), bias.data(), idx, w.data(), E, K, slices, maxs, 2.5f, 1 };
		dispatch( router_select, &p, rows );
		bool ok = true;
		std::vector<double> ref( rows * K );
		for ( int r = 0; r < rows; ++r ) {
			std::vector<float> raw( E ), sel( E );
			for ( int e = 0; e < E; ++e ) {
				float logit = 0; for ( int ks = 0; ks < slices; ++ks ) logit += partials[( static_cast<size_t>( r ) * maxs + ks ) * 288 + e];
				raw[e] = 1.0f / ( 1.0f + std::exp( -logit ) );
				sel[e] = raw[e] + static_cast<float>( bias[e] );
			}
			std::vector<int> chosen; double total = 0;
			for ( int k = 0; k < K; ++k ) {
				int bi = -1; for ( int e = 0; e < E; ++e ) { bool taken = false; for ( int c : chosen ) taken = taken || c == e; if ( !taken && ( bi < 0 || sel[e] > sel[bi] ) ) bi = e; }
				chosen.push_back( bi ); total += raw[bi];
			}
			for ( int k = 0; k < K; ++k ) { ok = ok && idx[r * K + k] == chosen[k]; ref[r * K + k] = raw[chosen[k]] * 2.5 / ( total + 1e-9 ); }
		}
		std::printf( "  %-22s expert ids  %s\n", "router_select", ok ? "PASS" : "FAIL" );
		pass = ok && check( "router_select weights", ref, w ) && pass;
	}
	{ // l2_norm_heads and gated_rmsnorm against FP64 references
		const int rows = 2, kh = 3, hd = 96, stride = 2 * kh * hd + 64;
		auto qkv = make( rows * stride, 1.0f );
		auto in = qkv;
		L2Params p{ qkv.data(), kh, hd, 1, stride };
		dispatch( l2_norm_heads, &p, 2 * kh, rows );
		std::vector<double> ref( rows * stride );
		for ( int r = 0; r < rows; ++r ) for ( int i = 0; i < stride; ++i ) ref[r * stride + i] = h2d( in[r * stride + i] );
		for ( int r = 0; r < rows; ++r ) for ( int b = 0; b < 2 * kh; ++b ) {
			double sq = 0; for ( int i = 0; i < hd; ++i ) { const double v = h2d( in[r * stride + b * hd + i] ); sq += v * v; }
			double inv = 1.0 / std::sqrt( sq + 1e-6 );
			if ( b < kh ) inv *= 1.0 / std::sqrt( static_cast<double>( hd ) );
			for ( int i = 0; i < hd; ++i ) ref[r * stride + b * hd + i] = h2d( in[r * stride + b * hd + i] ) * inv;
		}
		pass = check( "l2_norm_heads", ref, qkv ) && pass;
		const int vh = 5;
		for ( int sig = 0; sig < 2; ++sig ) {
			auto o = make( rows * vh * hd, 1.0f ), z = make( rows * vh * hd, 2.0f ), gw = make( hd, 0.5f );
			auto o0 = o;
			GatedNormParams q{ o.data(), z.data(), gw.data(), vh, hd, sig, 1, 1 };
			dispatch( gated_rmsnorm, &q, vh, rows );
			std::vector<double> r2( o.size() );
			for ( int g = 0; g < rows * vh; ++g ) {
				double sq = 0; for ( int i = 0; i < hd; ++i ) { const double v = h2d( o0[g * hd + i] ); sq += v * v; }
				const double inv = 1.0 / std::sqrt( sq / hd + 1e-6 );
				for ( int i = 0; i < hd; ++i ) {
					const double zv = h2d( z[g * hd + i] );
					const double gate = sig ? 1.0 / ( 1.0 + std::exp( -zv ) ) : zv / ( 1.0 + std::exp( -zv ) );
					r2[g * hd + i] = h2d( o0[g * hd + i] ) * inv * h2d( gw[i] ) * gate;
				}
			}
			pass = check( sig ? "gated_rmsnorm sigmoid" : "gated_rmsnorm silu", r2, o ) && pass;
		}
	}
	{ // GDN gate projection (softplus gate, sigmoid beta)
		const int rows = 2, vh = 3, hid = 96;
		auto x = make( rows * hid, 1.0f ), wa = make( vh * hid, 0.3f ), wb = make( vh * hid, 0.3f ), alog = make( vh, 0.5f ), dt = make( vh, 0.5f );
		std::vector<float> gate( rows * vh ), beta( rows * vh );
		GatesParams p{ x.data(), wa.data(), wb.data(), alog.data(), dt.data(), gate.data(), beta.data(), vh, hid, 1, 1 };
		dispatch( project_gates, &p, vh, rows );
		std::vector<double> rg, rb;
		for ( int r = 0; r < rows; ++r ) for ( int h = 0; h < vh; ++h ) {
			double da = 0, db = 0;
			for ( int i = 0; i < hid; ++i ) {
				da += h2d( wa[h * hid + i] ) * h2d( x[r * hid + i] );
				db += h2d( wb[h * hid + i] ) * h2d( x[r * hid + i] );
			}
			rb.push_back( 1.0 / ( 1.0 + std::exp( -db ) ) );
			rg.push_back( -std::exp( h2d( alog[h] ) ) * std::log1p( std::exp( da + h2d( dt[h] ) ) ) );
		}
		pass = checkd( "project_gates gate", rg, tod( gate ) ) && pass;
		pass = checkd( "project_gates beta", rb, tod( beta ) ) && pass;
	}
	{ // multi-row causal conv (two slots, snapshots) against a double reference
		const int ch = 40, rows = 5;
		const int64_t ss = ch * 3 + 7;
		auto qkv = make( rows * ch, 1.0f ), w = make( ch * 4, 0.5f );
		const auto in = qkv;
		std::vector<float> hist( 2 * ss ), snap( rows * ch * 3, -9.0f );
		for ( auto& v : hist ) v = frand( 1.0f );
		std::vector<int> slots = { 0, 0, 1, 1, 1 };
		int on = 1;
		const auto h0 = hist;
		ConvRowsParams p{ qkv.data(), hist.data(), w.data(), slots.data(), snap.data(), &on, ch, rows, rows, 1, ss, ch * 3 };
		dispatch( causal_conv1d_silu_rows, &p, ( ch + 255 ) / 256 );
		std::vector<double> ro( rows * ch ), rs( rows * ch * 3 ), rh;
		auto h = h0;
		for ( int m = 0; m < rows; ++m ) for ( int c = 0; c < ch; ++c ) {
			float* hs = &h[slots[m] * ss + c * 3];
			const double x = h2d( in[m * ch + c] );
			const double sum = hs[0] * h2d( w[c * 4] ) + hs[1] * h2d( w[c * 4 + 1] ) + hs[2] * h2d( w[c * 4 + 2] ) + x * h2d( w[c * 4 + 3] );
			ro[m * ch + c] = sum / ( 1.0 + std::exp( -sum ) );
			hs[0] = hs[1]; hs[1] = hs[2]; hs[2] = static_cast<float>( x );
			for ( int k = 0; k < 3; ++k ) rs[m * ch * 3 + c * 3 + k] = hs[k];
		}
		pass = check( "causal_conv1d rows", ro, qkv ) && pass;
		pass = checkd( "causal_conv1d snap", rs, tod( snap ) ) && pass;
		pass = checkd( "causal_conv1d history", tod( h ), tod( hist ) ) && pass;
	}
	{ // prefill tiled conv + history update
		const int ch = 33, rows = 40, tile = 16;
		auto src = make( rows * ch, 1.0f ), w = make( ch * 4, 0.5f ), dst = make( rows * ch, 0.0f );
		std::vector<float> hist( ch * 3 );
		for ( auto& v : hist ) v = frand( 1.0f );
		std::vector<int> slots = { 0 };
		ConvTiledParams p{ src.data(), dst.data(), hist.data(), w.data(), slots.data(), ch, rows, tile, 1, ch * 3 };
		const auto h0 = hist;
		dispatch( causal_conv1d_silu_tiled, &p, ( ch + 127 ) / 128, ( rows + tile - 1 ) / tile );
		std::vector<double> ro( rows * ch ), rh( ch * 3 );
		for ( int c = 0; c < ch; ++c ) {
			auto xs = [&]( int t ) { return t >= 0 ? h2d( src[t * ch + c] ) : static_cast<double>( h0[c * 3 + 3 + t] ); };
			for ( int t = 0; t < rows; ++t ) {
				double sum = 0;
				for ( int k = 0; k < 4; ++k ) sum += h2d( w[c * 4 + k] ) * xs( t - 3 + k );
				ro[t * ch + c] = sum / ( 1.0 + std::exp( -sum ) );
			}
			for ( int k = 0; k < 3; ++k ) rh[c * 3 + k] = xs( rows - 3 + k );
		}
		pass = check( "causal_conv1d tiled", ro, dst ) && pass;
		dispatch( causal_conv1d_history, &p, ( ch + 127 ) / 128 );
		pass = checkd( "causal_conv1d hist upd", rh, tod( hist ) ) && pass;
	}
	{ // delta-rule recurrence, 3 rows over 2 slots, with snapshots
		const int hk = 2, hv = 4, rows = 3, snap_rows = 2;
		const int64_t ss = int64_t( hv ) * 128 * 128;
		const int stride = ( 2 * hk + hv ) * 128;
		auto qkv = make( rows * stride, 0.5f );
		std::vector<float> state( 2 * ss ), gate( rows * hv ), beta( rows * hv ), snap( snap_rows * ss, -9.0f );
		for ( auto& v : state ) v = frand( 0.1f );
		for ( auto& v : gate ) v = -std::fabs( frand( 0.5f ) );
		for ( auto& v : beta ) v = std::fabs( frand( 1.0f ) );
		std::vector<half> out( rows * hv * 128 );
		std::vector<int> slots = { 0, 0, 1 };
		int on = 1;
		const auto s0 = state;
		RecurrentParams p{ state.data(), qkv.data(), gate.data(), beta.data(), out.data(), slots.data(), snap.data(), &on, hk, hv, rows, snap_rows, 1.0f / std::sqrt( 128.0f ), ss, ss };
		dispatch( recurrent_step_rows, &p, hv, 1, 128 );
		std::vector<double> ro( rows * hv * 128 ), rsnap( snap_rows * ss ), rstate = tod( s0 );
		for ( int m = 0; m < rows; ++m ) for ( int head = 0; head < hv; ++head ) for ( int col = 0; col < 128; ++col ) {
			double* s = &rstate[slots[m] * ss + ( int64_t( head ) * 128 + col ) * 128];
			const int qh = head / ( hv / hk );
			const half* q = &qkv[m * stride + qh * 128];
			const half* k = &qkv[m * stride + hk * 128 + qh * 128];
			const half* v = &qkv[m * stride + 2 * hk * 128 + head * 128];
			const double g = std::exp( static_cast<double>( gate[m * hv + head] ) );
			double kv = 0;
			for ( int i = 0; i < 128; ++i ) kv += s[i] * h2d( k[i] );
			const double delta = ( h2d( v[col] ) - g * kv ) * beta[m * hv + head];
			double attn = 0;
			for ( int i = 0; i < 128; ++i ) {
				s[i] = g * s[i] + h2d( k[i] ) * delta;
				attn += s[i] * h2d( q[i] );
			}
			ro[( m * hv + head ) * 128 + col] = attn / std::sqrt( 128.0 );
			if ( m < snap_rows ) for ( int i = 0; i < 128; ++i ) rsnap[m * ss + ( int64_t( head ) * 128 + col ) * 128 + i] = s[i];
		}
		pass = check( "recurrent_step_rows out", ro, out ) && pass;
		pass = checkd( "recurrent_step_rows state", rstate, tod( state ) ) && pass;
		pass = checkd( "recurrent_step_rows snap", rsnap, tod( snap ) ) && pass;
	}
	{ // FP16 paged KV store
		const int count = 3, total = 70, max_seq = 8, layer = 1;
		const int64_t ss = 2 * max_seq * total + 5;
		auto src = make( count * total, 1.0f );
		std::vector<half> data( 2 * ss, static_cast<half>( 0.0f ) ), ref = data;
		std::vector<int> slots = { 0, 1, 1 }, pos = { 2, 0, 1 };
		KvStoreParams p{ data.data(), src.data(), slots.data(), pos.data(), ss, layer, count, total, max_seq };
		dispatch( kv_store_f16, &p, ( count * total + 127 ) / 128 );
		for ( int m = 0; m < count; ++m ) for ( int e = 0; e < total; ++e ) ref[slots[m] * ss + ( layer * max_seq + pos[m] ) * total + e] = src[m * total + e];
		std::vector<double> rd, gd;
		for ( size_t i = 0; i < ref.size(); ++i ) { rd.push_back( h2d( ref[i] ) ); }
		pass = check( "kv_store_f16", rd, data ) && pass;
	}
	{ // mHC: stream init, collapse mix (partials + finalize with Sinkhorn), update, hyper-head mean
		const int D = 256, rows = 2, row_len = 4 * D, chunk_cols = 256, chunk_cols_c = 128, iters = 3;
		const int nch = row_len / chunk_cols;
		auto in = make( rows * D, 1.0f );
		std::vector<float> streams( rows * row_len );
		InitStreamsParams ip{ in.data(), streams.data(), D };
		dispatch( init_streams, &ip, ( D + 255 ) / 256, rows );
		std::vector<double> rs;
		for ( int r = 0; r < rows; ++r ) for ( int h = 0; h < 4; ++h ) for ( int d = 0; d < D; ++d ) rs.push_back( h2d( in[r * D + d] ) );
		pass = checkd( "init_streams", rs, tod( streams ) ) && pass;

		for ( auto& v : streams ) v = frand( 1.0f );
		std::vector<float> fn( 24 * row_len ), base( 24 ), scale = { 0.7f, 0.9f, 1.1f }, partials( rows * nch * 25 ), post( rows * 4 ), comb( rows * 16 ), inv_rms( rows ), logits( rows * 24 ), pre( rows * 4 );
		for ( auto& v : fn ) v = frand( 0.05f );
		for ( auto& v : base ) v = frand( 0.5f );
		PartialsParams pp{ streams.data(), fn.data(), partials.data(), row_len, chunk_cols, nch };
		dispatch( hc_mix_partials, &pp, nch, rows );
		std::vector<half> collapsed( rows * D );
		const float rms_eps = 1e-5f, hc_eps = 1e-6f;
		FinalizeParams fp{ streams.data(), partials.data(), base.data(), scale.data(), post.data(), comb.data(), collapsed.data(), inv_rms.data(), logits.data(), pre.data(),
			D, nch, chunk_cols_c, iters, rms_eps, hc_eps };
		dispatch( hc_mix_finalize, &fp, ( D + chunk_cols_c - 1 ) / chunk_cols_c, rows );
		std::vector<double> rpost, rcomb, rcol, rinv, rlog, rpre;
		for ( int r = 0; r < rows; ++r ) {
			double mix[25] = {};
			for ( int c = 0; c < row_len; ++c ) {
				const double sv = streams[r * row_len + c];
				for ( int j = 0; j < 24; ++j ) mix[j] += sv * fn[j * row_len + c];
				mix[24] += sv * sv;
			}
			const double rmr = 1.0 / std::sqrt( mix[24] / row_len + rms_eps );
			auto sig = []( double x ) { return 1.0 / ( 1.0 + std::exp( -x ) ); };
			double pr[4];
			for ( int h = 0; h < 4; ++h ) { pr[h] = sig( mix[h] * rmr * scale[0] + base[h] ) + hc_eps; rpre.push_back( pr[h] ); }
			for ( int h = 0; h < 4; ++h ) rpost.push_back( 2.0 * sig( mix[4 + h] * rmr * scale[1] + base[4 + h] ) );
			double v[16];
			for ( int i = 0; i < 16; ++i ) v[i] = mix[8 + i] * rmr * scale[2] + base[8 + i];
			for ( int a = 0; a < 4; ++a ) {
				double m = v[a * 4], s = 0;
				for ( int b = 1; b < 4; ++b ) m = std::fmax( m, v[a * 4 + b] );
				for ( int b = 0; b < 4; ++b ) { v[a * 4 + b] = std::exp( v[a * 4 + b] - m ); s += v[a * 4 + b]; }
				for ( int b = 0; b < 4; ++b ) v[a * 4 + b] = v[a * 4 + b] / s + hc_eps;
			}
			auto colnorm = [&]() { for ( int b = 0; b < 4; ++b ) { double cs = 0; for ( int a = 0; a < 4; ++a ) cs += v[a * 4 + b]; for ( int a = 0; a < 4; ++a ) v[a * 4 + b] /= cs + hc_eps; } };
			auto rownorm = [&]() { for ( int a = 0; a < 4; ++a ) { double rsum = 0; for ( int b = 0; b < 4; ++b ) rsum += v[a * 4 + b]; for ( int b = 0; b < 4; ++b ) v[a * 4 + b] /= rsum + hc_eps; } };
			colnorm();
			for ( int it = 0; it < iters - 1; ++it ) { rownorm(); colnorm(); }
			for ( int i = 0; i < 16; ++i ) rcomb.push_back( v[i] );
			for ( int d = 0; d < D; ++d ) {
				double o = 0;
				for ( int h = 0; h < 4; ++h ) o += pr[h] * streams[r * row_len + h * D + d];
				rcol.push_back( o );
			}
			rinv.push_back( rmr );
			for ( int k = 0; k < 24; ++k ) rlog.push_back( mix[k] );
		}
		pass = checkd( "hc_mix post", rpost, tod( post ) ) && pass;
		pass = checkd( "hc_mix comb (sinkhorn)", rcomb, tod( comb ) ) && pass;
		pass = checkd( "hc_mix pre", rpre, tod( pre ) ) && pass;
		pass = checkd( "hc_mix inv_rms", rinv, tod( inv_rms ) ) && pass;
		pass = checkd( "hc_mix logits", rlog, tod( logits ) ) && pass;
		pass = check( "hc_mix collapsed", rcol, collapsed ) && pass;

		auto y = make( rows * D, 1.0f );
		for ( int has_comb = 0; has_comb < 2; ++has_comb ) {
			std::vector<float> xo( streams.size() );
			ApplyParams ap{ streams.data(), xo.data(), y.data(), post.data(), comb.data(), D, has_comb };
			dispatch( hc_apply, &ap, ( D + 255 ) / 256, rows );
			std::vector<double> ra;
			for ( int r = 0; r < rows; ++r ) for ( int h = 0; h < 4; ++h ) for ( int d = 0; d < D; ++d ) {
				double o = post[r * 4 + h] * h2d( y[r * D + d] );
				if ( has_comb ) for ( int g = 0; g < 4; ++g ) o += comb[( r * 4 + g ) * 4 + h] * streams[r * row_len + g * D + d];
				else o += streams[r * row_len + h * D + d];
				ra.push_back( o );
			}
			pass = checkd( has_comb ? "hc_apply comb" : "hc_apply residual", ra, tod( xo ) ) && pass;
		}
		std::vector<half> mean( rows * D );
		HeadMeanParams hp{ streams.data(), mean.data(), D };
		dispatch( hyperhead_mean, &hp, ( D + 255 ) / 256, rows );
		std::vector<double> rm;
		for ( int r = 0; r < rows; ++r ) for ( int d = 0; d < D; ++d ) {
			double o = 0;
			for ( int h = 0; h < 4; ++h ) o += streams[r * row_len + h * D + d];
			rm.push_back( o * 0.25 );
		}
		pass = check( "hyperhead_mean", rm, mean ) && pass;
	}
	{ // DSA indexer prep (LayerNorm into the plane ring) and pooling
		const int rows = 3, layer = 1, max_seq = 2048;
		const int64_t pss = 2 * 1024 * 256, qss = 2 * ( max_seq / 4 ) * 128;
		auto proj = make( rows * 288, 1.0f ), lw = make( 128, 1.0f ), lb = make( 128, 0.5f );
		auto plane = make( 2 * pss, 1.0f );
		std::vector<half> wout( rows * 32 );
		std::vector<int> slots = { 0, 1, 0 }, pos = { 5, 1030, 7 };
		const auto plane0 = plane;
		PrepParams pp{ proj.data(), lw.data(), lb.data(), plane.data(), wout.data(), slots.data(), pos.data(), layer, pss };
		dispatch( dsa_prep, &pp, rows );
		auto rplane = plane0;
		std::vector<double> rw;
		for ( int r = 0; r < rows; ++r ) {
			double mean = 0, var = 0;
			for ( int d = 0; d < 128; ++d ) mean += h2d( proj[r * 288 + d] );
			mean /= 128;
			for ( int d = 0; d < 128; ++d ) var += ( h2d( proj[r * 288 + d] ) - mean ) * ( h2d( proj[r * 288 + d] ) - mean );
			var /= 128;
			half* dst = &rplane[slots[r] * pss + ( int64_t( layer ) * 1024 + ( pos[r] & 1023 ) ) * 256];
			for ( int d = 0; d < 128; ++d ) {
				dst[d] = static_cast<half>( static_cast<float>( ( h2d( proj[r * 288 + d] ) - mean ) / std::sqrt( var + 1e-6 ) * h2d( lw[d] ) + h2d( lb[d] ) ) );
				dst[128 + d] = proj[r * 288 + 160 + d];
			}
			for ( int h = 0; h < 32; ++h ) rw.push_back( h2d( proj[r * 288 + 128 + h] ) );
		}
		std::vector<double> rp;
		for ( auto v : rplane ) rp.push_back( h2d( v ) );
		pass = check( "dsa_prep plane", rp, plane ) && pass;
		pass = check( "dsa_prep head weights", rw, wout ) && pass;

		auto ape = make( 4 * 128, 0.5f );
		std::vector<half> pool( 2 * qss, static_cast<half>( 0.0f ) );
		std::vector<int> pos2 = { 7, 1027, 6 };
		PoolParams qp{ plane.data(), pool.data(), ape.data(), slots.data(), pos2.data(), layer, max_seq, pss, qss };
		dispatch( dsa_pool_update, &qp, rows );
		std::vector<double> rpool( 2 * qss, 0.0 );
		for ( int r = 0; r < 2; ++r ) for ( int d = 0; d < 128; ++d ) {
			double k[4], g[4], mx = -1e30, sum = 0, acc = 0;
			for ( int j = 0; j < 4; ++j ) {
				const half* src = &plane[slots[r] * pss + ( int64_t( layer ) * 1024 + ( ( pos2[r] - 3 + j ) & 1023 ) ) * 256];
				k[j] = h2d( src[d] );
				g[j] = h2d( src[128 + d] ) + h2d( ape[j * 128 + d] );
				mx = std::fmax( mx, g[j] );
			}
			for ( int j = 0; j < 4; ++j ) { const double e = std::exp( g[j] - mx ); sum += e; acc += e * k[j]; }
			rpool[slots[r] * qss + ( int64_t( layer ) * ( max_seq / 4 ) + pos2[r] / 4 ) * 128 + d] = acc / sum;
		}
		pass = check( "dsa_pool_update", rpool, pool ) && pass;
	}
	{ // speculative tap ring store and stream reduction
		const int m = 3, tap = 2, H = 70, cap = 8;
		auto hidden = make( m * H, 1.0f );
		std::vector<half> taps( cap * 5 * H, static_cast<half>( 0.0f ) );
		std::vector<int> slots = { 0, 1, 0 }, pos = { 3, 9, -1 };
		int owner = 0;
		TapStoreParams tp{ taps.data(), hidden.data(), slots.data(), pos.data(), &owner, m, tap, H, cap };
		dispatch( dflash_store_taps, &tp, ( H + 255 ) / 256, m );
		std::vector<double> rt( taps.size(), 0.0 );
		for ( int i = 0; i < H; ++i ) rt[( 3 % cap ) * 5 * H + tap * H + i] = h2d( hidden[i] );
		pass = check( "dflash_store_taps", rt, taps ) && pass;
		std::vector<float> streams( m * 4 * H );
		for ( auto& v : streams ) v = frand( 1.0f );
		for ( int mode = 0; mode < 3; ++mode ) {
			std::vector<half> out( m * H );
			ReduceStreamsParams rp{ streams.data(), out.data(), H, mode };
			dispatch( reduce_streams, &rp, ( H + 255 ) / 256, m );
			std::vector<double> rr;
			for ( int r = 0; r < m; ++r ) for ( int i = 0; i < H; ++i ) {
				const float* in = &streams[r * 4 * H];
				rr.push_back( mode == 0 ? ( in[i] + in[H + i] + in[2 * H + i] + in[3 * H + i] ) * 0.25 : in[( mode - 1 ) * H + i] );
			}
			pass = check( mode == 0 ? "reduce_streams mean" : mode == 1 ? "reduce_streams s0" : "reduce_streams s1", rr, out ) && pass;
		}
	}
	std::printf( pass ? "[+] Slang CPU target parity PASS\n" : "[-] Slang CPU target parity FAIL\n" );
	return pass ? 0 : 1;
}
