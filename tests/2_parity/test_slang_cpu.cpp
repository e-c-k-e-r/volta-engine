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
}

struct NormParams { half* src; half* dst; half* weight; int hidden_size; int has_weight; float eps; float bias; };
struct HeadNormParams { half* qk; half* weight; int num_heads; int head_dim; int has_weight; float eps; float bias; };
struct AddParams { half* hidden; half* residual; int n; };
struct SwigluParams { half* gate; half* up; half* out_; int n; float limit; };
struct SwigluFusedParams { half* gate_up; half* out_; int inter_dim; int total; float limit; };
struct EmbedParams { uint16_t* table; int* token_ids; half* out_; int hidden_size; };
struct RopeParams { half* q; half* k; int* pos; int num_q_heads; int num_kv_heads; int head_dim; int rot_dim; int interleaved; float freq_base; };

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
static void dispatch( void ( *fn )( ComputeVaryingInput*, void*, void* ), void* params, uint32_t gx, uint32_t gy = 1 ) {
	ComputeVaryingInput vi = {};
	vi.startGroupID = { 0, 0, 0 };
	vi.endGroupID = { gx, gy, 1 };
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
	std::printf( pass ? "[+] Slang CPU target parity PASS\n" : "[-] Slang CPU target parity FAIL\n" );
	return pass ? 0 : 1;
}
