#include "engine/modules/attention.h"
#include "engine/modules/mla.h"
#include "engine/backend/runtime.h"
#include "ref_attn_cuda.h"
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

// Tier 2 parity of the portable reference attention kernels (src/kernels/slang/ref_attention.slang, ref_latent.slang).
//  (a) host target (generated C++, no GPU) against an FP64 CPU reference;
//  (b) with --gpu: the Slang-generated CUDA kernels against Attention::forward_decode and MLALatent::attend (and FP64).
// The Slang C++ prelude is not included here (it clashes with the CUDA half types); the host entry points are declared with
// a layout-compatible copy of ComputeVaryingInput and parameter structs that mirror the Slang ones.

using namespace engine;

namespace {

struct U3 { uint32_t x, y, z; };
struct VaryingInput { U3 start, end; };

extern "C" {
void ref_attn_decode_cpu( VaryingInput*, void*, void* );
void ref_latent_decode_cpu( VaryingInput*, void*, void* );
}

struct RefAttnParams {
	__half *q, *k, *v, *gate, *out_;
	int *slots, *pos;
	int64_t slot_stride;
	int layer_idx, num_q_heads, num_kv_heads, head_dim, max_seq, has_gate;
	float scale;
};
struct RefLatentParams {
	__half *q_lat, *cache, *out_;
	int *slots, *pos;
	int64_t slot_stride;
	int layer, heads, max_seq;
	float scale;
};

uint32_t g_state = 2024u;
float frand( float scale ) {
	g_state = g_state * 1664525u + 1013904223u;
	return ( static_cast<int>( g_state >> 16 ) % 2001 - 1000 ) / 1000.0f * scale;
}
std::vector<__half> make( size_t n, float scale ) {
	std::vector<__half> v( n );
	for ( auto& x : v ) x = __float2half( frand( scale ) );
	return v;
}
double d( __half h ) { return static_cast<double>( __half2float( h ) ); }

bool check( const char* name, const std::vector<double>& ref, const std::vector<__half>& got ) {
	double dot = 0, na = 0, nb = 0, maxd = 0, maxabs = 0;
	for ( size_t i = 0; i < ref.size(); ++i ) {
		const double b = d( got[i] );
		dot += ref[i] * b; na += ref[i] * ref[i]; nb += b * b;
		maxd = std::fmax( maxd, std::fabs( ref[i] - b ) );
		maxabs = std::fmax( maxabs, std::fabs( ref[i] ) );
	}
	const double cosv = dot / ( std::sqrt( na ) * std::sqrt( nb ) + 1e-30 );
	const bool ok = ref.size() == got.size() && std::isfinite( cosv ) && cosv >= 0.9999 && maxd <= 0.005 * maxabs + 1e-3;
	std::printf( "  %-40s cos %.7f max|d| %.3g (max %.3g)  %s\n", name, cosv, maxd, maxabs, ok ? "PASS" : "FAIL" );
	return ok;
}
// Compare two FP16 outputs (the engine kernel as the reference)
bool check_h( const char* name, const std::vector<__half>& a, const std::vector<__half>& b ) {
	std::vector<double> ref( a.size() );
	for ( size_t i = 0; i < a.size(); ++i ) ref[i] = d( a[i] );
	return check( name, ref, b );
}

// ---- standard attention ----
struct AttnCase {
	const char* name;
	int hq, hkv, hd, max_seq, layer, nslot;
	std::vector<int> slots, pos;
	bool gate;
};

struct AttnData {
	std::vector<__half> q, k, v, gate;
	size_t slot_stride;
	int rows;
};

AttnData make_attn( const AttnCase& c ) {
	AttnData a;
	a.rows = static_cast<int>( c.pos.size() );
	a.slot_stride = static_cast<size_t>( c.layer + 2 ) * c.max_seq * c.hkv * c.hd;   // layers beyond `layer` exist too
	a.q = make( static_cast<size_t>( a.rows ) * c.hq * c.hd, 1.0f );
	a.k = make( a.slot_stride * c.nslot, 1.0f );
	a.v = make( a.slot_stride * c.nslot, 1.0f );
	if ( c.gate ) a.gate = make( a.q.size(), 2.0f );
	return a;
}

std::vector<double> attn_fp64( const AttnCase& c, const AttnData& a ) {
	std::vector<double> ref( a.q.size() );
	const int per_kv = std::max( c.hq / c.hkv, 1 );
	const int row_elems = c.hkv * c.hd;
	const double scale = 1.0 / std::sqrt( static_cast<double>( c.hd ) );
	for ( int m = 0; m < a.rows; ++m ) {
		const int total = std::min( c.pos[m] + 1, c.max_seq );
		for ( int h = 0; h < c.hq; ++h ) {
			const size_t qo = ( static_cast<size_t>( m ) * c.hq + h ) * c.hd;
			const size_t base = static_cast<size_t>( c.slots[m] ) * a.slot_stride + static_cast<size_t>( c.layer ) * c.max_seq * row_elems + ( h / per_kv ) * c.hd;
			std::vector<double> s( total );
			double mx = -1e300;
			for ( int t = 0; t < total; ++t ) {
				double dot = 0;
				for ( int e = 0; e < c.hd; ++e ) dot += d( a.q[qo + e] ) * d( a.k[base + static_cast<size_t>( t ) * row_elems + e] );
				s[t] = dot * scale;
				mx = std::fmax( mx, s[t] );
			}
			double l = 0;
			for ( int t = 0; t < total; ++t ) { s[t] = std::exp( s[t] - mx ); l += s[t]; }
			for ( int e = 0; e < c.hd; ++e ) {
				double o = 0;
				for ( int t = 0; t < total; ++t ) o += s[t] * d( a.v[base + static_cast<size_t>( t ) * row_elems + e] );
				o /= l;
				if ( c.gate ) o *= 1.0 / ( 1.0 + std::exp( -d( a.gate[qo + e] ) ) );
				ref[qo + e] = o;
			}
		}
	}
	return ref;
}

bool run_attn_host( AttnCase c ) {
	AttnData a = make_attn( c );
	std::vector<__half> out( a.q.size() );
	RefAttnParams p{ a.q.data(), a.k.data(), a.v.data(), c.gate ? a.gate.data() : nullptr, out.data(), c.slots.data(), c.pos.data(),
		static_cast<int64_t>( a.slot_stride ), c.layer, c.hq, c.hkv, c.hd, c.max_seq, c.gate ? 1 : 0, 1.0f / std::sqrt( static_cast<float>( c.hd ) ) };
	VaryingInput vi = { { 0, 0, 0 }, { static_cast<uint32_t>( c.hq ), static_cast<uint32_t>( a.rows ), 1 } };
	ref_attn_decode_cpu( &vi, &p, nullptr );
	char name[96];
	std::snprintf( name, sizeof( name ), "host attn %s", c.name );
	return check( name, attn_fp64( c, a ), out );
}

template <typename T>
T* upload( const std::vector<T>& h ) {
	void* p = backend::device_malloc( std::max<size_t>( h.size() * sizeof( T ), 16 ) );
	backend::memcpy_sync( p, h.data(), h.size() * sizeof( T ), backend::Copy::H2D );
	return static_cast<T*>( p );
}
template <typename T>
std::vector<T> download( const void* p, size_t n ) {
	std::vector<T> h( n );
	backend::memcpy_sync( h.data(), p, n * sizeof( T ), backend::Copy::D2H );
	return h;
}

bool run_attn_gpu( AttnCase c, backend::stream_t s ) {
	AttnData a = make_attn( c );
	__half *dq = upload( a.q ), *dk = upload( a.k ), *dv = upload( a.v ), *dg = c.gate ? upload( a.gate ) : nullptr;
	__half *o_ref = upload( std::vector<__half>( a.q.size() ) ), *o_eng = upload( std::vector<__half>( a.q.size() ) );
	int *dslots = upload( c.slots ), *dpos = upload( c.pos );
	pod::BatchRows br = pod::BatchRows::batched( a.rows, c.slots.data(), c.pos.data() );
	pod::BatchRows* d_rows = static_cast<pod::BatchRows*>( backend::device_malloc( sizeof( br ) ) );
	backend::memcpy_sync( d_rows, &br, sizeof( br ), backend::Copy::H2D );

	ref_attn_launch_decode( dq, dk, dv, dg, o_ref, dslots, dpos, static_cast<long long>( a.slot_stride ), c.layer, a.rows, c.hq, c.hkv, c.hd, c.max_seq, s );
	pod::TensorView tq = pod::make_tensor_2d( dq, pod::DType::FP16, a.rows, c.hq * c.hd ), tout = pod::make_tensor_2d( o_eng, pod::DType::FP16, a.rows, c.hq * c.hd );
	pod::TensorView tg = {};
	if ( dg ) tg = pod::make_tensor_2d( dg, pod::DType::FP16, a.rows, c.hq * c.hd );
	modules::Attention::forward_decode( pod::kv_ref_f16( dk, dv ), tq, tg, tout, a.slot_stride, c.layer, d_rows, a.rows, c.hq, c.hkv, c.hd, c.max_seq, s );
	backend::stream_sync( s );
	const auto got = download<__half>( o_ref, a.q.size() ), eng = download<__half>( o_eng, a.q.size() );
	char name[96];
	std::snprintf( name, sizeof( name ), "gpu attn %s vs engine", c.name );
	bool ok = check_h( name, eng, got );
	std::snprintf( name, sizeof( name ), "gpu attn %s vs fp64", c.name );
	ok = check( name, attn_fp64( c, a ), got ) && ok;
	for ( void* p : { (void*) dq, (void*) dk, (void*) dv, (void*) dg, (void*) o_ref, (void*) o_eng, (void*) dslots, (void*) dpos, (void*) d_rows } ) {
		if ( p ) backend::device_free( p );
	}
	return ok;
}

// ---- latent attention ----
struct LatCase {
	const char* name;
	int heads, max_seq, layer, nslot;
	std::vector<int> slots, pos;
};
constexpr int LORA = 512;

struct LatData {
	std::vector<__half> q, cache;
	size_t slot_stride;
	int rows;
};

LatData make_lat( const LatCase& c ) {
	LatData a;
	a.rows = static_cast<int>( c.pos.size() );
	a.slot_stride = static_cast<size_t>( c.layer + 2 ) * c.max_seq * LORA;
	a.q = make( static_cast<size_t>( a.rows ) * c.heads * LORA, 0.5f );
	a.cache = make( a.slot_stride * c.nslot, 1.7f );
	return a;
}

std::vector<double> lat_fp64( const LatCase& c, const LatData& a ) {
	std::vector<double> ref( a.q.size() );
	const double scale = 1.0 / 16.0;   // 1/sqrt(256)
	for ( int m = 0; m < a.rows; ++m ) {
		const int total = std::min( c.pos[m] + 1, c.max_seq );
		const size_t base = static_cast<size_t>( c.slots[m] ) * a.slot_stride + static_cast<size_t>( c.layer ) * c.max_seq * LORA;
		for ( int h = 0; h < c.heads; ++h ) {
			const size_t qo = ( static_cast<size_t>( m ) * c.heads + h ) * LORA;
			std::vector<double> s( total );
			double mx = -1e300;
			for ( int t = 0; t < total; ++t ) {
				double dot = 0;
				for ( int e = 0; e < LORA; ++e ) dot += d( a.q[qo + e] ) * d( a.cache[base + static_cast<size_t>( t ) * LORA + e] );
				s[t] = dot * scale;
				mx = std::fmax( mx, s[t] );
			}
			double l = 0;
			for ( int t = 0; t < total; ++t ) { s[t] = std::exp( s[t] - mx ); l += s[t]; }
			for ( int e = 0; e < LORA; ++e ) {
				double o = 0;
				for ( int t = 0; t < total; ++t ) o += s[t] * d( a.cache[base + static_cast<size_t>( t ) * LORA + e] );
				ref[qo + e] = o / l;
			}
		}
	}
	return ref;
}

bool run_lat_host( LatCase c ) {
	LatData a = make_lat( c );
	std::vector<__half> out( a.q.size() );
	RefLatentParams p{ a.q.data(), a.cache.data(), out.data(), c.slots.data(), c.pos.data(), static_cast<int64_t>( a.slot_stride ), c.layer, c.heads, c.max_seq, 0.0625f };
	VaryingInput vi = { { 0, 0, 0 }, { static_cast<uint32_t>( c.heads ), static_cast<uint32_t>( a.rows ), 1 } };
	ref_latent_decode_cpu( &vi, &p, nullptr );
	char name[96];
	std::snprintf( name, sizeof( name ), "host latent %s", c.name );
	return check( name, lat_fp64( c, a ), out );
}

bool run_lat_gpu( LatCase c, backend::stream_t s ) {
	LatData a = make_lat( c );
	__half *dq = upload( a.q ), *dc = upload( a.cache );
	__half *o_ref = upload( std::vector<__half>( a.q.size() ) ), *o_eng = upload( std::vector<__half>( a.q.size() ) );
	int *dslots = upload( c.slots ), *dpos = upload( c.pos );
	pod::BatchRows br = pod::BatchRows::batched( a.rows, c.slots.data(), c.pos.data() );
	pod::BatchRows* d_rows = static_cast<pod::BatchRows*>( backend::device_malloc( sizeof( br ) ) );
	backend::memcpy_sync( d_rows, &br, sizeof( br ), backend::Copy::H2D );
	float* ws = static_cast<float*>( backend::device_malloc( modules::MLALatent::workspace_floats( c.max_seq, c.heads ) * sizeof( float ) ) );

	ref_latent_launch_decode( dq, dc, o_ref, dslots, dpos, static_cast<long long>( a.slot_stride ), c.layer, a.rows, c.heads, c.max_seq, s );
	pod::TensorView tq = pod::make_tensor_2d( dq, pod::DType::FP16, a.rows, c.heads * LORA ), tout = pod::make_tensor_2d( o_eng, pod::DType::FP16, a.rows, c.heads * LORA );
	pod::SlotRef sr;
	sr.rows = d_rows; sr.row = 0; sr.stride = a.slot_stride;
	modules::MLALatent::attend( dc, tq, tout, c.layer, c.max_seq, a.rows, c.heads, sr, ws, 0, s );
	backend::stream_sync( s );
	const auto got = download<__half>( o_ref, a.q.size() ), eng = download<__half>( o_eng, a.q.size() );
	char name[96];
	std::snprintf( name, sizeof( name ), "gpu latent %s vs engine", c.name );
	bool ok = check_h( name, eng, got );
	std::snprintf( name, sizeof( name ), "gpu latent %s vs fp64", c.name );
	ok = check( name, lat_fp64( c, a ), got ) && ok;
	for ( void* p : { (void*) dq, (void*) dc, (void*) o_ref, (void*) o_eng, (void*) dslots, (void*) dpos, (void*) d_rows, (void*) ws } ) backend::device_free( p );
	return ok;
}

std::vector<AttnCase> attn_cases() {
	return {
		// Qwen-like: 24 q heads over 4 kv heads, head_dim 256, positions spanning one and several 256-key tiles, two slots
		{ "qwen 24/4x256 gate", 24, 4, 256, 768, 1, 2, { 0, 1, 0, 1, 0 }, { 0, 5, 255, 256, 700 }, true },
		{ "qwen 24/4x256 nogate", 24, 4, 256, 768, 0, 2, { 1, 1, 1 }, { 511, 512, 767 }, false },
		{ "tiny 4/2x8", 4, 2, 8, 40, 1, 2, { 0, 1, 0, 1 }, { 0, 1, 39, 17 }, false },
		{ "mha 8/8x64 gate", 8, 8, 64, 300, 2, 1, { 0, 0, 0 }, { 0, 150, 299 }, true },
		{ "gqa 6/2x128", 6, 2, 128, 520, 1, 3, { 2, 0, 1 }, { 300, 1, 519 }, false },
		{ "prefill chunk 8/2x128", 8, 2, 128, 600, 0, 1, { 0, 0, 0, 0, 0, 0 }, { 250, 251, 252, 253, 254, 255 }, true },
	};
}
std::vector<LatCase> lat_cases() {
	return {
		{ "16 heads", 16, 700, 1, 2, { 0, 1, 0 }, { 0, 300, 699 } },
		{ "16 heads long", 16, 1536, 1, 2, { 1, 0, 1, 0, 0 }, { 0, 255, 256, 1000, 1535 } },
	};
}

} // namespace

int main( int argc, char** argv ) {
	bool gpu = false;
	for ( int i = 1; i < argc; ++i ) gpu = gpu || std::strcmp( argv[i], "--gpu" ) == 0;
	bool pass = true;
	std::printf( "=== reference attention, Slang host target vs FP64 ===\n" );
	for ( const auto& c : attn_cases() ) pass = run_attn_host( c ) && pass;
	for ( const auto& c : lat_cases() ) pass = run_lat_host( c ) && pass;
	if ( gpu ) {
		std::printf( "=== reference attention, Slang CUDA vs engine kernels ===\n" );
		backend::stream_t s = backend::stream_create();
		for ( const auto& c : attn_cases() ) pass = run_attn_gpu( c, s ) && pass;
		for ( const auto& c : lat_cases() ) pass = run_lat_gpu( c, s ) && pass;
	}
	else {
		std::printf( "GPU comparison not run (pass --gpu)\n" );
	}
	std::printf( pass ? "ALL PASS\n" : "FAILED\n" );
	return pass ? 0 : 1;
}
