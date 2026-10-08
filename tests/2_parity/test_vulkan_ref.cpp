// Portable reference kernels (ref_gemm_f16, ref_trellis_reconstruct, ref_had128, ref_attn_decode, ref_latent_decode) as SPIR-V through
// the Vulkan layer on a real device, compared with FP64 references. Also reports each Params block against the push-constant limit.
// Skips (exit 0) when no Vulkan device with the required features exists. Device: VOLTA_VK_DEVICE (default 0).
#include "engine/backend/numeric.h"
#include "engine/backend/vulkan/device.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace engine::backend;

#ifndef SLANG_SPV_DIR
	#error "SLANG_SPV_DIR must point at the generated .spv directory"
#endif

// Push-constant blocks: natural C layout == Slang's std430 emission (8-byte pointers, 4-byte scalars); mirrors src/shaders/ref_*.slang
struct RefGemmParams { uint64_t x, w, out_; int m, k, n; };
struct RefReconParams { uint64_t codes, out_; int k_dim, n_dim, bits, cb; };
struct RefHadParams { uint64_t src, dst, pre, post; int cols; float r_scale; };
struct RefAttnParams { uint64_t q, k, v, gate, out_, slots, pos; int64_t slot_stride; int layer_idx, num_q_heads, num_kv_heads, head_dim, max_seq, has_gate; float scale; };
struct RefLatentParams { uint64_t q_lat, cache, out_, slots, pos; int64_t slot_stride; int layer, heads, max_seq; float scale; };

static uint32_t g_state = 99u;
static uint32_t rnd() {
	g_state = g_state * 1664525u + 1013904223u;
	return g_state;
}
static float frand( float scale ) { return ( static_cast<int>( rnd() >> 16 ) % 2001 - 1000 ) / 1000.0f * scale; }
static std::vector<half_t> make( size_t n, float scale ) {
	std::vector<half_t> v( n );
	for ( auto& x : v ) x = from_float_half( frand( scale ) );
	return v;
}
static std::vector<half_t> make_signs( size_t n ) {
	std::vector<half_t> v( n );
	for ( auto& x : v ) x = from_float_half( ( rnd() >> 20 ) & 1 ? 1.0f : -1.0f );
	return v;
}
static double d( half_t h ) { return static_cast<double>( to_float( h ) ); }

template<typename T> struct Dev {
	void* p = nullptr;
	size_t n = 0;
	explicit Dev( const std::vector<T>& v ) : n( v.size() ) {
		p = device_malloc( std::max<size_t>( n * sizeof( T ), 16 ) );
		memcpy_sync( p, v.data(), n * sizeof( T ), Copy::H2D );
	}
	explicit Dev( size_t count ) : n( count ) {
		p = device_malloc( std::max<size_t>( n * sizeof( T ), 16 ) );
		memset_sync( p, 0, ( ( n * sizeof( T ) + 3 ) / 4 ) * 4 );
	}
	std::vector<T> get() const {
		std::vector<T> v( n );
		memcpy_sync( v.data(), p, n * sizeof( T ), Copy::D2H );
		return v;
	}
	uint64_t addr() const { return reinterpret_cast<uint64_t>( p ); }
	~Dev() { device_free( p ); }
};

static vk::Pipeline* load( const char* name, uint32_t push_bytes ) {
	const std::string path = std::string( SLANG_SPV_DIR ) + "/" + name + ".spv";
	auto* p = vk::pipeline_create( path.c_str(), push_bytes );
	if ( !p ) std::printf( "  cannot create pipeline %s\n", path.c_str() );
	return p;
}

static bool check( const char* name, const std::vector<double>& ref, const std::vector<half_t>& got ) {
	double dot = 0, na = 0, nb = 0, maxd = 0, maxabs = 0;
	for ( size_t i = 0; i < ref.size(); ++i ) {
		const double b = d( got[i] );
		dot += ref[i] * b; na += ref[i] * ref[i]; nb += b * b;
		maxd = std::fmax( maxd, std::fabs( ref[i] - b ) );
		maxabs = std::fmax( maxabs, std::fabs( ref[i] ) );
	}
	const double cosv = dot / ( std::sqrt( na ) * std::sqrt( nb ) + 1e-30 );
	const bool ok = ref.size() == got.size() && std::isfinite( cosv ) && cosv >= 0.9999 && maxd <= 0.005 * maxabs + 1e-3;
	std::printf( "  %-38s cos %.7f max|d| %.3g (max %.3g)  %s\n", name, cosv, maxd, maxabs, ok ? "PASS" : "FAIL" );
	return ok;
}
static std::vector<double> tod( const std::vector<half_t>& v ) {
	std::vector<double> o( v.size() );
	for ( size_t i = 0; i < v.size(); ++i ) o[i] = d( v[i] );
	return o;
}

// ---- FP64 references (same formulations as test_ref_linear.cpp / test_ref_attention.cpp) ----

static double half_bits_to_double( uint32_t h ) {
	const double sign = ( h & 0x8000u ) ? -1.0 : 1.0;
	const int e = static_cast<int>( ( h >> 10 ) & 31u );
	const double m = static_cast<double>( h & 1023u );
	if ( e == 0 ) return sign * m * std::ldexp( 1.0, -24 );
	return sign * ( 1.0 + m / 1024.0 ) * std::ldexp( 1.0, e - 15 );
}
static double codebook_ref( uint32_t w, int cb ) {
	if ( cb == 2 ) {
		const uint32_t x = w * 0x83DCD12Du;
		const uint32_t s = ( x & 255u ) + ( ( x >> 8 ) & 255u ) + ( ( x >> 16 ) & 255u ) + ( x >> 24 );
		return static_cast<double>( 1024u + s ) * half_bits_to_double( 0x1eeeu ) + half_bits_to_double( 0xc931u );
	}
	const uint32_t x = cb == 0 ? w * 89226354u + 64248484u : w * 0xCBAC1FEDu;
	const uint32_t y = ( x & 0x8fff8fffu ) ^ 0x3b603b60u;
	return half_bits_to_double( y & 0xffffu ) + half_bits_to_double( y >> 16 );
}
// Dequantized [k, n] weight in the rotated basis; window extraction follows dq() (funnel shift over uint32 words), the scatter follows
// the forward map of the engine's reconstruct kernel (a different formulation from the Slang kernel's inverse map)
static std::vector<double> recon_ref( const std::vector<uint16_t>& codes, int k, int n, int bits, int cb ) {
	std::vector<double> w( static_cast<size_t>( k ) * n );
	const int nw = bits * 256 / 32;
	for ( int kt = 0; kt < k / 16; ++kt ) {
		for ( int nt = 0; nt < n / 16; ++nt ) {
			const uint16_t* tile = codes.data() + ( static_cast<size_t>( kt ) * ( n / 16 ) + nt ) * 16 * bits;
			auto word = [&]( int i ) { i %= nw; return static_cast<uint32_t>( tile[2 * i] ) | ( static_cast<uint32_t>( tile[2 * i + 1] ) << 16 ); };
			for ( int lane = 0; lane < 32; ++lane ) {
				for ( int i = 0; i < 8; ++i ) {
					const int p = lane * 8 + i;
					const int b0 = p * bits + bits - 16 + 256 * bits;
					const int b1 = b0 + 16;
					const int i0 = b0 / 32, i1 = ( b1 - 1 ) / 32;
					const int s0 = ( i1 + 1 ) * 32 - b1;
					const uint64_t merged = ( static_cast<uint64_t>( word( i0 ) ) << 32 ) | word( i1 );
					const uint32_t win = static_cast<uint32_t>( merged >> s0 ) & 0xffffu;
					const int row = ( lane % 4 ) * 2 + ( i & 1 ) + 8 * ( ( i >> 1 ) & 1 );
					const int col = 2 * ( lane / 8 + 4 * ( i >> 2 ) ) + ( ( lane >> 2 ) & 1 );
					w[static_cast<size_t>( kt * 16 + row ) * n + nt * 16 + col] = codebook_ref( win, cb );
				}
			}
		}
	}
	return w;
}
static void had128_ref( std::vector<double>& v, int rows, int cols ) {
	for ( int r = 0; r < rows; ++r ) {
		for ( int b = 0; b < cols / 128; ++b ) {
			double* p = v.data() + static_cast<size_t>( r ) * cols + b * 128;
			for ( int len = 1; len < 128; len <<= 1 )
				for ( int i = 0; i < 128; i += 2 * len )
					for ( int j = i; j < i + len; ++j ) {
						const double a = p[j], c = p[j + len];
						p[j] = a + c;
						p[j + len] = a - c;
					}
			for ( int i = 0; i < 128; ++i ) p[i] /= std::sqrt( 128.0 );
		}
	}
}
static std::vector<double> gemm_ref( const std::vector<double>& x, const std::vector<double>& w, int m, int k, int n ) {
	std::vector<double> o( static_cast<size_t>( m ) * n, 0.0 );
	for ( int i = 0; i < m; ++i )
		for ( int p = 0; p < k; ++p )
			for ( int j = 0; j < n; ++j ) o[static_cast<size_t>( i ) * n + j] += x[static_cast<size_t>( i ) * k + p] * w[static_cast<size_t>( p ) * n + j];
	return o;
}

int main() {
	if ( !vk::available() ) {
		std::printf( "test_vulkan_ref: no Vulkan device with BDA/Float16/Int64, skipping\n" );
		return 0;
	}
	const uint32_t limit = vk::max_push_constants_bytes();
	std::printf( "=== reference kernels through Vulkan: %s (push constants max %u B) ===\n", vk::device_name(), limit );
	bool pass = true;

	{ // push-constant budget
		struct Row { const char* name; size_t bytes; };
		const Row rows[] = { { "RefGemmParams", sizeof( RefGemmParams ) }, { "RefReconParams", sizeof( RefReconParams ) }, { "RefHadParams", sizeof( RefHadParams ) },
			{ "RefAttnParams", sizeof( RefAttnParams ) }, { "RefLatentParams", sizeof( RefLatentParams ) } };
		for ( const Row& r : rows ) {
			const bool ok = r.bytes <= limit && r.bytes <= 256;
			std::printf( "  push block %-18s %3zu B (limit %u, spec minimum 128)  %s\n", r.name, r.bytes, limit, ok ? "FITS" : "TOO LARGE" );
			pass = ok && pass;
		}
	}

	const char* kRecon = "ref_trellis_reconstruct";
	auto* gemm = load( "ref_gemm_f16", sizeof( RefGemmParams ) );
	auto* recon = load( kRecon, sizeof( RefReconParams ) );
	auto* had = load( "ref_had128", sizeof( RefHadParams ) );
	auto* attn = load( "ref_attn_decode", sizeof( RefAttnParams ) );
	auto* lat = load( "ref_latent_decode", sizeof( RefLatentParams ) );
	if ( !gemm || !recon || !had || !attn || !lat ) {
		std::printf( "test_vulkan_ref: FAILED (pipelines)\n" );
		return 1;
	}

	{ // ref_gemm_f16, ragged shapes
		const int shapes[3][3] = { { 1, 128, 128 }, { 7, 200, 150 }, { 33, 513, 65 } };
		for ( const auto& s : shapes ) {
			const int m = s[0], k = s[1], n = s[2];
			auto x = make( size_t( m ) * k, 1.0f ), w = make( size_t( k ) * n, 0.3f );
			Dev<half_t> dx( x ), dw( w ), out( size_t( m ) * n );
			RefGemmParams p{ dx.addr(), dw.addr(), out.addr(), m, k, n };
			vk::dispatch( nullptr, gemm, &p, ( n + 15 ) / 16, ( m + 15 ) / 16 );
			stream_sync( nullptr );
			std::vector<double> ref( size_t( m ) * n, 0.0 );
			for ( int i = 0; i < m; ++i )
				for ( int pp = 0; pp < k; ++pp )
					for ( int j = 0; j < n; ++j ) ref[size_t( i ) * n + j] += d( x[size_t( i ) * k + pp] ) * d( w[size_t( pp ) * n + j] );
			char name[64];
			std::snprintf( name, sizeof( name ), "ref_gemm_f16 %dx%dx%d", m, k, n );
			pass = check( name, ref, out.get() ) && pass;
		}
	}
	{ // ref_trellis_reconstruct for 2..4 bpw and the three codebooks
		const int k = 256, n = 384;
		for ( int bits = 2; bits <= 4; ++bits ) {
			for ( int cb = 0; cb < 3; ++cb ) {
				std::vector<uint16_t> codes( size_t( k / 16 ) * ( n / 16 ) * 16 * bits );
				for ( auto& c : codes ) c = static_cast<uint16_t>( rnd() >> 8 );
				Dev<uint16_t> dc( codes );
				Dev<half_t> out( size_t( k ) * n );
				RefReconParams p{ dc.addr(), out.addr(), k, n, bits, cb };
				vk::dispatch( nullptr, recon, &p, uint32_t( ( size_t( k ) * n + 255 ) / 256 ) );
				stream_sync( nullptr );
				char name[64];
				std::snprintf( name, sizeof( name ), "ref_trellis_reconstruct %d bpw cb %d", bits, cb );
				pass = check( name, recon_ref( codes, k, n, bits, cb ), out.get() ) && pass;
			}
		}
	}
	{ // ref_had128 with pre and post scales
		const int rows = 5, cols = 384;
		auto src = make( size_t( rows ) * cols, 1.0f ), pre = make_signs( cols ), post = make_signs( cols );
		Dev<half_t> ds( src ), dpre( pre ), dpost( post ), out( src.size() );
		RefHadParams p{ ds.addr(), out.addr(), dpre.addr(), dpost.addr(), cols, 0.08838834764831845f };
		vk::dispatch( nullptr, had, &p, cols / 128, rows );
		stream_sync( nullptr );
		std::vector<double> ref( src.size() );
		for ( int r = 0; r < rows; ++r )
			for ( int c = 0; c < cols; ++c ) ref[size_t( r ) * cols + c] = d( src[size_t( r ) * cols + c] ) * d( pre[c] );
		had128_ref( ref, rows, cols );
		for ( int r = 0; r < rows; ++r )
			for ( int c = 0; c < cols; ++c ) ref[size_t( r ) * cols + c] *= d( post[c] );
		pass = check( "ref_had128 pre+post", ref, out.get() ) && pass;
	}
	{ // composed trellis linear: had( x * suh ) @ W, had( ... ) * svh, intermediates FP16 as in LinearTrellis::forward_recon
		const int cfgs[3][5] = { { 1, 256, 384, 4, 2 }, { 3, 256, 256, 3, 2 }, { 4, 384, 128, 2, 1 } };
		for ( const auto& c : cfgs ) {
			const int m = c[0], k = c[1], n = c[2], bits = c[3], cb = c[4];
			auto x = make( size_t( m ) * k, 1.0f ), suh = make_signs( k ), svh = make_signs( n );
			std::vector<uint16_t> codes( size_t( k / 16 ) * ( n / 16 ) * 16 * bits );
			for ( auto& v : codes ) v = static_cast<uint16_t>( rnd() >> 8 );
			Dev<half_t> dx( x ), dsuh( suh ), dsvh( svh ), xh( size_t( m ) * k ), w( size_t( k ) * n ), y( size_t( m ) * n ), out( size_t( m ) * n );
			Dev<uint16_t> dc( codes );
			RefHadParams h1{ dx.addr(), xh.addr(), dsuh.addr(), 0, k, 0.08838834764831845f };
			RefReconParams rp{ dc.addr(), w.addr(), k, n, bits, cb };
			RefGemmParams gp{ xh.addr(), w.addr(), y.addr(), m, k, n };
			RefHadParams h2{ y.addr(), out.addr(), 0, dsvh.addr(), n, 0.08838834764831845f };
			vk::dispatch( nullptr, had, &h1, k / 128, m );
			vk::dispatch( nullptr, recon, &rp, uint32_t( ( size_t( k ) * n + 255 ) / 256 ) );
			vk::dispatch( nullptr, gemm, &gp, ( n + 15 ) / 16, ( m + 15 ) / 16 );
			vk::dispatch( nullptr, had, &h2, n / 128, m );
			stream_sync( nullptr );
			std::vector<double> xd( size_t( m ) * k );
			for ( int i = 0; i < m; ++i )
				for ( int p = 0; p < k; ++p ) xd[size_t( i ) * k + p] = d( x[size_t( i ) * k + p] ) * d( suh[p] );
			had128_ref( xd, m, k );
			auto yd = gemm_ref( xd, recon_ref( codes, k, n, bits, cb ), m, k, n );
			had128_ref( yd, m, n );
			for ( int i = 0; i < m; ++i )
				for ( int j = 0; j < n; ++j ) yd[size_t( i ) * n + j] *= d( svh[j] );
			char name[64];
			std::snprintf( name, sizeof( name ), "trellis linear m%d k%d n%d %dbpw", m, k, n, bits );
			pass = check( name, yd, out.get() ) && pass;
		}
	}

	{ // ref_attn_decode: GQA with gate, MHA without, multi-slot rows, context beyond one 256-key tile
		struct Case { const char* name; int hq, hkv, hd, max_seq, layer, nslot; std::vector<int> slots, pos; bool gate; };
		const Case cases[] = {
			{ "gqa gate", 8, 2, 128, 700, 1, 2, { 0, 0, 1 }, { 3, 500, 699 }, true },
			{ "mha hd64", 4, 4, 64, 300, 0, 1, { 0, 0 }, { 0, 299 }, false },
			{ "gqa hd256", 6, 1, 256, 520, 2, 2, { 1, 0 }, { 258, 10 }, true },
		};
		for ( const Case& c : cases ) {
			const int rows = int( c.pos.size() ), row_elems = c.hkv * c.hd;
			const size_t slot_stride = size_t( c.layer + 2 ) * c.max_seq * row_elems;
			auto q = make( size_t( rows ) * c.hq * c.hd, 1.0f ), k = make( slot_stride * c.nslot, 1.0f ), v = make( slot_stride * c.nslot, 1.0f );
			auto gate = make( q.size(), 2.0f );
			Dev<half_t> dq( q ), dk( k ), dv( v ), dg( gate ), out( q.size() );
			Dev<int> dslots( c.slots ), dpos( c.pos );
			RefAttnParams p{ dq.addr(), dk.addr(), dv.addr(), c.gate ? dg.addr() : 0, out.addr(), dslots.addr(), dpos.addr(), int64_t( slot_stride ), c.layer, c.hq, c.hkv, c.hd,
				c.max_seq, c.gate ? 1 : 0, 1.0f / std::sqrt( float( c.hd ) ) };
			vk::dispatch( nullptr, attn, &p, c.hq, rows );
			stream_sync( nullptr );
			std::vector<double> ref( q.size() );
			const int per_kv = std::max( c.hq / c.hkv, 1 );
			const double scale = 1.0 / std::sqrt( double( c.hd ) );
			for ( int m = 0; m < rows; ++m ) {
				const int total = std::min( c.pos[m] + 1, c.max_seq );
				for ( int h = 0; h < c.hq; ++h ) {
					const size_t qo = ( size_t( m ) * c.hq + h ) * c.hd;
					const size_t base = size_t( c.slots[m] ) * slot_stride + size_t( c.layer ) * c.max_seq * row_elems + ( h / per_kv ) * c.hd;
					std::vector<double> s( total );
					double mx = -1e300;
					for ( int t = 0; t < total; ++t ) {
						double dot = 0;
						for ( int e = 0; e < c.hd; ++e ) dot += d( q[qo + e] ) * d( k[base + size_t( t ) * row_elems + e] );
						s[t] = dot * scale;
						mx = std::fmax( mx, s[t] );
					}
					double l = 0;
					for ( int t = 0; t < total; ++t ) { s[t] = std::exp( s[t] - mx ); l += s[t]; }
					for ( int e = 0; e < c.hd; ++e ) {
						double o = 0;
						for ( int t = 0; t < total; ++t ) o += s[t] * d( v[base + size_t( t ) * row_elems + e] );
						o /= l;
						if ( c.gate ) o *= 1.0 / ( 1.0 + std::exp( -d( gate[qo + e] ) ) );
						ref[qo + e] = o;
					}
				}
			}
			char name[64];
			std::snprintf( name, sizeof( name ), "ref_attn_decode %s", c.name );
			pass = check( name, ref, out.get() ) && pass;
		}
	}
	{ // ref_latent_decode (GLM MLA latent cache, 512 wide)
		const int heads = 6, max_seq = 600, layer = 1, nslot = 2, lora = 512;
		const std::vector<int> slots = { 0, 1, 1 }, pos = { 2, 300, 599 };
		const int rows = int( pos.size() );
		const size_t slot_stride = size_t( layer + 2 ) * max_seq * lora;
		auto q = make( size_t( rows ) * heads * lora, 0.5f ), cache = make( slot_stride * nslot, 1.7f );
		Dev<half_t> dq( q ), dc( cache ), out( q.size() );
		Dev<int> dslots( slots ), dpos( pos );
		RefLatentParams p{ dq.addr(), dc.addr(), out.addr(), dslots.addr(), dpos.addr(), int64_t( slot_stride ), layer, heads, max_seq, 0.0625f };
		vk::dispatch( nullptr, lat, &p, heads, rows );
		stream_sync( nullptr );
		std::vector<double> ref( q.size() );
		for ( int m = 0; m < rows; ++m ) {
			const int total = std::min( pos[m] + 1, max_seq );
			const size_t base = size_t( slots[m] ) * slot_stride + size_t( layer ) * max_seq * lora;
			for ( int h = 0; h < heads; ++h ) {
				const size_t qo = ( size_t( m ) * heads + h ) * lora;
				std::vector<double> s( total );
				double mx = -1e300;
				for ( int t = 0; t < total; ++t ) {
					double dot = 0;
					for ( int e = 0; e < lora; ++e ) dot += d( q[qo + e] ) * d( cache[base + size_t( t ) * lora + e] );
					s[t] = dot * 0.0625;
					mx = std::fmax( mx, s[t] );
				}
				double l = 0;
				for ( int t = 0; t < total; ++t ) { s[t] = std::exp( s[t] - mx ); l += s[t]; }
				for ( int e = 0; e < lora; ++e ) {
					double o = 0;
					for ( int t = 0; t < total; ++t ) o += s[t] * d( cache[base + size_t( t ) * lora + e] );
					ref[qo + e] = o / l;
				}
			}
		}
		pass = check( "ref_latent_decode", ref, out.get() ) && pass;
	}

	vk::pipeline_destroy( gemm ); vk::pipeline_destroy( recon ); vk::pipeline_destroy( had ); vk::pipeline_destroy( attn ); vk::pipeline_destroy( lat );
	device_sync();
	std::printf( "%s\n", pass ? "test_vulkan_ref: PASSED" : "test_vulkan_ref: FAILED" );
	return pass ? 0 : 1;
}
