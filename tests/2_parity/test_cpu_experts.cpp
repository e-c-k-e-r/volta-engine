// Routed experts on the host CPU (engine/runtime/cpu_experts.h): the repacked AVX2 trellis GEMV and the expert pipeline against an independent
// scalar model of the EXL3 MUL1 format. Part 1 needs no files (random tiles, 1 to 8 bits, 1 to 16 rows, AVX2 vs scalar kernel); part 2 reads one
// GLM routed expert from the checkpoint directory given as argv[1] (layer argv[2], expert argv[3]) and checks the whole expert and its speed.
#include "engine/runtime/cpu_experts.h"
#include "engine/kernels/cpu/cpu_util.h"
#include "engine/kernels/cpu/trellis_linear.h"
#include "engine/kernels/cpu/nvfp4_linear.h"
#include "engine/quant/fp4_fp8.h"
#include "engine/loader/safetensors.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <sched.h>
#include <thread>
#include <string>
#include <vector>

using namespace engine::cpu_moe;
using namespace engine::kernels::cpu;

static std::mt19937 g_rng( 20261018 );
static float rnd() { return std::uniform_real_distribution<float>( -1.0f, 1.0f )( g_rng ); }
static int g_fail = 0;
#define CHECK( cond, ... ) do { if ( !( cond ) ) { ++g_fail; std::printf( "FAIL: " __VA_ARGS__ ); std::printf( "\n" ); } } while ( 0 )

// Independent model of the stored format (the byte-stream formulation): out[r][n] = sum_k W[k][n] * x[r][k] with W decoded tile by tile
static void model_gemv( const uint16_t* trellis, int kt, int nt, int bits, int rows, const float* x, double* out ) {
	const int total = 256 * bits;
	std::vector<uint8_t> stream( total / 8 + 4 );
	std::vector<double> w( static_cast<size_t>( kt ) * 16 * nt * 16 );
	const double c1 = 0.0067672729492187, c2 = -10.3828125;   // fp16 0x1eee, 0xc931 (checked below)
	for ( int k = 0; k < kt; ++k )
		for ( int n = 0; n < nt; ++n ) {
			const uint16_t* tile = trellis + ( static_cast<size_t>( k ) * nt + n ) * 16 * bits;
			for ( int wd = 0; wd < total / 32; ++wd ) {
				const uint32_t word = static_cast<uint32_t>( tile[wd * 2] ) | ( static_cast<uint32_t>( tile[wd * 2 + 1] ) << 16 );
				stream[wd * 4 + 0] = word >> 24; stream[wd * 4 + 1] = word >> 16; stream[wd * 4 + 2] = word >> 8; stream[wd * 4 + 3] = word;
			}
			for ( int e = 0; e < 4; ++e ) stream[total / 8 + e] = stream[e];
			for ( int p = 0; p < 256; ++p ) {
				const int q0 = ( p * bits + bits - 16 + total ) % total;
				const uint8_t* b = stream.data() + ( q0 >> 3 );
				const uint32_t v = ( static_cast<uint32_t>( b[0] ) << 24 ) | ( static_cast<uint32_t>( b[1] ) << 16 ) | ( static_cast<uint32_t>( b[2] ) << 8 ) | b[3];
				const uint32_t window = ( v >> ( 16 - ( q0 & 7 ) ) ) & 0xffffu;
				const uint32_t xx = window * 0x83DCD12Du;
				const uint32_t sum = ( xx & 255u ) + ( ( xx >> 8 ) & 255u ) + ( ( xx >> 16 ) & 255u ) + ( xx >> 24 );
				const int L = p >> 3, i = p & 7;
				const int row = ( L & 3 ) * 2 + ( i & 1 ) + 8 * ( ( i >> 1 ) & 1 );
				const int col = 2 * ( ( L >> 3 ) + 4 * ( i >> 2 ) ) + ( ( L >> 2 ) & 1 );
				w[( static_cast<size_t>( k ) * 16 + row ) * nt * 16 + n * 16 + col] = ( 1024.0 + sum ) * c1 + c2;
			}
		}
	for ( int r = 0; r < rows; ++r )
		for ( int n = 0; n < nt * 16; ++n ) {
			double a = 0;
			for ( int k = 0; k < kt * 16; ++k ) a += w[static_cast<size_t>( k ) * nt * 16 + n] * x[static_cast<size_t>( r ) * kt * 16 + k];
			out[static_cast<size_t>( r ) * nt * 16 + n] = a;
		}
}

static double rel_err( const float* a, const double* b, size_t n ) {
	double num = 0, den = 0;
	for ( size_t i = 0; i < n; ++i ) { num += ( a[i] - b[i] ) * ( a[i] - b[i] ); den += b[i] * b[i]; }
	return std::sqrt( num / ( den + 1e-30 ) );
}

static void part1() {
	std::printf( "part 1: random tiles, AVX2 %s\n", avx2_available() ? "yes" : "no (scalar kernel)" );
	const int kt = 8, nt = 8;
	for ( int bits = 1; bits <= 8; ++bits ) {
		std::vector<uint16_t> tr( static_cast<size_t>( kt ) * nt * 16 * bits ), su( kt * 16, 0x3c00 ), sv( nt * 16, 0x3c00 );
		for ( auto& v : tr ) v = static_cast<uint16_t>( g_rng() );
		TrellisLinear lin( tr.data(), kt, nt, bits, su.data(), sv.data() );
		for ( int rows : { 1, 2, 3, 4, 7, 16 } ) {
			std::vector<float> x( static_cast<size_t>( rows ) * kt * 16 ), out( static_cast<size_t>( rows ) * nt * 16 ), rs( rows );
			std::vector<double> ref( out.size() );
			for ( size_t i = 0; i < x.size(); ++i ) x[i] = rnd();
			for ( int r = 0; r < rows; ++r ) { double a = 0; for ( int i = 0; i < kt * 16; ++i ) a += x[static_cast<size_t>( r ) * kt * 16 + i]; rs[r] = static_cast<float>( a ); }
			lin.run( rows, x.data(), rs.data(), 0, nt, out.data() );
			model_gemv( tr.data(), kt, nt, bits, rows, x.data(), ref.data() );
			const double e = rel_err( out.data(), ref.data(), out.size() );
			CHECK( e < 2e-5, "bits %d rows %d: relative error %.3g", bits, rows, e );
		}
	}
	std::printf( "  done\n" );
}

static const void* find_tensor( const std::string& dir, const std::string& name, std::vector<pod::SafeTensorsFile>& keep, const pod::SafeTensorMeta** meta ) {
	for ( int s = 1; s <= 15; ++s ) {
		char fn[512];
		std::snprintf( fn, sizeof fn, "%s/model-%05d-of-00015.safetensors", dir.c_str(), s );
		keep.emplace_back();
		if ( !engine::loader::safetensors_open( keep.back(), fn ) ) { keep.pop_back(); continue; }
		if ( const auto* t = engine::loader::safetensors_find( keep.back(), name.c_str() ) ) { *meta = t; return engine::loader::safetensors_get_data_cpu( keep.back(), t ); }
	}
	return nullptr;
}

static void part2( const std::string& dir, int layer, int expert ) {
	std::vector<pod::SafeTensorsFile> keep;
	keep.reserve( 64 );
	const char* parts[3] = { "gate_proj", "up_proj", "down_proj" };
	std::vector<std::vector<uint16_t>> suh_h( 3 ), svh_h( 3 );
	const uint16_t* raw[3];
	std::vector<std::unique_ptr<TrellisLinear>> lin( 3 );
	std::vector<float> suh_f[3], svh_f[3];
	for ( int p = 0; p < 3; ++p ) {
		char base[256];
		std::snprintf( base, sizeof base, "model.language_model.layers.%d.mlp.experts.%d.%s", layer, expert, parts[p] );
		const pod::SafeTensorMeta *mt, *ms, *mv;
		const void* tr = find_tensor( dir, std::string( base ) + ".trellis", keep, &mt );
		const void* su = find_tensor( dir, std::string( base ) + ".suh", keep, &ms );
		const void* sv = find_tensor( dir, std::string( base ) + ".svh", keep, &mv );
		if ( !tr || !su || !sv ) { std::printf( "part 2: tensors of %s not found under %s, skipped\n", base, dir.c_str() ); return; }
		const int kt = mt->shape[0], nt = mt->shape[1], bits = mt->shape[2] / 16;
		raw[p] = static_cast<const uint16_t*>( tr );
		lin[p].reset( new TrellisLinear( raw[p], kt, nt, bits, static_cast<const uint16_t*>( su ), static_cast<const uint16_t*>( sv ) ) );
		suh_f[p].resize( kt * 16 ); svh_f[p].resize( nt * 16 );
		half_to_float( static_cast<const uint16_t*>( su ), suh_f[p].data(), suh_f[p].size() );
		half_to_float( static_cast<const uint16_t*>( sv ), svh_f[p].data(), svh_f[p].size() );
	}
	Expert ex; ex.gate = lin[0].get(); ex.up = lin[1].get(); ex.down = lin[2].get();
	struct Mats { int kt, nt, bits; } mats[3];
	for ( int p = 0; p < 3; ++p ) mats[p] = { lin[p]->in() / 16, lin[p]->out() / 16, lin[p]->bits() };
	struct Vec { const float *suh_g, *svh_g, *suh_u, *svh_u, *suh_d, *svh_d; } ev = { suh_f[0].data(), svh_f[0].data(), suh_f[1].data(), svh_f[1].data(), suh_f[2].data(), svh_f[2].data() };
	const int H = ex.hidden(), I = ex.inter();
	std::printf( "part 2: layer %d expert %d: hidden %d inter %d bits %d/%d/%d\n", layer, expert, H, I, mats[0].bits, mats[1].bits, mats[2].bits );

	// double-precision model of the whole expert
	auto rot_in = []( const std::vector<double>& x, const float* pre, int k ) {
		std::vector<double> o( k );
		for ( int i = 0; i < k; ++i ) o[i] = x[i] * pre[i];
		for ( int b = 0; b < k / 128; ++b )
			for ( int len = 1; len < 128; len <<= 1 )
				for ( int i = 0; i < 128; i += 2 * len )
					for ( int j = i; j < i + len; ++j ) { const double a = o[b * 128 + j], c = o[b * 128 + j + len]; o[b * 128 + j] = a + c; o[b * 128 + j + len] = a - c; }
		for ( auto& v : o ) v /= std::sqrt( 128.0 );
		return o;
	};
	auto rot_out = []( std::vector<double> y, const float* post, int n ) {
		for ( int b = 0; b < n / 128; ++b )
			for ( int len = 1; len < 128; len <<= 1 )
				for ( int i = 0; i < 128; i += 2 * len )
					for ( int j = i; j < i + len; ++j ) { const double a = y[b * 128 + j], c = y[b * 128 + j + len]; y[b * 128 + j] = a + c; y[b * 128 + j + len] = a - c; }
		for ( int i = 0; i < n; ++i ) y[i] = y[i] / std::sqrt( 128.0 ) * post[i];
		return y;
	};
	for ( int rows : { 1, 4, 8 } ) {
		std::vector<float> xf( static_cast<size_t>( rows ) * H );
		std::vector<uint16_t> xh( xf.size() );
		for ( auto& v : xf ) v = rnd() * 2;
		float_to_half( xf.data(), xh.data(), xf.size() );
		half_to_float( xh.data(), xf.data(), xf.size() );
		std::vector<float> wts( rows ), out( static_cast<size_t>( rows ) * H, 0.0f );
		for ( auto& w : wts ) w = 0.5f + rnd() * 0.25f;
		Pool pool( 4, {} );
		Scratch scratch;
		expert_forward( pool, ex, rows, xh.data(), wts.data(), out.data(), scratch );
		std::vector<double> ref( out.size() );
		for ( int r = 0; r < rows; ++r ) {
			std::vector<double> x( xf.begin() + static_cast<size_t>( r ) * H, xf.begin() + static_cast<size_t>( r + 1 ) * H );
			std::vector<double> xg = rot_in( x, ev.suh_g, H ), xu = rot_in( x, ev.suh_u, H );
			std::vector<float> xgf( xg.begin(), xg.end() ), xuf( xu.begin(), xu.end() );
			std::vector<double> g( I ), u( I ), a( I );
			model_gemv( raw[0], mats[0].kt, mats[0].nt, mats[0].bits, 1, xgf.data(), g.data() );
			model_gemv( raw[1], mats[1].kt, mats[1].nt, mats[1].bits, 1, xuf.data(), u.data() );
			g = rot_out( g, ev.svh_g, I ); u = rot_out( u, ev.svh_u, I );
			for ( int i = 0; i < I; ++i ) { const double gv = std::min( g[i], 10.0 ), uv = std::max( -10.0, std::min( u[i], 10.0 ) ); a[i] = gv / ( 1.0 + std::exp( -gv ) ) * uv; }
			std::vector<double> ad = rot_in( a, ev.suh_d, I );
			std::vector<float> adf( ad.begin(), ad.end() );
			std::vector<double> d( H );
			model_gemv( raw[2], mats[2].kt, mats[2].nt, mats[2].bits, 1, adf.data(), d.data() );
			d = rot_out( d, ev.svh_d, H );
			for ( int i = 0; i < H; ++i ) ref[static_cast<size_t>( r ) * H + i] = wts[r] * d[i];
		}
		const double e = rel_err( out.data(), ref.data(), out.size() );
		CHECK( e < 4e-3, "expert, %d rows: relative error %.3g", rows, e );
		std::printf( "  rows %d: relative error vs the double model %.3g\n", rows, e );
	}
	// speed: one expert on 1..10 threads
	for ( int thr : { 1, 5, 10 } ) {
		std::vector<int> cpus;
		for ( int t = 1; t < thr; ++t ) cpus.push_back( 10 + t );
		Pool pool( thr, cpus );
		Scratch scratch;
		std::vector<uint16_t> xh( H, 0x3c00 );
		std::vector<float> out( H ), wts( 1, 1.0f );
		for ( int i = 0; i < 3; ++i ) expert_forward( pool, ex, 1, xh.data(), wts.data(), out.data(), scratch );
		const int reps = 20;
		double ms = 0;
		for ( int i = 0; i < reps; ++i ) {
			const auto t0 = std::chrono::steady_clock::now();
			expert_forward( pool, ex, 1, xh.data(), wts.data(), out.data(), scratch );
			ms += std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t0 ).count() / reps;
		}
		const double bytes = 3.0 * mats[0].kt * mats[0].nt * mats[0].bits * 32;
		std::printf( "  %2d threads: %.3f ms per expert (1 row), %.2f GB/s of codes\n", thr, ms, bytes / ms * 1e-6 );
	}
}


// NVFP4: random prepacked streams; the model decodes the checkpoint layout column by column (the CPU backend's formulation) in double
struct Fp4Mat { int k, n; float gs; std::vector<uint8_t> codes, scales; };
static Fp4Mat make_fp4( int k, int n ) {
	Fp4Mat m; m.k = k; m.n = n; m.gs = 0.01f + 0.01f * rnd();
	m.codes.resize( static_cast<size_t>( n / 32 ) * ( k / 16 ) * 32 * 8 );
	m.scales.resize( static_cast<size_t>( n / 32 ) * ( k / 16 ) * 32 );
	for ( auto& c : m.codes ) c = static_cast<uint8_t>( g_rng() );
	for ( auto& c : m.scales ) c = static_cast<uint8_t>( 0x28 + g_rng() % 0x20 );   // positive, normal
	return m;
}
static std::vector<double> fp4_weights( const Fp4Mat& m ) {   // [k][n]
	std::vector<double> w( static_cast<size_t>( m.k ) * m.n );
	const int G = m.k / 16;
	for ( int col = 0; col < m.n; ++col ) {
		const int t = col / 32, j = col % 32, qp = j >> 3, r = j & 7, lane = qp * 4 + ( r & 3 ) + ( r >= 4 ? 16 : 0 );
		for ( int g = 0; g < G; ++g ) {
			const size_t at = ( static_cast<size_t>( t ) * G + g ) * 32 + lane;
			uint32_t cw[2];
			std::memcpy( cw, &m.codes[at * 8], 8 );
			const double s = engine::quant::e4m3_to_float( m.scales[at] ) * static_cast<double>( m.gs );
			for ( int h = 0; h < 2; ++h )
				for ( int i = 0; i < 4; ++i ) {
					w[static_cast<size_t>( g * 16 + h * 8 + 2 * i ) * m.n + col] = engine::quant::e2m1_to_float( ( cw[h] >> ( 4 * i ) ) & 15 ) * s;
					w[static_cast<size_t>( g * 16 + h * 8 + 2 * i + 1 ) * m.n + col] = engine::quant::e2m1_to_float( ( cw[h] >> ( 16 + 4 * i ) ) & 15 ) * s;
				}
		}
	}
	return w;
}
static std::vector<double> matvec( const std::vector<double>& w, const std::vector<double>& x, int k, int n ) {
	std::vector<double> y( n, 0.0 );
	for ( int i = 0; i < k; ++i ) for ( int c = 0; c < n; ++c ) y[c] += w[static_cast<size_t>( i ) * n + c] * x[i];
	return y;
}

// run() takes the prepared form of the input
static std::vector<float> prepared( const Linear& lin, const std::vector<float>& x, int rows ) {
	std::vector<float> xin( x.size() );
	for ( int r = 0; r < rows; ++r )
		for ( int b = 0; b < lin.in() / kBlock; ++b ) lin.prepare_block( &x[static_cast<size_t>( r ) * lin.in() + b * kBlock], b, &xin[static_cast<size_t>( r ) * lin.in() + b * kBlock] );
	return xin;
}

static void part3() {
	std::printf( "part 3: NVFP4 linear and expert\n" );
	const int H = 512, I = 256;
	Fp4Mat mg = make_fp4( H, I ), mu = make_fp4( H, I ), md = make_fp4( I, H );
	for ( int rows : { 1, 2, 5, 16 } ) {
		Nvfp4Linear lg( mg.codes.data(), mg.scales.data(), mg.gs, H, I );
		std::vector<float> x( static_cast<size_t>( rows ) * H ), out( static_cast<size_t>( rows ) * I ), rs( rows, 0.0f );
		for ( auto& v : x ) v = rnd();
		{ const auto xp = prepared( lg, x, rows ); lg.run( rows, xp.data(), rs.data(), 0, I / 16, out.data() ); }
		const auto w = fp4_weights( mg );
		std::vector<double> ref( out.size() );
		for ( int r = 0; r < rows; ++r ) {
			std::vector<double> xd( x.begin() + static_cast<size_t>( r ) * H, x.begin() + static_cast<size_t>( r + 1 ) * H );
			const auto y = matvec( w, xd, H, I );
			for ( int c = 0; c < I; ++c ) ref[static_cast<size_t>( r ) * I + c] = y[c];
		}
		const double e = rel_err( out.data(), ref.data(), out.size() );
		CHECK( e < 2e-4, "nvfp4 linear, %d rows: relative error %.3g", rows, e );
	}
	{
		const auto w = fp4_weights( mg );
		std::unique_ptr<Nvfp4Linear> nat( new Nvfp4Linear( mg.codes.data(), mg.scales.data(), mg.gs, H, I ) );
		for ( int rows : { 1, 3, 4, 7 } ) {
			std::vector<float> x( static_cast<size_t>( rows ) * H ), out( static_cast<size_t>( rows ) * I ), rs( rows, 0.0f );
			for ( auto& v : x ) v = rnd();
			{ const auto xp = prepared( *nat, x, rows ); nat->run( rows, xp.data(), rs.data(), 0, I / 16, out.data() ); }
			std::vector<double> ref( out.size() );
			for ( int r = 0; r < rows; ++r ) {
				std::vector<double> xd( x.begin() + static_cast<size_t>( r ) * H, x.begin() + static_cast<size_t>( r + 1 ) * H );
				const auto y = matvec( w, xd, H, I );
				for ( int c = 0; c < I; ++c ) ref[static_cast<size_t>( r ) * I + c] = y[c];
			}
			const double e = rel_err( out.data(), ref.data(), out.size() );
			CHECK( e < 2e-4, "nvfp4 linear, %d rows: relative error %.3g", rows, e );
		}
	}
	Nvfp4Linear lg( mg.codes.data(), mg.scales.data(), mg.gs, H, I ), lu( mu.codes.data(), mu.scales.data(), mu.gs, H, I ), ld( md.codes.data(), md.scales.data(), md.gs, I, H );
	Expert ex; ex.gate = &lg; ex.up = &lu; ex.down = &ld;
	const auto wg = fp4_weights( mg ), wu = fp4_weights( mu ), wd = fp4_weights( md );
	for ( int rows : { 1, 3 } ) {
		std::vector<float> xf( static_cast<size_t>( rows ) * H ), wts( rows ), out( static_cast<size_t>( rows ) * H, 0.0f );
		std::vector<uint16_t> xh( xf.size() );
		for ( auto& v : xf ) v = rnd();
		float_to_half( xf.data(), xh.data(), xf.size() );
		half_to_float( xh.data(), xf.data(), xf.size() );
		for ( auto& w : wts ) w = 0.6f;
		Pool pool( 3, {} );
		Scratch sc;
		expert_forward( pool, ex, rows, xh.data(), wts.data(), out.data(), sc );
		std::vector<double> ref( out.size() );
		for ( int r = 0; r < rows; ++r ) {
			std::vector<double> xd( xf.begin() + static_cast<size_t>( r ) * H, xf.begin() + static_cast<size_t>( r + 1 ) * H );
			auto g = matvec( wg, xd, H, I ), u = matvec( wu, xd, H, I );
			std::vector<double> a( I );
			for ( int i = 0; i < I; ++i ) { const double gv = std::min( g[i], 10.0 ), uv = std::max( -10.0, std::min( u[i], 10.0 ) ); a[i] = gv / ( 1.0 + std::exp( -gv ) ) * uv; }
			const auto d = matvec( wd, a, I, H );
			for ( int i = 0; i < H; ++i ) ref[static_cast<size_t>( r ) * H + i] = wts[r] * d[i];
		}
		const double e = rel_err( out.data(), ref.data(), out.size() );
		CHECK( e < 4e-3, "nvfp4 expert, %d rows: relative error %.3g", rows, e );
		std::printf( "  expert rows %d: relative error %.3g\n", rows, e );
	}
}

// Experts evaluated together give the same sums as one after the other
static void part3_many() {
	const int H = 512, I = 256, E = 6;
	std::vector<Fp4Mat> m;
	m.reserve( 3 * E );
	std::vector<std::unique_ptr<Nvfp4Linear>> lin;
	for ( int e = 0; e < E; ++e ) {
		m.push_back( make_fp4( H, I ) ); m.push_back( make_fp4( H, I ) ); m.push_back( make_fp4( I, H ) );
		lin.emplace_back( new Nvfp4Linear( m[3 * e].codes.data(), m[3 * e].scales.data(), m[3 * e].gs, H, I ) );
		lin.emplace_back( new Nvfp4Linear( m[3 * e + 1].codes.data(), m[3 * e + 1].scales.data(), m[3 * e + 1].gs, H, I ) );
		lin.emplace_back( new Nvfp4Linear( m[3 * e + 2].codes.data(), m[3 * e + 2].scales.data(), m[3 * e + 2].gs, I, H ) );
	}
	std::vector<std::vector<uint16_t>> xh( E );
	std::vector<std::vector<float>> w( E ), one( E ), many( E );
	std::vector<ExpertCall> calls;
	Pool pool( 4, {} );
	Scratch sc;
	for ( int e = 0; e < E; ++e ) {
		const int rows = 1 + e % 3;
		std::vector<float> xf( static_cast<size_t>( rows ) * H );
		for ( auto& v : xf ) v = rnd();
		xh[e].resize( xf.size() );
		float_to_half( xf.data(), xh[e].data(), xf.size() );
		w[e].assign( rows, 0.5f );
		one[e].assign( xf.size(), 0.0f ); many[e].assign( xf.size(), 0.0f );
		Expert ex; ex.gate = lin[3 * e].get(); ex.up = lin[3 * e + 1].get(); ex.down = lin[3 * e + 2].get();
		expert_forward( pool, ex, rows, xh[e].data(), w[e].data(), one[e].data(), sc, 0.0f );
		calls.push_back( ExpertCall{ ex, rows, xh[e].data(), w[e].data(), many[e].data() } );
	}
	for ( int rep = 0; rep < 50; ++rep ) {
		for ( auto& v : many ) std::fill( v.begin(), v.end(), 0.0f );
		expert_forward_many( pool, calls.data(), E, sc, 0.0f );
		for ( int e = 0; e < E; ++e ) CHECK( one[e] == many[e], "many: expert %d differs from the single call (rep %d)", e, rep );
	}
	std::printf( "many: done\n" );
}

static void part3_speed() {
	const int H = 4096, I = 2048;
	Fp4Mat mg = make_fp4( H, I ), mu = make_fp4( H, I ), md = make_fp4( I, H );
	Nvfp4Linear lg( mg.codes.data(), mg.scales.data(), mg.gs, H, I ), lu( mu.codes.data(), mu.scales.data(), mu.gs, H, I ), ld( md.codes.data(), md.scales.data(), md.gs, I, H );
	Expert ex; ex.gate = &lg; ex.up = &lu; ex.down = &ld;
	for ( int thr : { 1, 5, 10 } ) {
		std::vector<int> cpus;
		for ( int t = 1; t < thr; ++t ) cpus.push_back( 10 + t );
		Pool pool( thr, cpus );
		Scratch sc;
		std::vector<uint16_t> xh( H, 0x3c00 );
		std::vector<float> out( H ), wts( 1, 1.0f );
		for ( int i = 0; i < 3; ++i ) expert_forward( pool, ex, 1, xh.data(), wts.data(), out.data(), sc );
		const auto t0 = std::chrono::steady_clock::now();
		for ( int i = 0; i < 20; ++i ) expert_forward( pool, ex, 1, xh.data(), wts.data(), out.data(), sc );
		const double ms = std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t0 ).count() / 20;
		const double bytes = 3.0 * H * I * 0.5625;
		std::printf( "  nvfp4 %2d threads: %.3f ms per expert (1 row), %.2f GB/s of codes and scales\n", thr, ms, bytes / ms * 1e-6 );
	}
}

// Workers that park between tasks must still wake for the next one (a missed wake-up hangs the run), and a task finishes without the workers that
// never came back
struct CountTask : Task {
	std::atomic<int> next{ 0 }, done{ 0 };
	bool work() override { if ( next.fetch_add( 1 ) >= 50 ) return false; done.fetch_add( 1 ); return true; }
	bool finished() const override { return done.load() >= 50; }
};
static void pool_stress() {
	Pool pool( 6, {}, 20 );
	long total = 0;
	std::vector<CountTask> tasks( 8 );   // tasks outlive the call: a worker may still look at the last one
	for ( int i = 0; i < 20000; ++i ) {
		CountTask& t = tasks[i % 8];
		while ( t.refs.load() != 0 ) {}   // nobody holds it
		t.next = 0; t.done = 0;
		pool.run( t );
		total += t.done.load();
		if ( i % 7 == 0 ) { const auto t0 = std::chrono::steady_clock::now(); while ( std::chrono::steady_clock::now() - t0 < std::chrono::microseconds( 20 + i % 90 ) ) {} }
	}
	CHECK( total == 20000L * 50, "pool: %ld", total );
	std::printf( "pool stress done\n" );
}

// Many distinct experts (more than the L3 holds): the bytes come from DRAM as in serving. Run with CPUS="a,b,c" (the pool's CPUs, the first is the caller;
// combine with numactl --membind) to time that instead of the tests
static void part3_stream() {
	const int H = 2560, I = 640, N = 160;
	std::vector<Fp4Mat> keep;
	keep.reserve( N * 3 );
	std::vector<std::unique_ptr<Nvfp4Linear>> lin;
	for ( int i = 0; i < N * 3; ++i ) {
		const bool dn = i % 3 == 2;
		keep.push_back( make_fp4( dn ? I : H, dn ? H : I ) );
		lin.emplace_back( new Nvfp4Linear( keep.back().codes.data(), keep.back().scales.data(), keep.back().gs, dn ? I : H, dn ? H : I ) );
	}
	std::vector<int> all;
	{ std::string c = std::getenv( "CPUS" ); size_t p0 = 0; while ( p0 < c.size() ) { size_t q = c.find( ',', p0 ); if ( q == std::string::npos ) q = c.size(); all.push_back( std::atoi( c.substr( p0, q - p0 ).c_str() ) ); p0 = q + 1; } }
	std::vector<int> rest( all.begin() + 1, all.end() );
	Pool pool( static_cast<int>( all.size() ), rest );
	cpu_set_t cs; CPU_ZERO( &cs ); CPU_SET( all[0], &cs ); sched_setaffinity( 0, sizeof cs, &cs );
	Scratch sc;
	std::vector<uint16_t> xh( H, 0x3c00 );
	std::vector<float> out( H ), wts( 1, 1.0f );
	double ms = 0; int cnt = 0;
	for ( int pass = 0; pass < 3; ++pass ) {
		const auto t0 = std::chrono::steady_clock::now();
		for ( int e = 0; e < N; ++e ) {
			Expert ex; ex.gate = lin[3 * e].get(); ex.up = lin[3 * e + 1].get(); ex.down = lin[3 * e + 2].get();
			expert_forward( pool, ex, 1, xh.data(), wts.data(), out.data(), sc, 0.0f );
		}
		if ( pass > 0 ) { ms += std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t0 ).count(); cnt += N; }
	}
	std::printf( "  stream: %zu threads: %.3f ms per expert (2.8 MB codes+scales each: %.2f GB/s)\n", all.size(), ms / cnt, 3.0 * H * I * 0.5625 / ( ms / cnt ) * 1e-6 );
}

int main( int argc, char** argv ) {
	if ( std::getenv( "CPUS" ) ) { part3_stream(); return 0; }
	pool_stress();
	part1();
	part3();
	part3_many();
	part3_speed();
	const std::string dir = argc > 1 ? argv[1] : "/home/mrq/Documents/models/GLM-5.3-Flash-exl3-3.05bpw";
	part2( dir, argc > 2 ? std::atoi( argv[2] ) : 10, argc > 3 ? std::atoi( argv[3] ) : 0 );
	std::printf( g_fail ? "%d FAILED\n" : "all passed\n", g_fail );
	return g_fail ? 1 : 0;
}
