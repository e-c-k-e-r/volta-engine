// Tier 2: the EXL3 trellis encoder (engine/quant/exl3_encode.h) on i.i.d. Gaussian tiles, every width code and the three codebooks.
//   (a) error: the best normalized MSE over a scan of input scales, as a ratio to the Gaussian rate-distortion bound 2^(-2R); a tail-biting
//       trellis with a 16-bit window sits about 10% above it at 2 bits (exllamav3 / QTIP)
//   (b) decode: the packed windows read back by reconstruct_slice in the stock and the Volta tile order equal the encoder's own decoded
//       values, element for element (both use the inference codebook)
//   (c) weighted metric: weights of 1 give the plain path's windows bit for bit; per-row weights from 0.25 to 4 give a lower weighted squared
//       error than the plain path's
//   (d) open trellis (a free start state): its error is at most the tail-biting path's (it searches a superset of paths)
//   (e) extended trellis: the open trellis' packed codes with each tile's start state (its first window's bits before the stream) decode through
//       reconstruct_slice_ext in both tile orders to the encoder's values bit for bit, and with random tile gains to those values times the gain
#include "engine/config.h"
#include "engine/quant/exl3_encode.h"
#include "engine/modules/trellis.h"
#include "engine/modules/trellis_bits.h"
#include "reconstruct.cuh"
#include "trellis_context.h"
#include "exl3_vl.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

struct Case {
	int code;
	int cb;
	double max_ratio;   // normalized MSE over the bound
};

double rate( int code ) { return engine::modules::trellis_k2( code ) / 2.0; }

const char* cb_name( int cb ) { return cb == 0 ? "3inst" : cb == 1 ? "mcg" : "mul1"; }

// Normalized MSE of encoding s * x, decoded / s against x
double nmse( const std::vector<float>& x, const std::vector<float>& dec, float s ) {
	double e = 0.0;
	for ( size_t i = 0; i < x.size(); ++i ) {
		const double d = dec[i] / s - x[i];
		e += d * d;
	}
	return e / static_cast<double>( x.size() );
}

int run_case( const Case& cs, bool quick, bool volta_ok ) {
	const int tk = quick ? 8 : 32, tn = 16;
	const int k = 16 * tk, n = 16 * tn, tiles = tk * tn;
	const size_t count = static_cast<size_t>( tiles ) * 256;
	const int words = engine::modules::trellis_tile_u16( cs.code );
	const bool mcg = cs.cb == 1, mul1 = cs.cb == 2;

	std::mt19937 rng( 4321u + cs.code * 13u + cs.cb );
	std::normal_distribution<float> gauss( 0.0f, 1.0f );
	std::vector<float> x( count ), xs( count ), dec( count ), best_dec( count );
	for ( auto& v : x ) v = gauss( rng );

	float *d_in = nullptr, *d_dec = nullptr;
	uint16_t *d_win = nullptr, *d_packed = nullptr;
	half* d_w = nullptr;
	CUDA_CHECK( cudaMalloc( &d_in, count * sizeof( float ) ) );
	CUDA_CHECK( cudaMalloc( &d_dec, count * sizeof( float ) ) );
	CUDA_CHECK( cudaMalloc( &d_win, count * sizeof( uint16_t ) ) );
	CUDA_CHECK( cudaMalloc( &d_packed, static_cast<size_t>( tiles ) * words * sizeof( uint16_t ) ) );
	CUDA_CHECK( cudaMalloc( &d_w, static_cast<size_t>( k ) * n * sizeof( half ) ) );
	engine::quant::Exl3EncodeScratch scratch;
	ENGINE_ASSERT( engine::quant::exl3_encode_scratch_alloc( scratch, cs.code, cs.cb ), "test_exl3_encode: no memory for the encoder scratch" );

	int encodes = 0;
	auto encode = [&]( float s ) {
		for ( size_t i = 0; i < count; ++i ) xs[i] = s * x[i];
		CUDA_CHECK( cudaMemcpy( d_in, xs.data(), count * sizeof( float ), cudaMemcpyHostToDevice ) );
		engine::quant::exl3_encode_tiles( d_in, d_dec, d_win, tiles, cs.code, cs.cb, scratch, nullptr );
		CUDA_CHECK( cudaMemcpy( dec.data(), d_dec, count * sizeof( float ), cudaMemcpyDeviceToHost ) );
		++encodes;
		return nmse( x, dec, s );
	};

	// Coarse scan of the input scale, then a fine one around the best
	const auto t0 = std::chrono::steady_clock::now();
	float best_s = 1.0f;
	double best = 1e30;
	for ( float s = 0.5f; s <= 2.01f; s += 0.1f ) {
		const double e = encode( s );
		if ( e < best ) best = e, best_s = s;
	}
	const float centre = best_s;
	for ( int j = -5; j <= 5; ++j ) {
		const float s = centre + 0.02f * j;
		if ( s <= 0.0f ) continue;
		const double e = encode( s );
		if ( e < best ) best = e, best_s = s;
	}
	const double secs = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
	// Keep the windows of the best scale
	encode( best_s );
	best_dec = dec;

	const double bound = std::pow( 2.0, -2.0 * rate( cs.code ) );
	const double ratio = best / bound;
	int fails = 0;
	const bool err_ok = ratio >= 1.0 && ratio <= cs.max_ratio;
	fails += !err_ok;
	std::printf( "  width %2d (%.1f bpw) %-5s  nmse %.6f  bound %.6f  ratio %.4f (max %.2f) at scale %.2f  %.0f tiles/s  %s\n", cs.code,
		rate( cs.code ), cb_name( cs.cb ), best, bound, ratio, cs.max_ratio, best_s, encodes * tiles / secs, err_ok ? "OK" : "FAIL" );

	// Pack, then read back through the inference decode in both orders
	engine::quant::exl3_pack_tiles( d_packed, d_win, tiles, cs.code, nullptr );
	std::vector<half> w( static_cast<size_t>( k ) * n );
	for ( int order = 0; order < ( volta_ok ? 2 : 1 ); ++order ) {
		const bool volta = order == 1;
		EXL3_TILES_CALL( volta, reconstruct_slice, d_w, d_packed, cs.code, mcg, mul1, tk, n, 0, nullptr );
		CUDA_CHECK( cudaMemcpy( w.data(), d_w, w.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
		size_t mism = 0;
		for ( int t = 0; t < tiles; ++t ) {
			const int kt = t / tn, nt = t % tn;
			for ( int p = 0; p < 256; ++p ) {
				int r = 0, c = 0;
				engine::quant::exl3_tile_pos( volta, p, &r, &c );
				const half want = __float2half( best_dec[static_cast<size_t>( t ) * 256 + p] );
				const half got = w[static_cast<size_t>( kt * 16 + r ) * n + nt * 16 + c];
				uint16_t a = 0, b = 0;
				std::memcpy( &a, &want, 2 );
				std::memcpy( &b, &got, 2 );
				mism += a != b;
			}
		}
		std::printf( "  width %2d %-5s  %s order: %zu of %zu decoded elements differ from the encoder's %s\n", cs.code, cb_name( cs.cb ),
			volta ? "volta" : "stock", mism, count, mism ? "FAIL" : "OK" );
		fails += mism != 0;
	}

	engine::quant::exl3_encode_scratch_free( scratch );
	CUDA_CHECK( cudaFree( d_in ) );
	CUDA_CHECK( cudaFree( d_dec ) );
	CUDA_CHECK( cudaFree( d_win ) );
	CUDA_CHECK( cudaFree( d_packed ) );
	CUDA_CHECK( cudaFree( d_w ) );
	return fails;
}

// (c) at one width: the plain encode, all-ones weights, and per-row weights (a tile row's weight, as LDLQ gives them)
int run_weighted( int code, bool quick ) {
	const int tiles = quick ? 128 : 512;
	const size_t count = static_cast<size_t>( tiles ) * 256;
	std::mt19937 rng( 977u + code );
	std::normal_distribution<float> gauss( 0.0f, 1.0f );
	std::uniform_real_distribution<float> logw( std::log( 0.25f ), std::log( 4.0f ) );
	std::vector<float> x( count ), ones( count, 1.0f ), w( count );
	for ( auto& v : x ) v = gauss( rng );
	for ( int t = 0; t < tiles; ++t ) {
		float row_w[16];
		for ( auto& v : row_w ) v = std::exp( logw( rng ) );
		for ( int p = 0; p < 256; ++p ) {
			int r = 0, c = 0;
			engine::quant::exl3_tile_pos( true, p, &r, &c );
			w[static_cast<size_t>( t ) * 256 + p] = row_w[r];
		}
	}
	float *d_in = nullptr, *d_dec = nullptr, *d_w = nullptr;
	uint16_t* d_win = nullptr;
	CUDA_CHECK( cudaMalloc( &d_in, count * sizeof( float ) ) );
	CUDA_CHECK( cudaMalloc( &d_dec, count * sizeof( float ) ) );
	CUDA_CHECK( cudaMalloc( &d_w, count * sizeof( float ) ) );
	CUDA_CHECK( cudaMalloc( &d_win, count * sizeof( uint16_t ) ) );
	CUDA_CHECK( cudaMemcpy( d_in, x.data(), count * sizeof( float ), cudaMemcpyHostToDevice ) );
	engine::quant::Exl3EncodeScratch scratch;
	ENGINE_ASSERT( engine::quant::exl3_encode_scratch_alloc( scratch, code, 2 ), "test_exl3_encode: no memory for the encoder scratch" );
	std::vector<float> dec( count );
	std::vector<uint16_t> win_plain( count ), win( count );
	auto encode = [&]( const std::vector<float>* weights, std::vector<uint16_t>& windows ) {
		if ( weights ) CUDA_CHECK( cudaMemcpy( d_w, weights->data(), count * sizeof( float ), cudaMemcpyHostToDevice ) );
		engine::quant::exl3_encode_tiles( d_in, d_dec, d_win, tiles, code, 2, scratch, nullptr, weights ? d_w : nullptr );
		CUDA_CHECK( cudaMemcpy( dec.data(), d_dec, count * sizeof( float ), cudaMemcpyDeviceToHost ) );
		CUDA_CHECK( cudaMemcpy( windows.data(), d_win, count * sizeof( uint16_t ), cudaMemcpyDeviceToHost ) );
		double e = 0.0;
		for ( size_t i = 0; i < count; ++i ) e += w[i] * ( dec[i] - x[i] ) * ( dec[i] - x[i] );
		return e / static_cast<double>( count );
	};
	const double e_plain = encode( nullptr, win_plain );
	encode( &ones, win );
	size_t diff = 0;
	for ( size_t i = 0; i < count; ++i ) diff += win[i] != win_plain[i];
	const double e_weighted = encode( &w, win );
	// (d) the open trellis against the tail-biting one, plain metric
	auto plain_err = [&]( bool open ) {
		engine::quant::exl3_encode_tiles( d_in, d_dec, d_win, tiles, code, 2, scratch, nullptr, nullptr, open );
		CUDA_CHECK( cudaMemcpy( dec.data(), d_dec, count * sizeof( float ), cudaMemcpyDeviceToHost ) );
		double e = 0.0;
		for ( size_t i = 0; i < count; ++i ) e += ( dec[i] - x[i] ) * ( dec[i] - x[i] );
		return e / static_cast<double>( count );
	};
	const double e_ring = plain_err( false ), e_open = plain_err( true );
	int fails = 0;
	std::printf( "  width %2d mul1   weights of 1: %zu of %zu windows differ from the plain path's %s\n", code, diff, count, diff ? "FAIL" : "OK" );
	fails += diff != 0;
	const bool lower = e_weighted < e_plain;
	std::printf( "  width %2d mul1   per-row weights 0.25 .. 4: weighted error %.6f, plain path %.6f (%.1f%%) %s\n", code, e_weighted, e_plain,
		100.0 * ( e_weighted / e_plain - 1.0 ), lower ? "OK" : "FAIL" );
	fails += !lower;
	const bool open_ok = e_open <= e_ring * ( 1.0 + 1e-6 );
	std::printf( "  width %2d mul1   open trellis: error %.6f, tail-biting %.6f (%+.2f%%) %s\n", code, e_open, e_ring, 100.0 * ( e_open / e_ring - 1.0 ),
		open_ok ? "OK" : "FAIL" );
	fails += !open_ok;
	engine::quant::exl3_encode_scratch_free( scratch );
	CUDA_CHECK( cudaFree( d_in ) );
	CUDA_CHECK( cudaFree( d_dec ) );
	CUDA_CHECK( cudaFree( d_w ) );
	CUDA_CHECK( cudaFree( d_win ) );
	return fails;
}

// (e) at one width: open-trellis windows, packed, decoded with their start states, then with gains as well
int run_extended( int code, bool volta_ok ) {
	const int tk = 8, tn = 16, k = 16 * tk, n = 16 * tn, tiles = tk * tn;
	const size_t count = static_cast<size_t>( tiles ) * 256;
	const int words = engine::modules::trellis_tile_u16( code );
	const int d0 = engine::modules::trellis_base_bits( code );
	std::mt19937 rng( 555u + code );
	std::normal_distribution<float> gauss( 0.0f, 1.0f );
	std::vector<float> x( count ), dec( count );
	for ( auto& v : x ) v = gauss( rng );
	float *d_in = nullptr, *d_dec = nullptr;
	uint16_t *d_win = nullptr, *d_packed = nullptr, *d_starts = nullptr;
	uint8_t* d_gains = nullptr;
	half* d_w = nullptr;
	CUDA_CHECK( cudaMalloc( &d_in, count * sizeof( float ) ) );
	CUDA_CHECK( cudaMalloc( &d_dec, count * sizeof( float ) ) );
	CUDA_CHECK( cudaMalloc( &d_win, count * sizeof( uint16_t ) ) );
	CUDA_CHECK( cudaMalloc( &d_packed, static_cast<size_t>( tiles ) * words * sizeof( uint16_t ) ) );
	CUDA_CHECK( cudaMalloc( &d_starts, tiles * sizeof( uint16_t ) ) );
	CUDA_CHECK( cudaMalloc( &d_gains, tiles ) );
	CUDA_CHECK( cudaMalloc( &d_w, static_cast<size_t>( k ) * n * sizeof( half ) ) );
	CUDA_CHECK( cudaMemcpy( d_in, x.data(), count * sizeof( float ), cudaMemcpyHostToDevice ) );
	engine::quant::Exl3EncodeScratch scratch;
	ENGINE_ASSERT( engine::quant::exl3_encode_scratch_alloc( scratch, code, 2 ), "test_exl3_encode: no memory for the encoder scratch" );
	engine::quant::exl3_encode_tiles( d_in, d_dec, d_win, tiles, code, 2, scratch, nullptr, nullptr, true );
	engine::quant::exl3_pack_tiles( d_packed, d_win, tiles, code, nullptr );
	std::vector<uint16_t> win( count ), starts( tiles );
	CUDA_CHECK( cudaMemcpy( dec.data(), d_dec, count * sizeof( float ), cudaMemcpyDeviceToHost ) );
	CUDA_CHECK( cudaMemcpy( win.data(), d_win, count * sizeof( uint16_t ), cudaMemcpyDeviceToHost ) );
	for ( int t = 0; t < tiles; ++t ) starts[t] = static_cast<uint16_t>( win[static_cast<size_t>( t ) * 256] >> d0 );
	CUDA_CHECK( cudaMemcpy( d_starts, starts.data(), tiles * sizeof( uint16_t ), cudaMemcpyHostToDevice ) );
	std::vector<uint8_t> gains( tiles );
	std::uniform_int_distribution<int> gd( 0, 255 );
	for ( auto& g : gains ) g = static_cast<uint8_t>( gd( rng ) );
	CUDA_CHECK( cudaMemcpy( d_gains, gains.data(), tiles, cudaMemcpyHostToDevice ) );
	int fails = 0;
	std::vector<half> w( static_cast<size_t>( k ) * n );
	for ( int order = 0; order < ( volta_ok ? 2 : 1 ); ++order ) {
		const bool volta = order == 1;
		for ( int gained = 0; gained < 2; ++gained ) {
			EXL3_TILES_CALL( volta, reconstruct_slice_ext, d_w, d_packed, d_starts, gained ? d_gains : nullptr, code, false, true, tk, n, 0, nullptr );
			CUDA_CHECK( cudaMemcpy( w.data(), d_w, w.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
			size_t mism = 0;
			for ( int t = 0; t < tiles; ++t ) {
				const int kt = t / tn, nt = t % tn;
				const float g = gained ? ( 896.0f + gains[t] ) / 1024.0f : 1.0f;
				for ( int p = 0; p < 256; ++p ) {
					int r = 0, c = 0;
					engine::quant::exl3_tile_pos( volta, p, &r, &c );
					const half want = __float2half( __half2float( __float2half( dec[static_cast<size_t>( t ) * 256 + p] ) ) * g );
					const half got = w[static_cast<size_t>( kt * 16 + r ) * n + nt * 16 + c];
					uint16_t a = 0, b = 0;
					std::memcpy( &a, &want, 2 );
					std::memcpy( &b, &got, 2 );
					mism += a != b;
				}
			}
			std::printf( "  width %2d mul1   extended, %s order, start states%s: %zu of %zu decoded elements differ %s\n", code, volta ? "volta" : "stock",
				gained ? " and gains" : "", mism, count, mism ? "FAIL" : "OK" );
			fails += mism != 0;
		}
	}
	engine::quant::exl3_encode_scratch_free( scratch );
	for ( void* p : { (void*) d_in, (void*) d_dec, (void*) d_win, (void*) d_packed, (void*) d_starts, (void*) d_gains, (void*) d_w } ) CUDA_CHECK( cudaFree( p ) );
	return fails;
}

} // namespace

int main( int argc, char** argv ) {
	const bool quick = argc > 1 && !std::strcmp( argv[1], "--quick" );
	engine::modules::trellis_context_init_all();
#if ENGINE_EXL3_VL
	const bool volta_ok = engine::modules::trellis_volta_tiles_supported();
#else
	const bool volta_ok = false;
#endif
	std::vector<Case> cases;
	if ( quick ) {
		cases = { { 2, 2, 1.20 }, { 10, 2, 1.30 } };
	} else {
		// From 7 bits on, the mul1 codebook's ~1021 output levels (a byte sum times 0x1eee) add a floor of step^2 / 12 that is a sizable share
		// of the error
		for ( int b = 1; b <= 8; ++b ) cases.push_back( { b, 2, b == 1 ? 1.60 : b >= 7 ? 1.45 : 1.30 } );
		for ( int b : { 9, 10, 11 } ) cases.push_back( { b, 2, 1.40 } );
		for ( int b : { 2, 4 } ) cases.push_back( { b, 1, 1.30 } );
		cases.push_back( { 3, 0, 1.30 } );
	}
	std::printf( "[*] test_exl3_encode: %s, %s\n", quick ? "quick" : "full", volta_ok ? "stock and Volta order" : "stock order only" );
	int fails = 0;
	for ( const Case& c : cases ) fails += run_case( c, quick, volta_ok );
	for ( int code : quick ? std::vector<int>{ 2 } : std::vector<int>{ 1, 2, 4, 9, 10 } ) fails += run_weighted( code, quick );
	for ( int code : quick ? std::vector<int>{ 2 } : std::vector<int>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 } ) fails += run_extended( code, volta_ok );
	std::printf( fails ? "=== EXL3 ENCODE: %d FAILURE(S) ===\n" : "=== EXL3 ENCODE: PASS ===\n", fails );
	return fails ? 1 : 0;
}
