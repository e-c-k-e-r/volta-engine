#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/stat.h>

// Cross-backend golden files for op-level tests. The module API is backend-neutral, so ONE test source builds in the CUDA and
// the CPU configurations: `--dump <dir>` (run it in the CUDA build) writes inputs' outputs as golden files, `--check <dir>` (run
// it in any build, in particular the CPU one) recomputes and compares against them. Inputs are generated from fixed seeds
// with the portable generator below, so both builds see identical data without storing it.
//
//   golden_rng:   deterministic LCG (same sequence everywhere)
//   golden_write: <dir>/<name>.bin, raw bytes
//   golden_read:  false when the file is missing
//   golden_compare: tier 2 tolerance (cos >= 0.9999, max|d| <= 0.005 * max + 1e-3) on float data

struct GoldenRng {
	uint32_t state;
	explicit GoldenRng( uint32_t seed ) : state( seed ) {}
	uint32_t next() { state = state * 1664525u + 1013904223u; return state >> 8; }
	float uniform( float scale = 1.0f ) { return ( static_cast<int>( next() % 20001u ) - 10000 ) / 10000.0f * scale; }
};

struct GoldenArgs {
	std::string dir = "/tmp/volta_golden";
	bool dump = false;
	bool check = false;
	bool gpu = false;
	GoldenArgs( int argc, char** argv ) {
		for ( int i = 1; i < argc; ++i ) {
			if ( !std::strcmp( argv[i], "--dump" ) ) { dump = true; if ( i + 1 < argc && argv[i + 1][0] != '-' ) dir = argv[++i]; }
			else if ( !std::strcmp( argv[i], "--check" ) ) { check = true; if ( i + 1 < argc && argv[i + 1][0] != '-' ) dir = argv[++i]; }
			else if ( !std::strcmp( argv[i], "--gpu" ) ) gpu = true;
		}
		if ( dump ) mkdir( dir.c_str(), 0755 );
	}
};

inline bool golden_write( const std::string& dir, const std::string& name, const void* data, size_t bytes ) {
	std::FILE* f = std::fopen( ( dir + "/" + name + ".bin" ).c_str(), "wb" );
	if ( !f ) return false;
	const bool ok = std::fwrite( data, 1, bytes, f ) == bytes;
	std::fclose( f );
	return ok;
}

inline bool golden_read( const std::string& dir, const std::string& name, std::vector<uint8_t>& out ) {
	std::FILE* f = std::fopen( ( dir + "/" + name + ".bin" ).c_str(), "rb" );
	if ( !f ) return false;
	std::fseek( f, 0, SEEK_END );
	const long n = std::ftell( f );
	std::fseek( f, 0, SEEK_SET );
	out.resize( static_cast<size_t>( n ) );
	const bool ok = std::fread( out.data(), 1, out.size(), f ) == out.size();
	std::fclose( f );
	return ok;
}

inline bool golden_compare( const char* name, const std::vector<float>& ref, const std::vector<float>& got ) {
	if ( ref.size() != got.size() ) { std::printf( "  %-28s size mismatch %zu vs %zu  FAIL\n", name, ref.size(), got.size() ); return false; }
	double dot = 0, na = 0, nb = 0, maxd = 0, maxabs = 0;
	for ( size_t i = 0; i < ref.size(); ++i ) {
		dot += static_cast<double>( ref[i] ) * got[i]; na += static_cast<double>( ref[i] ) * ref[i]; nb += static_cast<double>( got[i] ) * got[i];
		maxd = std::fmax( maxd, std::fabs( static_cast<double>( ref[i] ) - got[i] ) );
		maxabs = std::fmax( maxabs, std::fabs( static_cast<double>( ref[i] ) ) );
	}
	const double cosv = dot / ( std::sqrt( na ) * std::sqrt( nb ) + 1e-30 );
	const bool ok = cosv >= 0.9999 && maxd <= 0.005 * maxabs + 1e-3;
	std::printf( "  %-28s cos %.7f max|d| %.3g (max %.3g)  %s\n", name, cosv, maxd, maxabs, ok ? "PASS" : "FAIL" );
	return ok;
}
