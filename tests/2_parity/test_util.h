#pragma once

#include <fstream>
#include <iterator>
#include <string>
#ifdef __CUDA_FP16_H__
	#define TEST_UTIL_HAS_HALF 1
#endif

// Helpers the parity tests share: whole-file reads and (when <cuda_fp16.h> is included first) half <-> float conversions

inline std::string read_file( const std::string& path ) {
	std::ifstream f( path, std::ios::binary );
	return std::string( std::istreambuf_iterator<char>( f ), std::istreambuf_iterator<char>() );
}

#ifdef TEST_UTIL_HAS_HALF
inline float half_to_float( half h ) { return __half2float( h ); }
inline half float_to_half( float f ) { return __float2half_rn( f ); }
#endif
