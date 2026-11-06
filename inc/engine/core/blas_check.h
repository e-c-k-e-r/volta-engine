#pragma once

#include "engine/config.h"
#include <cstdio>
#include <cstdlib>

// Aborts on a cuBLAS error (the caller includes <cublas_v2.h>)
#define CUBLAS_CHECK( call ) \
	do { \
		cublasStatus_t st_ = ( call ); \
		if ( ENGINE_UNLIKELY( st_ != CUBLAS_STATUS_SUCCESS ) ) { \
			std::fprintf( stderr, "[CUBLAS ERROR] %s (%s:%d): %d\n", #call, __FILE__, __LINE__, static_cast<int>( st_ ) ); \
			std::_Exit( 1 ); \
		} \
	} while ( 0 )
