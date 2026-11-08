// The few ggml runtime symbols the vendored CPU quantization files (ggml/) use, without the rest of ggml: the abort, the conversion tables
// ggml-cpu.c fills at init (filled here before main), and the block geometry of the types they read.
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml-impl.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

float ggml_table_f32_f16[1 << 16];
float ggml_table_f32_e8m0_half[1 << 8];
float ggml_table_f32_ue4m3[1 << 8];
float ggml_table_f32_f8_e4m3[1 << 8];

__attribute__(( constructor )) static void ggml_shim_tables( void ) {
	for ( int i = 0; i < ( 1 << 16 ); ++i ) {
		union { uint16_t u16; ggml_fp16_t fp16; } u = { (uint16_t) i };
		ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32( u.fp16 );
	}
	for ( int i = 0; i < ( 1 << 8 ); ++i ) {
		ggml_table_f32_e8m0_half[i] = GGML_E8M0_TO_FP32_HALF( i );
		ggml_table_f32_ue4m3[i] = ggml_ue4m3_to_fp32( (uint8_t) i );
		ggml_table_f32_f8_e4m3[i] = ggml_f8_e4m3_to_fp32( (uint8_t) i );
	}
}

void ggml_abort( const char* file, int line, const char* fmt, ... ) {
	fprintf( stderr, "[ggml] %s:%d: ", file, line );
	va_list ap;
	va_start( ap, fmt );
	vfprintf( stderr, fmt, ap );
	va_end( ap );
	fputc( '\n', stderr );
	abort();
}

// Values and bytes per block of the types the vendored files name (the others: 0)
static void geometry( enum ggml_type t, int64_t* elems, size_t* bytes ) {
	*elems = 0;
	*bytes = 0;
	switch ( t ) {
#define G( T, E, B ) case T: *elems = E; *bytes = B; break;
		G( GGML_TYPE_F32, 1, 4 ) G( GGML_TYPE_F16, 1, 2 ) G( GGML_TYPE_BF16, 1, 2 )
		G( GGML_TYPE_Q4_0, QK4_0, sizeof( block_q4_0 ) ) G( GGML_TYPE_Q4_1, QK4_1, sizeof( block_q4_1 ) )
		G( GGML_TYPE_Q5_0, QK5_0, sizeof( block_q5_0 ) ) G( GGML_TYPE_Q5_1, QK5_1, sizeof( block_q5_1 ) )
		G( GGML_TYPE_Q8_0, QK8_0, sizeof( block_q8_0 ) ) G( GGML_TYPE_Q8_1, QK8_1, sizeof( block_q8_1 ) )
		G( GGML_TYPE_Q2_0, QK2_0, sizeof( block_q2_0 ) )
		G( GGML_TYPE_Q2_K, QK_K, sizeof( block_q2_K ) ) G( GGML_TYPE_Q3_K, QK_K, sizeof( block_q3_K ) ) G( GGML_TYPE_Q4_K, QK_K, sizeof( block_q4_K ) )
		G( GGML_TYPE_Q5_K, QK_K, sizeof( block_q5_K ) ) G( GGML_TYPE_Q6_K, QK_K, sizeof( block_q6_K ) ) G( GGML_TYPE_Q8_K, QK_K, sizeof( block_q8_K ) )
		G( GGML_TYPE_IQ2_XXS, QK_K, sizeof( block_iq2_xxs ) ) G( GGML_TYPE_IQ2_XS, QK_K, sizeof( block_iq2_xs ) ) G( GGML_TYPE_IQ2_S, QK_K, sizeof( block_iq2_s ) )
		G( GGML_TYPE_IQ3_XXS, QK_K, sizeof( block_iq3_xxs ) ) G( GGML_TYPE_IQ3_S, QK_K, sizeof( block_iq3_s ) )
		G( GGML_TYPE_IQ1_S, QK_K, sizeof( block_iq1_s ) ) G( GGML_TYPE_IQ1_M, QK_K, sizeof( block_iq1_m ) )
		G( GGML_TYPE_IQ4_NL, QK4_NL, sizeof( block_iq4_nl ) ) G( GGML_TYPE_IQ4_XS, QK_K, sizeof( block_iq4_xs ) )
#undef G
		default: break;
	}
}

int64_t ggml_blck_size( enum ggml_type type ) {
	int64_t e;
	size_t b;
	geometry( type, &e, &b );
	return e;
}

size_t ggml_type_size( enum ggml_type type ) {
	int64_t e;
	size_t b;
	geometry( type, &e, &b );
	return b;
}

size_t ggml_row_size( enum ggml_type type, int64_t ne ) {
	int64_t e;
	size_t b;
	geometry( type, &e, &b );
	return e > 0 ? b * (size_t) ( ne / e ) : 0;
}

const char* ggml_type_name( enum ggml_type type ) {
	static char buf[16];
	snprintf( buf, sizeof( buf ), "type %d", (int) type );
	return buf;
}
