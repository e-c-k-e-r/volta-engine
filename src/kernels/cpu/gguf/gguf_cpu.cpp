// The GGUF block kernels of the CPU backend, a thin layer over llama.cpp's own CPU kernels (vendored in ggml/: the generic and x86 dot
// products of ggml-cpu/quants.c and arch/x86/quants.c, the block decoders of ggml-quants.c), kept apart from the backend's Slang prelude
// (both define int4)
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-cpu/quants.h"

#include "gguf_cpu.h"

namespace engine::modules::gguf_cpu {

namespace {

using VecDot = void ( * )( int n, float* s, size_t bs, const void* x, size_t bx, const void* y, size_t by, int nrc );
using FromFloat = void ( * )( const float* x, void* y, int64_t k );
using ToFloat = void ( * )( const void* x, float* y, int64_t k );

// A type's dot product, the activation type it pairs with and the block decoder (ggml-cpu.c's type_traits_cpu)
struct CpuType {
	VecDot dot = nullptr;
	GgmlType act = GgmlType::F32;
	ToFloat to_float = nullptr;
};

CpuType cpu_type( GgmlType t ) {
	switch ( t ) {
#define T( E, D, A, F ) case GgmlType::E: return { reinterpret_cast<VecDot>( D ), GgmlType::A, reinterpret_cast<ToFloat>( F ) };
		T( Q4_0, ggml_vec_dot_q4_0_q8_0, Q8_0, dequantize_row_q4_0 )
		T( Q4_1, ggml_vec_dot_q4_1_q8_1, Q8_1, dequantize_row_q4_1 )
		T( Q5_0, ggml_vec_dot_q5_0_q8_0, Q8_0, dequantize_row_q5_0 )
		T( Q5_1, ggml_vec_dot_q5_1_q8_1, Q8_1, dequantize_row_q5_1 )
		T( Q8_0, ggml_vec_dot_q8_0_q8_0, Q8_0, dequantize_row_q8_0 )
		T( Q2_0, ggml_vec_dot_q2_0_q8_0, Q8_0, dequantize_row_q2_0 )
		T( Q2_K, ggml_vec_dot_q2_K_q8_K, Q8_K, dequantize_row_q2_K )
		T( Q3_K, ggml_vec_dot_q3_K_q8_K, Q8_K, dequantize_row_q3_K )
		T( Q4_K, ggml_vec_dot_q4_K_q8_K, Q8_K, dequantize_row_q4_K )
		T( Q5_K, ggml_vec_dot_q5_K_q8_K, Q8_K, dequantize_row_q5_K )
		T( Q6_K, ggml_vec_dot_q6_K_q8_K, Q8_K, dequantize_row_q6_K )
		T( IQ2_XXS, ggml_vec_dot_iq2_xxs_q8_K, Q8_K, dequantize_row_iq2_xxs )
		T( IQ2_XS, ggml_vec_dot_iq2_xs_q8_K, Q8_K, dequantize_row_iq2_xs )
		T( IQ2_S, ggml_vec_dot_iq2_s_q8_K, Q8_K, dequantize_row_iq2_s )
		T( IQ3_XXS, ggml_vec_dot_iq3_xxs_q8_K, Q8_K, dequantize_row_iq3_xxs )
		T( IQ3_S, ggml_vec_dot_iq3_s_q8_K, Q8_K, dequantize_row_iq3_s )
		T( IQ1_S, ggml_vec_dot_iq1_s_q8_K, Q8_K, dequantize_row_iq1_s )
		T( IQ1_M, ggml_vec_dot_iq1_m_q8_K, Q8_K, dequantize_row_iq1_m )
		T( IQ4_NL, ggml_vec_dot_iq4_nl_q8_0, Q8_0, dequantize_row_iq4_nl )
		T( IQ4_XS, ggml_vec_dot_iq4_xs_q8_K, Q8_K, dequantize_row_iq4_xs )
#undef T
		default: return {};
	}
}

} // namespace

bool supported( GgmlType type ) { return cpu_type( type ).dot != nullptr && cpu_type( type ).to_float != nullptr; }

GgmlType act_type( GgmlType type ) { return cpu_type( type ).act; }

void quantize_act( GgmlType act, const float* x, void* out, int k ) {
	const FromFloat q = act == GgmlType::Q8_0 ? quantize_row_q8_0 : act == GgmlType::Q8_1 ? quantize_row_q8_1 : quantize_row_q8_K;
	q( x, out, k );
}

float dot( GgmlType type, const void* w_row, const void* act_row, int k ) {
	float s = 0.0f;
	cpu_type( type ).dot( k, &s, 0, w_row, 0, act_row, 0, 1 );
	return s;
}

void to_float( GgmlType type, const void* blocks, float* out, int64_t elems ) { cpu_type( type ).to_float( blocks, out, elems ); }

} // namespace engine::modules::gguf_cpu
