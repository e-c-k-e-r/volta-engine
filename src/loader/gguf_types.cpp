#include "engine/modules/gguf.h"

namespace engine::modules {

// Block geometry of the GGUF types (src/kernels/cuda/gguf/gguf_linear.cu checks it against the vendored block layouts)
GgmlBlock ggml_block( GgmlType type ) {
	switch ( type ) {
		case GgmlType::F32: return { 1, 4 };
		case GgmlType::F16: return { 1, 2 };
		case GgmlType::BF16: return { 1, 2 };
		case GgmlType::Q4_0: return { 32, 18 };
		case GgmlType::Q4_1: return { 32, 20 };
		case GgmlType::Q5_0: return { 32, 22 };
		case GgmlType::Q5_1: return { 32, 24 };
		case GgmlType::Q8_0: return { 32, 34 };
		case GgmlType::Q8_1: return { 32, 36 };
		case GgmlType::Q2_K: return { 256, 84 };
		case GgmlType::Q3_K: return { 256, 110 };
		case GgmlType::Q4_K: return { 256, 144 };
		case GgmlType::Q5_K: return { 256, 176 };
		case GgmlType::Q6_K: return { 256, 210 };
		case GgmlType::Q8_K: return { 256, 292 };
		case GgmlType::IQ2_XXS: return { 256, 66 };
		case GgmlType::IQ2_XS: return { 256, 74 };
		case GgmlType::IQ2_S: return { 256, 82 };
		case GgmlType::IQ3_XXS: return { 256, 98 };
		case GgmlType::IQ3_S: return { 256, 110 };
		case GgmlType::IQ1_S: return { 256, 50 };
		case GgmlType::IQ1_M: return { 256, 56 };
		case GgmlType::IQ4_NL: return { 32, 18 };
		case GgmlType::IQ4_XS: return { 256, 136 };
		case GgmlType::Q2_0: return { 64, 18 };
		default: return {};
	}
}

const char* ggml_type_name( GgmlType type ) {
	switch ( type ) {
		case GgmlType::F32: return "F32";
		case GgmlType::F16: return "F16";
		case GgmlType::BF16: return "BF16";
		case GgmlType::Q4_0: return "Q4_0";
		case GgmlType::Q4_1: return "Q4_1";
		case GgmlType::Q5_0: return "Q5_0";
		case GgmlType::Q5_1: return "Q5_1";
		case GgmlType::Q8_0: return "Q8_0";
		case GgmlType::Q8_1: return "Q8_1";
		case GgmlType::Q2_K: return "Q2_K";
		case GgmlType::Q3_K: return "Q3_K";
		case GgmlType::Q4_K: return "Q4_K";
		case GgmlType::Q5_K: return "Q5_K";
		case GgmlType::Q6_K: return "Q6_K";
		case GgmlType::Q8_K: return "Q8_K";
		case GgmlType::IQ2_XXS: return "IQ2_XXS";
		case GgmlType::IQ2_XS: return "IQ2_XS";
		case GgmlType::IQ2_S: return "IQ2_S";
		case GgmlType::IQ3_XXS: return "IQ3_XXS";
		case GgmlType::IQ3_S: return "IQ3_S";
		case GgmlType::IQ1_S: return "IQ1_S";
		case GgmlType::IQ1_M: return "IQ1_M";
		case GgmlType::IQ4_NL: return "IQ4_NL";
		case GgmlType::IQ4_XS: return "IQ4_XS";
		case GgmlType::Q2_0: return "Q2_0";
		case GgmlType::MXFP4: return "MXFP4";
		case GgmlType::NVFP4: return "NVFP4";
		default: return "unknown";
	}
}

} // namespace engine::modules
