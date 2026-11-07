#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/backend/types.h"
#include <cstddef>
#include <cstdint>

// GGUF quantized weights (llama.cpp's block formats), read as they are stored in the file: a weight of n output rows and k inputs is n rows
// of k / block_elems blocks, each block_bytes long. Up to 8 rows multiply through the block types' integer dot products against
// activations quantized to q8_1 (llama.cpp's vec_dot kernels); 9 to 16 rows decode every block once into shared memory (llama.cpp's
// block decoders) and multiply it with all rows on the tensor cores; Q4_K from 4 rows on, Q5_K, Q6_K and IQ4_XS from 6, decode each lane's weight row straight
// into Volta's m8n8k4 tensor-core fragments instead; prefill chunks dequantize the weight to FP16 (llama.cpp's converters) and run one
// tensor-core GEMM (Gemm::tn). The llama.cpp files are vendored in src/kernels/cuda/gguf/ggml.
namespace engine::modules {

// Tensor type ids of the GGUF format (ggml_type)
enum class GgmlType : uint32_t {
	F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
	Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
	IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ1_S = 19, IQ4_NL = 20, IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23,
	I8 = 24, I16 = 25, I32 = 26, I64 = 27, F64 = 28, IQ1_M = 29, BF16 = 30, MXFP4 = 39, NVFP4 = 40, Q2_0 = 42,
};

struct GgmlBlock {
	int elems = 0;    // values per block (0: unknown type)
	int bytes = 0;    // bytes per block
};

// Block geometry of a type (elems 0 for types this engine does not read)
GgmlBlock ggml_block( GgmlType type );
const char* ggml_type_name( GgmlType type );
// The quantized types LinearGguf multiplies on the GPU
bool gguf_linear_supported( GgmlType type );

// A weight whose input columns are in llama.cpp's tiled V-head order, [v][key head][dim], against an activation in grouped order,
// [key head][v][dim] (Qwen3.5's linear-attention out_proj): heads key heads of dim columns each, k / (heads * dim) values of v
struct GgufColTiling {
	int heads = 0, dim = 0;   // 0: the columns follow the activation
};

struct LinearGguf {
	// out[m, n] (FP16) = x[m, k] (FP16) times the weight w, n rows of k / elems blocks of `type`
	static void forward( const pod::TensorView& x, const void* w, GgmlType type, int n, int k, pod::TensorView& out, backend::stream_t stream = nullptr,
		GgufColTiling tiling = {} );
	// The current device's activation workspace and GEMM handle, made at load so no forward allocates (decode graphs, collectives)
	static void prepare();
	// out (FP16, `elems` values) = the blocks of w dequantized, in their stored order (elems a multiple of the block size)
	static void dequant( const void* w, GgmlType type, int64_t elems, backend::half_t* out, backend::stream_t stream = nullptr );
};

// GGUF routed experts (MoE::forward with QuantType::GGUF), route by route: the rows quantized to q8_1 once, each route's gate and up
// projections through the vec_dot of their block types, SiLU(gate) * up quantized to q8_1, the down projection scaled by the route's
// weight, then each row's routes summed in route order (deterministic). Rows run in passes of kMoERowsPerPass.
struct MoEGguf {
	static constexpr int kMoERowsPerPass = 64;
	// Device pointer table of a layer: rows gate, up, down of num_local pointers to each expert's blocks
	struct Table {
		const void* const* ptrs = nullptr;
		int num_local = 0, first_expert = 0;   // global id of local expert 0
		GgmlType type[3] = {};                 // gate, up, down
		int hidden = 0, inter = 0;
	};
	static size_t scratch_bytes( int top_k, int hidden, int inter );
	// out[m, hidden] (FP16) = this rank's experts over the routes of x[m, hidden]; topk ids are global, routes to other ranks' experts add 0.
	// heat (optional): per local expert, incremented once per route
	static void forward( const Table& t, const backend::half_t* x, int m, const int* topk_idx, const backend::half_t* topk_w, int top_k,
		backend::half_t* out, void* scratch, uint32_t* heat, backend::stream_t stream = nullptr );
};

} // namespace engine::modules
