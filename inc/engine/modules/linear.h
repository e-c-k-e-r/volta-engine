#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/modules/trellis.h"
#include "engine/modules/nvfp4.h"
#include "engine/modules/gguf.h"
#include "engine/backend/types.h"
#include <cstdint>

namespace pod {

enum class QuantType : uint8_t {
	FP16,
	TRELLIS,
	NVFP4,
	FP8,     // the original block-scaled weight, applied by DSV4::linear_fp8 (DeepSeek-V4.1 SAGE packs): w0 FP8 E4M3 [n, k], w1 E8M0 scales [n / 32, k / 32]
	GGUF     // a GGUF block type as stored in the file: w0 = n rows of k / elems blocks (bytes), gguf_type the block type, k and n in gguf_k / gguf_n
};

struct LinearWeight {
	QuantType  type = QuantType::TRELLIS;
	TensorView w0{};    // Trellis: codes / NVFP4: packed nibbles
	TensorView w1{};    // Trellis: suh   / NVFP4: fp8 block scales
	TensorView w2{};    // Trellis: svh
	float      alpha = 1.0f;
	int        bits = 4;
	bool       mul1 = false;
	bool       mcg = false;   // MCG codebook (the quantized DFlash2 drafters); everything else decodes as mul1
	uint32_t   gguf_type = 0; // GGUF: engine::modules::GgmlType of w0's blocks
	int        gguf_k = 0, gguf_n = 0;
	int        gguf_tile_heads = 0, gguf_tile_dim = 0;   // GGUF: input columns in tiled V-head order (see GgufColTiling); 0: as the activation

	// Trellis and GGUF projections run through one workspace per device (rotated or q8_1 activations, split-K partials): two of them must not
	// run on different streams at once
	bool shares_workspace() const { return type == QuantType::TRELLIS || type == QuantType::GGUF; }

	static LinearWeight make_trellis(
		const TensorView& codes,
		const TensorView& suh,
		const TensorView& svh,
		int bits = 4,
		bool mul1 = false,
		bool mcg = false
	) {
		LinearWeight lw;
		lw.type = QuantType::TRELLIS;
		lw.w0 = codes;
		lw.w1 = suh;
		lw.w2 = svh;
		lw.alpha = 1.0f;
		lw.bits = bits;
		lw.mul1 = mul1;
		lw.mcg = mcg;
		return lw;
	}

	static LinearWeight make_nvfp4(
		const TensorView& packed_nibbles,
		const TensorView& fp8_scales,
		float alpha = 1.0f
	) {
		LinearWeight lw;
		lw.type = QuantType::NVFP4;
		lw.w0 = packed_nibbles;
		lw.w1 = fp8_scales;
		lw.alpha = alpha;
		lw.bits = 4;
		return lw;
	}

	static LinearWeight make_fp8( const TensorView& codes, const TensorView& scales ) {
		LinearWeight lw;
		lw.type = QuantType::FP8;
		lw.w0 = codes;   // shape (k, n) like the FP16 form, the bytes laid out [n][k]
		lw.w1 = scales;
		return lw;
	}

	static LinearWeight make_gguf( const TensorView& blocks, uint32_t ggml_type, int k, int n ) {
		LinearWeight lw;
		lw.type = QuantType::GGUF;
		lw.w0 = blocks;   // [n, k / elems * bytes] raw bytes
		lw.gguf_type = ggml_type;
		lw.gguf_k = k;
		lw.gguf_n = n;
		return lw;
	}

	static LinearWeight make_fp16( const TensorView& w ) {
		LinearWeight lw;
		lw.type = QuantType::FP16;
		lw.w0 = w;   // [k, n] row-major, dense FP16
		return lw;
	}

	bool is_valid() const {
		return w0.data != nullptr;
	}
};

} // namespace pod

namespace engine::modules {

// Dense FP16 [k, n] weight times [m, k] activations. Implemented alongside the drafter's FP16 GEMV
// in src/spec/dflash.cu rather than duplicating the kernel; used only by unquantized projections and
// by tests that need a backbone they can also evaluate on the CPU.
struct LinearPlainGEMV {
	static void forward(
		const backend::half_t* w,
		const backend::half_t* x,
		backend::half_t* out,
		int m,
		int n,
		int k,
		backend::stream_t stream = nullptr
	);
};

struct Linear {
	static void forward(
		const pod::TensorView& x,
		const pod::LinearWeight& weight,
		pod::TensorView& out,
		backend::stream_t stream = nullptr
	) {
		if ( !weight.is_valid() ) return;

		switch ( weight.type ) {
			case pod::QuantType::TRELLIS: {
				LinearTrellis::forward(
					x,
					weight.w0,
					weight.w1,
					weight.w2,
					out,
					weight.bits,
					weight.mcg,
					!weight.mcg,
					stream
				);
			} break;
			case pod::QuantType::NVFP4: {
				LinearNVFP4::forward(
					x,
					weight.w0,
					weight.w1,
					weight.alpha,
					out,
					stream
				);
			} break;
			case pod::QuantType::FP16: {
				// Dense fallback. Only used by tests and by the drafter's own unquantized
				// projections; every real target weight is Trellis.
				ENGINE_ASSERT( x.shape[1] == weight.w0.shape[0], "Linear::forward FP16: K mismatch" );
				const int rows = x.shape[0];
				const int n = weight.w0.shape[1];
				const int k = weight.w0.shape[0];
				// Weight first, activations second: LinearPlainGEMV::forward(w, x, out, ...).
				// Passing them the other way round computes x @ w^T, which is the transpose of
				// the product and produces plausible magnitudes -- it survived every gate, because
				// test_f16_gemv_nt calls dflash_f16_gemv_nt directly and never went through here.
				LinearPlainGEMV::forward(
					weight.w0.ptr<const backend::half_t>(),
					x.ptr<const backend::half_t>(),
					out.ptr<backend::half_t>(),
					rows, n, k, stream
				);
			} break;
			case pod::QuantType::GGUF: {
				LinearGguf::forward( x, weight.w0.data, static_cast<GgmlType>( weight.gguf_type ), weight.gguf_n, weight.gguf_k, out, stream,
					{ weight.gguf_tile_heads, weight.gguf_tile_dim } );
			} break;
			default: {
				ENGINE_ASSERT( false, "Unimplemented format" );
			} break;
		}
	}
};

} // namespace engine::modules
