#pragma once

#include "engine/config.h"
#include "engine/backend/types.h"
#include <cstddef>
#include <cstdint>

namespace engine::modules {

// FP8 routed experts as the original checkpoint holds them (DeepSeek-V3-style block FP8, GLM-5.3-Flash): each projection an [n, k] E4M3 weight and
// F32 scales [n / 128, k / 128], the multiplier of the decoded value of each 128 x 128 block. The pointer table has NVFP4ExpertTable's layout
// (codes and scales per projection), so the expert tier moves the codes the same way
struct FP8ExpertTable {
	const uint8_t* const* gate_c;
	const float* const* gate_s;
	const uint8_t* const* up_c;
	const float* const* up_s;
	const uint8_t* const* down_c;
	const float* const* down_s;
	int num_local;
	int min_expert;   // global id of local expert 0; routes outside [min_expert, min_expert + num_local) are skipped
	int num_global;   // the layer's experts over every rank (0: four ranks of num_local each)
	int hidden;       // H
	int inter;        // I
};

struct FP8MoE {
	static size_t scratch_bytes( int top_k, int hidden, int inter );
	// out[m, hidden] (FP16) = this rank's experts over the routes of x[m, hidden] (routes to other ranks' experts add 0). Passes of up to 1024 rows:
	// the routes grouped by expert into work items, and per item a tile of output rows whose E4M3 bytes are decoded to FP16 in shared memory
	// (exactly: the value times 2^-8) and multiplied with all of the item's routes on the tensor cores, each 128-wide step's FP32 sum scaled by
	// its block scale. Gate and up in one pass (SiLU( gate ) * up of the FP16-rounded pair; act_limit > 0 the SwiGLU clamp), the down projection
	// into FP32 partials weighted by the routes, summed per token in route order. heat (optional): per local expert, incremented once per route
	static void forward( const FP8ExpertTable& t, const backend::half_t* x, int m, const int* topk_idx, const backend::half_t* topk_w, int top_k,
		backend::half_t* out, void* scratch, uint32_t* heat, float act_limit, backend::stream_t stream = nullptr );
};

} // namespace engine::modules
