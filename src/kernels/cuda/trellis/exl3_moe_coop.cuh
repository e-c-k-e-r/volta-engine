#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Fused decode-shaped routed-expert MoE (1..MOE_COOP_MAX_ROWS rows, 3 bpw mul1 experts): ported from exllamav3's
// exl3_moe_coop (two-stage cooperative GEMV kernels) onto our expert pointer tables. Three launches per layer replace the
// dispatch / persistent expert kernel / plane sum chain:
//   rot : (slot, 128-chunk, projection) warps: x row * suh -> Hadamard into had_g / had_u (skipped at one row, where the
//         gate/up blocks rotate into shared memory themselves)
//   A   : gate/up GEMV, one block per (expert run, column group, projection); a run is up to 8 slots that picked the same
//         expert, the rows of one MMA tile, so the expert's weights are read once per run. The last block to finish a
//         (slot, 128-chunk) applies the output rotation, svh, the activation and the down-input rotation
//   B   : down GEMV on the same runs; the last block to finish a (token, 128-chunk) rotates every slot's result, applies
//         svh and the routing weight, sums the slots in a fixed order and stores the row (half)
// The rows of a run are independent in the tile, and the slot sum order is fixed, so a row's result does not depend on how many
// rows share the call.

#define MOE_COOP_THREADS 512
#define MOE_COOP_WNT 2                  // adjacent 16-column tiles per warp
#define MOE_COOP_COLS ( MOE_COOP_WNT * 16 )
#define MOE_COOP_MAX_ROWS 16
#define MOE_COOP_TOPK 10                // the largest routes-per-token a call may have (scratch is sized by it); 3 bpw experts with 8 or fewer routes also take the int8 tile
#define MOE_COOP_MAX_SLOTS ( MOE_COOP_MAX_ROWS * MOE_COOP_TOPK )

struct MoeCoopParams {
	// Per call
	const half* x;                      // [bsz, H]
	const int* sel;                     // [bsz, topk] global expert ids
	const half* rw;                     // [bsz, topk] routing weights (a zero weight is an inactive slot)
	half* out;                          // [bsz, H] routed output, every row with an active slot is written
	uint32_t* heat;                     // optional per-expert route counters (local index)
	int bsz;
	int min_expert;                     // sel in [min, max) is local; local index = sel - min
	int max_expert;
	// Static
	int H;                              // hidden width (gate/up in_features, down out_features), multiple of 128
	int I;                              // this rank's intermediate width, multiple of 128
	int topk;
	int bits;                           // 3 or 4 bits per weight
	float act_limit;
	const uint16_t* const* g_trellis; const half* const* g_suh; const half* const* g_svh;
	const uint16_t* const* u_trellis; const half* const* u_suh; const half* const* u_svh;
	const uint16_t* const* d_trellis; const half* const* d_suh; const half* const* d_svh;
	// Scratch (slots = MOE_COOP_MAX_SLOTS)
	half* had_g;                        // [slots, H]
	half* had_u;
	float* gu_g;                        // [ksplit_a * slots, I]
	float* gu_u;
	half* act_out;                      // [slots, I]
	float* d_out;                       // [ksplit_b * slots, H]
	int* ctr_a;                         // [slots, I / 128] completion counters (zero between calls)
	int* ctr_b;                         // [MOE_COOP_MAX_ROWS, H / 128]
	int* runs;                          // [2 + slots + 1 + slots]
	// Launch geometry (filled by the launcher)
	int ksplit_a;
	int ksplit_b;
	bool a_global;                      // rows > 1: the gate/up blocks read had_g / had_u
	int ctr_a_len;
	int ctr_b_len;
};

inline int moe_coop_ctr_len( int H, int I ) {
	return MOE_COOP_MAX_SLOTS * ( I / 128 ) + MOE_COOP_MAX_ROWS * ( H / 128 ) + 2 + ( MOE_COOP_MAX_SLOTS + 1 ) + MOE_COOP_MAX_SLOTS;
}

// Scratch sizing (elements) for the arena; ksplit_max bounds the split-k factor of either stage
#define MOE_COOP_KSPLIT_MAX 2

void exl3_moe_coop_launch( const MoeCoopParams& p, int device, cudaStream_t stream );
