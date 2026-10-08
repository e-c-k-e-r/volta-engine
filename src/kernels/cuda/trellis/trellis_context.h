#pragma once

#include "engine/config.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstddef>

#define MAX_TILES_C (1024 * 1024)
#define MAX_BARRIERS 1024
#define BARRIER_LOCKS_OFFSET MAX_TILES_C
#define MOE_SCHED_OFFSET (MAX_TILES_C + 2 * MAX_BARRIERS)

// dp4a int8 GEMV workspace: counters, per-slice scales and per-slice partials. Fixed size so the
// pointer baked into captured graphs stays valid.
#define INT8_WS_BYTES ( 16 * 1024 * 1024 )
#define INT8_WS_INTS ( INT8_WS_BYTES / sizeof( int ) )

namespace engine::modules {

int trellis_get_num_sms( int device = 0 );
int trellis_get_smem_request( int device = 0 );
int trellis_get_cc( int device = 0 );
int* trellis_get_locks( int device = 0 );
// Workspace for the dp4a int8 GEMV path. Fixed size, allocated once: its pointer is baked into
// captured CUDA graphs, so it must never move or be reallocated.
int* trellis_get_int8_ws( int device = 0 );
half* trellis_get_workspace( int device = 0, size_t bytes = 16 * 1024 * 1024 );
// Scratch for a dequantized FP16 weight (large-M path). Grows on demand; never captured in a CUDA graph, so it may move.
half* trellis_get_recon_ws( int device, size_t bytes );
// Per-device cuBLAS handle (created on first use, destroyed with the context).
cublasHandle_t trellis_get_cublas( int device );
void trellis_context_init_all();
void trellis_context_destroy();
// Forgets per-process memos keyed on kernel addresses (shared-memory attributes, occupancy): cudaDeviceReset drops the
// attributes, so a second cluster in the same process must set them again.
void trellis_reset_launch_caches();

} // namespace engine::modules
