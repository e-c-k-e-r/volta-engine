#pragma once

#include "engine/config.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstddef>

#define MAX_TILES_C (1024 * 1024)
#define MAX_BARRIERS 1024
#define BARRIER_LOCKS_OFFSET MAX_TILES_C

namespace engine::modules {

int trellis_get_num_sms( int device = 0 );
int trellis_get_smem_request( int device = 0 );
int trellis_get_cc( int device = 0 );
int* trellis_get_locks( int device = 0 );
half* trellis_get_workspace( int device = 0, size_t bytes = 16 * 1024 * 1024 );
void trellis_context_init_all();
void trellis_context_destroy();

} // namespace engine::modules
