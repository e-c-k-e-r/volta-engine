#pragma once
// ROCM configuration: HIP's cooperative groups (this_grid().sync() needs a cooperative launch, as on CUDA)
#include "../compat.h"
#include <hip/hip_cooperative_groups.h>
