#pragma once
// ROCM configuration: profiler range markers are no-ops (rocprof traces whole processes)
#include "../compat.h"
inline hipError_t cudaProfilerStart() { return hipSuccess; }
inline hipError_t cudaProfilerStop() { return hipSuccess; }
