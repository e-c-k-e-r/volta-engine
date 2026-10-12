#pragma once

#include <cstddef>
#include <functional>

namespace engine::video {

// Collectives of a tensor-parallel group, enqueued on the caller's compute stream (the group provides them, e.g. over NCCL)
struct H3TpComm {
	std::function<void( float* buf, size_t count, void* stream )> sum_f32; // in place, summed over the ranks
	std::function<void( float* buf, size_t count, void* stream )> max_f32; // in place, maximum over the ranks
	std::function<void( void* buf, size_t count, void* stream )> sum_f16;  // in place, FP16 values summed over the ranks
};

} // namespace engine::video
