#pragma once

// CPU backend vocabulary (selected by engine/backend/types.h). half_t is _Float16, the type the Slang C++ prelude uses for
// half, so TensorView data passes to the generated host kernels unchanged. No vendor headers.
#include <cstdint>
#include <cstddef>
#include <cstring>

// Qualifier and vector-type spellings that shared headers (core/batch.h, modules/*.h) use in declarations. Host code on this
// backend sees them as no-ops / plain structs.
#ifndef __host__
	#define __host__
#endif
#ifndef __device__
	#define __device__
#endif
// (the Slang C++ prelude defines its own int4; translation units that use it include it before any engine header)
#ifndef SLANG_PRELUDE_CPP_TYPES_H
struct int4 { int x, y, z, w; };
#endif

namespace engine::backend {
	using stream_t = void*;
	using graph_t = void*;
	using graph_exec_t = void*;
	using half_t = _Float16;

	// bf16 storage: upper 16 bits of an IEEE float
	struct bf16_t {
		uint16_t bits;
	};

	enum class Copy { H2D, D2H, D2D, Auto };
	enum class Pin { Default, Mapped, MappedPortable };
}
