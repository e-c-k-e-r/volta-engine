#pragma once

#define EXL3_SM75 0

// EXL3_VOLTA selects the m8n8k4 HMMA inner loop of the EXL3 GEMM (exl3_gemm_inner.cuh). Decided per device pass: SM_70 has no
// m16n8k16 / ldmatrix / cp.async, SM_80+ runs the upstream loop on them. Only device code reads it. -DEXL3_VOLTA=1 forces the Volta
// loop everywhere (CMake option ENGINE_EXL3_FORCE_VOLTA_MMA).
#ifndef EXL3_VOLTA
	#if defined(USE_ROCM) || ( defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) )
		#define EXL3_VOLTA 0
	#else
		#define EXL3_VOLTA 1
	#endif
#endif

// EXL3_VOLTA_LAYOUT=1 reads the weight tiles in the Volta order (tile_layout "volta" checkpoints, tests/tools/exl3_quantize.py --order volta):
// the eight values a lane decodes are exactly its m8n8k4 B operands, so the GEMM needs no shuffles. SM_70 builds compile the tile-decoding
// units a second time with it, in namespace exl3_vl (exl3_vl.cuh), and every weight picks its copy.
#ifndef EXL3_VOLTA_LAYOUT
	#define EXL3_VOLTA_LAYOUT 0
#endif

#ifndef EXL3_SM70_HMMA
	#define EXL3_SM70_HMMA 1
#endif

