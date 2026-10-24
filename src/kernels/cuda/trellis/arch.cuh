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

// EXL3_VOLTA_LAYOUT=1 (proof of concept, CMake option ENGINE_EXL3_VOLTA_LAYOUT) reads the weight tiles in the Volta order written by
// tests/tools/exl3_volta_layout.py: the eight values a lane decodes are exactly its m8n8k4 B operands, so the GEMM needs no shuffles.
// A build with it only runs checkpoints in that order, and only the dense GEMM paths (gemm kernels, reconstruct) know it.
#ifndef EXL3_VOLTA_LAYOUT
	#define EXL3_VOLTA_LAYOUT 0
#endif

#ifndef EXL3_SM70_HMMA
	#define EXL3_SM70_HMMA 1
#endif

