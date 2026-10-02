#pragma once

#define EXL3_SM75 0

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 700)
	#define EXL3_VOLTA 1
#else
	#define EXL3_VOLTA 1
#endif

#ifndef EXL3_SM70_HMMA
	#define EXL3_SM70_HMMA 1
#endif

#define EXL3_SMEM_MAX_DEFAULT (96 * 1024)
