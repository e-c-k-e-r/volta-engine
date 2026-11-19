#pragma once

// The integer-rate encoder instances, split over two units to keep each compile short (exl3_encode_lo.cu: K 1..4, exl3_encode_hi.cu: K 5..8)
#include <cuda_fp16.h>
#include <cstdint>

using exl3e_kernel_fn = void ( * )( const float*, const float*, float*, uint16_t*, half*, uint16_t*, int );

exl3e_kernel_fn exl3e_kernel_lo( int K, int cb );
exl3e_kernel_fn exl3e_kernel_hi( int K, int cb );
