// ggml symbols the vendored iqk kernels (ik/) reference on paths this engine does not take: weight quantization, float and 1-bit
// types, tensor bookkeeping. The engine calls iqk_mul_mat only for block-quantized weights against q8_K activations; reaching one of
// these is a bug, so each aborts with its name.
#include <stdio.h>
#include <stdlib.h>

#define UNUSED_SYMBOL( name ) void name( void ) { fprintf( stderr, "[iqk] %s is not provided by this engine\n", #name ); abort(); }
UNUSED_SYMBOL( ggml_bf16_to_fp32 )
UNUSED_SYMBOL( ggml_get_quantize_fudge_factor )
UNUSED_SYMBOL( ggml_internal_get_type_traits )
UNUSED_SYMBOL( ggml_is_contiguous )
UNUSED_SYMBOL( ggml_nrows )
UNUSED_SYMBOL( ggml_row_blck_size )
UNUSED_SYMBOL( iq1m_process_1block )
UNUSED_SYMBOL( iq1s_process_1block )
UNUSED_SYMBOL( quantize_q6_0 )
UNUSED_SYMBOL( quantize_row_pq2_0_ref )
UNUSED_SYMBOL( quantize_row_ptq1_0_ref )
