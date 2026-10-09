// Vision-tower operators exist only in the CUDA backend; the others report them unsupported (the server then rejects image input).
#include "engine/modules/vision.h"
#include "engine/backend/tiers.h"

namespace engine::modules {

void Vision::patch_embed( const backend::half_t*, const backend::half_t*, const backend::half_t*, const backend::half_t*, float*, int, int, int, backend::stream_t ) {
	ENGINE_UNSUPPORTED( "Vision::patch_embed" );
}
void Vision::layer_norm( const float*, const backend::half_t*, const backend::half_t*, backend::half_t*, int, int, float, backend::stream_t ) {
	ENGINE_UNSUPPORTED( "Vision::layer_norm" );
}
void Vision::qkv_bias_rope( backend::half_t*, backend::half_t*, backend::half_t*, const backend::half_t*, const backend::half_t*, const backend::half_t*, const float*, const float*, int, int, int, backend::stream_t ) {
	ENGINE_UNSUPPORTED( "Vision::qkv_bias_rope" );
}
void Vision::attention( const backend::half_t*, const backend::half_t*, const backend::half_t*, backend::half_t*, int, int, int, float, backend::stream_t ) {
	ENGINE_UNSUPPORTED( "Vision::attention" );
}
void Vision::attention_simt( const backend::half_t*, const backend::half_t*, const backend::half_t*, backend::half_t*, int, int, int, float, backend::stream_t ) {
	ENGINE_UNSUPPORTED( "Vision::attention_simt" );
}
void Vision::rms_norm( const float*, const backend::half_t*, backend::half_t*, int, int, float, backend::stream_t ) {
	ENGINE_UNSUPPORTED( "Vision::rms_norm" );
}
void Vision::layer_norm_half( const backend::half_t*, const backend::half_t*, const backend::half_t*, backend::half_t*, int, int, float, backend::stream_t ) {
	ENGINE_UNSUPPORTED( "Vision::layer_norm_half" );
}
void Vision::qkv_bias_norm_rope( backend::half_t*, backend::half_t*, backend::half_t*, const backend::half_t*, const backend::half_t*, const backend::half_t*, const backend::half_t*, const backend::half_t*, const float*, const float*, int, int, int, float, backend::stream_t ) {
	ENGINE_UNSUPPORTED( "Vision::qkv_bias_norm_rope" );
}
void Vision::merge_gather( const backend::half_t*, backend::half_t*, int, int, int, backend::stream_t ) {
	ENGINE_UNSUPPORTED( "Vision::merge_gather" );
}
void Vision::dense_gemm( const backend::half_t*, const backend::half_t*, backend::half_t*, int, int, int, backend::stream_t ) {
	ENGINE_UNSUPPORTED( "Vision::dense_gemm" );
}
void Vision::bias_act( backend::half_t*, const backend::half_t*, int, int, int, backend::stream_t ) {
	ENGINE_UNSUPPORTED( "Vision::bias_act" );
}
void Vision::bias_residual( float*, const backend::half_t*, const backend::half_t*, int, int, backend::stream_t ) {
	ENGINE_UNSUPPORTED( "Vision::bias_residual" );
}
void Vision::layer_norm_merge( const float*, const backend::half_t*, const backend::half_t*, backend::half_t*, int, int, int, float, backend::stream_t ) {
	ENGINE_UNSUPPORTED( "Vision::layer_norm_merge" );
}

} // namespace engine::modules
