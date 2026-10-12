// Are two Qwen3-VL text checkpoints the same weights? Compares the tensors that EXL3 stores unquantized (RMSNorm weights, the embedding table) between a BF16 source and an
// EXL3 conversion, converted to FP32.   test_h3_text_identity <bf16 dir> <exl3 dir> [layers=50]
#include "engine/config.h"
#include "engine/loader/safetensors.h"
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static float to_f32( const void* p, pod::DType t, size_t i ) {
	if ( t == pod::DType::FP32 ) { return static_cast<const float*>( p )[i]; }
	if ( t == pod::DType::FP16 ) { return __half2float( static_cast<const half*>( p )[i] ); }
	ENGINE_ASSERT( t == pod::DType::BF16, "dtype" );
	const uint32_t bits = static_cast<uint32_t>( static_cast<const uint16_t*>( p )[i] ) << 16;
	float f;
	std::memcpy( &f, &bits, 4 );
	return f;
}

int main( int argc, char** argv ) {
	if ( argc < 3 ) { std::printf( "usage: test_h3_text_identity <bf16 dir> <exl3 dir> [layers]\n" ); return 1; }
	const int layers = argc > 3 ? std::atoi( argv[3] ) : 50;
	engine::loader::SafeTensorsRepo a, b;
	ENGINE_ASSERT( a.open( argv[1] ) && b.open( argv[2] ), "open" );
	int bad = 0;
	size_t tensors = 0;
	auto cmp = [&]( const std::string& name ) {
		const pod::SafeTensorMeta* ma = a.find( name.c_str() );
		const pod::SafeTensorMeta* mb = b.find( name.c_str() );
		if ( !ma || !mb ) { std::printf( "%s: missing in %s\n", name.c_str(), ma ? "the EXL3 repo" : "the source" ); ++bad; return; }
		size_t n = 1;
		for ( int i = 0; i < ma->ndim; ++i ) { n *= static_cast<size_t>( ma->shape[i] ); }
		const void* pa = a.cpu_data( name.c_str() );
		const void* pb = b.cpu_data( name.c_str() );
		double worst = 0, scale = 0;
		size_t diff = 0;
		for ( size_t i = 0; i < n; ++i ) {
			const float x = to_f32( pa, ma->dtype, i ), y = to_f32( pb, mb->dtype, i );
			const double d = std::fabs( static_cast<double>( x ) - y );
			if ( d > 0 ) { ++diff; }
			worst = std::fmax( worst, d );
			scale = std::fmax( scale, std::fabs( static_cast<double>( x ) ) );
		}
		++tensors;
		// BF16 -> FP16 is exact except for values below FP16's normal range
		if ( worst > 1e-4 * scale + 1e-6 ) {
			++bad;
			std::printf( "%s: DIFFERENT (%zu of %zu values differ, max abs %.3g, max |x| %.3g)\n", name.c_str(), diff, n, worst, scale );
		} else if ( diff && name.find( "embed" ) != std::string::npos ) {
			std::printf( "%s: %zu of %zu values differ by at most %.3g (FP16 rounding of tiny values)\n", name.c_str(), diff, n, worst );
		}
	};
	cmp( "model.language_model.embed_tokens.weight" );
	cmp( "model.language_model.norm.weight" );
	for ( int i = 0; i < layers; ++i ) {
		const std::string p = "model.language_model.layers." + std::to_string( i ) + ".";
		for ( const char* t : { "input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight" } ) { cmp( p + t ); }
	}
	std::printf( "%zu tensors compared: %s\n", tensors, bad ? "DIFFERENT" : "identical" );
	return bad ? 1 : 0;
}
