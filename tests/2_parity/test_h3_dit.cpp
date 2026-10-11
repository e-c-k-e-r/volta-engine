// H3 DiT parity: runs the packed forward on a golden file made by tests/tools/h3_golden.py and compares the stream after the
// embedding / selected blocks / the final layer and both outputs against the FP32 reference.
//   test_h3_dit <comfy checkpoint> <golden.safetensors> [lora.safetensors]   (the golden must have been made with the same adapter)
#include "engine/config.h"
#include "engine/loader/safetensors.h"
#include "engine/video/h3_dit.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

pod::SafeTensorsFile g_st;

const pod::SafeTensorMeta* need( const std::string& name ) {
	const pod::SafeTensorMeta* m = engine::loader::safetensors_find( g_st, name.c_str() );
	ENGINE_ASSERT( m, name.c_str() );
	return m;
}

template <typename T>
T* to_device( const std::string& name ) {
	const pod::SafeTensorMeta* m = need( name );
	void* d = nullptr;
	CUDA_CHECK( cudaMalloc( &d, m->size_bytes ) );
	CUDA_CHECK( cudaMemcpy( d, engine::loader::safetensors_get_data_cpu( g_st, m ), m->size_bytes, cudaMemcpyHostToDevice ) );
	return static_cast<T*>( d );
}

struct Stats {
	double cosine, rel;
};

Stats compare( const float* a, const float* b, size_t n ) {
	double dot = 0, na = 0, nb = 0, nd = 0;
	for ( size_t i = 0; i < n; ++i ) {
		dot += static_cast<double>( a[i] ) * b[i];
		na += static_cast<double>( a[i] ) * a[i];
		nb += static_cast<double>( b[i] ) * b[i];
		nd += ( static_cast<double>( a[i] ) - b[i] ) * ( static_cast<double>( a[i] ) - b[i] );
	}
	return { dot / std::sqrt( na * nb + 1e-30 ), std::sqrt( nd / ( nb + 1e-30 ) ) };
}

} // namespace

int main( int argc, char** argv ) {
	if ( argc < 3 ) { std::printf( "usage: test_h3_dit <checkpoint> <golden>\n" ); return 1; }
	ENGINE_ASSERT( engine::loader::safetensors_open( g_st, argv[2] ), "golden file" );
	const pod::SafeTensorMeta* embed = need( "s0.embed" );
	const int used = embed->shape[0];
	const int text_len = need( "prompt_embeds" )->shape[0];
	const int n_img = need( "img_pos" )->shape[0];
	const int n_audio = need( "audio_pos" )->shape[0];
	int layers = 0;
	for ( int i = 0; i < 64; ++i ) {
		if ( engine::loader::safetensors_find( g_st, ( "s0.block" + std::to_string( i ) ).c_str() ) ) { layers = i + 1; }
	}
	std::printf( "golden: used %d text %d video rows %d audio rows %d, blocks dumped up to %d\n", used, text_len, n_img, n_audio, layers );

	engine::video::H3Config cfg;
	if ( layers > 0 && layers < 50 ) { cfg.layers = layers; }
	engine::video::H3DiT dit;
	auto t0 = std::chrono::steady_clock::now();
	ENGINE_ASSERT( dit.load( argv[1], 0, ( used + 63 ) / 64 * 64, cfg ), "load" );
	std::printf( "loaded %.2f GB of weights in %.1f s\n", dit.weight_bytes() / 1e9, std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count() );

	// inputs (device)
	if ( argc > 3 ) { ENGINE_ASSERT( dit.add_lora( argv[3], 1.0f ) >= 0, "lora" ); }
	float* d_video = to_device<float>( "x" ); // [S,96]: gather the video rows on the host to match the request contract
	const float* h_x = static_cast<const float*>( engine::loader::safetensors_get_data_cpu( g_st, need( "x" ) ) );
	const float* h_ax = static_cast<const float*>( engine::loader::safetensors_get_data_cpu( g_st, need( "audio_x" ) ) );
	const int* h_img = static_cast<const int*>( engine::loader::safetensors_get_data_cpu( g_st, need( "img_pos" ) ) );
	const int* h_aud = static_cast<const int*>( engine::loader::safetensors_get_data_cpu( g_st, need( "audio_pos" ) ) );
	std::vector<float> vrows( static_cast<size_t>( n_img ) * 96 ), arows( static_cast<size_t>( n_audio ) * 32 );
	for ( int i = 0; i < n_img; ++i ) { std::memcpy( &vrows[static_cast<size_t>( i ) * 96], h_x + static_cast<size_t>( h_img[i] ) * 96, 96 * sizeof( float ) ); }
	for ( int i = 0; i < n_audio; ++i ) { std::memcpy( &arows[static_cast<size_t>( i ) * 32], h_ax + static_cast<size_t>( h_aud[i] ) * 32, 32 * sizeof( float ) ); }
	cudaFree( d_video );
	float *d_vrows, *d_arows;
	CUDA_CHECK( cudaMalloc( &d_vrows, vrows.size() * 4 ) );
	CUDA_CHECK( cudaMalloc( &d_arows, arows.size() * 4 ) );
	CUDA_CHECK( cudaMemcpy( d_vrows, vrows.data(), vrows.size() * 4, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( d_arows, arows.data(), arows.size() * 4, cudaMemcpyHostToDevice ) );

	engine::video::H3Request req;
	req.used = used;
	req.text_len = text_len;
	req.n_img = n_img;
	req.n_audio = n_audio;
	req.video_rows = d_vrows;
	req.audio_rows = d_arows;
	req.prompt = to_device<float>( "prompt_embeds" );
	req.position_ids = to_device<float>( "img_position_ids" );
	req.img_pos = to_device<int>( "img_pos" );
	req.audio_pos = to_device<int>( "audio_pos" );
	req.text_pos = to_device<int>( "text_pos" );
	req.update_mask = to_device<uint8_t>( "update_mask" );
	req.tags = to_device<int>( "token_tags" );
	float *d_vo, *d_ao;
	CUDA_CHECK( cudaMalloc( &d_vo, static_cast<size_t>( n_img ) * 96 * 4 ) );
	CUDA_CHECK( cudaMalloc( &d_ao, static_cast<size_t>( n_audio ) * 32 * 4 ) );

	dit.probe_blocks = { 0, 1, cfg.layers / 2, cfg.layers - 1 };
	int bad = 0;
	for ( int step = 0; step < 2; ++step ) {
		const std::string pre = "s" + std::to_string( step ) + ".";
		const pod::SafeTensorMeta* ut = need( pre + "unique_timesteps" );
		const float* uv = static_cast<const float*>( engine::loader::safetensors_get_data_cpu( g_st, ut ) );
		req.unique_t.assign( uv, uv + ut->shape[0] );
		req.inverse = to_device<int>( pre + "inverse_indices" );
		dit.probe = [&]( const char* tag, const float* d_x, int rows ) {
			const pod::SafeTensorMeta* m = engine::loader::safetensors_find( g_st, ( pre + tag ).c_str() );
			if ( !m ) { return; }
			std::vector<float> got( static_cast<size_t>( rows ) * 5376 );
			CUDA_CHECK( cudaMemcpy( got.data(), d_x, got.size() * 4, cudaMemcpyDeviceToHost ) );
			const Stats s = compare( got.data(), static_cast<const float*>( engine::loader::safetensors_get_data_cpu( g_st, m ) ), got.size() );
			std::printf( "  step %d %-13s cosine %.6f rel %.4g\n", step, tag, s.cosine, s.rel );
		};
		auto t1 = std::chrono::steady_clock::now();
		dit.forward( req, d_vo, d_ao );
		const double secs = std::chrono::duration<double>( std::chrono::steady_clock::now() - t1 ).count();
		std::vector<float> vo( static_cast<size_t>( n_img ) * 96 ), ao( static_cast<size_t>( n_audio ) * 32 );
		CUDA_CHECK( cudaMemcpy( vo.data(), d_vo, vo.size() * 4, cudaMemcpyDeviceToHost ) );
		CUDA_CHECK( cudaMemcpy( ao.data(), d_ao, ao.size() * 4, cudaMemcpyDeviceToHost ) );
		const Stats sv = compare( vo.data(), static_cast<const float*>( engine::loader::safetensors_get_data_cpu( g_st, need( pre + "video_out" ) ) ), vo.size() );
		const Stats sa = compare( ao.data(), static_cast<const float*>( engine::loader::safetensors_get_data_cpu( g_st, need( pre + "audio_out" ) ) ), ao.size() );
		std::printf( "step %d (%.2f s): video cosine %.6f rel %.4g | audio cosine %.6f rel %.4g\n", step, secs, sv.cosine, sv.rel, sa.cosine, sa.rel );
		if ( sv.cosine < 0.999 || sa.cosine < 0.999 ) { ++bad; }
	}
	std::printf( bad ? "FAIL\n" : "ok\n" );
	return bad ? 1 : 0;
}
