// H3 DiT forward timing at a production-sized packed sequence (synthetic inputs, real weights, one GPU).
//   bench_h3_dit <comfy checkpoint> [rows=34000] [layers=50] [repeats=2]
#include "engine/config.h"
#include "engine/video/h3_dit.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

template <typename T>
static T* up( const std::vector<T>& v ) {
	T* d = nullptr;
	CUDA_CHECK( cudaMalloc( &d, v.size() * sizeof( T ) ) );
	CUDA_CHECK( cudaMemcpy( d, v.data(), v.size() * sizeof( T ), cudaMemcpyHostToDevice ) );
	return d;
}

int main( int argc, char** argv ) {
	if ( argc < 2 ) { std::printf( "usage: bench_h3_dit <checkpoint> [rows] [layers] [repeats]\n" ); return 1; }
	const int rows = argc > 2 ? std::atoi( argv[2] ) : 34000;
	const int layers = argc > 3 ? std::atoi( argv[3] ) : 50;
	const int reps = argc > 4 ? std::atoi( argv[4] ) : 2;
	engine::video::H3Config cfg;
	cfg.layers = layers;
	engine::video::H3DiT dit;
	ENGINE_ASSERT( dit.load( argv[1], 0, rows, cfg ), "load" );
	const int text = 64, aud = 414, vid = rows - text - aud;
	std::vector<int> img( vid ), ap( aud ), tp( text ), tags( rows ), inv( rows );
	std::vector<uint8_t> mask( vid, 1 );
	for ( int i = 0; i < text; ++i ) { tp[i] = i; tags[i] = 1; inv[i] = 1; }
	for ( int i = 0; i < aud; ++i ) { ap[i] = text + i; tags[text + i] = 2; inv[text + i] = 0; }
	for ( int i = 0; i < vid; ++i ) { img[i] = text + aud + i; tags[text + aud + i] = 0; inv[text + aud + i] = 1; }
	std::vector<float> pos( static_cast<size_t>( rows ) * 3 ), vr( static_cast<size_t>( vid ) * 96 ), ar( static_cast<size_t>( aud ) * 32 ), pr( static_cast<size_t>( text ) * 5120 );
	uint32_t s = 7;
	auto rnd = [&]() { s = s * 1664525u + 1013904223u; return ( ( s >> 8 ) / 16777216.0f - 0.5f ) * 2.0f; };
	for ( auto& v : pos ) { v = std::abs( rnd() ) * 30.0f; }
	for ( auto& v : vr ) { v = rnd(); }
	for ( auto& v : ar ) { v = rnd(); }
	for ( auto& v : pr ) { v = rnd() * 3.0f; }
	engine::video::H3Request req;
	req.used = rows; req.text_len = text; req.n_img = vid; req.n_audio = aud;
	req.video_rows = up( vr ); req.audio_rows = up( ar ); req.prompt = up( pr ); req.position_ids = up( pos );
	req.img_pos = up( img ); req.audio_pos = up( ap ); req.text_pos = up( tp ); req.update_mask = up( mask ); req.tags = up( tags ); req.inverse = up( inv );
	req.unique_t = { 0.4f, 0.5f };
	float *vo, *ao;
	CUDA_CHECK( cudaMalloc( &vo, static_cast<size_t>( vid ) * 96 * 4 ) );
	CUDA_CHECK( cudaMalloc( &ao, static_cast<size_t>( aud ) * 32 * 4 ) );
	for ( int r = 0; r < reps; ++r ) {
		auto t0 = std::chrono::steady_clock::now();
		dit.forward( req, vo, ao );
		const double sec = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
		// linear 2*M*params + attention 4*M^2*d*heads per layer
		const double fl = layers * ( 2.0 * rows * 385e6 + 4.0 * rows * rows * 128.0 * 56 );
		std::printf( "rows %d layers %d: %.2f s per forward, %.1f TFLOP/s effective (est)\n", rows, layers, sec, fl / sec / 1e12 );
	}
	size_t fr, tot;
	CUDA_CHECK( cudaMemGetInfo( &fr, &tot ) );
	std::printf( "device memory in use: %.1f GB\n", ( tot - fr ) / 1e9 );
	return 0;
}
