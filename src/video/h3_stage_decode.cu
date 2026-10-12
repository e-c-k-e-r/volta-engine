// H3 stage: video and audio decoders, then the MP4 mux through ffmpeg.
#include "engine/core/log.h"
#include "engine/video/h3_audio.h"
#include "engine/video/h3_vae.h"
#include "h3_run.h"
#include <cstdio>
#include <cstring>

namespace engine::video {

namespace {

// rows [T*h2*w2, 96] -> latent [24, T, 2*h2, 2*w2]; row = (t, hh, ww), col = c * 4 + ph * 2 + pw
__global__ void k_unpatchify( const float* rows, float* z, int T, int h2, int w2 ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	const size_t total = static_cast<size_t>( T ) * h2 * w2 * 96;
	if ( i >= total ) { return; }
	const size_t n = i / 96;
	const int col = static_cast<int>( i % 96 );
	const int c = col / 4, ph = ( col / 2 ) % 2, pw = col % 2;
	const int ww = static_cast<int>( n % w2 ), hh = static_cast<int>( ( n / w2 ) % h2 ), t = static_cast<int>( n / ( static_cast<size_t>( w2 ) * h2 ) );
	const int H = 2 * h2, W = 2 * w2;
	z[( ( static_cast<size_t>( c ) * T + t ) * H + hh * 2 + ph ) * W + ww * 2 + pw] = rows[i];
}

} // namespace

bool H3Run::decode() {
	{ // a fingerprint of the denoised latents: equal seeds and settings give equal values (regression checks compare it before the decoders run)
		uint64_t h = 1469598103934665603ULL;
		for ( float v : vrows ) { uint32_t b; std::memcpy( &b, &v, 4 ); h = ( h ^ b ) * 1099511628211ULL; }
		for ( float v : arows ) { uint32_t b; std::memcpy( &b, &v, 4 ); h = ( h ^ b ) * 1099511628211ULL; }
		LOG_INFO( "H3: denoised latents fingerprint %016llx", static_cast<unsigned long long>( h ) );
	}
	comp.trim_transformer(); // a resident transformer gives its scratch back for the decoders
	CUDA_CHECK( cudaSetDevice( paths.device ) ); // the DiT group leaves the calling thread on its last GPU
	float* d_vrows = upload( vrows );
	float* d_arows = upload( arows );
	report( "decode", 0, 1 );
	auto t0 = std::chrono::steady_clock::now();
	float* d_z = nullptr;
	CUDA_CHECK( cudaMalloc( &d_z, static_cast<size_t>( 24 ) * latent_t * lh * lw * sizeof( float ) ) );
	const size_t target_floats = vrows.size() - static_cast<size_t>( n_cond ) * 96;
	k_unpatchify<<<( target_floats + 255 ) / 256, 256>>>( d_vrows + static_cast<size_t>( n_cond ) * 96, d_z, latent_t, lh / 2, lw / 2 );
	CUDA_CHECK( cudaDeviceSynchronize() );
	{
		std::vector<int> devices = paths.devices;
		if ( devices.empty() ) { devices.push_back( paths.device ); }
		const auto tvl = std::chrono::steady_clock::now();
		std::vector<H3Lease<H3VideoVAE>> leases = comp.video_vaes( devices );
		LOG_INFO( "H3: %zu video decoder(s) loaded in %.1f s", leases.size(), seconds_since( tvl ) );
		std::vector<H3VideoVAE*> vaes;
		for ( auto& l : leases ) { vaes.push_back( &*l ); }
		if ( vaes.empty() || !H3VideoVAE::decode_video( vaes, d_z, latent_t, lh, lw, rgb, out_frames ) ) { LOG_ERROR( "H3: video decode failed" ); return false; }
	}
	{
		H3Lease<H3AudioVAE> avae = comp.audio_vae();
		if ( !avae || !avae->decode( d_arows + static_cast<size_t>( n_aref ) * 32, audio_t, pcm ) ) { LOG_ERROR( "H3: audio decode failed" ); return false; }
	}
	res.t_decode = seconds_since( t0 );
	cudaFree( d_z );
	cudaFree( d_vrows );
	cudaFree( d_arows );
	LOG_INFO( "H3: decoded %d frames (%dx%d) and %zu audio samples in %.1f s", out_frames, width, height, pcm.size() / 2, res.t_decode );
	return true;
}

bool H3Run::mux() {
	report( "mux", 0, 1 );
	auto t0 = std::chrono::steady_clock::now();
	const std::string pcm_path = req.out + ".f32";
	{
		const size_t L = pcm.size() / 2;
		std::vector<float> inter( pcm.size() );
		for ( size_t i = 0; i < L; ++i ) { inter[2 * i] = pcm[i]; inter[2 * i + 1] = pcm[L + i]; }
		FILE* f = std::fopen( pcm_path.c_str(), "wb" );
		if ( !f ) { return false; }
		std::fwrite( inter.data(), sizeof( float ), inter.size(), f );
		std::fclose( f );
	}
	char cmd[2048];
	std::snprintf( cmd, sizeof( cmd ),
		"ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgb24 -s %dx%d -r 24 -i - -f f32le -ar 32000 -ac 2 -i '%s' -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p "
		"-c:a aac -b:a 192k -shortest '%s'", width, height, pcm_path.c_str(), req.out.c_str() );
	FILE* pipe = popen( cmd, "w" );
	if ( !pipe ) { return false; }
	// the decoder returns frames-last RGB [F, H, W, 3]
	std::fwrite( rgb.data(), 1, rgb.size(), pipe );
	const int rc = pclose( pipe );
	if ( !req.keep_pcm ) { std::remove( pcm_path.c_str() ); }
	res.t_mux = seconds_since( t0 );
	res.frames = out_frames; res.width = width; res.height = height; res.latent_t = latent_t; res.rows = lay.used;
	LOG_INFO( "H3: wrote %s (ffmpeg rc %d)", req.out.c_str(), rc );
	return rc == 0;
}

} // namespace engine::video
