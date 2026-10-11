// h3_generate: MiniMax-H3 text-to-video-and-audio from the command line.
//   h3_generate --root <FL2VA dir> --dit <comfy int8 checkpoint> --prompt TEXT --out out.mp4 [--lora PATH[:scale]]... [--steps 4] [--size WxH]
//               [--frames 124] [--seed 1] [--video-shift S] [--audio-shift S] [--device 0]
#include "engine/core/log.h"
#include "engine/loader/image.h"
#include "engine/video/h3_pipeline.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main( int argc, char** argv ) {
	engine::video::H3Paths paths;
	engine::video::H3GenRequest req;
	bool size_given = false;
	for ( int i = 1; i < argc; ++i ) {
		const std::string a = argv[i];
		auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
		if ( a == "--root" ) { paths.root = next(); }
		else if ( a == "--dit" ) { paths.dit = next(); }
		else if ( a == "--device" ) { paths.device = std::atoi( next() ); }
		else if ( a == "--prompt" ) { req.prompt = next(); }
		else if ( a == "--out" ) { req.out = next(); }
		else if ( a == "--steps" ) { req.steps = std::atoi( next() ); }
		else if ( a == "--frames" ) { req.frames = std::atoi( next() ); }
		else if ( a == "--seed" ) { req.seed = std::strtoull( next(), nullptr, 10 ); }
		else if ( a == "--video-shift" ) { req.video_shift = static_cast<float>( std::atof( next() ) ); }
		else if ( a == "--audio-shift" ) { req.audio_shift = static_cast<float>( std::atof( next() ) ); }
		else if ( a == "--size" ) { std::sscanf( next(), "%dx%d", &req.width, &req.height ); size_given = true; }
		else if ( a == "--image" ) { req.images.push_back( next() ); }
		else if ( a == "--frame-index" ) { req.frame_indices.push_back( std::atoi( next() ) ); }
		else if ( a == "--lora" ) {
			std::string v = next();
			engine::video::H3Lora l;
			const size_t c = v.rfind( ':' );
			if ( c != std::string::npos && c > 1 && v.find( '/', c ) == std::string::npos ) { l.scale = static_cast<float>( std::atof( v.c_str() + c + 1 ) ); v.resize( c ); }
			l.path = v;
			req.loras.push_back( l );
		}
		else { std::fprintf( stderr, "unknown argument %s\n", a.c_str() ); return 1; }
	}
	if ( paths.root.empty() || paths.dit.empty() || req.prompt.empty() || req.out.empty() ) {
		std::fprintf( stderr, "usage: h3_generate --root DIR --dit CKPT --prompt TEXT --out FILE.mp4 [--lora PATH[:scale]] [--steps N] [--size WxH] [--frames N] [--seed N]\n" );
		return 1;
	}
	if ( !req.images.empty() && !size_given ) {
		// FL2VA: the canvas follows the first image's aspect ratio
		FILE* f = std::fopen( req.images[0].c_str(), "rb" );
		if ( f ) {
			std::vector<uint8_t> bytes;
			uint8_t buf[65536];
			size_t n;
			while ( ( n = std::fread( buf, 1, sizeof( buf ), f ) ) > 0 ) { bytes.insert( bytes.end(), buf, buf + n ); }
			std::fclose( f );
			engine::loader::DecodedImage di;
			std::string err;
			if ( engine::loader::decode_image( bytes.data(), bytes.size(), &di, &err ) ) { engine::video::h3_output_canvas( static_cast<double>( di.width ) / di.height, &req.height, &req.width ); }
		}
	}
	engine::video::H3GenResult r;
	const bool ok = engine::video::h3_generate( paths, req, &r );
	if ( ok ) {
		std::printf( "done: %d frames %dx%d, %d rows; text %.1f s, load %.1f s, denoise %.1f s, decode %.1f s, mux %.1f s\n", r.frames, r.width, r.height, r.rows, r.t_text, r.t_load,
			r.t_denoise, r.t_decode, r.t_mux );
	}
	return ok ? 0 : 1;
}
