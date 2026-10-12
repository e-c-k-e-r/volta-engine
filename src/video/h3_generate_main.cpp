// h3_generate: MiniMax-H3 text-to-video-and-audio from the command line.
//   h3_generate --root <FL2VA dir> --dit <comfy int8 checkpoint> --prompt TEXT --out out.mp4 [--lora PATH[:scale]]... [--steps 4] [--size WxH]
//               [--frames 124] [--seed 1] [--video-shift S] [--audio-shift S] [--device 0]
//   keyframes (FL2VA): --image PATH [--frame-index 0|-1]; references (Ref2VA): --dit-ref2va <ref2va checkpoint> --ref-image PATH ... [--ref-audio PATH ...]
#include "engine/core/log.h"
#include "engine/loader/image.h"
#include "engine/video/h3_pipeline.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main( int argc, char** argv ) {
	engine::video::H3Paths paths;
	engine::video::H3GenRequest req;
	bool size_given = false, resident = false;
	int repeat = 1;
	for ( int i = 1; i < argc; ++i ) {
		const std::string a = argv[i];
		auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
		if ( a == "--root" ) { paths.root = next(); }
		else if ( a == "--dit" ) { paths.dit = next(); }
		else if ( a == "--resident" ) { resident = true; }
		else if ( a == "--no-trim" ) { paths.trim_scratch = false; }
		else if ( a == "--tp" ) { paths.tensor_parallel = true; }
		else if ( a == "--text-encoder" ) { paths.text_encoder = next(); }
		else if ( a == "--repeat" ) { repeat = std::max( 1, std::atoi( next() ) ); }
		else if ( a == "--dit-ref2va" ) { paths.dit_ref2va = next(); }
		else if ( a == "--device" ) { paths.device = std::atoi( next() ); }
		else if ( a == "--devices" ) {
			paths.devices.clear();
			for ( const char* p = next(); *p; ) { paths.devices.push_back( std::atoi( p ) ); while ( *p && *p != ',' ) { ++p; } if ( *p == ',' ) { ++p; } }
		}
		else if ( a == "--prompt" ) { req.prompt = next(); }
		else if ( a == "--out" ) { req.out = next(); }
		else if ( a == "--steps" ) { req.steps = std::atoi( next() ); }
		else if ( a == "--frames" ) { req.frames = std::atoi( next() ); }
		else if ( a == "--seed" ) { req.seed = std::strtoull( next(), nullptr, 10 ); }
		else if ( a == "--video-shift" ) { req.video_shift = static_cast<float>( std::atof( next() ) ); }
		else if ( a == "--audio-shift" ) { req.audio_shift = static_cast<float>( std::atof( next() ) ); }
		else if ( a == "--size" ) { std::sscanf( next(), "%dx%d", &req.width, &req.height ); size_given = true; }
		else if ( a == "--image" ) { req.images.push_back( next() ); }
		else if ( a == "--ref-image" ) { req.ref_images.push_back( next() ); }
		else if ( a == "--ref-video" ) { req.ref_videos.push_back( next() ); }
		else if ( a == "--ref-audio" ) { req.ref_audios.push_back( next() ); }
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
	const bool ref_task = !req.ref_images.empty() || !req.ref_audios.empty() || !req.ref_videos.empty();
	if ( paths.root.empty() || ( paths.dit.empty() && !( ref_task && !paths.dit_ref2va.empty() ) ) || req.prompt.empty() || req.out.empty() ) {
		std::fprintf( stderr, "usage: h3_generate --root DIR --dit CKPT --prompt TEXT --out FILE.mp4 [--lora PATH[:scale]] [--steps N] [--size WxH] [--frames N] [--seed N] [--devices 0,1,2,3] [--tp] [--text-encoder EXL3_DIR]\n" );
		return 1;
	}
	const std::string* first_image = !req.images.empty() ? &req.images[0] : ( !req.ref_images.empty() ? &req.ref_images[0] : nullptr );
	if ( first_image && !size_given ) {
		// the canvas follows the first image's aspect ratio
		FILE* f = std::fopen( first_image->c_str(), "rb" );
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
	// --repeat N runs the same job N times in this process (with --resident the transformer stays loaded in between): a regression check for the resident path
	engine::video::H3Engine engine( paths, resident );
	engine::video::H3GenResult r;
	bool ok = true;
	const std::string out = req.out;
	for ( int i = 0; i < repeat && ok; ++i ) {
		if ( repeat > 1 ) { req.out = out.substr( 0, out.rfind( '.' ) ) + "_" + std::to_string( i ) + out.substr( out.rfind( '.' ) ); }
		ok = engine.generate( req, &r );
		if ( ok ) {
			std::printf( "done: %d frames %dx%d, %d rows; text %.1f s, load %.1f s, denoise %.1f s, decode %.1f s, mux %.1f s\n", r.frames, r.width, r.height, r.rows, r.t_text, r.t_load,
				r.t_denoise, r.t_decode, r.t_mux );
		}
	}
	return ok ? 0 : 1;
}
