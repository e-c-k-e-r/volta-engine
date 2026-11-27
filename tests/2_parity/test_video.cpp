// Video preprocessing against transformers' video processors (tests/tools/gen_video_goldens.py): Qwen3-VL's and GLM-5's frame samples and
// canvases, the engine's token budget, a step's temporal patch layout and times, and (with FFmpeg) a decoded clip's steps
#include "engine/loader/video.h"

#include <nlohmann/json.hpp>

#include <climits>
#include <cmath>
#include <cstdio>
#include "test_util.h"
#include <string>

#ifndef TEST_DATA_DIR
#error "TEST_DATA_DIR must point at tests/data"
#endif

using engine::loader::DecodedImage;
using engine::loader::VisionParams;

namespace {

int failures = 0;

void check( bool ok, const std::string& what ) {
	std::printf( "%s %s\n", ok ? "PASS" : "FAIL", what.c_str() );
	failures += ok ? 0 : 1;
}

VisionParams qwen_params() {
	VisionParams p;
	p.patch_size = 16;
	p.merge_size = 2;
	p.temporal_patch_size = 2;
	p.video = true;
	p.max_pixels = INT_MAX;   // no tower limit: the processor's canvas alone
	return p;
}

VisionParams glm_params() {
	VisionParams p;
	p.patch_size = 14;
	p.merge_size = 2;
	p.temporal_patch_size = 2;
	p.canvas = true;
	p.video = true;
	p.video_max_frames = 2048;
	p.max_pixels = INT_MAX;
	return p;
}

DecodedImage solid( int w, int h, uint8_t r, uint8_t g, uint8_t b ) {
	DecodedImage img;
	img.width = w;
	img.height = h;
	img.rgb.resize( static_cast<size_t>( w ) * h * 3 );
	for ( size_t i = 0; i < img.rgb.size(); i += 3 ) {
		img.rgb[i] = r;
		img.rgb[i + 1] = g;
		img.rgb[i + 2] = b;
	}
	return img;
}

// The mean of one channel over a step's temporal slot t (patches of 3 x 2 x P x P values, channel-major)
double slot_mean( const engine::loader::PreparedImage& s, int P, int c, int t ) {
	const int dim = 3 * 2 * P * P;
	double sum = 0.0;
	const size_t rows = s.patches.size() / dim;
	for ( size_t r = 0; r < rows; ++r )
		for ( int k = 0; k < P * P; ++k ) sum += s.patches[r * dim + ( c * 2 + t ) * P * P + k];
	return sum / ( rows * P * P );
}

int plan_tokens( const engine::loader::VideoPlan& plan, const VisionParams& p ) {
	const int f = p.patch_size * p.merge_size;
	return plan.steps() * ( plan.height / f ) * ( plan.width / f );
}

} // namespace

int main() {
	const std::string dir = TEST_DATA_DIR;
	const auto cases = nlohmann::json::parse( read_file( dir + "/video_cases.json" ) );
	const VisionParams qp = qwen_params(), gp = glm_params();

	for ( const auto& c : cases["qwen_sample"] ) {
		const std::vector<int> got = engine::loader::sample_video_frames( qp, c["total"].get<int>(), c["fps"].get<double>(), 0.0 );
		check( got == c["frames"].get<std::vector<int>>(), "qwen sample " + std::to_string( c["total"].get<int>() ) + " frames at " + c["fps"].dump() + " fps -> " + std::to_string( got.size() ) );
	}
	for ( const auto& c : cases["glm_sample"] ) {
		const std::vector<int> got = engine::loader::sample_video_frames( gp, c["total"].get<int>(), c["fps"].get<double>(), c["duration"].get<double>() );
		std::vector<int> want = c["frames"].get<std::vector<int>>();
		if ( want.empty() ) want = { 0, 0 };   // the processor samples nothing from a clip shorter than an interval; the engine takes its first frame
		check( got == want, "glm sample " + std::to_string( c["total"].get<int>() ) + " frames at " + c["fps"].dump() + " fps, " + c["duration"].dump() + " s -> " + std::to_string( got.size() ) );
	}
	for ( const char* family : { "qwen_canvas", "glm_canvas" } ) {
		const VisionParams& p = family[0] == 'q' ? qp : gp;
		for ( const auto& c : cases[family] ) {
			engine::loader::VideoPlan plan;
			std::string err;
			const int n = c["frames"].get<int>(), h = c["height"].get<int>(), w = c["width"].get<int>();
			const auto want = c["canvas"].get<std::vector<int>>();
			const bool ok = engine::loader::plan_video( n, h, w, p, INT_MAX, &plan, &err ) && plan.height == want[0] && plan.width == want[1] &&
				plan.content_h == want[2] && plan.content_w == want[3] && plan.steps() == ( n + 1 ) / 2;
			char what[160];
			std::snprintf( what, sizeof( what ), "%s %d x %dx%d -> %dx%d (content %dx%d), want %dx%d (%dx%d) %s", family, n, h, w, plan.height, plan.width,
				plan.content_h, plan.content_w, want[0], want[1], want[2], want[3], err.c_str() );
			check( ok, what );
		}
	}

	// The engine's budget: whole-video tokens and the tower's step, frames dropped where the smallest step still exceeds the budget
	for ( const VisionParams* p : { &qp, &gp } ) {
		VisionParams capped = *p;
		const int f = p->patch_size * p->merge_size;
		capped.max_pixels = 256 * f * f;
		struct Case { int frames, h, w, budget; };
		for ( const Case& c : { Case{ 120, 1080, 1920, 4096 }, Case{ 20, 2160, 3840, 100000 }, Case{ 768, 720, 1280, 256 }, Case{ 7, 480, 640, 3 } } ) {
			engine::loader::VideoPlan plan;
			std::string err;
			const bool planned = engine::loader::plan_video( c.frames, c.h, c.w, capped, c.budget, &plan, &err );
			const int tokens = planned ? plan_tokens( plan, capped ) : 0;
			const bool ok = planned && tokens <= c.budget && static_cast<int64_t>( plan.height ) * plan.width <= capped.max_pixels && plan.steps() > 0 &&
				plan.frames.back() <= c.frames - 1;
			char what[200];
			std::snprintf( what, sizeof( what ), "%s budget: %d x %dx%d within %d tokens -> %d steps of %dx%d, %d tokens %s", p->canvas ? "glm" : "qwen", c.frames, c.h, c.w,
				c.budget, plan.steps(), plan.height, plan.width, tokens, err.c_str() );
			check( ok, what );
		}
	}

	// A step's temporal slots hold its two frames; times: Qwen3-VL the pair's mean, GLM-5 the first frame's
	{
		engine::loader::VideoFrames v;
		v.frames = { solid( 64, 64, 255, 0, 0 ), solid( 64, 64, 0, 0, 255 ), solid( 64, 64, 0, 255, 0 ) };
		v.times = { 0.0, 0.5, 1.0 };
		engine::loader::PreparedVideo q, g;
		std::string err;
		const bool okq = engine::loader::prepare_video( v, qp, INT_MAX, &q, &err );
		const bool okg = engine::loader::prepare_video( v, gp, INT_MAX, &g, &err );
		check( okq && q.steps.size() == 2 && std::fabs( q.times[0] - 0.25 ) < 1e-9 && std::fabs( q.times[1] - 1.0 ) < 1e-9, "qwen step times" );
		check( okg && g.steps.size() == 2 && g.times[0] == 0.0 && g.times[1] == 1.0, "glm step times" );
		if ( okq && q.steps.size() == 2 ) {
			const auto& s0 = q.steps[0];
			const bool layout = std::fabs( slot_mean( s0, 16, 0, 0 ) - 1.0 ) < 1e-5 && std::fabs( slot_mean( s0, 16, 0, 1 ) + 1.0 ) < 1e-5 &&
				std::fabs( slot_mean( s0, 16, 2, 0 ) + 1.0 ) < 1e-5 && std::fabs( slot_mean( s0, 16, 2, 1 ) - 1.0 ) < 1e-5 &&
				std::fabs( slot_mean( q.steps[1], 16, 1, 0 ) - 1.0 ) < 1e-5 && std::fabs( slot_mean( q.steps[1], 16, 1, 1 ) - 1.0 ) < 1e-5 && s0.hash != q.steps[1].hash;
			check( layout, "qwen step slots: red then blue, the odd last frame repeated" );
		}
	}

	// A decoded clip (30 frames at 10 fps: a red, a green and a blue second)
	if ( engine::loader::video_decoding_available() ) {
		const std::string clip = read_file( dir + "/video_rgb.mp4" );
		for ( const VisionParams* p : { &qp, &gp } ) {
			engine::loader::PreparedVideo v;
			std::string err;
			const bool ok = engine::loader::decode_video( reinterpret_cast<const uint8_t*>( clip.data() ), clip.size(), *p, INT_MAX, &v, &err );
			const std::vector<double> want_t = p->canvas ? std::vector<double>{ 0.0, 1.0, 2.0 } : std::vector<double>{ 0.3, 1.45, 2.6 };
			bool good = ok && v.steps.size() == 3;
			for ( size_t s = 0; good && s < 3; ++s ) {
				good = std::fabs( v.times[s] - want_t[s] ) < 1e-9;
				// GLM-5's canvas pads black past the content: its corner patch is content
				for ( int c = 0; c < 3 && good; ++c ) {
					const engine::loader::PreparedImage& st = v.steps[s];
					const int P = p->patch_size, dim = 3 * 2 * P * P;
					double sum = 0.0;
					for ( int k = 0; k < P * P; ++k ) sum += st.patches[( c * 2 ) * P * P + k] * p->stddev[c] + p->mean[c];   // patch 0, slot 0, in [0, 1]
					const double want = c == static_cast<int>( s ) ? 1.0 : 0.0;
					good = std::fabs( sum / ( P * P ) - want ) < 0.1;
					(void) dim;
				}
			}
			char what[160];
			std::snprintf( what, sizeof( what ), "%s decode: %zu steps, times %.2f %.2f %.2f, red green blue %s", p->canvas ? "glm" : "qwen", v.steps.size(),
				v.times.size() > 0 ? v.times[0] : -1.0, v.times.size() > 1 ? v.times[1] : -1.0, v.times.size() > 2 ? v.times[2] : -1.0, err.c_str() );
			check( good, what );
		}
	}
	else {
		std::printf( "SKIP decode (built without FFmpeg)\n" );
	}

	std::printf( failures ? "test_video: FAIL\n" : "test_video: PASS\n" );
	return failures ? 1 : 0;
}
