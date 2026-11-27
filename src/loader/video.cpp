// Video input for the Qwen3-VL and GLM-5 vision towers: the checkpoints' video processors (transformers' video_processing_qwen3_vl.py and
// video_processing_glm5_next.py): frame sampling, the canvas of a video's frames, and the frames cut into steps of two (one temporal patch)
#include "engine/loader/video.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>

namespace engine::loader {

namespace {

nlohmann::json read_json_file( const std::string& path ) {
	std::ifstream f( path );
	if ( !f ) return nlohmann::json::value_t::discarded;
	return nlohmann::json::parse( std::string( std::istreambuf_iterator<char>( f ), std::istreambuf_iterator<char>() ), nullptr, false );
}

// numpy.linspace( start, stop, num )[i]
double linspace_at( double start, double stop, int num, int i ) {
	if ( num > 1 && i == num - 1 ) return stop;
	return num > 1 ? i * ( ( stop - start ) / ( num - 1 ) ) + start : start;
}

// Qwen3-VL's smart_resize of a video: dimensions rounded to multiples of factor, frames x height x width within [min_pixels, max_pixels]
void qwen_video_resize( int num_frames, int height, int width, int temporal_factor, int factor, int64_t min_pixels, int64_t max_pixels, int* out_h, int* out_w ) {
	if ( height < factor || width < factor ) {
		const double scale = std::max( static_cast<double>( factor ) / height, static_cast<double>( factor ) / width );
		height = static_cast<int>( height * scale );
		width = static_cast<int>( width * scale );
	}
	int h_bar = static_cast<int>( std::nearbyint( static_cast<double>( height ) / factor ) ) * factor;
	int w_bar = static_cast<int>( std::nearbyint( static_cast<double>( width ) / factor ) ) * factor;
	const int t_bar = static_cast<int>( std::nearbyint( static_cast<double>( num_frames ) / temporal_factor ) ) * temporal_factor;
	const double volume = static_cast<double>( num_frames ) * height * width;
	if ( static_cast<int64_t>( t_bar ) * h_bar * w_bar > max_pixels ) {
		const double beta = std::sqrt( volume / static_cast<double>( max_pixels ) );
		h_bar = std::max( factor, static_cast<int>( std::floor( height / beta / factor ) ) * factor );
		w_bar = std::max( factor, static_cast<int>( std::floor( width / beta / factor ) ) * factor );
	}
	else if ( static_cast<int64_t>( t_bar ) * h_bar * w_bar < min_pixels ) {
		const double beta = std::sqrt( static_cast<double>( min_pixels ) / volume );
		h_bar = static_cast<int>( std::ceil( height * beta / factor ) ) * factor;
		w_bar = static_cast<int>( std::ceil( width * beta / factor ) ) * factor;
	}
	*out_h = h_bar;
	*out_w = w_bar;
}

// GLM-5's smart_resize: an aligned canvas within [min_tokens, max_tokens] tokens of temporal_factor x factor x factor pixels
void glm_video_resize( int num_frames, int height, int width, int temporal_factor, int factor, int64_t min_tokens, int64_t max_tokens, int* out_h, int* out_w ) {
	const int64_t per_token = static_cast<int64_t>( temporal_factor ) * factor * factor;
	const int64_t min_pixels = min_tokens * per_token, max_pixels = max_tokens * per_token;
	const auto align = [factor]( int64_t v ) { return ( v + factor - 1 ) / factor * factor; };
	const int64_t frames = std::max<int64_t>( temporal_factor, static_cast<int64_t>( std::nearbyint( static_cast<double>( num_frames ) / temporal_factor ) ) * temporal_factor );
	int64_t ah = align( height ), aw = align( width );
	if ( frames * ah * aw < min_pixels ) {
		const double scale = std::sqrt( static_cast<double>( min_pixels ) / ( static_cast<double>( num_frames ) * height * width ) );
		ah = align( std::max<int64_t>( 1, static_cast<int64_t>( std::ceil( height * scale ) ) ) );
		aw = align( std::max<int64_t>( 1, static_cast<int64_t>( std::ceil( width * scale ) ) ) );
	}
	if ( frames * ah * aw > max_pixels ) {
		int64_t lo = 1, hi = height, bh = factor, bw = factor;
		while ( lo <= hi ) {
			const int64_t ch = ( lo + hi ) / 2;
			const int64_t cw = std::max<int64_t>( 1, static_cast<int64_t>( std::floor( static_cast<double>( width ) * ch / height ) ) );
			const int64_t th = align( ch ), tw = align( cw );
			if ( frames * th * tw <= max_pixels ) {
				bh = th;
				bw = tw;
				lo = ch + 1;
			}
			else {
				hi = ch - 1;
			}
		}
		ah = bh;
		aw = bw;
	}
	*out_h = static_cast<int>( ah );
	*out_w = static_cast<int>( aw );
}

// A frame on the plan's canvas as float RGB in [0, 255] (the step's patches read it)
std::vector<float> frame_canvas( const DecodedImage& img, const VideoPlan& plan, const VisionParams& p ) {
	if ( !p.canvas ) {
		if ( img.height == plan.height && img.width == plan.width ) return std::vector<float>( img.rgb.begin(), img.rgb.end() );
		return resize_bicubic( img.rgb.data(), img.height, img.width, plan.height, plan.width );
	}
	const std::vector<float> content = ( img.height == plan.content_h && img.width == plan.content_w ) ? std::vector<float>( img.rgb.begin(), img.rgb.end() )
	                                                                                                : resize_bicubic( img.rgb.data(), img.height, img.width, plan.content_h, plan.content_w );
	std::vector<float> px( static_cast<size_t>( plan.height ) * plan.width * 3, 0.0f );
	for ( int y = 0; y < plan.content_h; ++y )
		std::copy_n( &content[static_cast<size_t>( y ) * plan.content_w * 3], static_cast<size_t>( plan.content_w ) * 3, &px[static_cast<size_t>( y ) * plan.width * 3] );
	return px;
}

} // namespace

void read_video_params( const std::string& model_dir, VisionParams* p ) {
	p->video = false;
	if ( p->deepseek || p->gemma || p->temporal_patch_size != 2 ) return;
	const nlohmann::json pc = read_json_file( model_dir + "/processor_config.json" );
	nlohmann::json vp;
	if ( !p->canvas ) {
		vp = read_json_file( model_dir + "/video_preprocessor_config.json" );
		if ( vp.is_discarded() && !pc.is_discarded() && pc.contains( "video_processor" ) ) vp = pc["video_processor"];
	}
	else if ( !pc.is_discarded() && pc.contains( "video_processor" ) ) {
		vp = pc["video_processor"];
	}
	if ( vp.is_discarded() || !vp.is_object() ) return;
	p->video = true;
	p->video_fps = vp.value( "fps", 2.0 );
	if ( p->canvas ) {
		p->video_max_frames = vp.value( "max_frames", 2048 );
		p->video_min_tokens = vp.value( "min_image_tokens", 16 );
		p->video_max_tokens = vp.value( "max_image_tokens", 240000 );
	}
	else {
		p->video_min_frames = vp.value( "min_frames", 4 );
		p->video_max_frames = vp.value( "max_frames", 768 );
		p->video_frame_tokens = vp.value( "max_video_tokens", 768 );
		if ( vp.contains( "size" ) && vp["size"].is_object() ) {
			p->video_min_pixels = vp["size"].value( "shortest_edge", p->video_min_pixels );
			p->video_max_pixels = vp["size"].value( "longest_edge", p->video_max_pixels );
		}
	}
}

std::vector<int> sample_video_frames( const VisionParams& p, int total, double fps, double duration ) {
	std::vector<int> out;
	if ( total < 1 ) return out;
	if ( !( fps > 0.0 ) ) fps = 24.0;
	if ( !p.canvas ) {
		// Qwen3VLVideoProcessor.sample_frames: video_fps frames a second, evenly spaced and rounded
		int n = static_cast<int>( total / fps * p.video_fps );
		n = std::min( { std::max( n, p.video_min_frames ), p.video_max_frames, total } );
		for ( int i = 0; i < n; ++i ) out.push_back( static_cast<int>( std::nearbyint( linspace_at( 0.0, total - 1, n, i ) ) ) );
		return out;
	}
	// Glm5NextVideoProcessor.sample_frames: the first frame of every 1 / video_fps seconds, a count of duration x video_fps (evenly spaced
	// when the walk finds another), unique and even
	if ( !( duration > 0.0 ) ) duration = std::nearbyint( ( total - 1 ) / fps ) + 1;
	const int max_seconds = static_cast<int>( duration );
	const int extract = std::min( static_cast<int>( duration * p.video_fps ), p.video_max_frames );
	const double per_frame = 1.0 / fps;
	std::vector<int> idx;
	auto spaced = [&]( int start, int stop, int num ) {
		std::vector<int> v;
		for ( int i = 0; i < num; ++i ) v.push_back( static_cast<int>( std::floor( linspace_at( start, stop, num, i ) ) ) );
		return v;
	};
	if ( total < extract ) {
		idx = spaced( 0, total - 1, extract );
	}
	else {
		double current = 0.0;
		const double inv = 1.0 / p.video_fps;
		for ( int i = 0; i < total; ++i ) {
			if ( i * per_frame >= current ) {
				current += inv;
				idx.push_back( i );
				if ( current >= max_seconds ) break;
			}
		}
	}
	if ( static_cast<int>( idx.size() ) < extract ) idx = idx.empty() ? spaced( 0, std::max( total - 1, 0 ), extract ) : spaced( idx.front(), idx.back(), extract );
	else if ( static_cast<int>( idx.size() ) > extract ) idx = spaced( 0, total - 1, extract );
	for ( int i : idx ) {
		if ( std::find( out.begin(), out.end(), i ) == out.end() ) out.push_back( i );
	}
	if ( out.empty() ) out.push_back( 0 );   // a clip shorter than one sampling interval: its first frame
	if ( out.size() & 1 ) out.push_back( out.back() );
	return out;
}

bool plan_video( int count, int height, int width, const VisionParams& p, int max_tokens, VideoPlan* out, std::string* error ) {
	auto fail = [error]( const char* why ) {
		if ( error ) *error = why;
		return false;
	};
	if ( count < 1 ) return fail( "the video has no frames" );
	if ( height < 1 || width < 1 ) return fail( "empty video frame" );
	if ( std::max( width, height ) > 200 * std::min( width, height ) ) return fail( "video aspect ratio above 200:1" );
	if ( max_tokens < 1 ) return fail( "video input is off (max_video_tokens 0)" );
	const int f = p.patch_size * p.merge_size, tp = p.temporal_patch_size;
	const int64_t ff = static_cast<int64_t>( f ) * f, step_px = std::max<int64_t>( p.max_pixels, ff );
	int n = std::min( count, std::max( 1, p.video_max_frames ) );
	double squeeze = 1.0;   // the engine's per-step limit scaled down while rounding overshoots it
	for ( int guard = 0; guard < 200; ++guard ) {
		const int steps = ( n + tp - 1 ) / tp;
		// the engine's limits on a step's pixels: the tower's largest image and an even share of max_tokens
		const double step_cap = std::min( static_cast<double>( step_px ), static_cast<double>( max_tokens ) * ff / steps ) * squeeze;
		int h = 0, w = 0, ch = 0, cw = 0;
		int64_t step_tokens = 1;   // what a step costs at its smallest
		if ( step_cap >= ff ) {
			if ( p.canvas ) {
				// GLM-5: the processor's token budget of the even frame count, within the engine's
				const int frames = steps * tp;
				const int64_t cap = std::min<int64_t>( p.video_max_tokens, static_cast<int64_t>( step_cap * steps / ff ) );
				glm_video_resize( frames, height, width, tp, f, p.video_min_tokens, cap, &h, &w );
				double scale = std::min( static_cast<double>( h ) / height, static_cast<double>( w ) / width );
				if ( static_cast<double>( frames ) * height * width >= static_cast<double>( tp ) * ff * p.video_min_tokens ) scale = std::min( 1.0, scale );
				ch = std::max( 1, std::min( h, static_cast<int>( std::floor( height * scale ) ) ) );
				cw = std::max( 1, std::min( w, static_cast<int>( std::floor( width * scale ) ) ) );
			}
			else {
				// Qwen3-VL with qwen-vl-utils' per-frame cap (at most video_frame_tokens tokens or the budget's even share, at least 1.05 x
				// the minimum), then the engine's
				int64_t per_frame = std::max( std::min( static_cast<int64_t>( p.video_frame_tokens ) * ff, p.video_max_pixels / n ),
					static_cast<int64_t>( p.video_min_pixels * 1.05 ) );
				per_frame = std::min( per_frame, static_cast<int64_t>( step_cap ) );
				qwen_video_resize( std::max( n, tp ), height, width, tp, f, p.video_min_pixels, per_frame * n, &h, &w );
				ch = h;
				cw = w;
			}
			const int64_t px = static_cast<int64_t>( h ) * w, tokens = steps * ( px / ff );
			if ( px <= step_px && tokens <= max_tokens ) {
				out->height = h;
				out->width = w;
				out->content_h = ch;
				out->content_w = cw;
				out->frames.clear();
				for ( int i = 0; i < n; ++i ) out->frames.push_back( n == count ? i : static_cast<int>( std::nearbyint( linspace_at( 0.0, count - 1, n, i ) ) ) );
				while ( out->frames.size() % tp ) out->frames.push_back( out->frames.back() );
				return true;
			}
			if ( px > ff ) {
				squeeze *= 0.97 * std::min( static_cast<double>( step_px ) / px, static_cast<double>( max_tokens ) / tokens );
				continue;
			}
			step_tokens = px / ff;
		}
		// a step at its smallest still over the budget: fewer frames, as many steps as fit
		if ( n <= 1 ) break;
		n = static_cast<int>( std::max<int64_t>( 1, std::min<int64_t>( n - 1, max_tokens / step_tokens * tp ) ) );
		squeeze = 1.0;
	}
	return fail( "the video does not fit the video token budget (max_video_tokens)" );
}

void prepare_video_step( const DecodedImage& a, const DecodedImage& b, const VideoPlan& plan, const VisionParams& p, PreparedImage* out ) {
	const int h = plan.height, w = plan.width, ps = p.patch_size, m = p.merge_size, tp = p.temporal_patch_size;
	const std::vector<float> fa = frame_canvas( a, plan, p );
	const std::vector<float> fb = &a == &b ? fa : frame_canvas( b, plan, p );
	const std::vector<float>* frame[2] = { &fa, &fb };
	const int gh = h / ps, gw = w / ps, dim = 3 * tp * ps * ps;
	out->grid_h = gh;
	out->grid_w = gw;
	out->llm_h = out->llm_w = 0;
	out->patches.assign( static_cast<size_t>( gh ) * gw * dim, 0.0f );
	// prepare_image's layout, the temporal slots from the step's two frames
	size_t row = 0;
	for ( int by = 0; by < gh / m; ++by ) {
		for ( int bx = 0; bx < gw / m; ++bx ) {
			for ( int my = 0; my < m; ++my ) {
				for ( int mx = 0; mx < m; ++mx, ++row ) {
					float* dst = &out->patches[row * dim];
					const int y0 = ( by * m + my ) * ps, x0 = ( bx * m + mx ) * ps;
					for ( int c = 0; c < 3; ++c ) {
						for ( int t = 0; t < tp; ++t ) {
							const std::vector<float>& px = *frame[std::min( t, 1 )];
							for ( int y = 0; y < ps; ++y ) {
								for ( int x = 0; x < ps; ++x ) {
									const float v = px[( static_cast<size_t>( y0 + y ) * w + x0 + x ) * 3 + c] * ( 1.0f / 255.0f );
									*dst++ = ( v - p.mean[c] ) / p.stddev[c];
								}
							}
						}
					}
				}
			}
		}
	}
	const int dims[2] = { gh, gw };
	out->hash = hash_bytes( dims, sizeof( dims ), hash_bytes( b.rgb.data(), b.rgb.size(), hash_bytes( a.rgb.data(), a.rgb.size() ) ) );
}

bool prepare_video( const VideoFrames& video, const VisionParams& p, int max_tokens, PreparedVideo* out, std::string* error ) {
	const auto& frames = video.frames;
	if ( frames.empty() ) {
		if ( error ) *error = "the video has no frames";
		return false;
	}
	if ( video.times.size() != frames.size() ) {
		if ( error ) *error = "a video needs one time per frame";
		return false;
	}
	for ( const DecodedImage& f : frames ) {
		if ( f.width != frames[0].width || f.height != frames[0].height || f.rgb.size() != static_cast<size_t>( f.width ) * f.height * 3 ) {
			if ( error ) *error = "the frames of a video need one size (and width x height x 3 bytes of RGB)";
			return false;
		}
	}
	VideoPlan plan;
	if ( !plan_video( static_cast<int>( frames.size() ), frames[0].height, frames[0].width, p, max_tokens, &plan, error ) ) return false;
	out->steps.assign( plan.steps(), PreparedImage() );
	out->times.assign( plan.steps(), 0.0 );
	for ( int s = 0; s < plan.steps(); ++s ) {
		const int ia = plan.frames[2 * s], ib = plan.frames[2 * s + 1];
		prepare_video_step( frames[ia], frames[ib], plan, p, &out->steps[s] );
		out->times[s] = p.canvas ? video.times[ia] : 0.5 * ( video.times[ia] + video.times[ib] );
	}
	return true;
}

} // namespace engine::loader
