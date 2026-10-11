// H3 generation: output canvas and frame bookkeeping, packed-layout builders, request validation and the stage driver (see h3_pipeline.h, h3_run.h).
#include "engine/video/h3_pipeline.h"
#include "h3_run.h"
#include "engine/core/log.h"
#include "engine/loader/image.h"
#include "engine/video/h3_image.h"
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace engine::video {

namespace {

constexpr double kFrameRescale = 5.0 / 3.0;
constexpr int kFramePerToken[5] = { 1, 4, 4, 4, 4 };
constexpr int kInterp = 32;

std::vector<double> axis_from_sqrt_area( int dim, int patch, double sqrt_area ) {
	const double ratio = dim / sqrt_area;
	const double left = ( 1.0 - ratio ) / 2.0, right = left + ratio;
	const int n = dim / patch;
	std::vector<double> g( n );
	for ( int i = 0; i < n; ++i ) { g[i] = ( left + ( right - left ) * i / n ) * kInterp; } // linspace, endpoint excluded
	return g;
}

double temporal_position_span( int n ) {
	// numpy pairwise sum in the checkpoint; the sequential sum differs only in the last ulp for n >= 16 and only enters the last-frame anchor
	double s = 0;
	for ( int k = 0; k < n; ++k ) { s += kFrameRescale * kFramePerToken[k % 5]; }
	return s;
}

} // namespace

void h3_output_canvas( double aspect, int* height, int* width ) {
	double w, h;
	if ( aspect >= 1.0 ) { w = 768.0 * aspect; h = 768.0; } else { w = 768.0; h = 768.0 / aspect; }
	const double area = w * h, cap = 768.0 * 1344.0;
	if ( area > cap ) { const double s = std::sqrt( cap / area ); w *= s; h *= s; }
	auto align = []( double v ) { return std::max( 32, static_cast<int>( std::lround( v / 32.0 ) ) * 32 ); };
	*height = align( h );
	*width = align( w );
}

int h3_align_frames( int frames ) {
	if ( frames <= 0 ) { return 1; }
	while ( frames % 17 != 5 ) { ++frames; }
	return frames;
}
int h3_video_latent_t( int frames ) { return frames <= 5 ? 2 : ( ( frames - 5 ) / 17 ) * 5 + 2; }
int h3_audio_latent_t( double seconds ) { return static_cast<int>( std::lround( seconds * 40.0 ) ); }

H3Layout h3_build_layout( int text_len, int latent_t, int latent_h, int latent_w, int audio_t, const std::vector<int>& keyframes, int frame_count ) {
	H3Layout L;
	L.text_len = text_len; L.latent_t = latent_t; L.latent_h = latent_h; L.latent_w = latent_w; L.audio_t = audio_t;
	const int ph = latent_h / 2, pw = latent_w / 2, frame_rows = ph * pw;
	const int cond_rows = static_cast<int>( keyframes.size() ) * frame_rows;
	const int video_rows = latent_t * frame_rows, audio_rows = audio_t * 2;
	L.used = text_len + cond_rows + audio_rows + video_rows;
	L.n_img = cond_rows + video_rows;
	L.n_audio = audio_rows;
	L.tags.assign( L.used, 0 );
	L.position_ids.assign( static_cast<size_t>( L.used ) * 3, 0.0f );
	std::vector<double> g( static_cast<size_t>( L.used ) * 3, 0.0 );
	const int cond0 = text_len, audio0 = cond0 + cond_rows, video0 = audio0 + audio_rows;
	for ( int i = 0; i < text_len; ++i ) { L.text_pos.push_back( i ); L.tags[i] = 1; g[static_cast<size_t>( i ) * 3] = i; }
	for ( int i = 0; i < cond_rows; ++i ) { L.img_pos.push_back( cond0 + i ); L.update_mask.push_back( 0 ); }
	for ( int i = 0; i < video_rows; ++i ) { L.img_pos.push_back( video0 + i ); L.update_mask.push_back( 1 ); }
	for ( int i = 0; i < audio_rows; ++i ) { L.audio_pos.push_back( audio0 + i ); L.tags[audio0 + i] = 2; }
	// video grid
	std::vector<double> t_grid( latent_t );
	{
		double acc = static_cast<double>( text_len );
		for ( int k = 0; k < latent_t; ++k ) {
			t_grid[k] = acc;
			acc += kFrameRescale * kFramePerToken[k % 5];
		}
	}
	const double sqrt_area = std::sqrt( static_cast<double>( latent_h ) * latent_w );
	const std::vector<double> hg = axis_from_sqrt_area( latent_h, 2, sqrt_area ), wg = axis_from_sqrt_area( latent_w, 2, sqrt_area );
	for ( int t = 0; t < latent_t; ++t ) {
		for ( int h = 0; h < ph; ++h ) {
			for ( int w = 0; w < pw; ++w ) {
				const size_t r = static_cast<size_t>( video0 ) + ( static_cast<size_t>( t ) * ph + h ) * pw + w;
				g[r * 3] = t_grid[t]; g[r * 3 + 1] = hg[h]; g[r * 3 + 2] = wg[w];
			}
		}
	}
	for ( size_t b = 0; b < keyframes.size(); ++b ) {
		const bool first = keyframes[b] == 0;
		const double cond_t = first ? static_cast<double>( text_len ) : static_cast<double>( text_len ) + temporal_position_span( latent_t ) - kFrameRescale;
		( void )frame_count;
		for ( int h = 0; h < ph; ++h ) {
			for ( int w = 0; w < pw; ++w ) {
				const size_t r = static_cast<size_t>( cond0 ) + b * frame_rows + static_cast<size_t>( h ) * pw + w;
				g[r * 3] = cond_t; g[r * 3 + 1] = hg[h]; g[r * 3 + 2] = wg[w];
			}
		}
	}
	for ( int c = 0; c < 2; ++c ) {
		for ( int t = 0; t < audio_t; ++t ) {
			const size_t r = static_cast<size_t>( audio0 ) + static_cast<size_t>( c ) * audio_t + t;
			g[r * 3] = static_cast<double>( text_len ) + t;
			g[r * 3 + 2] = c == 0 ? wg.front() : wg.back();
		}
	}
	for ( size_t i = 0; i < g.size(); ++i ) { L.position_ids[i] = static_cast<float>( g[i] ); }
	return L;
}

H3Layout h3_build_layout_ref( int text_len, int latent_t, int latent_h, int latent_w, int audio_t, const std::vector<H3RefBlock>& blocks ) {
	H3Layout L;
	L.text_len = text_len; L.latent_t = latent_t; L.latent_h = latent_h; L.latent_w = latent_w; L.audio_t = audio_t;
	const int ph = latent_h / 2, pw = latent_w / 2, frame_rows = ph * pw;
	int ref_visual = 0, ref_audio = 0;
	for ( const H3RefBlock& b : blocks ) {
		if ( b.kind != H3RefBlock::Audio ) { ref_visual += b.latent_t * ( b.latent_h / 2 ) * ( b.latent_w / 2 ); }
		if ( b.kind != H3RefBlock::Image ) { ref_audio += b.ref_audio_t * 2; }
	}
	const int video_rows = latent_t * frame_rows, audio_rows = audio_t * 2;
	L.used = text_len + ref_visual + ref_audio + audio_rows + video_rows;
	L.n_img = ref_visual + video_rows;
	L.n_audio = ref_audio + audio_rows;
	L.tags.assign( L.used, 0 );
	std::vector<double> g( static_cast<size_t>( L.used ) * 3, 0.0 );
	for ( int i = 0; i < text_len; ++i ) { L.text_pos.push_back( i ); L.tags[i] = 1; g[static_cast<size_t>( i ) * 3] = i; }
	const double target_area = std::sqrt( static_cast<double>( latent_h ) * latent_w );
	const std::vector<double> hg = axis_from_sqrt_area( latent_h, 2, target_area ), wg = axis_from_sqrt_area( latent_w, 2, target_area );
	auto t_grid = [&]( int n, double origin ) {
		std::vector<double> v( n );
		double acc = origin;
		for ( int k = 0; k < n; ++k ) { v[k] = acc; acc += kFrameRescale * kFramePerToken[k % 5]; }
		return v;
	};
	auto video_rows_at = [&]( int row0, int T, int h, int w, double origin ) {
		const double area = std::sqrt( static_cast<double>( h ) * w );
		const std::vector<double> rh = axis_from_sqrt_area( h, 2, area ), rw = axis_from_sqrt_area( w, 2, area ), tg = t_grid( T, origin );
		const int rph = h / 2, rpw = w / 2;
		for ( int t = 0; t < T; ++t ) {
			for ( int y = 0; y < rph; ++y ) {
				for ( int x = 0; x < rpw; ++x ) {
					const size_t r = static_cast<size_t>( row0 ) + ( static_cast<size_t>( t ) * rph + y ) * rpw + x;
					g[r * 3] = tg[t]; g[r * 3 + 1] = rh[y]; g[r * 3 + 2] = rw[x];
					L.img_pos.push_back( static_cast<int>( r ) );
					L.update_mask.push_back( 0 );
				}
			}
		}
	};
	auto audio_rows_at = [&]( int row0, int T, double origin, double w_first, double w_last, bool target ) {
		for ( int c = 0; c < 2; ++c ) {
			for ( int t = 0; t < T; ++t ) {
				const size_t r = static_cast<size_t>( row0 ) + static_cast<size_t>( c ) * T + t;
				g[r * 3] = origin + t;
				g[r * 3 + 2] = c == 0 ? w_first : w_last;
				L.audio_pos.push_back( static_cast<int>( r ) );
				L.audio_update.push_back( target ? 1 : 0 );
				L.tags[r] = 2;
			}
		}
	};
	double t_cursor = text_len;
	int cursor = text_len;
	for ( const H3RefBlock& b : blocks ) {
		if ( b.kind == H3RefBlock::Image ) {
			video_rows_at( cursor, 1, b.latent_h, b.latent_w, t_cursor );
			cursor += ( b.latent_h / 2 ) * ( b.latent_w / 2 );
			t_cursor += 1.0;
		} else if ( b.kind == H3RefBlock::Audio ) {
			if ( b.ref_audio_t > 0 ) { audio_rows_at( cursor, b.ref_audio_t, t_cursor, wg.front(), wg.back(), false ); }
			cursor += b.ref_audio_t * 2;
			t_cursor += b.ref_audio_t;
		} else {
			const double area = std::sqrt( static_cast<double>( b.latent_h ) * b.latent_w );
			const std::vector<double> rw = axis_from_sqrt_area( b.latent_w, 2, area );
			if ( b.ref_audio_t > 0 ) { audio_rows_at( cursor, b.ref_audio_t, t_cursor, rw.front(), rw.back(), false ); }
			cursor += b.ref_audio_t * 2;
			video_rows_at( cursor, b.latent_t, b.latent_h, b.latent_w, t_cursor );
			cursor += b.latent_t * ( b.latent_h / 2 ) * ( b.latent_w / 2 );
			double span = 0;
			for ( int k = 0; k < b.latent_t; ++k ) { span += kFrameRescale * kFramePerToken[k % 5]; }
			t_cursor += std::max( static_cast<double>( b.ref_audio_t ), span );
		}
	}
	// the target audio and video (the tail of img_pos / audio_pos)
	audio_rows_at( cursor, audio_t, t_cursor, wg.front(), wg.back(), true );
	cursor += audio_rows;
	{
		const std::vector<double> tg = t_grid( latent_t, t_cursor );
		for ( int t = 0; t < latent_t; ++t ) {
			for ( int y = 0; y < ph; ++y ) {
				for ( int x = 0; x < pw; ++x ) {
					const size_t r = static_cast<size_t>( cursor ) + ( static_cast<size_t>( t ) * ph + y ) * pw + x;
					g[r * 3] = tg[t]; g[r * 3 + 1] = hg[y]; g[r * 3 + 2] = wg[x];
					L.img_pos.push_back( static_cast<int>( r ) );
					L.update_mask.push_back( 1 );
				}
			}
		}
	}
	L.position_ids.resize( g.size() );
	for ( size_t i = 0; i < g.size(); ++i ) { L.position_ids[i] = static_cast<float>( g[i] ); }
	return L;
}

bool H3Run::plan_shape() {
	// reference images: decoded here. FL2VA keyframes follow the caller's canvas size (the CLI derives it from the first image's aspect ratio);
	// Ref2VA references are resized to a 2048 px short edge and keep their own size
	ref2va = !req.ref_images.empty() || !req.ref_audios.empty() || !req.ref_videos.empty();
	auto read_image = [&]( const std::string& path, engine::loader::DecodedImage* img ) {
		std::ifstream f( path, std::ios::binary );
		std::stringstream ss;
		ss << f.rdbuf();
		const std::string bytes = ss.str();
		std::string err;
		if ( bytes.empty() || !engine::loader::decode_image( reinterpret_cast<const uint8_t*>( bytes.data() ), bytes.size(), img, &err ) ) {
			LOG_ERROR( "H3: cannot read image %s: %s", path.c_str(), err.c_str() );
			return false;
		}
		return true;
	};
	for ( const std::string& path : req.images ) {
		engine::loader::DecodedImage img;
		if ( !read_image( path, &img ) ) { return false; }
		refs.push_back( std::move( img ) );
	}
	if ( refs.size() > 2 ) { LOG_ERROR( "H3: FL2VA takes at most two images (first and last frame)" ); return false; }
	if ( ref2va && !refs.empty() ) { LOG_ERROR( "H3: keyframe images (FL2VA) and reference images / audio (Ref2VA) are different tasks" ); return false; }
	if ( ref2va && req.ref_images.empty() && req.ref_videos.empty() ) { LOG_ERROR( "H3: reference audio needs a reference image or video" ); return false; }
	if ( req.ref_images.size() > 9 || req.ref_videos.size() > 3 || req.ref_audios.size() > 3 || req.ref_images.size() + req.ref_videos.size() + req.ref_audios.size() > 12 ) {
		LOG_ERROR( "H3: Ref2VA takes at most 9 reference images, 3 videos and 3 audio clips (12 references in all)" );
		return false;
	}
	for ( const std::string& path : req.ref_images ) {
		engine::loader::DecodedImage img;
		if ( !read_image( path, &img ) ) { return false; }
		const double ratio = static_cast<double>( img.width ) / img.height;
		if ( ratio < 0.4 || ratio > 2.5 || std::min( img.width, img.height ) < 256 || std::max( img.width, img.height ) > 5760 ) {
			LOG_ERROR( "H3: reference image %s (%dx%d) must have an aspect ratio in [0.4, 2.5] and sides in [256, 5760] px", path.c_str(), img.width, img.height );
			return false;
		}
		ref_src.push_back( std::move( img ) );
	}
	keyframes = req.frame_indices;
	if ( !refs.empty() && keyframes.empty() ) { keyframes = refs.size() == 1 ? std::vector<int>{ 0 } : std::vector<int>{ 0, -1 }; }
	if ( keyframes.size() != refs.size() ) { LOG_ERROR( "H3: one frame index per image" ); return false; }
	width = req.width / 32 * 32;
	height = req.height / 32 * 32;
	frames = h3_align_frames( req.frames );
	latent_t = h3_video_latent_t( frames );
	audio_t = h3_audio_latent_t( frames / 24.0 );
	if ( width <= 0 || height <= 0 || req.steps < 1 ) { return false; }
	lh = height / 16;
	lw = width / 16;
	return true;
}

} // namespace engine::video
