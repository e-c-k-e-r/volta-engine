// H3 T2VA pipeline orchestration (see h3_pipeline.h).
#include "engine/video/h3_pipeline.h"
#include "engine/core/log.h"
#include "engine/loader/image.h"
#include "engine/loader/tokenizer.h"
#include "engine/video/h3_audio.h"
#include "engine/video/h3_dit.h"
#include "engine/video/h3_dit_group.h"
#include "engine/video/h3_image.h"
#include "engine/video/h3_text.h"
#include "engine/video/h3_vae.h"
#include "engine/video/h3_vae_enc.h"
#include "engine/video/h3_vision.h"
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>

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

__global__ void k_euler( float* x, const float* v, float coef, size_t n ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	if ( i < n ) { x[i] += coef * v[i]; }
}

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

double seconds_since( std::chrono::steady_clock::time_point t0 ) {
	return std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
}

template <typename T>
T* upload( const std::vector<T>& v ) {
	T* d = nullptr;
	CUDA_CHECK( cudaMalloc( &d, std::max<size_t>( v.size(), 1 ) * sizeof( T ) ) );
	CUDA_CHECK( cudaMemcpy( d, v.data(), v.size() * sizeof( T ), cudaMemcpyHostToDevice ) );
	return d;
}

void gaussian( std::mt19937_64& rng, std::vector<float>& out ) {
	std::normal_distribution<float> nd( 0.0f, 1.0f );
	for ( float& v : out ) { v = nd( rng ); }
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

bool h3_generate( const H3Paths& paths, const H3GenRequest& req, H3GenResult* result ) {
	H3GenResult res;
	auto report = [&]( const char* stage, int done, int total ) { if ( req.progress ) { req.progress( stage, done, total ); } };
	auto cancelled = [&]() { if ( req.cancel && req.cancel->load() ) { res.cancelled = true; if ( result ) { *result = res; } return true; } return false; };
	CUDA_CHECK( cudaSetDevice( paths.device ) );
	// ---- shape ----
	// reference images (FL2VA): decoded here; the canvas follows the caller's size (the CLI derives it from the first image's aspect ratio)
	std::vector<engine::loader::DecodedImage> refs;
	for ( const std::string& path : req.images ) {
		std::ifstream f( path, std::ios::binary );
		std::stringstream ss;
		ss << f.rdbuf();
		const std::string bytes = ss.str();
		engine::loader::DecodedImage img;
		std::string err;
		if ( bytes.empty() || !engine::loader::decode_image( reinterpret_cast<const uint8_t*>( bytes.data() ), bytes.size(), &img, &err ) ) {
			LOG_ERROR( "H3: cannot read image %s: %s", path.c_str(), err.c_str() );
			return false;
		}
		refs.push_back( std::move( img ) );
	}
	if ( refs.size() > 2 ) { LOG_ERROR( "H3: FL2VA takes at most two images (first and last frame)" ); return false; }
	std::vector<int> keyframes = req.frame_indices;
	if ( !refs.empty() && keyframes.empty() ) { keyframes = refs.size() == 1 ? std::vector<int>{ 0 } : std::vector<int>{ 0, -1 }; }
	if ( keyframes.size() != refs.size() ) { LOG_ERROR( "H3: one frame index per image" ); return false; }
	const int width = req.width / 32 * 32, height = req.height / 32 * 32;
	const int frames = h3_align_frames( req.frames );
	const int latent_t = h3_video_latent_t( frames );
	const int audio_t = h3_audio_latent_t( frames / 24.0 );
	if ( width <= 0 || height <= 0 || req.steps < 1 ) { return false; }
	const int lh = height / 16, lw = width / 16;

	// ---- stage 0: keyframe latents (video VAE image encoder) ----
	report( "images", 0, 1 );
	auto t0 = std::chrono::steady_clock::now();
	std::vector<std::vector<uint8_t>> canvas; // images resized to the canvas
	std::vector<float> cond_rows;             // [n_cond, 96] anchors
	for ( const auto& img : refs ) { canvas.push_back( h3_resize_lanczos( img.rgb.data(), img.height, img.width, height, width ) ); }
	if ( !refs.empty() ) {
		H3ImageEncoder enc;
		H3VaeParams vp;
		if ( !h3_read_vae_params( ( paths.root + "/video_vae" ).c_str(), vp ) || !enc.load( ( paths.root + "/video_vae" ).c_str(), paths.device ) ) { return false; }
		float *d_in = nullptr, *d_m = nullptr;
		CUDA_CHECK( cudaMalloc( &d_in, static_cast<size_t>( 3 ) * height * width * sizeof( float ) ) );
		CUDA_CHECK( cudaMalloc( &d_m, static_cast<size_t>( 48 ) * lh * lw * sizeof( float ) ) );
		for ( const auto& px : canvas ) {
			const std::vector<float> in = h3_vae_input( px.data(), height, width );
			CUDA_CHECK( cudaMemcpy( d_in, in.data(), in.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
			if ( !enc.encode( d_in, height, width, d_m ) ) { LOG_ERROR( "H3: image encode failed" ); return false; }
			std::vector<float> moments( static_cast<size_t>( 48 ) * lh * lw );
			CUDA_CHECK( cudaMemcpy( moments.data(), d_m, moments.size() * sizeof( float ), cudaMemcpyDeviceToHost ) );
			const std::vector<float> rows = h3_keyframe_rows( moments, lh, lw, vp.mean, vp.stdev, 42, 0.999f );
			cond_rows.insert( cond_rows.end(), rows.begin(), rows.end() );
		}
		cudaFree( d_in );
		cudaFree( d_m );
		LOG_INFO( "H3: %zu keyframe image(s) encoded in %.1f s", refs.size(), seconds_since( t0 ) );
	}

	if ( cancelled() ) { return false; }
	// ---- stage 1: text conditioning ----
	report( "text", 0, 1 );
	t0 = std::chrono::steady_clock::now();
	engine::loader::Tokenizer tok;
	if ( !tok.load( ( paths.root + "/tokenizer" ).c_str() ) ) { LOG_ERROR( "H3: tokenizer not found under %s", paths.root.c_str() ); return false; }
	std::vector<int> ids;
	std::vector<int> text_tags;
	std::vector<H3ImageFeatures> feats;
	const int image_pad = tok.get_special_token( "<|image_pad|>" );
	if ( refs.empty() ) {
		ids = tok.encode( req.prompt, false );
		text_tags.assign( ids.size(), 1 );
	} else {
		H3VisionTower tower;
		if ( !tower.load( ( paths.root + "/text_encoder" ).c_str(), paths.device, 8192 ) ) { return false; }
		engine::loader::VisionParams vparams;
		vparams.min_pixels = 65536;
		vparams.max_pixels = 16777216;
		auto push = [&]( const std::vector<int>& v, int tag ) {
			ids.insert( ids.end(), v.begin(), v.end() );
			text_tags.insert( text_tags.end(), v.size(), tag );
		};
		for ( size_t i = 0; i < canvas.size(); ++i ) {
			engine::loader::DecodedImage di;
			di.width = width;
			di.height = height;
			di.rgb = canvas[i];
			engine::loader::PreparedImage prep;
			std::string err;
			if ( !engine::loader::prepare_image( di, vparams, &prep, &err ) ) { LOG_ERROR( "H3: image preparation failed: %s", err.c_str() ); return false; }
			H3ImageFeatures f;
			f.grid_h = prep.grid_h;
			f.grid_w = prep.grid_w;
			if ( !tower.encode( prep.patches, prep.grid_h, prep.grid_w, &f.merged, f.deepstack ) ) { LOG_ERROR( "H3: vision tower failed" ); return false; }
			const int n = static_cast<int>( f.merged.size() / 5120 );
			push( tok.encode( "<Picture " + std::to_string( i + 1 ) + ">: ", false ), 1 );
			push( { tok.get_special_token( "<|vision_start|>" ) }, 0 );
			push( std::vector<int>( n, image_pad ), 0 );
			push( { tok.get_special_token( "<|vision_end|>" ) }, 0 );
			feats.push_back( std::move( f ) );
		}
		push( tok.encode( req.prompt, false ), 1 );
	}
	if ( ids.empty() ) { return false; }
	float* d_prompt = nullptr;
	CUDA_CHECK( cudaMalloc( &d_prompt, ids.size() * 5120 * sizeof( float ) ) );
	{
		H3TextEncoder enc;
		if ( !enc.load( ( paths.root + "/text_encoder" ).c_str(), paths.device, 8192, 50 ) || !enc.encode_mm( ids, image_pad, feats, d_prompt ) ) { return false; }
	}
	res.t_text = seconds_since( t0 );
	LOG_INFO( "H3: prompt %d tokens (%zu image(s)) encoded in %.1f s", static_cast<int>( ids.size() ), feats.size(), res.t_text );

	// ---- layout, noise, schedule ----
	H3Layout lay = h3_build_layout( static_cast<int>( ids.size() ), latent_t, lh, lw, audio_t, keyframes, frames );
	for ( size_t i = 0; i < ids.size(); ++i ) { lay.tags[i] = text_tags[i]; }
	const int n_cond = static_cast<int>( keyframes.size() ) * ( lh / 2 ) * ( lw / 2 );
	std::mt19937_64 rng( req.seed );
	std::vector<float> vrows( static_cast<size_t>( lay.n_img ) * 96 ), arows( static_cast<size_t>( lay.n_audio ) * 32 );
	gaussian( rng, vrows );
	gaussian( rng, arows );
	// condition rows are the fixed (noised) anchors; only the target rows are denoised
	std::copy( cond_rows.begin(), cond_rows.end(), vrows.begin() );
	float video_shift = req.video_shift;
	if ( video_shift <= 0.0f ) {
		video_shift = 12.0f;
		for ( const H3Lora& l : req.loras ) {
			if ( l.path.find( "_768p" ) != std::string::npos ) { video_shift = 6.0f; }
		}
	}
	const int n_sigma = req.steps + 1;
	std::vector<double> sv( n_sigma ), sa( n_sigma );
	for ( int i = 0; i < n_sigma; ++i ) {
		const double u = n_sigma > 1 ? 1.0 - static_cast<double>( i ) / ( n_sigma - 1 ) : 1.0;
		sv[i] = video_shift * u / ( 1.0 + ( video_shift - 1.0 ) * u );
		sa[i] = req.audio_shift * u / ( 1.0 + ( req.audio_shift - 1.0 ) * u );
	}

	if ( cancelled() ) { return false; }
	// ---- stage 2: DiT (one or several GPUs, sequence parallel) ----
	report( "load", 0, 1 );
	t0 = std::chrono::steady_clock::now();
	std::vector<float> host_prompt( ids.size() * 5120 );
	CUDA_CHECK( cudaMemcpy( host_prompt.data(), d_prompt, host_prompt.size() * sizeof( float ), cudaMemcpyDeviceToHost ) );
	cudaFree( d_prompt );
	std::vector<int> devices = paths.devices;
	if ( devices.empty() ) { devices.push_back( paths.device ); }
	{
		H3DiTGroup dit;
		if ( !dit.load( paths.dit.c_str(), devices, lay.used ) ) { LOG_ERROR( "H3: cannot load %s", paths.dit.c_str() ); return false; }
		for ( const H3Lora& l : req.loras ) {
			if ( dit.add_lora( l.path.c_str(), l.scale ) < 0 ) { LOG_ERROR( "H3: cannot use LoRA %s", l.path.c_str() ); return false; }
		}
		res.t_load = seconds_since( t0 );
		LOG_INFO( "H3: DiT and %zu adapter(s) ready on %zu GPU(s) in %.1f s; %d rows (%d video, %d audio, %d text), %d steps, shifts %g/%g", req.loras.size(), devices.size(), res.t_load, lay.used,
			lay.n_img, lay.n_audio, lay.text_len, req.steps, video_shift, req.audio_shift );
		H3HostRequest r;
		r.used = lay.used; r.text_len = lay.text_len; r.n_img = lay.n_img; r.n_audio = lay.n_audio;
		r.prompt = std::move( host_prompt );
		r.position_ids = lay.position_ids;
		r.img_pos = lay.img_pos; r.audio_pos = lay.audio_pos; r.text_pos = lay.text_pos;
		r.update_mask = lay.update_mask; r.tags = lay.tags;
		r.inverse.resize( lay.used );
		t0 = std::chrono::steady_clock::now();
		std::vector<float> vout, aout;
		report( "denoise", 0, req.steps );
		for ( int step = 0; step < req.steps; ++step ) {
			if ( cancelled() ) { return false; }
			const double tv = 1.0 - sv[step], ta = 1.0 - sa[step];
			// per-row timesteps: text rows inherit the video timestep, condition rows sit at the condition timestep, audio rows at the audio one
			const double tc = std::max( tv, 0.999 );
			std::vector<double> row_t( lay.used, tv );
			for ( int i = 0; i < lay.n_audio; ++i ) { row_t[lay.audio_pos[i]] = ta; }
			for ( int i = 0; i < n_cond; ++i ) { row_t[lay.img_pos[i]] = tc; }
			std::vector<float> uniq;
			for ( double t : row_t ) {
				const float f = static_cast<float>( t );
				if ( std::find( uniq.begin(), uniq.end(), f ) == uniq.end() ) { uniq.push_back( f ); }
			}
			std::sort( uniq.begin(), uniq.end() );
			for ( int i = 0; i < lay.used; ++i ) { r.inverse[i] = static_cast<int>( std::lower_bound( uniq.begin(), uniq.end(), static_cast<float>( row_t[i] ) ) - uniq.begin() ); }
			r.unique_t = uniq;
			r.video_rows = vrows;
			r.audio_rows = arows;
			if ( !dit.forward( r, &vout, &aout ) ) { return false; }
			{ double sum = 0, sa2 = 0; size_t bad = 0; for ( float v : vout ) { if ( !std::isfinite( v ) ) { ++bad; } else { sum += std::fabs( v ); } } for ( float v : aout ) { if ( std::isfinite( v ) ) { sa2 += std::fabs( v ); } }
				LOG_INFO( "H3: step %d velocity |v|_1 %.6f |a|_1 %.6f, %zu non-finite", step, sum, sa2, bad ); }
			// Euler (eta 0): x += (sigma_curr - sigma_next) * v; condition rows stay at their anchors
			const float cv = static_cast<float>( sv[step] - sv[step + 1] ), ca = static_cast<float>( sa[step] - sa[step + 1] );
			for ( size_t i = static_cast<size_t>( n_cond ) * 96; i < vrows.size(); ++i ) { vrows[i] += cv * vout[i]; }
			for ( size_t i = 0; i < arows.size(); ++i ) { arows[i] += ca * aout[i]; }
			LOG_INFO( "H3: step %d/%d done (%.1f s elapsed)", step + 1, req.steps, seconds_since( t0 ) );
			report( "denoise", step + 1, req.steps );
		}
		res.t_denoise = seconds_since( t0 );
	}
	CUDA_CHECK( cudaSetDevice( paths.device ) );
	float* d_vrows = upload( vrows );
	float* d_arows = upload( arows );

	if ( cancelled() ) { return false; }
	// ---- stage 3: decoders ----
	report( "decode", 0, 1 );
	t0 = std::chrono::steady_clock::now();
	std::vector<uint8_t> rgb;
	int out_frames = 0;
	float* d_z = nullptr;
	CUDA_CHECK( cudaMalloc( &d_z, static_cast<size_t>( 24 ) * latent_t * lh * lw * sizeof( float ) ) );
	const size_t target_floats = vrows.size() - static_cast<size_t>( n_cond ) * 96;
	k_unpatchify<<<( target_floats + 255 ) / 256, 256>>>( d_vrows + static_cast<size_t>( n_cond ) * 96, d_z, latent_t, lh / 2, lw / 2 );
	CUDA_CHECK( cudaDeviceSynchronize() );
	{
		H3VideoVAE vae;
		if ( !vae.load( ( paths.root + "/video_vae" ).c_str(), paths.device ) || !vae.decode_video( d_z, latent_t, lh, lw, rgb, out_frames ) ) { LOG_ERROR( "H3: video decode failed" ); return false; }
	}
	std::vector<float> pcm;
	{
		H3AudioVAE avae;
		if ( !avae.load( ( paths.root + "/audio_vae" ).c_str(), paths.device, audio_t + 8 ) || !avae.decode( d_arows, audio_t, pcm ) ) { LOG_ERROR( "H3: audio decode failed" ); return false; }
	}
	res.t_decode = seconds_since( t0 );
	LOG_INFO( "H3: decoded %d frames (%dx%d) and %zu audio samples in %.1f s", out_frames, width, height, pcm.size() / 2, res.t_decode );

	// ---- mux ----
	report( "mux", 0, 1 );
	t0 = std::chrono::steady_clock::now();
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
	if ( result ) { *result = res; }
	LOG_INFO( "H3: wrote %s (ffmpeg rc %d)", req.out.c_str(), rc );
	return rc == 0;
}

} // namespace engine::video
