#include "engine/video/h3_image.h"
#include "engine/config.h"
#include "engine/loader/image.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

namespace engine::video {

float H3Gaussian::next() {
	if ( has_spare_ ) {
		has_spare_ = false;
		return static_cast<float>( spare_ );
	}
	// (0, 1) uniforms from 53 random bits; each statement is a single multiply or add, so there is nothing for the compiler to contract
	const double inv = 1.0 / 9007199254740992.0;
	const double u1 = ( static_cast<double>( rng_() >> 11 ) + 0.5 ) * inv, u2 = static_cast<double>( rng_() >> 11 ) * inv;
	const double r = std::sqrt( -2.0 * std::log( u1 ) ), a = 6.283185307179586 * u2;
	spare_ = r * std::sin( a );
	has_spare_ = true;
	return static_cast<float>( r * std::cos( a ) );
}

namespace {

double sinc( double x ) {
	if ( x == 0.0 ) { return 1.0; }
	x *= M_PI;
	return std::sin( x ) / x;
}

double lanczos( double x ) {
	if ( x > -3.0 && x < 3.0 ) { return sinc( x ) * sinc( x / 3.0 ); }
	return 0.0;
}

struct Coeffs {
	std::vector<int> xmin, xsize;
	std::vector<double> w; // [out, ksize]
	int ksize = 0;
};

// Pillow's precompute_coeffs
Coeffs precompute( int in_size, int out_size ) {
	const double scale = static_cast<double>( in_size ) / out_size;
	const double filterscale = std::max( scale, 1.0 );
	const double support = 3.0 * filterscale;
	Coeffs c;
	c.ksize = static_cast<int>( std::ceil( support ) ) * 2 + 1;
	c.xmin.resize( out_size );
	c.xsize.resize( out_size );
	c.w.assign( static_cast<size_t>( out_size ) * c.ksize, 0.0 );
	const double ss = 1.0 / filterscale;
	for ( int xx = 0; xx < out_size; ++xx ) {
		const double center = ( xx + 0.5 ) * scale;
		int xmin = static_cast<int>( center - support + 0.5 );
		if ( xmin < 0 ) { xmin = 0; }
		int xmax = static_cast<int>( center + support + 0.5 );
		if ( xmax > in_size ) { xmax = in_size; }
		xmax -= xmin;
		double* k = &c.w[static_cast<size_t>( xx ) * c.ksize];
		double ww = 0.0;
		for ( int x = 0; x < xmax; ++x ) {
			const double w = lanczos( ( x + xmin - center + 0.5 ) * ss );
			k[x] = w;
			ww += w;
		}
		if ( ww != 0.0 ) { for ( int x = 0; x < xmax; ++x ) { k[x] /= ww; } }
		c.xmin[xx] = xmin;
		c.xsize[xx] = xmax;
	}
	return c;
}

uint8_t clip8( double v ) {
	const int i = static_cast<int>( std::lround( v ) ); // Pillow rounds half up on the fixed-point result; the difference is below one level
	return static_cast<uint8_t>( std::min( 255, std::max( 0, i ) ) );
}

} // namespace

std::vector<uint8_t> h3_resize_lanczos( const uint8_t* rgb, int in_h, int in_w, int out_h, int out_w ) {
	const Coeffs ch = precompute( in_w, out_w ), cv = precompute( in_h, out_h );
	std::vector<uint8_t> tmp( static_cast<size_t>( in_h ) * out_w * 3 );
	for ( int y = 0; y < in_h; ++y ) {
		for ( int x = 0; x < out_w; ++x ) {
			const double* k = &ch.w[static_cast<size_t>( x ) * ch.ksize];
			for ( int c = 0; c < 3; ++c ) {
				double acc = 0.0;
				for ( int i = 0; i < ch.xsize[x]; ++i ) { acc += k[i] * rgb[( static_cast<size_t>( y ) * in_w + ch.xmin[x] + i ) * 3 + c]; }
				tmp[( static_cast<size_t>( y ) * out_w + x ) * 3 + c] = clip8( acc );
			}
		}
	}
	std::vector<uint8_t> out( static_cast<size_t>( out_h ) * out_w * 3 );
	for ( int y = 0; y < out_h; ++y ) {
		const double* k = &cv.w[static_cast<size_t>( y ) * cv.ksize];
		for ( int x = 0; x < out_w; ++x ) {
			for ( int c = 0; c < 3; ++c ) {
				double acc = 0.0;
				for ( int i = 0; i < cv.xsize[y]; ++i ) { acc += k[i] * tmp[( static_cast<size_t>( cv.xmin[y] + i ) * out_w + x ) * 3 + c]; }
				out[( static_cast<size_t>( y ) * out_w + x ) * 3 + c] = clip8( acc );
			}
		}
	}
	return out;
}

std::vector<float> h3_vae_input( const uint8_t* rgb, int h, int w ) {
	const float mean[3] = { 0.485f, 0.456f, 0.406f }, sd[3] = { 0.229f, 0.224f, 0.225f };
	std::vector<float> out( static_cast<size_t>( 3 ) * h * w );
	for ( int c = 0; c < 3; ++c ) {
		for ( size_t p = 0; p < static_cast<size_t>( h ) * w; ++p ) { out[static_cast<size_t>( c ) * h * w + p] = ( rgb[p * 3 + c] / 255.0f - mean[c] ) / sd[c]; }
	}
	return out;
}

namespace {
float to_fp16_and_back( float v ) {
	// round-to-nearest-even FP16 (the reference keeps the sampled latent in half precision)
	uint32_t u;
	std::memcpy( &u, &v, 4 );
	const uint32_t sign = u & 0x80000000u;
	int32_t e = static_cast<int32_t>( ( u >> 23 ) & 0xff ) - 127 + 15;
	uint32_t m = u & 0x7fffff;
	uint16_t h;
	if ( e >= 31 ) { h = static_cast<uint16_t>( ( sign >> 16 ) | 0x7c00 ); }
	else if ( e <= 0 ) {
		if ( e < -10 ) { h = static_cast<uint16_t>( sign >> 16 ); }
		else {
			m |= 0x800000;
			const int shift = 14 - e;
			uint32_t r = m >> shift;
			const uint32_t rem = m & ( ( 1u << shift ) - 1 ), half = 1u << ( shift - 1 );
			if ( rem > half || ( rem == half && ( r & 1 ) ) ) { ++r; }
			h = static_cast<uint16_t>( ( sign >> 16 ) | r );
		}
	} else {
		uint32_t r = ( static_cast<uint32_t>( e ) << 10 ) | ( m >> 13 );
		const uint32_t rem = m & 0x1fff;
		if ( rem > 0x1000 || ( rem == 0x1000 && ( r & 1 ) ) ) { ++r; }
		h = static_cast<uint16_t>( ( sign >> 16 ) | r );
	}
	// back to float
	const uint32_t hs = ( h & 0x8000u ) << 16;
	uint32_t he = ( h >> 10 ) & 0x1f, hm = h & 0x3ff, out;
	if ( he == 0 ) {
		if ( hm == 0 ) { out = hs; }
		else {
			int sh = 0;
			while ( !( hm & 0x400 ) ) { hm <<= 1; ++sh; }
			hm &= 0x3ff;
			out = hs | ( ( 113 - sh ) << 23 ) | ( hm << 13 );
		}
	} else if ( he == 31 ) { out = hs | 0x7f800000 | ( hm << 13 ); }
	else { out = hs | ( ( he + 112 ) << 23 ) | ( hm << 13 ); }
	float f;
	std::memcpy( &f, &out, 4 );
	return f;
}
} // namespace

std::vector<float> h3_condition_rows( const std::vector<float>& moments, int T, int h, int w, const std::vector<float>& mean, const std::vector<float>& stdev, uint64_t seed, float noise_aug ) {
	ENGINE_ASSERT( moments.size() == static_cast<size_t>( T ) * 48 * h * w && h % 2 == 0 && w % 2 == 0, "condition rows: moments shape" );
	H3Gaussian gauss( seed );
	const size_t hw = static_cast<size_t>( h ) * w;
	const int h2 = h / 2, w2 = w / 2;
	std::vector<float> rows( static_cast<size_t>( T ) * h2 * w2 * 96 ), noise( rows.size() );
	std::vector<float> z( 24 * hw );
	for ( int t = 0; t < T; ++t ) {
		const float* m = &moments[static_cast<size_t>( t ) * 48 * hw];
		for ( int c = 0; c < 24; ++c ) {
			for ( size_t p = 0; p < hw; ++p ) {
				const float lv = std::min( 20.0f, std::max( -30.0f, m[static_cast<size_t>( 24 + c ) * hw + p] ) );
				const float v = m[static_cast<size_t>( c ) * hw + p] + std::exp( 0.5f * lv ) * gauss.next();
				z[static_cast<size_t>( c ) * hw + p] = ( to_fp16_and_back( v ) - mean[c] ) / stdev[c];
			}
		}
		// patchify (1, 2, 2): row = (t, hh, ww), col = c * 4 + ph * 2 + pw
		for ( int hh = 0; hh < h2; ++hh ) {
			for ( int ww = 0; ww < w2; ++ww ) {
				for ( int c = 0; c < 24; ++c ) {
					for ( int ph = 0; ph < 2; ++ph ) {
						for ( int pw = 0; pw < 2; ++pw ) {
							rows[( ( static_cast<size_t>( t ) * h2 + hh ) * w2 + ww ) * 96 + c * 4 + ph * 2 + pw] = z[static_cast<size_t>( c ) * hw + static_cast<size_t>( hh * 2 + ph ) * w + ww * 2 + pw];
						}
					}
				}
			}
		}
	}
	if ( noise_aug < 1.0f ) {
		H3Gaussian noise_gen( seed );
		noise_gen.fill( noise );
		for ( size_t i = 0; i < rows.size(); ++i ) { rows[i] = noise_aug * rows[i] + ( 1.0f - noise_aug ) * noise[i]; }
	}
	return rows;
}

std::vector<float> h3_keyframe_rows( const std::vector<float>& moments, int h, int w, const std::vector<float>& mean, const std::vector<float>& stdev, uint64_t seed, float noise_aug ) {
	return h3_condition_rows( moments, 1, h, w, mean, stdev, seed, noise_aug );
}

H3VideoBlocks h3_video_blocks( const std::vector<const uint8_t*>& frames, int h, int w ) {
	// Qwen3-VL video processor: smart_resize(num_frames, h, w, temporal 2, factor 32, min 4096, max 25165824), bicubic resize, 0.5 / 0.5 normalization,
	// frames padded to a multiple of 2 by repeating the last one, patches in (t, h/m, w/m, m, m) block order with (channel, 2 frames, 16, 16) inside
	const int n = static_cast<int>( frames.size() );
	const int factor = 32, tf = 2;
	const double min_pixels = 4096, max_pixels = 25165824;
	auto rnd = []( double v ) { return std::nearbyint( v ); }; // Python round(): half to even
	int hb = std::max( factor, static_cast<int>( rnd( static_cast<double>( h ) / factor ) ) * factor ), wb = std::max( factor, static_cast<int>( rnd( static_cast<double>( w ) / factor ) ) * factor );
	const int tb = static_cast<int>( rnd( static_cast<double>( n ) / tf ) ) * tf;
	if ( static_cast<double>( tb ) * hb * wb > max_pixels ) {
		const double beta = std::sqrt( static_cast<double>( n ) * h * w / max_pixels );
		hb = std::max( factor, static_cast<int>( std::floor( h / beta / factor ) ) * factor );
		wb = std::max( factor, static_cast<int>( std::floor( w / beta / factor ) ) * factor );
	} else if ( static_cast<double>( tb ) * hb * wb < min_pixels ) {
		const double beta = std::sqrt( min_pixels / ( static_cast<double>( n ) * h * w ) );
		hb = static_cast<int>( std::ceil( h * beta / factor ) ) * factor;
		wb = static_cast<int>( std::ceil( w * beta / factor ) ) * factor;
	}
	std::vector<std::vector<float>> px; // resized frames, float [0, 255], RGB
	for ( int i = 0; i < n + ( n % tf ); ++i ) {
		const uint8_t* f = frames[std::min( i, n - 1 )];
		px.push_back( hb == h && wb == w ? std::vector<float>( f, f + static_cast<size_t>( h ) * w * 3 ) : engine::loader::resize_bicubic( f, h, w, hb, wb ) );
	}
	H3VideoBlocks out;
	const int ps = 16, m = 2, gh = hb / ps, gw = wb / ps, dim = 3 * tf * ps * ps;
	out.grid_h = gh;
	out.grid_w = gw;
	for ( size_t b = 0; 2 * b < px.size(); ++b ) {
		std::vector<float> patches( static_cast<size_t>( gh ) * gw * dim );
		size_t row = 0;
		for ( int by = 0; by < gh / m; ++by ) {
			for ( int bx = 0; bx < gw / m; ++bx ) {
				for ( int my = 0; my < m; ++my ) {
					for ( int mx = 0; mx < m; ++mx, ++row ) {
						float* dst = &patches[row * dim];
						const int y0 = ( by * m + my ) * ps, x0 = ( bx * m + mx ) * ps;
						for ( int c = 0; c < 3; ++c ) {
							for ( int t = 0; t < tf; ++t ) {
								const std::vector<float>& f = px[2 * b + t];
								for ( int y = 0; y < ps; ++y ) {
									for ( int x = 0; x < ps; ++x ) { *dst++ = ( f[( static_cast<size_t>( y0 + y ) * wb + x0 + x ) * 3 + c] * ( 1.0f / 255.0f ) - 0.5f ) / 0.5f; }
								}
							}
						}
					}
				}
			}
		}
		out.patches.push_back( std::move( patches ) );
	}
	return out;
}

} // namespace engine::video
