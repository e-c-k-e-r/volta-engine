#include "engine/video/h3_image.h"
#include "engine/config.h"
#include "engine/loader/image.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

namespace engine::video {
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

std::vector<float> h3_keyframe_rows( const std::vector<float>& moments, int h, int w, const std::vector<float>& mean, const std::vector<float>& stdev, uint64_t seed, float noise_aug ) {
	ENGINE_ASSERT( moments.size() == static_cast<size_t>( 48 ) * h * w && h % 2 == 0 && w % 2 == 0, "keyframe rows: moments shape" );
	std::mt19937_64 rng( seed );
	std::normal_distribution<float> nd( 0.0f, 1.0f );
	const size_t hw = static_cast<size_t>( h ) * w;
	std::vector<float> z( 24 * hw );
	for ( int c = 0; c < 24; ++c ) {
		for ( size_t p = 0; p < hw; ++p ) {
			const float lv = std::min( 20.0f, std::max( -30.0f, moments[static_cast<size_t>( 24 + c ) * hw + p] ) );
			const float v = moments[static_cast<size_t>( c ) * hw + p] + std::exp( 0.5f * lv ) * nd( rng );
			z[static_cast<size_t>( c ) * hw + p] = ( to_fp16_and_back( v ) - mean[c] ) / stdev[c];
		}
	}
	// patchify (1, 2, 2): row = (hh, ww), col = c * 4 + ph * 2 + pw
	const int h2 = h / 2, w2 = w / 2;
	std::vector<float> rows( static_cast<size_t>( h2 ) * w2 * 96 ), noise( rows.size() );
	for ( int hh = 0; hh < h2; ++hh ) {
		for ( int ww = 0; ww < w2; ++ww ) {
			for ( int c = 0; c < 24; ++c ) {
				for ( int ph = 0; ph < 2; ++ph ) {
					for ( int pw = 0; pw < 2; ++pw ) {
						rows[( static_cast<size_t>( hh ) * w2 + ww ) * 96 + c * 4 + ph * 2 + pw] = z[static_cast<size_t>( c ) * hw + static_cast<size_t>( hh * 2 + ph ) * w + ww * 2 + pw];
					}
				}
			}
		}
	}
	if ( noise_aug < 1.0f ) {
		std::mt19937_64 rng2( seed );
		for ( float& v : noise ) { v = nd( rng2 ); }
		for ( size_t i = 0; i < rows.size(); ++i ) { rows[i] = noise_aug * rows[i] + ( 1.0f - noise_aug ) * noise[i]; }
	}
	return rows;
}

} // namespace engine::video
