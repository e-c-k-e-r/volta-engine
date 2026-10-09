#include "engine/loader/image.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_GIF
#include "../../deps/stb/stb_image.h"

namespace engine::loader {

uint64_t hash_bytes( const void* data, size_t size, uint64_t seed ) {
	const auto* p = static_cast<const uint8_t*>( data );
	uint64_t h = seed;
	for ( size_t i = 0; i < size; ++i ) {
		h ^= p[i];
		h *= 0x100000001b3ULL;
	}
	return h;
}

bool base64_decode( const std::string& in, std::string* out ) {
	static const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	out->clear();
	out->reserve( in.size() / 4 * 3 );
	uint32_t acc = 0;
	int bits = 0;
	for ( char c : in ) {
		if ( c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t' ) continue;
		const size_t v = alphabet.find( c );
		if ( v == std::string::npos ) return false;
		acc = ( acc << 6 ) | static_cast<uint32_t>( v );
		bits += 6;
		if ( bits >= 8 ) {
			bits -= 8;
			out->push_back( static_cast<char>( ( acc >> bits ) & 0xFF ) );
		}
	}
	return true;
}

bool decode_image( const uint8_t* data, size_t size, DecodedImage* out, std::string* error ) {
	int w = 0, h = 0, c = 0;
	uint8_t* px = stbi_load_from_memory( data, static_cast<int>( size ), &w, &h, &c, 4 );
	if ( !px ) {
		if ( error ) *error = std::string( "image decode failed: " ) + stbi_failure_reason();
		return false;
	}
	out->width = w;
	out->height = h;
	out->rgb.resize( static_cast<size_t>( w ) * h * 3 );
	for ( size_t i = 0, n = static_cast<size_t>( w ) * h; i < n; ++i ) {
		const int a = px[i * 4 + 3];
		for ( int k = 0; k < 3; ++k ) {
			out->rgb[i * 3 + k] = static_cast<uint8_t>( ( px[i * 4 + k] * a + 255 * ( 255 - a ) + 127 ) / 255 );
		}
	}
	stbi_image_free( px );
	return true;
}

void smart_resize( int height, int width, int factor, int min_pixels, int max_pixels, int* out_h, int* out_w ) {
	auto round_to = [&]( double v ) { return std::max( factor, static_cast<int>( std::nearbyint( v / factor ) ) * factor ); };
	int h_bar = round_to( height );
	int w_bar = round_to( width );
	if ( static_cast<int64_t>( h_bar ) * w_bar > max_pixels ) {
		const double beta = std::sqrt( static_cast<double>( height ) * width / max_pixels );
		h_bar = std::max( factor, static_cast<int>( std::floor( height / beta / factor ) ) * factor );
		w_bar = std::max( factor, static_cast<int>( std::floor( width / beta / factor ) ) * factor );
	}
	else if ( static_cast<int64_t>( h_bar ) * w_bar < min_pixels ) {
		const double beta = std::sqrt( static_cast<double>( min_pixels ) / ( static_cast<double>( height ) * width ) );
		h_bar = static_cast<int>( std::ceil( height * beta / factor ) ) * factor;
		w_bar = static_cast<int>( std::ceil( width * beta / factor ) ) * factor;
	}
	*out_h = h_bar;
	*out_w = w_bar;
}

namespace {

struct Taps {
	std::vector<int> first;
	std::vector<int> count;
	std::vector<float> weight; // [out, max_taps]
	int max_taps = 0;
};

float bicubic_kernel( float x ) {
	constexpr float a = -0.5f;
	x = std::fabs( x );
	if ( x < 1.0f ) return ( ( a + 2.0f ) * x - ( a + 3.0f ) ) * x * x + 1.0f;
	if ( x < 2.0f ) return ( ( ( x - 5.0f ) * x + 8.0f ) * x - 4.0f ) * a;
	return 0.0f;
}

Taps make_taps( int in_size, int out_size ) {
	const float scale = static_cast<float>( in_size ) / out_size;
	const float filter_scale = std::max( scale, 1.0f );
	const float support = 2.0f * filter_scale;
	Taps t;
	t.max_taps = static_cast<int>( std::ceil( support ) ) * 2 + 1;
	t.first.resize( out_size );
	t.count.resize( out_size );
	t.weight.assign( static_cast<size_t>( out_size ) * t.max_taps, 0.0f );
	for ( int i = 0; i < out_size; ++i ) {
		const float center = ( i + 0.5f ) * scale;
		const int lo = std::max( static_cast<int>( center - support + 0.5f ), 0 );
		const int hi = std::min( static_cast<int>( center + support + 0.5f ), in_size );
		float sum = 0.0f;
		float* w = &t.weight[static_cast<size_t>( i ) * t.max_taps];
		for ( int j = lo; j < hi; ++j ) {
			w[j - lo] = bicubic_kernel( ( j - center + 0.5f ) / filter_scale );
			sum += w[j - lo];
		}
		if ( sum != 0.0f ) {
			for ( int j = lo; j < hi; ++j ) w[j - lo] /= sum;
		}
		t.first[i] = lo;
		t.count[i] = hi - lo;
	}
	return t;
}

} // namespace

std::vector<float> resize_bicubic( const uint8_t* rgb, int in_h, int in_w, int out_h, int out_w ) {
	const Taps tx = make_taps( in_w, out_w );
	const Taps ty = make_taps( in_h, out_h );
	// Horizontal pass: [in_h, out_w, 3]
	std::vector<float> mid( static_cast<size_t>( in_h ) * out_w * 3 );
	for ( int y = 0; y < in_h; ++y ) {
		for ( int x = 0; x < out_w; ++x ) {
			const float* w = &tx.weight[static_cast<size_t>( x ) * tx.max_taps];
			float acc[3] = { 0.0f, 0.0f, 0.0f };
			for ( int j = 0; j < tx.count[x]; ++j ) {
				const uint8_t* px = rgb + ( static_cast<size_t>( y ) * in_w + tx.first[x] + j ) * 3;
				acc[0] += w[j] * px[0];
				acc[1] += w[j] * px[1];
				acc[2] += w[j] * px[2];
			}
			float* o = &mid[( static_cast<size_t>( y ) * out_w + x ) * 3];
			o[0] = acc[0];
			o[1] = acc[1];
			o[2] = acc[2];
		}
	}
	std::vector<float> out( static_cast<size_t>( out_h ) * out_w * 3 );
	for ( int y = 0; y < out_h; ++y ) {
		const float* w = &ty.weight[static_cast<size_t>( y ) * ty.max_taps];
		for ( int x = 0; x < out_w * 3; ++x ) {
			float acc = 0.0f;
			for ( int j = 0; j < ty.count[y]; ++j ) {
				acc += w[j] * mid[static_cast<size_t>( ty.first[y] + j ) * out_w * 3 + x];
			}
			out[static_cast<size_t>( y ) * out_w * 3 + x] = std::min( std::max( acc, 0.0f ), 255.0f );
		}
	}
	return out;
}

bool prepare_image( const DecodedImage& img, const VisionParams& p, PreparedImage* out, std::string* error ) {
	if ( img.width < 1 || img.height < 1 ) {
		if ( error ) *error = "empty image";
		return false;
	}
	const int factor = p.patch_size * p.merge_size;
	if ( std::max( img.width, img.height ) > 200 * std::min( img.width, img.height ) ) {
		if ( error ) *error = "image aspect ratio above 200:1";
		return false;
	}
	int h = 0, w = 0;
	smart_resize( img.height, img.width, factor, p.min_pixels, p.max_pixels, &h, &w );
	std::vector<float> px = ( h == img.height && w == img.width ) ? std::vector<float>( img.rgb.begin(), img.rgb.end() )
	                                                              : resize_bicubic( img.rgb.data(), img.height, img.width, h, w );

	const int ps = p.patch_size, m = p.merge_size, tp = p.temporal_patch_size;
	const int gh = h / ps, gw = w / ps;
	const int dim = 3 * tp * ps * ps;
	out->grid_h = gh;
	out->grid_w = gw;
	out->patches.assign( static_cast<size_t>( gh ) * gw * dim, 0.0f );
	// Row order: (gh / m, gw / m, m, m); within a patch: channel, temporal frame, y, x. A still image repeats its frame.
	size_t row = 0;
	for ( int by = 0; by < gh / m; ++by ) {
		for ( int bx = 0; bx < gw / m; ++bx ) {
			for ( int my = 0; my < m; ++my ) {
				for ( int mx = 0; mx < m; ++mx, ++row ) {
					float* dst = &out->patches[row * dim];
					const int y0 = ( by * m + my ) * ps, x0 = ( bx * m + mx ) * ps;
					for ( int c = 0; c < 3; ++c ) {
						for ( int t = 0; t < tp; ++t ) {
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
	out->hash = hash_bytes( dims, sizeof( dims ), hash_bytes( img.rgb.data(), img.rgb.size() ) );
	return true;
}

} // namespace engine::loader
