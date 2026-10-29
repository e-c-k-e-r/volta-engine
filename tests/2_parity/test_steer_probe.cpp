#include "engine/runtime/steer_probe.h"
#include "engine/config.h"
#include <cmath>
#include <cstdio>
#include <vector>

// Host maths of the steering probe on synthetic states: set a varies along axes 1 and 2 only, set b sits 3 units along axis 0 and also leans on
// axis 1; the axis must come out as axis 0 (the lean on axis 1 is spread inside a, so it is removed), a at fraction 0 and b at fraction 1.
using namespace engine::runtime;

int main() {
	const int H = 64;
	std::vector<std::vector<float>> a( 5, std::vector<float>( H, 0.0f ) ), b( 8, std::vector<float>( H, 0.0f ) );
	for ( int i = 0; i < 5; ++i ) { a[i][1] = 0.8f * ( i - 2 ); a[i][2] = 0.5f * ( ( i * 7 ) % 5 - 2 ); a[i][5] = 10.0f; }
	for ( int j = 0; j < 8; ++j ) { b[j][0] = 3.0f + 0.05f * ( j - 4 ); b[j][1] = 1.5f; b[j][5] = 10.0f; }
	StateSet sa, sb;
	for ( auto& v : a ) sa.push_back( v.data() );
	for ( auto& v : b ) sb.push_back( v.data() );
	const ProbeAxis ax = probe_axis( sa, sb, H, true );
	double nrm = 0.0;
	for ( float v : ax.dir ) nrm += v * v;
	ENGINE_ASSERT( std::fabs( nrm - 1.0 ) < 1e-5, "the axis is not a unit vector" );
	ENGINE_ASSERT( std::fabs( ax.dir[0] ) > 0.99f && std::fabs( ax.dir[1] ) < 0.1f, "the spread inside set a was not removed from the axis" );
	const ProbeAxis plain = probe_axis( sa, sb, H, false );
	ENGINE_ASSERT( std::fabs( plain.dir[1] ) > 0.2f, "without the orthogonalisation the lean on axis 1 should remain" );
	float fa = 0.0f, fb = 0.0f;
	for ( auto& v : a ) fa += probe_fraction( ax, v.data(), H ) / 5;
	for ( auto& v : b ) fb += probe_fraction( ax, v.data(), H ) / 8;
	std::printf( "fraction of a %.4f, of b %.4f, a's spread %.4f\n", fa, fb, ax.sd );
	ENGINE_ASSERT( std::fabs( fa ) < 0.05f && std::fabs( fb - 1.0f ) < 0.05f, "fractions should be 0 at set a and 1 at set b" );
	ENGINE_ASSERT( ax.sd < 0.1f, "set a varies too much along the axis" );
	// Extra directions: the declining samples split into two styles beyond axis 0 (3 samples lean on axis 3, 5 on axis 4); the extra axes must
	// stay in the span of axes 0, 3 and 4 (the first axis already holds part of both styles), be unit and orthogonal to axis 0 and to each other, 
	std::vector<std::vector<float>> b2 = b;
	for ( int j = 0; j < 8; ++j ) { if ( j < 3 ) b2[j][3] = 2.0f; else b2[j][4] = 1.0f; }
	StateSet sb2;
	for ( auto& v : b2 ) sb2.push_back( v.data() );
	const ProbeAxis ax2 = probe_axis( sa, sb2, H, true );
	const std::vector<ProbeAxis> extra = probe_extra_axes( sa, sb2, H, { ax2.dir }, 3 );
	ENGINE_ASSERT( extra.size() >= 2, "two further directions were planted" );
	for ( size_t k = 0; k < 2; ++k ) {
		double n2 = 0.0, plane = 0.0, c0 = 0.0, c01 = 0.0;
		for ( int i = 0; i < H; ++i ) { n2 += extra[k].dir[i] * extra[k].dir[i]; c0 += extra[k].dir[i] * ax2.dir[i]; if ( i == 0 || i == 3 || i == 4 ) plane += extra[k].dir[i] * extra[k].dir[i]; if ( k == 1 ) c01 += extra[k].dir[i] * extra[0].dir[i]; }
		ENGINE_ASSERT( std::fabs( n2 - 1.0 ) < 1e-4, "an extra axis is not a unit vector" );
		ENGINE_ASSERT( plane > 0.95, "an extra axis left the span of the planted axes" );
		ENGINE_ASSERT( std::fabs( c0 ) < 1e-3 && std::fabs( c01 ) < 1e-3, "extra axes are not orthogonal to the earlier ones" );
		ENGINE_ASSERT( extra[k].pb > extra[k].pa, "an extra axis should put the declining samples above the accepting ones on average" );
	}
		ENGINE_ASSERT( extra.size() == 2 || std::sqrt( extra[2].sd ) < 1.0, "a third axis, if any, must be sane" );
	std::printf( "extra axes: %zu, first (%.2f on axis 3, %.2f on axis 4)\n", extra.size(), extra[0].dir[3], extra[0].dir[4] );
	std::printf( "[+] steer probe maths PASS\n" );
	return 0;
}
