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
	std::printf( "[+] steer probe maths PASS\n" );
	return 0;
}
