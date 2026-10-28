#include "engine/runtime/steer_probe.h"
#include <algorithm>
#include <cmath>

namespace engine::runtime {

namespace {

double dot( const std::vector<double>& a, const float* b, int n ) { double s = 0.0; for ( int i = 0; i < n; ++i ) s += a[i] * b[i]; return s; }
double norm( const std::vector<double>& a ) { double s = 0.0; for ( double v : a ) s += v * v; return std::sqrt( s ); }

std::vector<double> mean_of( const StateSet& s, int n ) {
	std::vector<double> m( n, 0.0 );
	for ( const float* p : s ) for ( int i = 0; i < n; ++i ) m[i] += p[i];
	for ( double& v : m ) v /= static_cast<double>( s.size() );
	return m;
}

} // namespace

ProbeAxis probe_axis( const StateSet& a, const StateSet& b, int hidden, bool orthogonalize ) {
	const std::vector<double> ma = mean_of( a, hidden ), mb = mean_of( b, hidden );
	std::vector<double> g( hidden );
	for ( int i = 0; i < hidden; ++i ) g[i] = mb[i] - ma[i];
	if ( orthogonalize ) {
		// One pass over the unit deviations of a's samples from their mean, each projected out of g in turn (the deviations are not made
		// orthogonal to each other first: the samples keep a small spread along the axis, which the clamp's reference level is measured in)
		for ( const float* p : a ) {
			std::vector<double> u( hidden );
			for ( int i = 0; i < hidden; ++i ) u[i] = p[i] - ma[i];
			const double un = norm( u );
			if ( un < 1e-9 ) continue;
			double c = 0.0;
			for ( int i = 0; i < hidden; ++i ) c += g[i] * u[i] / un;
			for ( int i = 0; i < hidden; ++i ) g[i] -= c * u[i] / un;
		}
	}
	const double n = std::max( norm( g ), 1e-12 );
	ProbeAxis ax;
	ax.dir.resize( hidden );
	std::vector<double> d( hidden );
	for ( int i = 0; i < hidden; ++i ) { d[i] = g[i] / n; ax.dir[i] = static_cast<float>( d[i] ); }
	std::vector<double> pa;
	for ( const float* p : a ) pa.push_back( dot( d, p, hidden ) );
	double sa = 0.0, sb = 0.0;
	for ( double v : pa ) sa += v;
	for ( const float* p : b ) sb += dot( d, p, hidden );
	const double mean_a = sa / a.size();
	double var = 0.0;
	for ( double v : pa ) var += ( v - mean_a ) * ( v - mean_a );
	ax.pa = static_cast<float>( mean_a );
	ax.pb = static_cast<float>( sb / b.size() );
	ax.sd = static_cast<float>( std::sqrt( var / a.size() ) );
	return ax;
}

float probe_fraction( const ProbeAxis& axis, const float* state, int hidden ) {
	double s = 0.0;
	for ( int i = 0; i < hidden; ++i ) s += static_cast<double>( axis.dir[i] ) * state[i];
	return static_cast<float>( ( s - axis.pa ) / std::max( static_cast<double>( axis.pb - axis.pa ), 1e-3 ) );
}

} // namespace engine::runtime
