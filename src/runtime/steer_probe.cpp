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

std::vector<ProbeAxis> probe_extra_axes( const StateSet& a, const StateSet& b, int hidden, const std::vector<std::vector<float>>& used, int count ) {
	std::vector<ProbeAxis> out;
	const std::vector<double> ma = mean_of( a, hidden );
	std::vector<std::vector<double>> taken;
	for ( const auto& u : used ) taken.emplace_back( u.begin(), u.end() );
	auto strip = [&]( std::vector<double>& x ) {
		for ( const auto& t : taken ) {
			double c = 0.0;
			for ( int i = 0; i < hidden; ++i ) c += x[i] * t[i];
			for ( int i = 0; i < hidden; ++i ) x[i] -= c * t[i];
		}
	};
	// an orthonormal basis of a's own spread (its samples' deviations from the mean, without the taken directions): a refusal direction must not
	// be made of what varies among the accepting openings
	std::vector<std::vector<double>> spread;
	for ( const float* p : a ) {
		std::vector<double> u( hidden );
		for ( int i = 0; i < hidden; ++i ) u[i] = p[i] - ma[i];
		strip( u );
		for ( const auto& q : spread ) {
			double c = 0.0;
			for ( int i = 0; i < hidden; ++i ) c += u[i] * q[i];
			for ( int i = 0; i < hidden; ++i ) u[i] -= c * q[i];
		}
		const double un = norm( u );
		if ( un < 1e-6 ) continue;
		for ( double& v : u ) v /= un;
		spread.push_back( std::move( u ) );
	}
	auto strip_spread = [&]( std::vector<double>& x ) {
		for ( const auto& q : spread ) {
			double c = 0.0;
			for ( int i = 0; i < hidden; ++i ) c += x[i] * q[i];
			for ( int i = 0; i < hidden; ++i ) x[i] -= c * q[i];
		}
	};
	std::vector<std::vector<double>> dev;   // b's deviations from a's mean, without the taken directions and a's spread
	for ( const float* p : b ) {
		std::vector<double> d( hidden );
		for ( int i = 0; i < hidden; ++i ) d[i] = p[i] - ma[i];
		strip( d );
		strip_spread( d );
		dev.push_back( std::move( d ) );
	}
	for ( int round = 0; round < count; ++round ) {
		double energy = 0.0;
		for ( const auto& d : dev ) energy += norm( d ) * norm( d );
		if ( energy < 1e-9 ) break;
		// power iteration on the deviations' second-moment operator, started from the largest deviation
		std::vector<double> g = dev[0];
		for ( const auto& d : dev ) if ( norm( d ) > norm( g ) ) g = d;
		for ( int it = 0; it < 30; ++it ) {
			std::vector<double> next( hidden, 0.0 );
			for ( const auto& d : dev ) {
				double c = 0.0;
				for ( int i = 0; i < hidden; ++i ) c += d[i] * g[i];
				for ( int i = 0; i < hidden; ++i ) next[i] += c * d[i];
			}
			const double nn = norm( next );
			if ( nn < 1e-12 ) break;
			for ( int i = 0; i < hidden; ++i ) g[i] = next[i] / nn;
		}
		// the unit axis (the deviations are already free of a's spread and the taken directions)
		strip( g );
		const double n = norm( g );
		if ( n < 1e-9 ) break;
		for ( double& v : g ) v /= n;
		{
			// the sign of a power-iteration vector is arbitrary: point it from set a towards set b
			double along = 0.0;
			for ( const float* p : b ) along += dot( g, p, hidden );
			for ( const float* p : a ) along -= dot( g, p, hidden ) * static_cast<double>( b.size() ) / a.size();
			if ( along < 0.0 ) for ( double& v : g ) v = -v;
		}
		ProbeAxis ax;
		ax.dir.resize( hidden );
		for ( int i = 0; i < hidden; ++i ) ax.dir[i] = static_cast<float>( g[i] );
		std::vector<double> pa;
		double sa = 0.0, sb = 0.0;
		for ( const float* p : a ) { pa.push_back( dot( g, p, hidden ) ); sa += pa.back(); }
		for ( const float* p : b ) sb += dot( g, p, hidden );
		const double mean_a = sa / a.size();
		double var = 0.0;
		for ( double v : pa ) var += ( v - mean_a ) * ( v - mean_a );
		ax.pa = static_cast<float>( mean_a );
		ax.pb = static_cast<float>( sb / b.size() );
		ax.sd = static_cast<float>( std::sqrt( var / a.size() ) );
		out.push_back( std::move( ax ) );
		taken.push_back( g );
		for ( auto& d : dev ) strip( d );
	}
	return out;
}

float probe_fraction( const ProbeAxis& axis, const float* state, int hidden ) {
	double s = 0.0;
	for ( int i = 0; i < hidden; ++i ) s += static_cast<double>( axis.dir[i] ) * state[i];
	return static_cast<float>( ( s - axis.pa ) / std::max( static_cast<double>( axis.pb - axis.pa ), 1e-3 ) );
}

} // namespace engine::runtime
