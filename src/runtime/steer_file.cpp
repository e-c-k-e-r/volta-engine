#include "engine/runtime/steer_file.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace engine::modules {

namespace {

// One side of a layer range: an index, a fraction of the layer count (0.33, 1/3) or "last"
bool parse_layer_token( const std::string& t, int num_layers, int* out ) {
	if ( t == "last" ) { *out = num_layers - 1; return true; }
	if ( t.empty() ) return false;
	char* end = nullptr;
	const size_t slash = t.find( '/' );
	if ( slash != std::string::npos ) {
		const double a = std::strtod( t.substr( 0, slash ).c_str(), &end );
		if ( end == t.c_str() ) return false;
		const double b = std::strtod( t.c_str() + slash + 1, &end );
		if ( b == 0.0 || *end != '\0' ) return false;
		*out = static_cast<int>( std::lround( a / b * num_layers ) );
		return true;
	}
	const double v = std::strtod( t.c_str(), &end );
	if ( *end != '\0' ) return false;
	*out = t.find( '.' ) != std::string::npos ? static_cast<int>( std::lround( v * num_layers ) ) : static_cast<int>( v );
	return true;
}

} // namespace

bool parse_steer_layers( const std::string& spec, int num_layers, int* first, int* last ) {
	if ( spec.empty() ) {
		*first = static_cast<int>( std::lround( num_layers / 3.0 ) );
		*last = static_cast<int>( std::lround( num_layers * 2.0 / 3.0 ) );
		return true;
	}
	const size_t dash = spec.find( '-' );
	if ( dash == std::string::npos ) return false;
	if ( !parse_layer_token( spec.substr( 0, dash ), num_layers, first ) || !parse_layer_token( spec.substr( dash + 1 ), num_layers, last ) ) return false;
	return *first >= 0 && *last < num_layers && *first <= *last;
}

} // namespace engine::modules

namespace engine::runtime {

bool load_steer_file( const std::string& path, SteerFile* out, std::string* err ) {
	FILE* f = std::fopen( path.c_str(), "rb" );
	if ( !f ) { *err = "cannot open " + path; return false; }
	int32_t hdr[5] = {};
	char magic[4] = {};
	bool ok = std::fread( magic, 1, 4, f ) == 4 && std::memcmp( magic, "VSTR", 4 ) == 0 && std::fread( hdr, sizeof( int32_t ), 5, f ) == 5 && ( hdr[0] == 1 || hdr[0] == 2 );
	if ( !ok ) { std::fclose( f ); *err = path + ": not a control-vector file (version 1 or 2)"; return false; }
	SteerFile file;
	if ( hdr[0] == 2 && std::fread( &file.scale, sizeof( float ), 1, f ) != 1 ) { std::fclose( f ); *err = path + ": truncated"; return false; }
	file.hidden = hdr[1];
	file.layers = hdr[2];
	file.scope = hdr[3];
	if ( file.hidden <= 0 || file.hidden > 65536 || hdr[4] < 0 || hdr[4] > 4096 ) { std::fclose( f ); *err = path + ": bad header"; return false; }
	file.entries.resize( hdr[4] );
	for ( auto& e : file.entries ) {
		int32_t lk[2];
		ok = std::fread( lk, sizeof( int32_t ), 2, f ) == 2 && std::fread( &e.par, sizeof( float ), 1, f ) == 1;
		e.layer = lk[0];
		e.kind = lk[1];
		e.vec.resize( file.hidden );
		ok = ok && std::fread( e.vec.data(), sizeof( float ), file.hidden, f ) == static_cast<size_t>( file.hidden );
		if ( !ok ) break;
	}
	std::fclose( f );
	if ( !ok ) { *err = path + ": truncated"; return false; }
	*out = std::move( file );
	return true;
}

bool save_steer_file( const std::string& path, const SteerFile& file, std::string* err ) {
	FILE* f = std::fopen( path.c_str(), "wb" );
	if ( !f ) { *err = "cannot write " + path; return false; }
	const int32_t hdr[5] = { 2, file.hidden, file.layers, file.scope, static_cast<int32_t>( file.entries.size() ) };
	bool ok = std::fwrite( "VSTR", 1, 4, f ) == 4 && std::fwrite( hdr, sizeof( int32_t ), 5, f ) == 5 && std::fwrite( &file.scale, sizeof( float ), 1, f ) == 1;
	for ( const auto& e : file.entries ) {
		const int32_t lk[2] = { e.layer, e.kind };
		ok = ok && std::fwrite( lk, sizeof( int32_t ), 2, f ) == 2 && std::fwrite( &e.par, sizeof( float ), 1, f ) == 1
			&& std::fwrite( e.vec.data(), sizeof( float ), file.hidden, f ) == static_cast<size_t>( file.hidden );
	}
	ok = std::fclose( f ) == 0 && ok;
	if ( !ok ) *err = "writing " + path + " failed";
	return ok;
}

bool parse_steer_scope( const std::string& spec, int* scope ) {
	int bits = 0;
	size_t i = 0;
	while ( i <= spec.size() ) {
		size_t j = spec.find( ',', i );
		if ( j == std::string::npos ) j = spec.size();
		const std::string w = spec.substr( i, j - i );
		if ( w == "prompt" ) bits |= engine::modules::STEER_PROMPT;
		else if ( w == "think" ) bits |= engine::modules::STEER_THINK;
		else if ( w == "answer" ) bits |= engine::modules::STEER_ANSWER;
		else if ( w == "generated" ) bits |= engine::modules::STEER_GENERATED;
		else if ( w == "all" ) bits |= engine::modules::STEER_ALL;
		else if ( !w.empty() ) return false;
		i = j + 1;
	}
	if ( bits == 0 ) return false;
	*scope = bits;
	return true;
}

} // namespace engine::runtime
