#include "engine/runtime/options.h"
#include "engine/core/log.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>

namespace engine::runtime {

namespace {

bool parse_value( const std::string& text, int& out ) {
	char* end = nullptr;
	const long x = std::strtol( text.c_str(), &end, 10 );
	if ( end == text.c_str() || *end ) return false;
	out = static_cast<int>( x );
	return true;
}

bool parse_value( const std::string& text, double& out ) {
	char* end = nullptr;
	const double x = std::strtod( text.c_str(), &end );
	if ( end == text.c_str() || *end ) return false;
	out = x;
	return true;
}

bool parse_value( const std::string& text, bool& out ) {
	if ( text == "true" || text == "1" ) out = true;
	else if ( text == "false" || text == "0" ) out = false;
	else return false;
	return true;
}

bool parse_value( const std::string& text, std::string& out ) {
	out = text;
	return true;
}

const char* type_name( int ) { return "int"; }
const char* type_name( double ) { return "float"; }
const char* type_name( bool ) { return "bool"; }
const char* type_name( const std::string& ) { return "string"; }

std::string to_text( int v ) { return std::to_string( v ); }
std::string to_text( double v ) { char b[32]; std::snprintf( b, sizeof( b ), "%g", v ); return b; }
std::string to_text( bool v ) { return v ? "true" : "false"; }
std::string to_text( const std::string& v ) { return v.empty() ? "(empty)" : v; }

struct OptionInfo {
	const char* name;
	const char* type;
	std::string def;
	std::string env;
	const char* help;
	bool debug;
	std::function<bool( Options&, const std::string& )> set;
};

// Range / enum checks that a plain parse cannot express
bool valid( const char* name, const Options& o ) {
	if ( !std::strcmp( name, "kv_format" ) ) return o.kv_format.empty() || o.kv_format == "fp16" || o.kv_format == "fp8" || o.kv_format == "k8v4" || o.kv_format == "nvfp4";
	if ( !std::strcmp( name, "expert_cache" ) ) return o.expert_cache > 0.0 && o.expert_cache <= 1.0;
	if ( !std::strcmp( name, "speculator" ) ) return o.speculator == "dflash" || o.speculator == "mtp";
	if ( !std::strcmp( name, "mtp_depth" ) ) return o.mtp_depth >= 1 && o.mtp_depth <= 7;
	if ( !std::strcmp( name, "log_level" ) ) return o.log_level == "error" || o.log_level == "warn" || o.log_level == "info" || o.log_level == "debug";
	if ( !std::strcmp( name, "tp" ) ) return o.tp < 0 || o.tp == 1 || o.tp == 2 || o.tp == 4;
	return true;
}

std::string env_name( const char* name, bool debug ) {
	std::string e = debug ? "ENGINE_DEBUG_" : "ENGINE_";
	for ( const char* c = name; *c; ++c ) e.push_back( static_cast<char>( std::toupper( static_cast<unsigned char>( *c ) ) ) );
	return e;
}

const std::vector<OptionInfo>& table() {
	static const std::vector<OptionInfo> t = [] {
		std::vector<OptionInfo> v;
		const Options defaults;
#define ADD( type, name, def, help, dbg ) \
		v.push_back( { #name, type_name( defaults.name ), to_text( defaults.name ), env_name( #name, dbg ), help, dbg, \
			[]( Options& o, const std::string& text ) { Options trial = o; if ( !parse_value( text, trial.name ) || !valid( #name, trial ) ) return false; o.name = trial.name; return true; } } );
#define OPTION( type, name, def, help ) ADD( type, name, def, help, false )
#define DEBUG_OPTION( type, name, def, help ) ADD( type, name, def, help, true )
#include "engine/runtime/options.def"
#undef OPTION
#undef DEBUG_OPTION
#undef ADD
		return v;
	}();
	return t;
}

Options from_env() {
	Options o;
	for ( const OptionInfo& info : table() ) {
		const char* e = std::getenv( info.env.c_str() );
		if ( !e ) continue;
		if ( !info.set( o, e ) ) {
			std::fprintf( stderr, "[-] FATAL: invalid value '%s' for %s\n", e, info.env.c_str() );
			std::exit( 2 );
		}
	}
	return o;
}

} // namespace

Options& options() {
	static Options o = from_env();
	return o;
}

bool set_option( const std::string& key, const std::string& value ) {
	for ( const OptionInfo& info : table() ) {
		if ( key != info.name ) continue;
		// The launch file and --set come after the environment: say so when a variable that is set loses to them
		const char* e = std::getenv( info.env.c_str() );
		if ( e && value != e ) LOG_WARN( "[config] %s=%s is overridden by the launch file or --set (%s=%s)\n", info.env.c_str(), e, key.c_str(), value.c_str() );
		return info.set( options(), value );
	}
	return false;
}

std::string options_markdown() {
	std::string md = "| Option | Type | Default | Environment | Description |\n|---|---|---|---|---|\n";
	for ( int pass = 0; pass < 2; ++pass ) {
		if ( pass == 1 ) md += "| **Debug** | | | | |\n";
		for ( const OptionInfo& info : table() ) {
			if ( info.debug != ( pass == 1 ) ) continue;
			md += std::string( "| `" ) + info.name + "` | " + info.type + " | `" + info.def + "` | `" + info.env + "` | " + info.help + " |\n";
		}
	}
	return md;
}

} // namespace engine::runtime
