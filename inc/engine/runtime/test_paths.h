#pragma once

#include <cstdlib>
#include <fstream>
#include <list>
#include <map>
#include <string>
#include <unistd.h>

namespace engine::runtime::test {

// A variable of the test configuration: the process environment, else tests/local.env (per machine, not committed), else tests/models.env
// (the committed defaults). The two files are found next to the build directory the executable runs from (<root>/build*/x -> <root>/tests);
// lines are KEY=VALUE, optionally prefixed with `export`, # starts a comment.
inline const char* test_setting( const char* key ) {
	if ( const char* e = std::getenv( key ) ) return e;
	static const std::map<std::string, std::string> file = [] {
		std::map<std::string, std::string> m;
		char exe[4096];
		const ssize_t n = readlink( "/proc/self/exe", exe, sizeof( exe ) - 1 );
		if ( n <= 0 ) return m;
		std::string dir( exe, static_cast<size_t>( n ) );
		dir = dir.substr( 0, dir.rfind( '/' ) );                  // the build directory
		dir = dir.substr( 0, dir.rfind( '/' ) ) + "/tests";
		for ( const char* f : { "/models.env", "/local.env" } ) {   // local.env read last: it overrides
			std::ifstream in( dir + f );
			for ( std::string line; std::getline( in, line ); ) {
				if ( const size_t h = line.find( '#' ); h != std::string::npos ) line.resize( h );
				if ( line.rfind( "export ", 0 ) == 0 ) line = line.substr( 7 );
				const size_t eq = line.find( '=' );
				if ( eq == std::string::npos || eq == 0 ) continue;
				std::string k = line.substr( 0, eq ), v = line.substr( eq + 1 );
				while ( !v.empty() && ( v.back() == ' ' || v.back() == '\t' || v.back() == '\r' ) ) v.pop_back();
				if ( v.size() >= 2 && ( v.front() == '"' || v.front() == '\'' ) && v.back() == v.front() ) v = v.substr( 1, v.size() - 2 );
				m[k] = v;
			}
		}
		return m;
	}();
	const auto it = file.find( key );
	return it == file.end() ? nullptr : it->second.c_str();
}

// Checkpoint location for tests and benches. `role` is one of the variables of tests/models.env (ENGINE_TEST_QWEN_DIR, ENGINE_TEST_GLM_DIR,
// ...); its value is a directory under ENGINE_MODEL_ROOT or an absolute path. Anything else is taken as a directory name under the root.
// An unset root yields a "<ENGINE_MODEL_ROOT unset>/name" path, so a test that needs the checkpoint fails to open it with that name in the
// message, and tests that never load one are unaffected.
inline const char* model_path( const char* role ) {
	const char* name = role;
	if ( std::string( role ).rfind( "ENGINE_TEST_", 0 ) == 0 ) {
		name = test_setting( role );
		if ( !name ) name = role;    // a role models.env does not list: the message names it
	}
	static std::list<std::string> interned;
	if ( name[0] == '/' ) {
		interned.emplace_back( name );
		return interned.back().c_str();
	}
	const char* root = test_setting( "ENGINE_MODEL_ROOT" );
	if ( !root || !*root ) root = "<ENGINE_MODEL_ROOT unset>";
	interned.push_back( std::string( root ) + "/" + name );
	return interned.back().c_str();
}

} // namespace engine::runtime::test
