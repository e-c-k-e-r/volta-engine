#pragma once

#include <nlohmann/json.hpp>
#include <cstdio>
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


// The test checkpoint of an architecture (the model_type of its config.json), from tests/models.env; nullptr for an unknown name
inline const char* model_role( const std::string& type ) {
	if ( type == "qwen3_5" ) return "ENGINE_TEST_QWEN_DIR";
	if ( type == "qwen4_exp" ) return "ENGINE_TEST_FLASH_DIR";
	if ( type == "glm5_next" ) return "ENGINE_TEST_GLM_DIR";
	if ( type == "deepseek_v41" ) return "ENGINE_TEST_DSV41_DIR";
	if ( type == "gemma4" ) return "ENGINE_TEST_GEMMA_DIR";
	return nullptr;
}

// The top-level model_type of a checkpoint's config.json ("" when it has none; text_config / vision_config carry their own)
inline std::string config_model_type( const std::string& dir ) {
	std::ifstream in( dir + "/config.json" );
	const nlohmann::json j = nlohmann::json::parse( in, nullptr, false );
	return !j.is_discarded() && j.is_object() && j.contains( "model_type" ) && j["model_type"].is_string() ? j["model_type"].get<std::string>() : "";
}

// The checkpoint a test runs on: --model <name | dir>. An architecture name (qwen3_5, qwen4_exp, glm5_next, deepseek_v41, gemma4) takes
// that family's test checkpoint (tests/models.env); a directory (an argument with a '/') runs that checkpoint, its architecture read from
// its config.json. Without --model, `fallback` (an architecture name; empty: no model). Exits on an argument neither resolves.
struct TestModel {
	std::string type, dir;
	bool is( const char* t ) const { return type == t; }
	explicit operator bool() const { return !type.empty(); }
};

inline TestModel resolve_model( const std::string& arg ) {
	TestModel m;
	if ( arg.find( '/' ) != std::string::npos ) {
		m.dir = arg;
		m.type = config_model_type( arg );
	}
	else if ( const char* role = model_role( arg ) ) {
		m.type = arg;
		m.dir = model_path( role );
	}
	if ( m.type.empty() || !model_role( m.type ) ) {
		std::fprintf( stderr, "[-] --model %s: neither an architecture (qwen3_5, qwen4_exp, glm5_next, deepseek_v41, gemma4) nor a checkpoint directory with one in its config.json\n", arg.c_str() );
		std::exit( 2 );
	}
	return m;
}

// The family flags --model replaced: a tool that ignores unknown arguments would otherwise run its default model
inline void refuse_family_flags( int argc, char** argv ) {
	for ( int i = 1; i < argc; ++i ) {
		const std::string a = argv[i];
		if ( a == "--glm" || a == "--flash" || a == "--qwen" || a == "--glm-only" || a == "--qwen-only" || a == "--gemma" || a == "--type" ) {
			std::fprintf( stderr, "[-] %s is gone: --model <architecture | checkpoint dir> (qwen3_5, qwen4_exp, glm5_next, deepseek_v41, gemma4)\n", a.c_str() );
			std::exit( 2 );
		}
	}
}

// --model's value in argv (a tool's own argument loop skips "--model" and the value after it)
inline TestModel model_arg( int argc, char** argv, const char* fallback ) {
	refuse_family_flags( argc, argv );
	for ( int i = 1; i + 1 < argc; ++i ) {
		if ( std::string( argv[i] ) == "--model" ) return resolve_model( argv[i + 1] );
	}
	return fallback && *fallback ? resolve_model( fallback ) : TestModel{};
}
} // namespace engine::runtime::test
