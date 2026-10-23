// Tier 2: the engine's Jinja interpreter renders the checkpoint chat templates exactly like the reference Jinja2
// (cases from tests/tools/gen_chat_goldens.py). No weights or GPU needed.
#include "engine/loader/jinja.h"

#define JSON_NOEXCEPTION 1
#include <nlohmann/json.hpp>

#include <cstdio>
#include "test_util.h"
#include <fstream>
#include <sstream>
#include <string>

#ifndef TEST_DATA_DIR
#error "TEST_DATA_DIR must point at tests/data"
#endif

namespace jinja = engine::loader::jinja;

int main() {
	const std::string dir = TEST_DATA_DIR;
	auto cases = nlohmann::ordered_json::parse( read_file( dir + "/chat_template_cases.json" ), nullptr, false );
	if ( cases.is_discarded() ) {
		std::printf( "[-] cannot parse chat_template_cases.json\n" );
		return 1;
	}
	int total = 0, failed = 0;
	for ( auto& [name, list] : cases["templates"].items() ) {
		jinja::Template tpl;
		std::string err;
		if ( !tpl.compile( read_file( dir + "/" + name + "_chat_template.jinja" ), &err ) ) {
			std::printf( "[-] %s: compile failed: %s\n", name.c_str(), err.c_str() );
			return 1;
		}
		for ( auto& c : list ) {
			++total;
			const jinja::Value vars = jinja::from_json( c["vars"].dump() );
			std::string out, e;
			const bool ok = tpl.render( vars, &out, &e );
			const bool want_error = c.contains( "error" );
			bool pass;
			if ( want_error ) {
				pass = !ok;
			}
			else {
				pass = ok && out == c["expected"].get<std::string>();
			}
			if ( !pass ) {
				++failed;
				if ( failed <= 5 ) {
					std::printf( "[FAIL] %s / %s: %s\n", name.c_str(), c["name"].get<std::string>().c_str(), ok ? "output differs" : e.c_str() );
					if ( ok && !want_error ) {
						const std::string want = c["expected"].get<std::string>();
						size_t k = 0;
						while ( k < out.size() && k < want.size() && out[k] == want[k] ) ++k;
						std::printf( "  first difference at byte %zu\n  got : ...%s\n  want: ...%s\n", k,
							out.substr( k > 40 ? k - 40 : 0, 120 ).c_str(), want.substr( k > 40 ? k - 40 : 0, 120 ).c_str() );
					}
				}
			}
		}
	}
	std::printf( "%d cases, %d failed\n", total, failed );
	return failed ? 1 : 0;
}
