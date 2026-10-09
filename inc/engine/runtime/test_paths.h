#pragma once

#include <cstdlib>
#include <list>
#include <string>

namespace engine::runtime::test {

// Checkpoint locations for tests and benches: $ENGINE_MODEL_ROOT/<name>. Per-checkpoint overrides are
// ENGINE_TEST_QWEN_DIR / ENGINE_TEST_DRAFT_DIR / ENGINE_TEST_GLM_DIR / ENGINE_TEST_GLM_DRAFT_DIR (test_driver.h). An unset root yields a "<ENGINE_MODEL_ROOT unset>/name" path, so a test that
// needs the checkpoint fails to open it with that name in the message, and tests that never load one are unaffected.
inline const char* model_path( const char* name, const char* override_env = nullptr ) {
	if ( override_env )
		if ( const char* e = std::getenv( override_env ) ) return e;
	const char* root = std::getenv( "ENGINE_MODEL_ROOT" );
	if ( !root || !*root ) root = "<ENGINE_MODEL_ROOT unset>";
	static std::list<std::string> interned;
	interned.push_back( std::string( root ) + "/" + name );
	return interned.back().c_str();
}

} // namespace engine::runtime::test
