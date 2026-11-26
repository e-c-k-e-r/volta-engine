// Every given checkpoint's config.json passes its model's layout check (ModelPipeline::check_config: the layout the pipeline implements, or
// Gemma 4's read layout), and copies of it with one field changed fail it: another layer count, and for the fixed-layout families another
// hidden size and a list of layer kinds that differs (Gemma 4, which reads those: a MoE block, attention biases, another activation). Host
// only, no device.
//   test_config_check <model dir> [more dirs...]
#include "engine/models/model.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <utility>
#include <vector>

using namespace engine;

namespace {

nlohmann::json read_json( const std::string& path ) {
	std::ifstream f( path );
	return f ? nlohmann::json::parse( f, nullptr, false ) : nlohmann::json();
}

}   // namespace

int main( int argc, char** argv ) {
	if ( argc < 2 ) {
		std::printf( "usage: test_config_check <model dir> [more dirs...]\n" );
		return 2;
	}
	char tmpl[] = "/tmp/test_config_check.XXXXXX";
	const std::string scratch = mkdtemp( tmpl ) ? std::string( tmpl ) : std::string();
	if ( scratch.empty() ) {
		std::printf( "[-] no scratch directory\n" );
		return 2;
	}
	int fails = 0;
	for ( int a = 1; a < argc; ++a ) {
		const std::string dir = argv[a];
		const nlohmann::json cfg = read_json( dir + "/config.json" );
		if ( cfg.is_discarded() || !cfg.is_object() || !cfg.contains( "model_type" ) || !cfg["model_type"].is_string() ) {
			std::printf( "[FAIL] %s: no config.json with a model_type\n", dir.c_str() );
			++fails;
			continue;
		}
		const std::string type = cfg["model_type"].get<std::string>();
		const bool own = models::check_model_config( type, dir.c_str(), true );
		// one field changed in text_config (or the top level), each copy must fail
		const bool nested = cfg.contains( "text_config" ) && cfg["text_config"].is_object();
		// A fixed-layout family refuses another size or layer order; Gemma 4 reads those, and refuses what it doesn't implement
		std::vector<std::pair<std::string, std::function<void( nlohmann::json& )>>> mutations = {
			{ "num_hidden_layers + 1", []( nlohmann::json& t ) { t["num_hidden_layers"] = t.value( "num_hidden_layers", 0 ) + 1; } },
		};
		if ( type == "gemma4" ) {
			mutations.push_back( { "enable_moe_block", []( nlohmann::json& t ) { t["enable_moe_block"] = true; } } );
			mutations.push_back( { "attention_bias", []( nlohmann::json& t ) { t["attention_bias"] = true; } } );
			mutations.push_back( { "hidden_activation silu", []( nlohmann::json& t ) { t["hidden_activation"] = "silu"; } } );
		}
		else {
			mutations.push_back( { "hidden_size + 128", []( nlohmann::json& t ) { t["hidden_size"] = t.value( "hidden_size", 0 ) + 128; } } );
			if ( ( nested ? cfg["text_config"] : cfg ).contains( "layer_types" ) )
				mutations.push_back( { "layer_types reversed", []( nlohmann::json& t ) { std::reverse( t["layer_types"].begin(), t["layer_types"].end() ); } } );
		}
		int caught = 0;
		for ( const auto& [what, mutate] : mutations ) {
			nlohmann::json m = cfg;
			mutate( nested ? m["text_config"] : m );
			{
				std::ofstream o( scratch + "/config.json" );
				o << m.dump();
			}
			const bool passed = models::check_model_config( type, scratch.c_str(), false );
			if ( passed ) std::printf( "  %s: the copy with %s passed the check\n", dir.c_str(), what.c_str() );
			caught += !passed;
		}
		const bool ok = own && caught == static_cast<int>( mutations.size() );
		std::printf( "[%s] %s (%s): its config %s, %d of %zu changed copies refused\n", ok ? "PASS" : "FAIL", dir.c_str(), type.c_str(), own ? "passes" : "FAILS",
			caught, mutations.size() );
		fails += !ok;
	}
	std::filesystem::remove_all( scratch );
	std::printf( "[%s] test_config_check\n", fails ? "FAIL" : "PASS" );
	return fails ? 1 : 0;
}
