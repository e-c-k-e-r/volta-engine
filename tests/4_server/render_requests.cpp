// Renders recorded Anthropic Messages requests (req_*.json, as a client sent them) the way volta_server would, without loading weights:
// the checkpoint's tokenizer and chat template (or DeepSeek's built-in encoding), the server's request parsing and prompt rendering.
// Prints each request's prompt length and writes the prompt with its special tokens as text (what dump_requests writes) and, as
// req_N.vars.json, the variables the chat template rendered (messages, tools and options after the server's conversion), and as
// req_N.ids.txt the prompt's token ids, one a line.
//   render_requests <checkpoint dir> <dir of req_*.json> [out dir]
#include "engine/loader/chat_template.h"
#include "engine/loader/tokenizer.h"
#include "engine/server/chat.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

int main( int argc, char** argv ) {
	if ( argc < 3 ) {
		std::printf( "usage: render_requests <checkpoint dir> <dir of req_*.json> [out dir]\n" );
		return 2;
	}
	const std::string model_dir = argv[1], rec = argv[2], out = argc > 3 ? argv[3] : std::string();
	engine::loader::Tokenizer tok;
	engine::loader::ChatTemplate tmpl;
	if ( !tok.load( model_dir.c_str() ) || !tmpl.load( model_dir.c_str() ) ) {
		std::printf( "[-] cannot load the tokenizer and chat template of %s\n", model_dir.c_str() );
		return 1;
	}
	if ( !out.empty() ) fs::create_directories( out );
	std::vector<fs::path> reqs;
	for ( const auto& e : fs::directory_iterator( rec ) ) {
		const std::string n = e.path().filename().string();
		if ( n.rfind( "req_", 0 ) == 0 && e.path().extension() == ".json" ) reqs.push_back( e.path() );
	}
	std::sort( reqs.begin(), reqs.end() );
	const engine::server::ServerConfig cfg;
	const engine::server::PromptModel model{ tmpl, tok, nullptr, 1 << 30, nullptr };
	auto protocol = engine::server::make_anthropic_protocol();
	int failed = 0;
	for ( const fs::path& p : reqs ) {
		engine::server::ChatRequest req;
		engine::server::ApiError err;
		std::string prompt, vars;
		std::vector<int> tokens;
		std::ostringstream body;
		body << std::ifstream( p ).rdbuf();
		if ( !protocol->parse( body.str(), &req, &err ) || !engine::server::render_chat_prompt( model, cfg, req, &prompt, &tokens, &err, &vars ) ) {
			std::printf( "%s: error %s\n", p.filename().c_str(), err.message.c_str() );
			++failed;
			continue;
		}
		std::printf( "%s: %zu tokens\n", p.filename().c_str(), tokens.size() );
		if ( !out.empty() ) {
			const std::string shown = engine::server::shown_prompt( tok, prompt );
			std::string ids;
			for ( int t : tokens ) ids += std::to_string( t ) + "\n";
			for ( const auto& [suffix, text] : { std::pair<const char*, const std::string*>{ ".prompt.txt", &shown }, { ".vars.json", &vars }, { ".ids.txt", &ids } } ) {
				if ( FILE* f = std::fopen( ( fs::path( out ) / ( p.stem().string() + suffix ) ).c_str(), "wb" ) ) {
					std::fwrite( text->data(), 1, text->size(), f );
					std::fclose( f );
				}
			}
		}
	}
	return failed ? 1 : 0;
}
