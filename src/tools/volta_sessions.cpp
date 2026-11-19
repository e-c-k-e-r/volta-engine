// volta_sessions: Anthropic Messages request bodies from recorded Claude Code sessions (~/.claude/projects/*/*.jsonl transcripts), so the server
// renders them exactly as it rendered the live conversation (its count_tokens endpoint with dump_requests writes the rendered prompt, which
// test_ppl --calibrate --prompts replays). One body per session: its main-thread user and assistant entries in order, consecutive entries of
// one role joined into one message (thinking, text, tool_use and tool_result blocks as recorded), cut at a message boundary once the content
// passes --max-chars; side-chain (subagent) entries and meta entries are left out. Thinking is enabled on the request.
//
//   volta_sessions --out <dir> [--max-chars 120000] SESSION.jsonl ...      writes <dir>/<n>.json, n in argument order
//
// Single prompts answered by another server (a hosted copy of the model, for its own outputs as calibration text):
//   volta_sessions --requests PROMPTS.json --out <dir> --model NAME [--max-tokens 6000] [--temperature 1.0] [--top-p 0.95]
//       OpenAI chat completion bodies, <dir>/<n>.json for the n-th prompt ({"kind", "prompt"} entries, test_ear's prompt files)
//   volta_sessions --conversations PROMPTS.json --responses <dir> --out <dir>
//       from <responses>/<n>.json (the server's chat completion), Anthropic Messages bodies of the exchange: the prompt and the answer with its
//       reasoning as a thinking block, for rendering through count_tokens like the sessions; answers that are missing, empty or a proxy's error
//       text are skipped
#include "engine/core/log.h"
#include <nlohmann/json.hpp>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

using json = nlohmann::json;

// Content as a list of blocks (a string becomes one text block)
json blocks( const json& content ) {
	if ( content.is_string() ) return json::array( { { { "type", "text" }, { "text", content.get<std::string>() } } } );
	return content.is_array() ? content : json::array();
}

size_t chars( const json& content ) { return content.dump().size(); }

bool session_body( const std::string& path, size_t max_chars, json& body, int& n_msgs ) {
	std::ifstream in( path );
	if ( !in ) return false;
	json msgs = json::array();
	size_t total = 0;
	bool full = false;
	for ( std::string line; std::getline( in, line ) && !full; ) {
		const json e = json::parse( line, nullptr, false );
		if ( e.is_discarded() || !e.is_object() || !e.contains( "message" ) ) continue;
		if ( e.value( "isSidechain", false ) || e.value( "isMeta", false ) ) continue;
		const json& m = e["message"];
		if ( !m.is_object() ) continue;
		const std::string role = m.value( "role", "" );
		if ( role != "user" && role != "assistant" ) continue;
		json b = blocks( m.contains( "content" ) ? m["content"] : json() );
		if ( b.empty() ) continue;
		const size_t c = chars( b );
		if ( total + c > max_chars && !msgs.empty() ) {
			// The conversation up to here; it ends on an assistant turn or a user turn, either renders
			full = true;
			break;
		}
		total += c;
		if ( !msgs.empty() && msgs.back()["role"] == role ) {
			for ( auto& x : b ) msgs.back()["content"].push_back( x );
		} else {
			msgs.push_back( { { "role", role }, { "content", b } } );
		}
	}
	// The API starts with a user turn
	while ( !msgs.empty() && msgs.front()["role"] != "user" ) msgs.erase( msgs.begin() );
	if ( msgs.empty() ) return false;
	n_msgs = static_cast<int>( msgs.size() );
	body = { { "model", "volta" }, { "max_tokens", 1 }, { "messages", msgs }, { "thinking", { { "type", "enabled" }, { "budget_tokens", 16000 } } } };
	return true;
}

} // namespace

json read_json( const std::string& path ) {
	std::ifstream in( path );
	if ( !in ) return json();
	std::stringstream ss;
	ss << in.rdbuf();
	return json::parse( ss.str(), nullptr, false );
}

bool write_json( const std::string& path, const json& j ) {
	std::ofstream f( path );
	f << j.dump();
	return static_cast<bool>( f );
}

int main( int argc, char** argv ) {
	std::string out, requests, conversations, responses, model;
	size_t max_chars = 120000;
	int max_tokens = 6000;
	double temperature = 1.0, top_p = 0.95;
	std::vector<std::string> sessions;
	for ( int i = 1; i < argc; ++i ) {
		const std::string a = argv[i];
		if ( a == "--out" && i + 1 < argc ) out = argv[++i];
		else if ( a == "--max-chars" && i + 1 < argc ) max_chars = std::stoull( argv[++i] );
		else if ( a == "--requests" && i + 1 < argc ) requests = argv[++i];
		else if ( a == "--conversations" && i + 1 < argc ) conversations = argv[++i];
		else if ( a == "--responses" && i + 1 < argc ) responses = argv[++i];
		else if ( a == "--model" && i + 1 < argc ) model = argv[++i];
		else if ( a == "--max-tokens" && i + 1 < argc ) max_tokens = std::stoi( argv[++i] );
		else if ( a == "--temperature" && i + 1 < argc ) temperature = std::stod( argv[++i] );
		else if ( a == "--top-p" && i + 1 < argc ) top_p = std::stod( argv[++i] );
		else sessions.push_back( a );
	}
	if ( !requests.empty() || !conversations.empty() ) {
		const json prompts = read_json( requests.empty() ? conversations : requests );
		if ( out.empty() || !prompts.is_array() || ( !requests.empty() && model.empty() ) || ( !conversations.empty() && responses.empty() ) ) {
			LOG_ERROR( "usage: volta_sessions --requests PROMPTS.json --out <dir> --model NAME [...] | --conversations PROMPTS.json --responses <dir> --out <dir>" );
			return 2;
		}
		mkdir( out.c_str(), 0755 );
		int written = 0;
		for ( size_t n = 0; n < prompts.size(); ++n ) {
			const std::string prompt = prompts[n].value( "prompt", "" );
			const std::string path = out + "/" + std::to_string( n ) + ".json";
			if ( !requests.empty() ) {
				const json body = { { "model", model }, { "max_tokens", max_tokens }, { "temperature", temperature }, { "top_p", top_p },
					{ "messages", json::array( { { { "role", "user" }, { "content", prompt } } } ) } };
				written += write_json( path, body );
				continue;
			}
			const json r = read_json( responses + "/" + std::to_string( n ) + ".json" );
			if ( !r.is_object() || !r.contains( "choices" ) || r["choices"].empty() ) continue;
			if ( !r["choices"][0].contains( "message" ) || !r["choices"][0]["message"].is_object() ) continue;
			const json& m = r["choices"][0]["message"];
			// Fields may be null (an answer cut at max_tokens inside its reasoning has no content): the reasoning alone is still the model's output
			auto str_of = [&]( const char* key ) { return m.contains( key ) && m[key].is_string() ? m[key].get<std::string>() : std::string(); };
			const std::string text = str_of( "content" ), reasoning = str_of( "reasoning_content" ).empty() ? str_of( "reasoning" ) : str_of( "reasoning_content" );
			if ( ( text.empty() && reasoning.empty() ) || text.find( "[Proxy error" ) != std::string::npos ) continue;
			json content = json::array();
			if ( !reasoning.empty() ) content.push_back( { { "type", "thinking" }, { "thinking", reasoning } } );
			if ( !text.empty() ) content.push_back( { { "type", "text" }, { "text", text } } );
			const json body = { { "model", "volta" }, { "max_tokens", 1 },
				{ "messages", json::array( { { { "role", "user" }, { "content", prompt } }, { { "role", "assistant" }, { "content", content } } } ) },
				{ "thinking", { { "type", "enabled" }, { "budget_tokens", 16000 } } } };
			written += write_json( path, body );
		}
		LOG_INFO( "[volta_sessions] %d of %zu bodies written to %s", written, prompts.size(), out.c_str() );
		return 0;
	}
	if ( out.empty() || sessions.empty() ) {
		LOG_ERROR( "usage: volta_sessions --out <dir> [--max-chars 120000] SESSION.jsonl ..." );
		return 2;
	}
	mkdir( out.c_str(), 0755 );
	int written = 0;
	for ( size_t s = 0; s < sessions.size(); ++s ) {
		json body;
		int n = 0;
		if ( !session_body( sessions[s], max_chars, body, n ) ) {
			LOG_WARN( "[volta_sessions] %s: no conversation", sessions[s].c_str() );
			continue;
		}
		const std::string path = out + "/" + std::to_string( s ) + ".json";
		std::ofstream f( path );
		f << body.dump();
		if ( !f ) {
			LOG_ERROR( "[volta_sessions] cannot write %s", path.c_str() );
			return 1;
		}
		std::printf( "%s\t%d messages\t%s\n", path.c_str(), n, sessions[s].c_str() );
		++written;
	}
	LOG_INFO( "[volta_sessions] %d of %zu sessions written to %s", written, sessions.size(), out.c_str() );
	return 0;
}
