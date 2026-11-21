// volta_sessions: Anthropic Messages request bodies from recorded Claude Code sessions (~/.claude/projects/*/*.jsonl transcripts), so the server
// renders them exactly as it rendered the live conversation (its count_tokens endpoint with dump_requests writes the rendered prompt, which
// test_ppl --calibrate --prompts replays). One body per session: its main-thread user and assistant entries in order, consecutive entries of
// one role joined into one message (thinking, text, tool_use and tool_result blocks as recorded), cut at a message boundary once the content
// passes --max-chars; side-chain (subagent) entries and meta entries are left out. Thinking is enabled on the request. --split writes the
// rest of a session too, as further bodies of up to --max-chars each (cut before a user turn, so each starts with one): a long agent session
// renders whole in parts that fit the server's context.
//
//   volta_sessions --out <dir> [--max-chars 120000] [--split] SESSION.jsonl ...      writes <dir>/<n>.json, n in argument order (--split: in
//                                                                                    order of the sessions' parts)
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

// A session's main-thread messages in order, consecutive entries of one role joined, from its first user turn
json session_messages( const std::string& path ) {
	json msgs = json::array();
	std::ifstream in( path );
	if ( !in ) return msgs;
	for ( std::string line; std::getline( in, line ); ) {
		const json e = json::parse( line, nullptr, false );
		if ( e.is_discarded() || !e.is_object() || !e.contains( "message" ) ) continue;
		if ( e.value( "isSidechain", false ) || e.value( "isMeta", false ) ) continue;
		const json& m = e["message"];
		if ( !m.is_object() ) continue;
		const std::string role = m.value( "role", "" );
		if ( role != "user" && role != "assistant" ) continue;
		json b = blocks( m.contains( "content" ) ? m["content"] : json() );
		if ( b.empty() ) continue;
		if ( !msgs.empty() && msgs.back()["role"] == role ) {
			for ( auto& x : b ) msgs.back()["content"].push_back( x );
		} else {
			msgs.push_back( { { "role", role }, { "content", b } } );
		}
	}
	// The API starts with a user turn
	while ( !msgs.empty() && msgs.front()["role"] != "user" ) msgs.erase( msgs.begin() );
	return msgs;
}

json body_of( const json& msgs ) {
	return { { "model", "volta" }, { "max_tokens", 1 }, { "messages", msgs }, { "thinking", { { "type", "enabled" }, { "budget_tokens", 16000 } } } };
}

// The messages as parts of up to max_chars of content each: without all, the first part only, cut at the message boundary where the content
// passes max_chars (it ends on either turn, both render); with all, every part, each cut before a user turn so that the next starts with one
// (its first tool results answer the previous part's tool calls), a message larger than max_chars making a part of its own. A later part
// opens with the session's first message (the task), since a chat template wants a user query and the part's own first turn is tool results
std::vector<json> session_parts( const json& msgs, size_t max_chars, bool all ) {
	std::vector<json> parts;
	json cur = json::array();
	size_t total = 0;
	for ( const json& m : msgs ) {
		const size_t c = chars( m["content"] );
		if ( total + c > max_chars && !cur.empty() && ( !all || m["role"] == "user" ) ) {
			parts.push_back( cur );
			if ( !all ) return parts;
			cur = json::array( { msgs[0] } );
			total = chars( msgs[0]["content"] );
		}
		cur.push_back( m );
		total += c;
	}
	if ( !cur.empty() ) parts.push_back( cur );
	return parts;
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
	bool split = false;
	int max_tokens = 6000;
	double temperature = 1.0, top_p = 0.95;
	std::vector<std::string> sessions;
	for ( int i = 1; i < argc; ++i ) {
		const std::string a = argv[i];
		if ( a == "--out" && i + 1 < argc ) out = argv[++i];
		else if ( a == "--max-chars" && i + 1 < argc ) max_chars = std::stoull( argv[++i] );
		else if ( a == "--split" ) split = true;
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
		LOG_ERROR( "usage: volta_sessions --out <dir> [--max-chars 120000] [--split] SESSION.jsonl ..." );
		return 2;
	}
	mkdir( out.c_str(), 0755 );
	int written = 0, bodies = 0;
	for ( size_t s = 0; s < sessions.size(); ++s ) {
		const json msgs = session_messages( sessions[s] );
		if ( msgs.empty() ) {
			LOG_WARN( "[volta_sessions] %s: no conversation", sessions[s].c_str() );
			continue;
		}
		const std::vector<json> parts = session_parts( msgs, max_chars, split );
		for ( size_t p = 0; p < parts.size(); ++p ) {
			// one body per session as <s>.json, or with --split every part, numbered on over all the sessions' parts
			const std::string path = out + "/" + std::to_string( split ? bodies : static_cast<int>( s ) ) + ".json";
			std::ofstream f( path );
			f << body_of( parts[p] ).dump();
			if ( !f ) {
				LOG_ERROR( "[volta_sessions] cannot write %s", path.c_str() );
				return 1;
			}
			std::printf( "%s\t%zu messages\t%s (part %zu of %zu)\n", path.c_str(), parts[p].size(), sessions[s].c_str(), p + 1, parts.size() );
			++bodies;
		}
		++written;
	}
	LOG_INFO( "[volta_sessions] %d of %zu sessions written to %s (%d bodies)", written, sessions.size(), out.c_str(), bodies );
	return 0;
}
