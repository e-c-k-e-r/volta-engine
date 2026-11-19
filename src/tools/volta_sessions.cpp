// volta_sessions: Anthropic Messages request bodies from recorded Claude Code sessions (~/.claude/projects/*/*.jsonl transcripts), so the server
// renders them exactly as it rendered the live conversation (its count_tokens endpoint with dump_requests writes the rendered prompt, which
// test_ppl --calibrate --prompts replays). One body per session: its main-thread user and assistant entries in order, consecutive entries of
// one role joined into one message (thinking, text, tool_use and tool_result blocks as recorded), cut at a message boundary once the content
// passes --max-chars; side-chain (subagent) entries and meta entries are left out. Thinking is enabled on the request.
//
//   volta_sessions --out <dir> [--max-chars 120000] SESSION.jsonl ...      writes <dir>/<n>.json, n in argument order
#include "engine/core/log.h"
#include <nlohmann/json.hpp>
#include <cstdio>
#include <fstream>
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

int main( int argc, char** argv ) {
	std::string out;
	size_t max_chars = 120000;
	std::vector<std::string> sessions;
	for ( int i = 1; i < argc; ++i ) {
		const std::string a = argv[i];
		if ( a == "--out" && i + 1 < argc ) out = argv[++i];
		else if ( a == "--max-chars" && i + 1 < argc ) max_chars = std::stoull( argv[++i] );
		else sessions.push_back( a );
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
