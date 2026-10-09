// Tier 2: OutputParser splits model output into reasoning, content and tool calls identically for any chunking, and
// recovers what the checkpoint chat templates render for assistant turns (template -> text -> parser round trip).
#include "engine/server/output_parser.h"

#define JSON_NOEXCEPTION 1
#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#ifndef TEST_DATA_DIR
#error "TEST_DATA_DIR must point at tests/data"
#endif

using engine::loader::ModelFamily;
using engine::server::OutputEvent;
using engine::server::OutputParser;

static int g_failed = 0;

#define CHECK( cond, ... ) do { if ( !( cond ) ) { ++g_failed; std::printf( "[FAIL] " __VA_ARGS__ ); std::printf( "\n" ); } } while ( 0 )

struct Result {
	std::string reasoning, content;
	std::vector<std::pair<std::string, std::string>> calls;
	bool operator==( const Result& o ) const { return reasoning == o.reasoning && content == o.content && calls == o.calls; }
};

static Result run( ModelFamily fam, bool in_reasoning, const std::string& text, size_t chunk, const std::string& tools_json = "" ) {
	OutputParser p( fam, in_reasoning, engine::server::tool_schemas_from_json( tools_json ) );
	std::vector<OutputEvent> ev;
	for ( size_t k = 0; k < text.size(); k += chunk ) p.feed( text.substr( k, chunk ), ev );
	p.finish( ev );
	Result r;
	for ( const auto& e : ev ) {
		if ( e.type == OutputEvent::Type::Reasoning ) r.reasoning += e.text;
		else if ( e.type == OutputEvent::Type::Content ) r.content += e.text;
		else r.calls.emplace_back( e.call.name, e.call.arguments );
	}
	return r;
}

static void check_chunking( const char* what, ModelFamily fam, bool in_reasoning, const std::string& text, const Result& want, const std::string& tools = "" ) {
	for ( size_t chunk : { size_t( 1 ), size_t( 2 ), size_t( 5 ), size_t( 13 ), text.size() + 1 } ) {
		const Result got = run( fam, in_reasoning, text, chunk, tools );
		CHECK( got == want, "%s (chunk %zu): reasoning='%s' content='%s' calls=%zu", what, chunk, got.reasoning.c_str(), got.content.c_str(), got.calls.size() );
	}
}

static std::string read_file( const std::string& path ) {
	std::ifstream f( path, std::ios::binary );
	std::stringstream ss;
	ss << f.rdbuf();
	return ss.str();
}

int main() {
	const std::string tools = R"([{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"s":{"type":"string"},"n":{"type":"integer"},"o":{"type":"object"}}}}}])";

	// Qwen
	check_chunking( "qwen think+content", ModelFamily::QWEN, true, "Let me think.\n</think>\n\nThe answer.", { "Let me think.", "The answer.", {} } );
	check_chunking( "qwen no think", ModelFamily::QWEN, false, "\n\nJust text", { "", "Just text", {} } );
	check_chunking( "qwen think tag in content", ModelFamily::QWEN, false, "<think>\nhm\n</think>\n\nok", { "hm", "ok", {} } );
	check_chunking( "qwen literal angle brackets", ModelFamily::QWEN, false, "a < b and <b> c </tool> d", { "", "a < b and <b> c </tool> d", {} } );
	check_chunking( "qwen call", ModelFamily::QWEN, true, "plan\n</think>\n\nChecking.\n\n<tool_call>\n<function=f>\n<parameter=s>\n007\n</parameter>\n<parameter=n>\n42\n</parameter>\n<parameter=o>\n{\"a\": [1, 2]}\n</parameter>\n</function>\n</tool_call>",
		{ "plan", "Checking.", { { "f", R"({"s":"007","n":42,"o":{"a":[1,2]}})" } } }, tools );
	check_chunking( "qwen two calls", ModelFamily::QWEN, false, "<tool_call>\n<function=f>\n<parameter=n>\n1\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=f>\n<parameter=n>\n2\n</parameter>\n</function>\n</tool_call>",
		{ "", "", { { "f", R"({"n":1})" }, { "f", R"({"n":2})" } } }, tools );
	check_chunking( "qwen multiline string", ModelFamily::QWEN, false, "<tool_call>\n<function=f>\n<parameter=s>\nline1\nline2\n</parameter>\n</function>\n</tool_call>",
		{ "", "", { { "f", R"({"s":"line1\nline2"})" } } }, tools );
	check_chunking( "qwen unterminated call", ModelFamily::QWEN, false, "x <tool_call>\n<function=f>", { "", "x<tool_call>\n<function=f>", {} } );

	// Tolerated damage: the last closing tags missing (generation stopped early)
	check_chunking( "qwen missing closers", ModelFamily::QWEN, false, "<tool_call>\n<function=f>\n<parameter=n>\n7\n</parameter>\n<parameter=s>\nabc\n</tool_call>",
		{ "", "", { { "f", R"({"n":7,"s":"abc"})" } } }, tools );
	check_chunking( "qwen no function end", ModelFamily::QWEN, false, "<tool_call>\n<function=f>\n<parameter=n>\n7\n</parameter>\n</tool_call>",
		{ "", "", { { "f", R"({"n":7})" } } }, tools );
	check_chunking( "glm missing arg_value end", ModelFamily::GLM, false, "<tool_call>f<arg_key>n</arg_key><arg_value>7</tool_call>",
		{ "", "", { { "f", R"({"n":7})" } } }, tools );

	// GLM
	check_chunking( "glm think+content", ModelFamily::GLM, true, "reason</think>Answer", { "reason", "Answer", {} } );
	check_chunking( "glm call", ModelFamily::GLM, true, "r</think>go<tool_call>f<arg_key>s</arg_key><arg_value>007</arg_value><arg_key>n</arg_key><arg_value>42</arg_value></tool_call>",
		{ "r", "go", { { "f", R"({"s":"007","n":42})" } } }, tools );
	check_chunking( "glm no args", ModelFamily::GLM, false, "<tool_call>f</tool_call>", { "", "", { { "f", "{}" } } }, tools );

	// Template round trip: what the checkpoint template renders for an assistant turn must parse back
	const std::string data = TEST_DATA_DIR;
	const std::string rt_tools = R"([{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"},"opts":{"type":"object"},"tags":{"type":"array"}}}}}])";
	auto cases = nlohmann::ordered_json::parse( read_file( data + "/chat_template_cases.json" ), nullptr, false );
	int round_trips = 0;
	for ( const char* name : { "qwen3_5", "glm5_next" } ) {
		const bool glm = std::string( name ) == "glm5_next";
		for ( auto& c : cases["templates"][name] ) {
			// reasoning-bearing assistant turn with tool calls, rendered by the reference Jinja2, generation prompt off
			if ( c["name"] != "args_types" && c["name"] != "parallel" && c["name"] != "toolcall" ) continue;
			if ( !c.contains( "expected" ) || c["vars"]["add_generation_prompt"].get<bool>() || c["vars"].contains( "enable_thinking" ) || c["vars"].contains( "reasoning_effort" ) || c["vars"].contains( "preserve_thinking" ) || c["vars"].contains( "clear_thinking" ) ) continue;
			const std::string text = c["expected"].get<std::string>();
			// the cases hold a single assistant turn
			const std::string marker = glm ? "<|assistant|>" : "<|im_start|>assistant\n";
			const size_t at = text.rfind( marker );
			if ( at == std::string::npos ) continue;
			std::string turn = text.substr( at + marker.size() );
			size_t end = glm ? turn.find( "<|observation|>" ) : turn.find( "<|im_end|>" );
			if ( end != std::string::npos ) turn = turn.substr( 0, end );
			const std::string think_open = glm ? "<think>" : "<think>\n";
			const bool opens = turn.compare( 0, think_open.size(), think_open ) == 0;
			if ( opens ) turn.erase( 0, think_open.size() );
			const bool in_reasoning = opens && turn.find( "</think>" ) != std::string::npos;
			if ( !opens ) continue;
			const Result got = run( glm ? ModelFamily::GLM : ModelFamily::QWEN, in_reasoning, turn, 7, rt_tools );
			// find the originating assistant message with tool calls
			size_t want_calls = 0;
			std::vector<nlohmann::json> expect;
			for ( auto& m : c["vars"]["messages"] ) {
				if ( m["role"] == "assistant" && m.contains( "tool_calls" ) ) {
					for ( auto& tc : m["tool_calls"] ) {
						expect.push_back( tc["function"] );
						++want_calls;
					}
				}
			}
			CHECK( got.calls.size() == want_calls, "%s/%s: %zu calls parsed, %zu rendered", name, c["name"].get<std::string>().c_str(), got.calls.size(), want_calls );
			for ( size_t k = 0; k < std::min( got.calls.size(), expect.size() ); ++k ) {
				auto args = nlohmann::json::parse( got.calls[k].second, nullptr, false );
				CHECK( got.calls[k].first == expect[k]["name"].get<std::string>() && args == nlohmann::json::parse( expect[k]["arguments"].dump() ),
					"%s/%s: call %zu differs: %s vs %s", name, c["name"].get<std::string>().c_str(), k, got.calls[k].second.c_str(), expect[k]["arguments"].dump().c_str() );
			}
			++round_trips;
		}
	}
	CHECK( round_trips >= 4, "only %d template round trips ran", round_trips );
	std::printf( "%d round trips, %d failed checks\n", round_trips, g_failed );
	return g_failed ? 1 : 0;
}
