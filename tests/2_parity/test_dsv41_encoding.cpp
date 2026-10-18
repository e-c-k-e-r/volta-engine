// DeepSeek-V4.1 prompt encoding against the checkpoint's own encoding/ reference: the golden cases in encoding/tests (input JSON ->
// expected text) and cases from encoding/test_encoding.py (reasoning effort, system tokens, thinking drop, DSML tool calls, namespaces).
//   test_dsv41_encoding <model dir or its encoding/ directory>   (default: ENGINE_TEST_DSV41_DIR or the test model path)
#include "engine/loader/deepseek_encoding.h"
#include "engine/runtime/test_paths.h"

#define JSON_NOEXCEPTION 1
#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

using json = nlohmann::ordered_json;
using engine::loader::DeepseekEncodeOptions;
using engine::loader::deepseek_v41_encode;

static int failures = 0;

static std::string slurp( const std::string& path ) {
	std::ifstream f( path, std::ios::binary );
	std::stringstream ss;
	ss << f.rdbuf();
	return ss.str();
}

static std::string encode( const json& messages, bool thinking, const std::string& effort = std::string(), const json& tools = json(), bool drop = true,
	const json& context = json() ) {
	DeepseekEncodeOptions o;
	o.thinking = thinking;
	o.reasoning_effort = effort;
	o.drop_thinking = drop;
	std::string out, err;
	if ( !deepseek_v41_encode( messages.dump(), tools.is_null() ? std::string() : tools.dump(), o, &out, &err, context.is_null() ? std::string() : context.dump() ) ) {
		return "ERROR: " + err;
	}
	return out;
}

static void check( const char* name, const std::string& got, const std::string& want ) {
	if ( got == want ) {
		std::printf( "[+] %s\n", name );
		return;
	}
	++failures;
	size_t i = 0;
	while ( i < got.size() && i < want.size() && got[i] == want[i] ) ++i;
	std::printf( "[-] %s: differs at byte %zu\n    got:  ...%s\n    want: ...%s\n", name, i, got.substr( i > 40 ? i - 40 : 0, 160 ).c_str(), want.substr( i > 40 ? i - 40 : 0, 160 ).c_str() );
}

static void check_has( const char* name, const std::string& got, const std::string& part, bool want = true ) {
	const bool has = got.find( part ) != std::string::npos;
	if ( has == want ) { std::printf( "[+] %s\n", name ); return; }
	++failures;
	std::printf( "[-] %s: %s \"%s\"\n    in: %s\n", name, want ? "missing" : "unexpected", part.c_str(), got.substr( 0, 400 ).c_str() );
}

int main( int argc, char** argv ) {
	std::string dir = argc > 1 ? argv[1] : engine::runtime::test::model_path( "DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw", "ENGINE_TEST_DSV41_DIR" );
	if ( dir.size() < 9 || dir.compare( dir.size() - 9, 9, "/encoding" ) != 0 ) dir += "/encoding";

	// golden cases (encoding.py load_cases + encode_case with thinking_mode "chat")
	for ( int id = 1; id <= 64; ++id ) {
		const std::string in = slurp( dir + "/tests/test_input_" + std::to_string( id ) + ".json" );
		if ( in.empty() ) break;
		const std::string want = slurp( dir + "/tests/test_output_" + std::to_string( id ) + ".txt" );
		json data = json::parse( in, nullptr, false );
		json c = data.is_object() ? data : json { { "messages", data } };
		const bool thinking = c.value( "thinking_mode", std::string( "chat" ) ) == "thinking";
		std::string effort;
		if ( c.contains( "reasoning_effort" ) && !c["reasoning_effort"].is_null() ) effort = c["reasoning_effort"].is_string() ? c["reasoning_effort"].get<std::string>() : c["reasoning_effort"].dump();
		const std::string got = encode( c["messages"], thinking, effort, c.contains( "tools" ) ? c["tools"] : json(), true, c.contains( "context" ) ? c["context"] : json() );
		check( ( "golden " + std::to_string( id ) ).c_str(), got, want );
	}

	const std::string BOS = "<｜begin▁of▁sentence｜>", EFF = "<｜System｜>Reasoning Effort: %d (range 1-100, the higher the value, the more thorough the reasoning)\n\n";
	auto eff = [&]( int b ) { char buf[256]; std::snprintf( buf, sizeof( buf ), EFF.c_str(), b ); return std::string( buf ); };
	const json q = json::array( { { { "role", "user" }, { "content", "question" } } } );
	for ( auto [e, b] : std::initializer_list<std::pair<const char*, int>>{ { "", 75 }, { "low", 50 }, { "high", 75 }, { "max", 100 }, { "1", 1 }, { "42", 42 }, { "100", 100 } } ) {
		check( ( std::string( "reasoning effort " ) + ( *e ? e : "(default)" ) ).c_str(), encode( q, true, e ), BOS + eff( b ) + "<｜User｜>question<｜Assistant｜><think>" );
	}
	check_has( "reasoning effort 101 rejected", encode( q, true, "101" ), "ERROR:" );
	check_has( "reasoning effort medium rejected", encode( q, true, "medium" ), "ERROR:" );
	check( "chat mode: no effort, no system token", encode( json::array( { { { "role", "user" }, { "content", "hello" } } } ), false, "max" ),
		BOS + "<｜User｜>hello<｜Assistant｜></think>" );
	check( "leading system message", encode( json::array( { { { "role", "system" }, { "content", "You are a helpful assistant." } }, { { "role", "user" }, { "content", "hello" } } } ), false ),
		BOS + "<｜System｜>You are a helpful assistant.<｜User｜>hello<｜Assistant｜></think>" );
	check( "mid-conversation system message",
		encode( json::array( { { { "role", "system" }, { "content", "sys" } }, { { "role", "user" }, { "content", "q1" } }, { { "role", "assistant" }, { "content", "a1" }, { "reasoning_content", "r1" } },
			{ { "role", "system" }, { "content", "mid sys" } } } ), true, "88" ),
		BOS + eff( 88 ) + "sys<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜><｜System｜>mid sys<｜Assistant｜><think>" );

	// tool calls (V41_TOOL_CALL_OUTPUT of test_encoding.py) and the thinking drop across tool results
	const json tool = { { "type", "function" }, { "function", { { "name", "lookup" }, { "description", "Look up a value" },
		{ "parameters", { { "type", "object" }, { "properties", { { "query", { { "type", "string" } } }, { "limit", { { "type", "integer" } } } } } } } } } };
	const json call_turn = json::array( { { { "role", "user" }, { "content", "question" } },
		{ { "role", "assistant" }, { "reasoning_content", "  reason  " }, { "content", "summary" },
		  { "tool_calls", json::array( { { { "type", "function" }, { "function", { { "name", "lookup" }, { "arguments", "{\"query\":\"value\",\"limit\":2}" } } } } } ) } } } );
	const std::string v41_call = "  reason  </think>summary\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n<｜DSML｜ parameter name=\"query\" string=\"true\">value</｜DSML｜ parameter>\n"
		"<｜DSML｜ parameter name=\"limit\" string=\"false\">2</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜>";
	check( "assistant DSML tool call (thinking)", encode( json::array( { call_turn[1] } ), true, "", json(), true, json::array( { call_turn[0] } ) ), v41_call );
	{
		const json msgs = json::array( { { { "role", "system" }, { "content", "system" } }, { { "role", "user" }, { "content", "question" } } } );
		const std::string p = encode( msgs, false, "", json::array( { tool } ) );
		check_has( "tool instructions use spaced DSML tags", p, "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"$TOOL_NAME\">\n<｜DSML｜ parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜ parameter>\n...\n</｜DSML｜ invoke>" );
		check_has( "tool schema as json.dumps", p, "{\"name\": \"lookup\", \"description\": \"Look up a value\", \"parameters\": {\"type\": \"object\", \"properties\": {\"query\": {\"type\": \"string\"}, \"limit\": {\"type\": \"integer\"}}}}" );
	}
	{
		json t = tool;
		t["namespace"] = { { "name", "search" }, { "description", "Search tools." } };
		const std::string p = encode( json::array( { { { "role", "system" }, { "content", "system" } } } ), false, "", json::array( { t } ) );
		check_has( "namespaced tool schema", p, "{\"name\": \"search::lookup\", \"description\": \"Search tools.\\nLook up a value\"" );
		check_has( "namespace key not rendered", p, "\"namespace\":", false );
	}
	{
		const json msgs = json::array( { { { "role", "user" }, { "content", "q1" } }, { { "role", "assistant" }, { "content", "a1" }, { "reasoning_content", "r1" } },
			{ { "role", "user" }, { "content", "q2" } },
			{ { "role", "assistant" }, { "reasoning_content", "r2" }, { "tool_calls", json::array( { { { "id", "c1" }, { "type", "function" }, { "function", { { "name", "lookup" }, { "arguments", "{}" } } } } } ) } },
			{ { "role", "tool" }, { "tool_call_id", "c1" }, { "content", "value" } } } );
		const std::string p = encode( msgs, true );
		check_has( "drop thinking: finished turn", p, "<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜>" );
		check_has( "drop thinking: r1 gone", p, "r1", false );
		check_has( "drop thinking: current turn kept across tool results", p, "<think>r2</think>" );
		check_has( "tool result closes the prompt", p.substr( p.size() - 80 ), "<tool_result>value</tool_result><｜Assistant｜><think>" );
		json next = msgs;
		next.push_back( { { "role", "user" }, { "content", "q3" } } );
		const std::string p2 = encode( next, true );
		check_has( "next turn drops r2", p2, "r2", false );
	}
	{
		// tool results sorted by the call order of the assistant turn
		const json msgs = json::array( { { { "role", "user" }, { "content", "q" } },
			{ { "role", "assistant" }, { "tool_calls", json::array( { { { "id", "a" }, { "type", "function" }, { "function", { { "name", "a" }, { "arguments", "{}" } } } },
				{ { "id", "b" }, { "type", "function" }, { "function", { { "name", "b" }, { "arguments", "{}" } } } } } ) } },
			{ { "role", "tool" }, { "tool_call_id", "b" }, { "content", "result b" } }, { { "role", "tool" }, { "tool_call_id", "a" }, { "content", "result a" } } } );
		const std::string p = encode( msgs, false );
		check_has( "tool results in call order", p, "<｜User｜><tool_result>result a</tool_result>\n\n<tool_result>result b</tool_result><｜Assistant｜></think>" );
	}
	check( "json.dumps escapes", engine::loader::py_json_dumps( "{\"a\": \"x\\ny\\u0001\\\"\", \"b\": [1, 2.5, true, null], \"c\": \"日本\"}" ),
		"{\"a\": \"x\\ny\\u0001\\\"\", \"b\": [1, 2.5, true, null], \"c\": \"日本\"}" );

	std::printf( "%s: %d failure(s)\n", failures ? "[-] DSV41 ENCODING FAIL" : "[+] DSV41 ENCODING PASS", failures );
	return failures ? 1 : 0;
}
