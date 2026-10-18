#include "engine/loader/deepseek_encoding.h"

#define JSON_NOEXCEPTION 1
#include <nlohmann/json.hpp>

#include <cstdio>
#include <map>
#include <optional>
#include <utility>
#include <vector>

// A line-by-line port of DeepSeek-V4.1's encoding/encoding.py (text path; images become the placeholder token). Function names follow it.

namespace engine::loader {

namespace {

using json = nlohmann::ordered_json;

const std::string kBos = "<｜begin▁of▁sentence｜>";
const std::string kEos = "<｜end▁of▁sentence｜>";
const std::string kThinkStart = "<think>";
const std::string kThinkEnd = "</think>";
const std::string kDsml = "｜DSML｜";
const std::string kUser = "<｜User｜>";
const std::string kAssistant = "<｜Assistant｜>";
const std::string kSystem = "<｜System｜>";
const std::string kLatestReminder = "<｜latest_reminder｜>";
const std::string kImage = "<｜deepseek_image｜>";
const std::string kCallsBlock = " calls", kCallTag = " invoke", kParamTag = " parameter";
const std::map<std::string, std::string> kTasks = { { "action", "<｜action｜>" }, { "query", "<｜query｜>" }, { "authority", "<｜authority｜>" },
	{ "domain", "<｜domain｜>" }, { "title", "<｜title｜>" }, { "read_url", "<｜read_url｜>" } };

// The engine builds without exceptions: the first failed check is recorded here and the encode reports it at the end
thread_local std::string g_error;

void require( bool ok, const std::string& what ) {
	if ( !ok && g_error.empty() ) g_error = what;
}

std::string str_or( const json& m, const char* key, const std::string& dflt = std::string() ) {
	if ( !m.is_object() || !m.contains( key ) || m[key].is_null() ) return dflt;
	return m[key].is_string() ? m[key].get<std::string>() : dflt;
}

bool truthy( const json& m, const char* key ) {
	if ( !m.is_object() || !m.contains( key ) ) return false;
	const json& v = m[key];
	if ( v.is_null() ) return false;
	if ( v.is_boolean() ) return v.get<bool>();
	if ( v.is_string() ) return !v.get<std::string>().empty();
	if ( v.is_array() || v.is_object() ) return !v.empty();
	if ( v.is_number() ) return v.get<double>() != 0.0;
	return true;
}

// json.dumps( ..., ensure_ascii=False ): ", " / ": " separators, the escapes Python writes for control characters
void dumps_string( const std::string& s, std::string& o ) {
	o += '"';
	for ( unsigned char c : s ) {
		switch ( c ) {
			case '"': o += "\\\""; break;
			case '\\': o += "\\\\"; break;
			case '\n': o += "\\n"; break;
			case '\r': o += "\\r"; break;
			case '\t': o += "\\t"; break;
			case '\b': o += "\\b"; break;
			case '\f': o += "\\f"; break;
			default:
				if ( c < 0x20 ) {
					char buf[8];
					std::snprintf( buf, sizeof( buf ), "\\u%04x", c );
					o += buf;
				}
				else o += static_cast<char>( c );
		}
	}
	o += '"';
}

void dumps( const json& v, std::string& o ) {
	if ( v.is_object() ) {
		o += '{';
		bool first = true;
		for ( auto it = v.begin(); it != v.end(); ++it ) {
			if ( !first ) o += ", ";
			first = false;
			dumps_string( it.key(), o );
			o += ": ";
			dumps( it.value(), o );
		}
		o += '}';
	}
	else if ( v.is_array() ) {
		o += '[';
		for ( size_t i = 0; i < v.size(); ++i ) {
			if ( i ) o += ", ";
			dumps( v[i], o );
		}
		o += ']';
	}
	else if ( v.is_string() ) dumps_string( v.get<std::string>(), o );
	else if ( v.is_null() ) o += "null";
	else if ( v.is_boolean() ) o += v.get<bool>() ? "true" : "false";
	else if ( v.is_number_float() ) {
		const double d = v.get<double>();
		if ( d == static_cast<double>( static_cast<long long>( d ) ) && std::abs( d ) < 1e16 ) o += std::to_string( static_cast<long long>( d ) ) + ".0";   // Python writes 2.0
		else o += v.dump();
	}
	else o += v.dump();
}

std::string to_json( const json& v ) {
	std::string o;
	dumps( v, o );
	return o;
}

std::pair<std::optional<std::string>, std::string> split_tool_name( const std::string& name_in, std::optional<std::string> ns ) {
	std::string name = name_in;
	const size_t sep = name.find( "::" );
	if ( sep != std::string::npos ) {
		const std::string prefix = name.substr( 0, sep );
		require( !ns || *ns == prefix, "Conflicting tool namespaces: " + ( ns ? *ns : std::string() ) + " != " + prefix );
		ns = prefix;
		name = name.substr( sep + 2 );
	}
	require( name.find( "::" ) == std::string::npos, "Tool name must not contain '::': " + name );
	require( !ns || ns->find( "::" ) == std::string::npos, "Tool namespace must not contain '::': " + ( ns ? *ns : std::string() ) );
	return { ns, name };
}

std::optional<std::string> namespace_of( const json& v ) {
	if ( v.is_string() ) return v.get<std::string>();
	if ( v.is_object() && v.contains( "name" ) && v["name"].is_string() ) return v["name"].get<std::string>();
	return std::nullopt;
}

std::string tool_name_for_encoding( const json& tool ) {
	std::optional<std::string> ns;
	if ( tool.contains( "namespace" ) && !tool["namespace"].is_null() ) ns = namespace_of( tool["namespace"] );
	const auto [n, name] = split_tool_name( str_or( tool, "name" ), ns );
	return n ? *n + "::" + name : name;
}

json tools_from_openai_format( const json& tools ) {
	json functions = json::array();
	for ( const json& tool : tools ) {
		json function = tool.contains( "function" ) && tool["function"].is_object() ? tool["function"] : tool;
		if ( tool.contains( "namespace" ) && !tool["namespace"].is_null() ) function["namespace"] = tool["namespace"];
		function["name"] = tool_name_for_encoding( function );
		json ns = function.contains( "namespace" ) ? function["namespace"] : json();
		function.erase( "namespace" );
		if ( ns.is_object() && truthy( ns, "description" ) ) {
			function["description"] = str_or( ns, "description" ) + "\n" + str_or( function, "description" );
		}
		functions.push_back( function );
	}
	return functions;
}

json tool_calls_from_openai_format( const json& tool_calls ) {
	json calls = json::array();
	for ( const json& tc : tool_calls ) {
		const json function = tc.contains( "function" ) && tc["function"].is_object() ? tc["function"] : json::object();
		std::optional<std::string> ns;
		if ( tc.contains( "namespace" ) && !tc["namespace"].is_null() ) ns = namespace_of( tc["namespace"] );
		else if ( function.contains( "namespace" ) && !function["namespace"].is_null() ) ns = namespace_of( function["namespace"] );
		const auto [n, name] = split_tool_name( str_or( function, "name" ), ns );
		json call = { { "name", name }, { "arguments", function.contains( "arguments" ) ? function["arguments"] : json() } };
		if ( n ) call["namespace"] = *n;
		calls.push_back( call );
	}
	return calls;
}

std::string render_tools( const json& tools ) {
	std::string schemas;
	for ( size_t i = 0; i < tools.size(); ++i ) schemas += ( i ? "\n" : "" ) + to_json( tools[i] );
	const std::string calls = kDsml + kCallsBlock, invoke = kDsml + kCallTag, param = kDsml + kParamTag;
	return "## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<" + calls + ">\" block like the following:\n\n"
		"<" + calls + ">\n<" + invoke + " name=\"$TOOL_NAME\">\n<" + param + " name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</" + param + ">\n...\n</" + invoke + ">\n"
		"<" + invoke + " name=\"$TOOL_NAME2\">\n...\n</" + invoke + ">\n</" + calls + ">\n\n"
		"String parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n"
		"If thinking_mode is enabled (triggered by " + kThinkStart + "), you MUST output your complete reasoning inside " + kThinkStart + "..." + kThinkEnd + " BEFORE any tool calls or final response.\n\n"
		"Otherwise, output directly after " + kThinkEnd + " with tool calls or final response.\n\n### Available Tool Schemas\n\n" + schemas +
		"\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";
}

std::string encode_arguments_to_dsml( const json& tool_call ) {
	json arguments = tool_call["arguments"];
	if ( !arguments.is_object() ) {
		for ( int i = 0; i < 2 && arguments.is_string(); ++i ) {
			json parsed = json::parse( arguments.get<std::string>(), nullptr, false );
			if ( parsed.is_discarded() ) break;
			arguments = parsed;
		}
		if ( !arguments.is_object() ) arguments = json { { "arguments", tool_call["arguments"] } };
	}
	std::string out;
	bool first = true;
	for ( auto it = arguments.begin(); it != arguments.end(); ++it ) {
		const bool is_str = it.value().is_string();
		if ( !first ) out += "\n";
		first = false;
		out += "<" + kDsml + kParamTag + " name=\"" + it.key() + "\" string=\"" + ( is_str ? "true" : "false" ) + "\">" + ( is_str ? it.value().get<std::string>() : to_json( it.value() ) ) +
			"</" + kDsml + kParamTag + ">";
	}
	return out;
}

json merge_tool_messages( const json& messages ) {
	json merged = json::array();
	for ( json msg : messages ) {
		const std::string role = str_or( msg, "role" );
		if ( role == "tool" ) {
			json content = msg.contains( "content" ) ? msg["content"] : json( "" );
			if ( truthy( msg, "content_blocks" ) && !( msg.contains( "content" ) && msg["content"].is_string() ) ) content = msg["content_blocks"];
			json block = { { "type", "tool_result" }, { "tool_use_id", str_or( msg, "tool_call_id" ) }, { "content", content } };
			if ( !merged.empty() && str_or( merged.back(), "role" ) == "user" && merged.back().contains( "content_blocks" ) ) merged.back()["content_blocks"].push_back( block );
			else merged.push_back( { { "role", "user" }, { "content_blocks", json::array( { block } ) } } );
		}
		else if ( role == "user" ) {
			json blocks;
			if ( msg.contains( "content_blocks" ) && !msg["content_blocks"].is_null() ) blocks = msg["content_blocks"];
			else {
				const json content = msg.contains( "content" ) ? msg["content"] : json( "" );
				blocks = content.is_array() ? content : json::array( { { { "type", "text" }, { "text", content.is_string() ? content : json( "" ) } } } );
			}
			if ( !merged.empty() && str_or( merged.back(), "role" ) == "user" && merged.back().contains( "content_blocks" ) && !merged.back().contains( "task" ) ) {
				for ( const json& b : blocks ) merged.back()["content_blocks"].push_back( b );
			}
			else {
				msg["content_blocks"] = blocks;
				merged.push_back( msg );
			}
		}
		else merged.push_back( msg );
	}
	return merged;
}

void sort_tool_results_by_call_order( json& messages ) {
	std::map<std::string, int> order;
	for ( json& msg : messages ) {
		const std::string role = str_or( msg, "role" );
		if ( role == "assistant" && truthy( msg, "tool_calls" ) ) {
			order.clear();
			int idx = 0;
			for ( const json& tc : msg["tool_calls"] ) {
				std::string id = str_or( tc, "id" );
				if ( id.empty() && tc.contains( "function" ) ) id = str_or( tc["function"], "id" );
				if ( !id.empty() ) order[id] = idx;
				++idx;
			}
		}
		else if ( role == "user" && truthy( msg, "content_blocks" ) ) {
			std::vector<json> tool_blocks;
			for ( const json& b : msg["content_blocks"] ) if ( str_or( b, "type" ) == "tool_result" ) tool_blocks.push_back( b );
			if ( tool_blocks.size() > 1 && !order.empty() ) {
				auto key = [&]( const json& b ) { auto it = order.find( str_or( b, "tool_use_id" ) ); return it == order.end() ? 0 : it->second; };
				std::stable_sort( tool_blocks.begin(), tool_blocks.end(), [&]( const json& a, const json& b ) { return key( a ) < key( b ); } );
				size_t k = 0;
				for ( json& b : msg["content_blocks"] ) if ( str_or( b, "type" ) == "tool_result" ) b = tool_blocks[k++];
			}
		}
	}
}

json process_image_blocks( const json& blocks ) {
	json out = json::array();
	for ( const json& block : blocks ) {
		if ( !block.is_object() ) { out.push_back( block ); continue; }
		const std::string type = str_or( block, "type" );
		if ( type == "image" || type == "image_url" ) out.push_back( { { "type", "text" }, { "text", kImage } } );
		else if ( type == "tool_result" && block.contains( "content" ) && block["content"].is_array() ) {
			json b = block;
			b["content"] = process_image_blocks( block["content"] );
			out.push_back( b );
		}
		else {
			if ( type == "text" ) require( str_or( block, "text" ).find( kImage ) == std::string::npos, "Text block contains image placeholder" );
			out.push_back( block );
		}
	}
	return out;
}

json process_image_messages( const json& messages ) {
	json processed = json::array();
	for ( json msg : messages ) {
		require( !( msg.contains( "content" ) && msg["content"].is_string() && msg["content"].get<std::string>().find( kImage ) != std::string::npos ),
			"Message content contains the image special token" );
		require( str_or( msg, "reasoning_content" ).find( kImage ) == std::string::npos, "reasoning_content contains the image special token" );
		if ( msg.contains( "content" ) && msg["content"].is_array() && !msg.contains( "content_blocks" ) ) {
			msg["content_blocks"] = msg["content"];
			msg.erase( "content" );
		}
		if ( truthy( msg, "content_blocks" ) ) {
			msg["content_blocks"] = process_image_blocks( msg["content_blocks"] );
			if ( !( msg.contains( "content" ) && msg["content"].is_string() ) ) {
				std::string text;
				bool first = true;
				for ( const json& b : msg["content_blocks"] ) {
					if ( !b.is_object() || str_or( b, "type" ) != "text" ) continue;
					text += ( first ? "" : "\n\n" ) + str_or( b, "text" );
					first = false;
				}
				msg["content"] = text;
			}
		}
		processed.push_back( msg );
	}
	return processed;
}

int find_last_user_index( const json& messages ) {
	for ( int idx = static_cast<int>( messages.size() ) - 1; idx >= 0; --idx ) {
		const std::string role = str_or( messages[idx], "role" );
		if ( role == "user" || ( role == "system" && idx > 0 ) ) {
			const json& m = messages[idx];
			if ( truthy( m, "content_blocks" ) ) {
				bool any = false;
				for ( const json& b : m["content_blocks"] ) {
					const std::string t = str_or( b, "type" );
					any = any || t == "text" || t == "image" || t == "image_url";
				}
				if ( !any ) continue;
			}
			return idx;
		}
	}
	return -1;
}

std::string render_reasoning_effort( int index, bool thinking, const std::string& effort_in ) {
	const std::string effort = effort_in.empty() ? "high" : effort_in;
	int budget = 0;
	if ( effort == "low" ) budget = 50;
	else if ( effort == "high" ) budget = 75;
	else if ( effort == "max" ) budget = 100;
	else {
		char* end = nullptr;
		const long v = std::strtol( effort.c_str(), &end, 10 );
		require( end && *end == '\0' && v >= 1 && v <= 100, "Invalid reasoning effort for deepseek_v41: " + effort + ", should be int within [1,100] or low / high / max" );
		budget = static_cast<int>( v );
	}
	if ( index == 0 && thinking ) return "Reasoning Effort: " + std::to_string( budget ) + " (range 1-100, the higher the value, the more thorough the reasoning)\n\n";
	return std::string();
}

std::string render_message( int index, const json& messages, bool thinking, bool drop_thinking, const std::string& effort ) {
	const json& msg = messages[index];
	const int last_user = find_last_user_index( messages );
	const std::string role = str_or( msg, "role" );
	const std::string content = msg.contains( "content" ) && msg["content"].is_string() ? msg["content"].get<std::string>() : std::string();
	json tools = truthy( msg, "tools" ) ? tools_from_openai_format( msg["tools"] ) : json();
	json tool_calls = truthy( msg, "tool_calls" ) ? tool_calls_from_openai_format( msg["tool_calls"] ) : json();

	const std::string effort_prompt = render_reasoning_effort( index, thinking, effort );
	std::string prompt = ( index == 0 && ( !effort_prompt.empty() || role == "system" ) ) ? kSystem : std::string();
	prompt += effort_prompt;

	if ( role == "system" ) {
		if ( index > 0 ) prompt += kSystem;
		prompt += content;
		if ( !tools.is_null() ) prompt += "\n\n" + render_tools( tools );
		if ( truthy( msg, "response_format" ) ) prompt += "\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n" + to_json( msg["response_format"] );
	}
	else if ( role == "user" ) {
		prompt += kUser;
		if ( truthy( msg, "content_blocks" ) ) {
			std::string joined;
			bool first = true;
			for ( const json& block : msg["content_blocks"] ) {
				const std::string type = str_or( block, "type" );
				std::string part;
				if ( type == "text" ) part = str_or( block, "text" );
				else if ( type == "tool_result" ) {
					std::string tc;
					if ( block.contains( "content" ) && block["content"].is_array() ) {
						bool f2 = true;
						for ( const json& b : block["content"] ) {
							tc += f2 ? "" : "\n\n";
							f2 = false;
							tc += str_or( b, "type" ) == "text" ? str_or( b, "text" ) : "[Unsupported " + str_or( b, "type" ) + "]";
						}
					}
					else tc = block.contains( "content" ) && block["content"].is_string() ? block["content"].get<std::string>() : std::string();
					part = "<tool_result>" + tc + "</tool_result>";
				}
				else part = "[Unsupported " + type + "]";
				joined += ( first ? "" : "\n\n" ) + part;
				first = false;
			}
			prompt += joined;
		}
		else prompt += content;
	}
	else if ( role == "latest_reminder" ) {
		prompt += kLatestReminder + content;
	}
	else if ( role == "assistant" ) {
		std::string tc_content;
		if ( !tool_calls.is_null() ) {
			std::string list;
			for ( size_t i = 0; i < tool_calls.size(); ++i ) {
				list += ( i ? "\n" : "" ) + std::string( "<" ) + kDsml + kCallTag + " name=\"" + tool_name_for_encoding( tool_calls[i] ) + "\">\n" + encode_arguments_to_dsml( tool_calls[i] ) +
					"\n</" + kDsml + kCallTag + ">";
			}
			tc_content = "\n\n<" + kDsml + kCallsBlock + ">\n" + list + "\n</" + kDsml + kCallsBlock + ">";
		}
		std::string thinking_part;
		const bool prev_task = index >= 1 && messages[index - 1].contains( "task" ) && !messages[index - 1]["task"].is_null();
		if ( thinking && !prev_task && ( !drop_thinking || index > last_user ) ) thinking_part = str_or( msg, "reasoning_content" ) + kThinkEnd;
		prompt += thinking_part + content + tc_content;
		if ( !truthy( msg, "wo_eos" ) ) prompt += kEos;
	}
	else {
		require( false, role == "tool" ? "tool messages must be merged first" : "Unknown role: " + role );
	}

	if ( index + 1 < static_cast<int>( messages.size() ) ) {
		const std::string next = str_or( messages[index + 1], "role" );
		if ( next != "assistant" && next != "latest_reminder" ) return prompt;
	}
	if ( msg.contains( "task" ) && !msg["task"].is_null() ) {
		const std::string task = str_or( msg, "task" );
		auto it = kTasks.find( task );
		require( it != kTasks.end(), "Invalid task: " + task );
		if ( task != "action" ) prompt += it->second;
		else prompt += kAssistant + ( thinking ? kThinkStart : kThinkEnd ) + it->second;
	}
	else if ( role == "user" || ( role == "system" && index > 0 ) ) {
		prompt += kAssistant;
		if ( !drop_thinking && thinking ) prompt += kThinkStart;
		else if ( drop_thinking && thinking && index >= last_user ) prompt += kThinkStart;
		else prompt += kThinkEnd;
	}
	return prompt;
}

json drop_thinking_messages( const json& messages ) {
	const int last_user = find_last_user_index( messages );
	json result = json::array();
	for ( int idx = 0; idx < static_cast<int>( messages.size() ); ++idx ) {
		const std::string role = str_or( messages[idx], "role" );
		if ( role == "user" || role == "system" || role == "tool" || role == "latest_reminder" || role == "direct_search_results" || idx >= last_user ) result.push_back( messages[idx] );
		else if ( role == "assistant" ) {
			json m = messages[idx];
			m.erase( "reasoning_content" );
			result.push_back( m );
		}
	}
	return result;
}

std::string encode_messages_text( const json& messages, const json& context, bool thinking, bool drop_thinking, bool add_bos, const std::string& effort ) {
	json full = context;
	for ( const json& m : messages ) full.push_back( m );
	std::string prompt = add_bos && context.empty() ? kBos : std::string();
	bool eff_drop = drop_thinking;
	for ( const json& m : full ) if ( truthy( m, "tools" ) ) eff_drop = false;
	int num, ctx_len;
	if ( thinking && eff_drop ) {
		full = drop_thinking_messages( full );
		num = static_cast<int>( full.size() ) - static_cast<int>( drop_thinking_messages( context ).size() );
		ctx_len = static_cast<int>( full.size() ) - num;
	}
	else {
		num = static_cast<int>( messages.size() );
		ctx_len = static_cast<int>( context.size() );
	}
	for ( int i = 0; i < num; ++i ) prompt += render_message( i + ctx_len, full, thinking, eff_drop, effort );
	return prompt;
}

} // namespace

std::string py_json_dumps( const std::string& json_text ) {
	const json j = json::parse( json_text, nullptr, false );
	return j.is_discarded() ? std::string() : to_json( j );
}

bool deepseek_v41_encode( const std::string& messages_json, const std::string& tools_json, const DeepseekEncodeOptions& options, std::string* out, std::string* error,
	const std::string& context_json ) {
	json messages = json::parse( messages_json, nullptr, false );
	json context = context_json.empty() ? json::array() : json::parse( context_json, nullptr, false );
	if ( messages.is_discarded() || !messages.is_array() || context.is_discarded() || !context.is_array() ) {
		if ( error ) *error = "messages must be a JSON array";
		return false;
	}
	g_error.clear();
	{
		if ( !tools_json.empty() ) {
			json tools = json::parse( tools_json, nullptr, false );
			require( !tools.is_discarded() && tools.is_array(), "tools must be a JSON array" );
			if ( !tools.empty() ) {
				// the schemas render inside the first system message (encoding.py attaches them to the conversation's first message)
				json& host = !context.empty() ? context : messages;
				if ( host.empty() || str_or( host[0], "role" ) != "system" ) host.insert( host.begin(), json { { "role", "system" }, { "content", "" } } );
				host[0]["tools"] = tools;
			}
		}
		json ctx = merge_tool_messages( context );
		json msgs = merge_tool_messages( messages );
		json all = ctx;
		for ( const json& m : msgs ) all.push_back( m );
		sort_tool_results_by_call_order( all );
		json sorted_msgs = json::array();
		for ( size_t i = ctx.size(); i < all.size(); ++i ) sorted_msgs.push_back( all[i] );
		const json pctx = ctx.empty() ? json::array() : process_image_messages( ctx );
		const json pmsgs = process_image_messages( sorted_msgs );
		*out = encode_messages_text( pmsgs, pctx, options.thinking, options.drop_thinking, options.add_bos, options.reasoning_effort );
	}
	if ( !g_error.empty() ) {
		if ( error ) *error = g_error;
		return false;
	}
	return true;
}

} // namespace engine::loader
