#pragma once

#include <string>

namespace engine::loader {

// DeepSeek-V4.1's prompt format (the checkpoint's encoding/encoding.py, which it ships instead of a chat template): BOS, an optional
// <｜System｜> block (with the reasoning-effort prefix in thinking mode and the tool schemas), <｜User｜> / <｜Assistant｜> turns, tool
// results merged into the following user turn as <tool_result> blocks, assistant tool calls as DSML (<｜DSML｜ calls> / invoke / parameter),
// reasoning of finished turns dropped unless tools are declared. JSON is serialized like Python's json.dumps(ensure_ascii=False).
struct DeepseekEncodeOptions {
	bool thinking = false;                // thinking_mode "thinking" (else "chat")
	std::string reasoning_effort;         // "", "low", "high", "max" or "1".."100" (thinking mode; default "high")
	bool drop_thinking = true;
	bool add_bos = true;
};

// messages_json: an array of OpenAI-style messages (role system / user / assistant / tool / latest_reminder; content a string or a list
// of {type text|image|image_url|tool_result} parts; assistant reasoning_content and tool_calls; tool_call_id). tools_json: an optional
// array of tool definitions, rendered in the first system message (an empty one is inserted when the conversation has none).
// context_json: an optional array of earlier messages (encoding.py's `context`): rendered as history without the BOS.
bool deepseek_v41_encode( const std::string& messages_json, const std::string& tools_json, const DeepseekEncodeOptions& options, std::string* out, std::string* error,
	const std::string& context_json = std::string() );

// Python json.dumps( value, ensure_ascii=False ) of a JSON text (", " and ": " separators, keys in their given order)
std::string py_json_dumps( const std::string& json_text );

} // namespace engine::loader
