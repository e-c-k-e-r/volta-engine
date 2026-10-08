#pragma once

#include "engine/loader/chat_template.h"

#include <map>
#include <string>
#include <vector>

namespace engine::server {

struct ToolCallOut {
	std::string name;
	std::string arguments; // JSON object text
};

struct OutputEvent {
	enum class Type { Content, Reasoning, ToolCall };
	Type type = Type::Content;
	std::string text;   // Content, Reasoning
	ToolCallOut call;   // ToolCall
};

// Parameter types of the request's tools, used to decode argument values the model wrote as text.
// tool name -> parameter name -> JSON schema "type" ("" when absent or not a single string)
using ToolSchemas = std::map<std::string, std::map<std::string, std::string>>;

// Extracts the schemas from an OpenAI-style tools array ([{"type":"function","function":{...}}] or bare functions).
ToolSchemas tool_schemas_from_json( const std::string& tools_json );

// Incremental parser for the model's output: <think> reasoning, plain content, and <tool_call> blocks in the family's
// syntax (Qwen: <function=name><parameter=key>value</parameter></function>; GLM: name<arg_key>k</arg_key><arg_value>v</arg_value>).
// The result does not depend on how the text is split across feed() calls.
class OutputParser {
public:
	// starts_in_reasoning: the rendered prompt ended inside an open <think> block
	OutputParser( engine::loader::ModelFamily family, bool starts_in_reasoning, ToolSchemas schemas );

	void feed( const std::string& text, std::vector<OutputEvent>& events );
	void finish( std::vector<OutputEvent>& events );

	bool saw_tool_call() const { return saw_tool_call_; }

private:
	enum class State { Reasoning, Content, ToolCall };

	void process( std::vector<OutputEvent>& events, bool final );
	void emit_text( OutputEvent::Type type, const std::string& s, std::vector<OutputEvent>& events );
	void end_section();
	bool parse_tool_call( const std::string& body, ToolCallOut* out ) const;
	std::string decode_value( const std::string& tool, const std::string& key, const std::string& raw ) const;

	engine::loader::ModelFamily family_;
	ToolSchemas schemas_;
	State state_;
	std::string buf_;
	bool lead_trim_ = true;
	std::string pending_ws_;
	bool emitted_any_ = false;
	bool saw_tool_call_ = false;
};

} // namespace engine::server
