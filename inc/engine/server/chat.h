#pragma once

#include "engine/runtime/generator.h"
#include "engine/server/config.h"
#include "engine/server/http.h"
#include "engine/server/output_parser.h"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace engine::server {

struct ApiError {
	int status = 400;
	std::string type = "invalid_request_error";
	std::string message;
};

// Protocol-independent chat request. messages_json / tools_json are in the HF chat-template shape:
// messages carry role, content (string or text parts), reasoning_content, tool_calls[{function:{name, arguments(object)}}],
// tool_call_id; tools is a list of {"type":"function","function":{name, description, parameters}}.
struct ChatRequest {
	std::string model;
	std::string messages_json;
	std::string tools_json;      // empty: no tools
	bool stream = false;
	bool include_usage = false;  // OpenAI stream_options.include_usage
	int max_tokens = 0;          // 0: server default
	float temperature = -1.0f;   // <0: server default
	float top_p = 1.0f;
	int top_k = 16;
	uint64_t seed = 0;
	std::vector<std::string> stop;
	int enable_thinking = -1;    // -1: server default
	std::string reasoning_effort;
};

struct Usage {
	int prompt_tokens = 0;
	int completion_tokens = 0;
	double ttft_ms = 0.0;
	double decode_tok_s = 0.0;
};

// Writes one protocol's response for one request. Called from the scheduler thread.
class Responder {
public:
	virtual ~Responder() = default;
	// Parsed pieces in order. Returning false cancels the generation.
	virtual bool on_event( const OutputEvent& ev ) = 0;
	virtual void on_finish( engine::runtime::FinishReason reason, const std::string& matched_stop, bool had_tool_calls, const Usage& usage ) = 0;
};

// Renders the prompt, applies server defaults and starts the generation. On failure fills err and sends nothing.
bool start_chat( engine::runtime::Generator& gen, const ServerConfig& cfg, ChatRequest req, uint64_t request_id,
	std::shared_ptr<Responder> responder, const ConnPtr& conn, ApiError* err );

// Protocol front ends (openai.cpp, anthropic.cpp)
class ChatProtocol {
public:
	virtual ~ChatProtocol() = default;
	virtual bool parse( const std::string& body, ChatRequest* out, ApiError* err ) = 0;
	virtual std::shared_ptr<Responder> make_responder( const ChatRequest& req, const ConnPtr& conn, uint64_t request_id ) = 0;
	virtual bool anthropic_errors() const { return false; }
};

std::unique_ptr<ChatProtocol> make_openai_protocol();
std::unique_ptr<ChatProtocol> make_anthropic_protocol();

// Shared by the protocol parsers: JSON array text of the OpenAI "tools" value normalized to {"type","function"} form
// (bare functions and Anthropic {name, description, input_schema} are accepted). Empty on failure.
std::string normalize_tools( const std::string& tools_json, bool* ok );

} // namespace engine::server
