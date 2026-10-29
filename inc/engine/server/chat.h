#pragma once

#include "engine/runtime/generator.h"
#include "engine/server/config.h"
#include "engine/server/http.h"
#include "engine/server/output_parser.h"

#include <atomic>
#include <memory>
#include <cmath>
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
	std::vector<std::string> images; // encoded image files in prompt order; messages_json refers to them as {"type": "image"} parts
	bool stream = false;
	bool include_usage = false;  // OpenAI stream_options.include_usage
	int max_tokens = 0;          // 0: server default
	float temperature = -1.0f;   // <0: server default
	float top_p = -1.0f;         // <0: server default
	int top_k = -1;              // <0: server default
	uint64_t seed = 0;
	std::vector<std::string> stop;
	int enable_thinking = -1;    // -1: server default
	std::string reasoning_effort;
	// tool_choice "required" / a named function: the model is told to call a tool (best effort, no constrained decoding)
	int thinking_budget = 0;     // reasoning tokens before the thinking block is closed for the model (0 = unlimited)
	bool require_tool = false;
	std::string force_tool;
	// runtime steering (needs the steer option): control-vector file name ("" = the server's choice, "none" = unsteered), strength and scope
	// (<0 / empty = the server's), and the logit taken off reflection markers inside the reasoning block (<0 = the server's)
	std::string steer_vector;
	float steer_scale = -1.0f;
	std::string steer_scope;
	float probe_threshold = NAN, probe_edit_strength = NAN, probe_hedge_strength = NAN, probe_push = NAN;   // per-request values of the probe options (NaN = the option)
	std::string probe_edit_layers;   // per-request steer_edit_layers (empty = the option)
	bool steer_probe = false;   // the steering probe reads the prompt first and applies its direction clamps when the gate fires
	float think_penalty = -1.0f;
};

struct Usage {
	int prompt_tokens = 0;
	int cached_tokens = 0;   // of prompt_tokens: resumed from the prefix cache
	int completion_tokens = 0;
	double ttft_ms = 0.0;
	double decode_tok_s = 0.0;
	float gate_score = -1.0f;   // steering probe: the look-ahead score (-1 = no probe ran)
	bool gate_fired = false;
};

// Writes one protocol's response for one request. Called from the scheduler thread.
class Responder {
public:
	virtual ~Responder() = default;
	// Parsed pieces in order. Returning false cancels the generation.
	virtual void on_prompt( int prompt_tokens ) { (void) prompt_tokens; }
	virtual bool on_event( const OutputEvent& ev ) = 0;
	virtual void on_finish( engine::runtime::FinishReason reason, const std::string& matched_stop, bool had_tool_calls, const Usage& usage ) = 0;
};

// Renders the prompt, applies server defaults and starts the generation. On failure fills err and sends nothing.
bool start_chat( engine::runtime::Generator& gen, const ServerConfig& cfg, ChatRequest req, uint64_t request_id,
	std::shared_ptr<Responder> responder, const ConnPtr& conn, ApiError* err );

// Prompt length of a request without generating (count_tokens).
bool count_chat_tokens( engine::runtime::Generator& gen, const ServerConfig& cfg, const ChatRequest& req, int* tokens, ApiError* err );

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
