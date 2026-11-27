#pragma once

#include "engine/runtime/generator.h"
#include "engine/server/config.h"
#include "engine/server/http.h"
#include "engine/server/output_parser.h"
#include "engine/loader/video.h"
#include "engine/loader/audio.h"

#include <nlohmann/json.hpp>
#include <atomic>
#include <climits>
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
	// tools_json entries the template's tool list leaves out (Anthropic defer_loading): the output parser knows them, and a tool_reference in a
	// tool result shows their definition there
	std::vector<std::string> deferred_tools;
	std::vector<std::string> images; // encoded image files in prompt order; messages_json refers to them as {"type": "image"} parts
	// Videos in prompt order, {"type": "video"} parts in messages_json: a video file, or its frames (encoded images, or the host's decoded
	// ones through libvolta) with their times in seconds
	struct Video {
		std::string file;
		std::vector<std::string> frames;
		std::vector<double> times;
		std::shared_ptr<const engine::loader::VideoFrames> decoded;
	};
	std::vector<Video> videos;
	// Audio clips in prompt order, {"type": "audio"} parts in messages_json: an audio file, or (libvolta) the host's mono samples at `rate`
	struct Audio {
		std::string file;
		std::shared_ptr<const std::vector<float>> pcm;
		int rate = 0;
	};
	std::vector<Audio> audios;
	// a text completion (OpenAI /v1/completions): the prompt as given instead of messages through the chat template (its special-token strings
	// are those tokens, nothing is prepended), and the reply as plain text (no reasoning or tool-call parsing)
	bool raw = false;
	std::string raw_prompt;
	std::vector<int> raw_tokens;     // the prompt as token ids instead of raw_prompt
	bool echo = false;               // the reply starts with the prompt's text
	bool stream = false;
	bool include_usage = false;  // OpenAI stream_options.include_usage
	int max_tokens = 0;          // 0: server default
	float temperature = -1.0f;   // <0: server default
	float top_p = -1.0f;         // <0: server default
	int top_k = -1;              // <0: server default
	uint64_t seed = 0;
	float presence_penalty = -1000.0f;    // -1000: server default
	float frequency_penalty = 0.0f;
	float repetition_penalty = -1.0f;     // <0: server default
	int logprobs = -1;                    // >= 0: report token log-probabilities with this many alternatives
	int priority = INT_MIN;               // INT_MIN: from the model name (low_priority_models), else 0
	std::vector<std::string> probes;      // probe files scored on the request (empty: the probe option's)
	std::vector<std::string> stop;
	int stop_on_loop = -1;       // extension field stop_on_loop: -1 the loop_detect option, 0 off, 1 stop on a detected loop
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
	float probe_threshold = NAN, probe_edit_strength = NAN, probe_hedge_strength = NAN, probe_push = NAN, probe_directions = NAN, probe_keep_norm = NAN;   // per-request values of the probe options (NaN = the option)
	std::string probe_edit_layers;   // per-request steer_edit_layers (empty = the option)
	bool steer_probe = false;   // the steering probe reads the prompt first and applies its direction clamps when the gate fires
	float think_penalty = -1.0f;
	// Anthropic messages: a prompt or a prompt + max_tokens that does not fit the context window is refused with the API's own errors
	// ("prompt is too long: N tokens > W maximum", "input length and `max_tokens` exceed context limit: N + M > W"), which clients such as
	// Claude Code parse (compacting, or retrying with a smaller max_tokens); otherwise the reply is cut at the window (finish reason length)
	bool strict_context = false;
};

struct Usage {
	int prompt_tokens = 0;
	int cached_tokens = 0;   // of prompt_tokens: resumed from the prefix cache
	int completion_tokens = 0;
	double ttft_ms = 0.0;
	double decode_tok_s = 0.0;
	float gate_score = -1.0f;   // steering probe: the look-ahead score (-1 = no probe ran)
	bool gate_fired = false;
	std::vector<engine::runtime::ProbeSummary> probes;
};

// The usage entry of a request's probes: [{ name, prompt, max, mean, last, above, tokens }] (scores without data are null); empty array when none
nlohmann::json probes_json( const std::vector<engine::runtime::ProbeSummary>& probes );

// A generated token's log-probability and its top alternatives, as text
struct LogprobOut {
	std::string token;
	float logprob = 0.0f;
	std::vector<std::pair<std::string, float>> top;
};

// Writes one protocol's response for one request. Called from the scheduler thread.
class Responder {
public:
	virtual ~Responder() = default;
	// Parsed pieces in order. Returning false cancels the generation.
	virtual void on_prompt( int prompt_tokens ) { (void) prompt_tokens; }
	// A text completion's prompt as text (decoded when it came as token ids), before generation starts
	virtual void on_raw_prompt( const std::string& text ) { (void) text; }
	// When the request takes its slot: the prompt tokens read from the prefix cache (before the first on_event; on_wait may come earlier)
	virtual void on_admit( int cached_tokens ) { (void) cached_tokens; }
	virtual bool on_event( const OutputEvent& ev ) = 0;
	// With ChatRequest::logprobs: the entries of the tokens whose text the next on_event calls carry
	virtual void on_logprobs( const std::vector<LogprobOut>& entries ) { (void) entries; }
	// While queued or prefilling (every few seconds): a streaming response starts and sends keepalives. False when the client is gone.
	virtual bool on_wait() { return true; }
	virtual void on_finish( engine::runtime::FinishReason reason, const std::string& matched_stop, bool had_tool_calls, const Usage& usage ) = 0;
};

// Renders the prompt, applies server defaults and starts the generation. On failure fills err and sends nothing. conn may be null (the library:
// the responder alone decides when to stop)
bool start_chat( engine::runtime::Generator& gen, const ServerConfig& cfg, ChatRequest req, uint64_t request_id,
	std::shared_ptr<Responder> responder, const ConnPtr& conn, ApiError* err );

// Prompt length of a request without generating (count_tokens); with dump_requests its rendered prompt is written as a generation's is
bool count_chat_tokens( engine::runtime::Generator& gen, const ServerConfig& cfg, const ChatRequest& req, uint64_t request_id, int* tokens,
	ApiError* err );

// What rendering a request reads from the model: its chat template, tokenizer, vision tower (nullptr: an image becomes a note in the
// text) and context window. The server takes them from its generator; tools without weights load the template and tokenizer alone
struct PromptModel {
	const engine::loader::ChatTemplate& chat_template;
	const engine::loader::Tokenizer& tokenizer;
	const engine::loader::VisionParams* vision = nullptr;
	int max_context = 0;
	std::function<void( const engine::runtime::GenerationRequest::Image& )> prefetch_image;   // vision_cpu: encode before the request queues
	const engine::loader::AudioParams* audio = nullptr;   // the audio tower's feature extractor (nullptr: an audio part becomes a note)
};

// The prompt a request renders to, as the server feeds it to the model (prompt text with special tokens escaped, and its token ids);
// vars_json (optional) receives the variables the chat template renders (messages, tools and options, as JSON)
bool render_chat_prompt( const PromptModel& model, const ServerConfig& cfg, const ChatRequest& req, std::string* prompt, std::vector<int>* tokens,
	ApiError* err, std::string* vars_json = nullptr );

// The prompt text with its special tokens as their text (what dump_requests writes)
std::string shown_prompt( const engine::loader::Tokenizer& tok, const std::string& prompt );

// Protocol front ends (openai.cpp, anthropic.cpp)
class ChatProtocol {
public:
	virtual ~ChatProtocol() = default;
	virtual bool parse( const std::string& body, ChatRequest* out, ApiError* err ) = 0;
	virtual std::shared_ptr<Responder> make_responder( const ChatRequest& req, const ConnPtr& conn, uint64_t request_id ) = 0;
	virtual bool anthropic_errors() const { return false; }
};

std::unique_ptr<ChatProtocol> make_openai_protocol();
std::unique_ptr<ChatProtocol> make_openai_completion_protocol();   // /v1/completions
std::unique_ptr<ChatProtocol> make_anthropic_protocol();

// Shared by the protocol parsers: JSON array text of the OpenAI "tools" value normalized to {"type","function"} form
// (bare functions and Anthropic {name, description, input_schema} are accepted). Empty on failure.
std::string normalize_tools( const std::string& tools_json, bool* ok );

} // namespace engine::server
