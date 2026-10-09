#include "engine/server/chat.h"
#include "engine/loader/image.h"

#define JSON_NOEXCEPTION 1
#include <nlohmann/json.hpp>

namespace engine::server {

namespace {

using json = nlohmann::ordered_json;

bool fail( ApiError* err, const std::string& message, const char* type = "invalid_request_error" ) {
	err->status = 400;
	err->type = type;
	err->message = message;
	return false;
}

// Concatenated text of a string-or-blocks value (system, tool_result content)
bool text_of( const json& v, std::string* out, ApiError* err ) {
	if ( v.is_null() ) return true;
	if ( v.is_string() ) {
		*out += v.get<std::string>();
		return true;
	}
	if ( !v.is_array() ) return fail( err, "expected a string or an array of text blocks" );
	for ( const auto& b : v ) {
		if ( b.is_string() ) *out += b.get<std::string>();
		else if ( b.is_object() && b.value( "type", std::string() ) == "text" && b.contains( "text" ) && b["text"].is_string() ) *out += b["text"].get<std::string>();
		else if ( b.is_object() && b.contains( "type" ) && b["type"].is_string() ) return fail( err, "unsupported content block type '" + b["type"].get<std::string>() + "' (only text is supported)" );
		else return fail( err, "malformed content block" );
	}
	return true;
}

// An Anthropic image block ({type: image, source: {type: base64, media_type, data}}) -> template part; the bytes go to `images`
bool image_part( const json& b, json* part, std::vector<std::string>* images, ApiError* err ) {
	const json src = b.contains( "source" ) && b["source"].is_object() ? b["source"] : json::object();
	std::string bytes;
	if ( src.value( "type", std::string() ) != "base64" || !src.contains( "data" ) || !src["data"].is_string() ) {
		return fail( err, "image source must be {type: base64, media_type, data} (remote URLs are not fetched)" );
	}
	if ( !engine::loader::base64_decode( src["data"].get<std::string>(), &bytes ) || bytes.empty() ) return fail( err, "image source holds invalid base64" );
	images->push_back( std::move( bytes ) );
	*part = { { "type", "image" } };
	return true;
}

// tool_result content: a string, or text and image blocks (a string when there are no images)
bool tool_content( const json& v, json* out, std::vector<std::string>* images, ApiError* err ) {
	bool has_image = false;
	if ( v.is_array() ) {
		for ( const auto& b : v ) has_image = has_image || ( b.is_object() && b.value( "type", std::string() ) == "image" );
	}
	if ( !has_image ) {
		std::string text;
		if ( !text_of( v, &text, err ) ) return false;
		*out = text;
		return true;
	}
	json parts = json::array();
	for ( const auto& b : v ) {
		const std::string type = b.is_object() ? b.value( "type", std::string() ) : std::string( b.is_string() ? "text" : "" );
		if ( type == "text" ) parts.push_back( { { "type", "text" }, { "text", b.is_string() ? b.get<std::string>() : b.value( "text", std::string() ) } } );
		else if ( type == "image" ) {
			json part;
			if ( !image_part( b, &part, images, err ) ) return false;
			parts.push_back( part );
		}
		else return fail( err, "unsupported content block type '" + type + "' in a tool result" );
	}
	*out = parts;
	return true;
}

bool normalize_messages( const json& j, json* out, std::vector<std::string>* images, ApiError* err ) {
	*out = json::array();
	if ( j.contains( "system" ) && !j["system"].is_null() ) {
		std::string sys;
		if ( !text_of( j["system"], &sys, err ) ) return false;
		if ( !sys.empty() ) out->push_back( { { "role", "system" }, { "content", sys } } );
	}
	if ( !j.contains( "messages" ) || !j["messages"].is_array() || j["messages"].empty() ) return fail( err, "messages must be a non-empty array" );
	for ( const auto& m : j["messages"] ) {
		if ( !m.is_object() || !m.contains( "role" ) || !m["role"].is_string() ) return fail( err, "every message needs a role" );
		const std::string role = m["role"].get<std::string>();
		if ( role != "user" && role != "assistant" && role != "system" ) return fail( err, "message role must be 'user' or 'assistant'" );
		const json content = m.contains( "content" ) ? m["content"] : json( "" );
		if ( role == "system" ) {
			// Claude Code sends system content inside messages as well; start_chat folds it into the leading system prompt
			std::string text;
			if ( !text_of( content, &text, err ) ) return false;
			out->push_back( { { "role", "system" }, { "content", text } } );
			continue;
		}
		if ( content.is_string() ) {
			out->push_back( { { "role", role }, { "content", content } } );
			continue;
		}
		if ( !content.is_array() ) return fail( err, "message content must be a string or an array of content blocks" );
		if ( role == "user" ) {
			std::string pending;
			bool has_pending = false;
			json parts = json::array();
			bool has_image = false;
			auto flush_text = [&]() {
				if ( has_pending ) parts.push_back( { { "type", "text" }, { "text", pending } } );
				pending.clear();
				has_pending = false;
			};
			// Text-only messages keep the plain string form; a message with images carries its parts in order
			auto flush = [&]() {
				flush_text();
				if ( !parts.empty() ) {
					if ( has_image ) out->push_back( { { "role", "user" }, { "content", parts } } );
					else out->push_back( { { "role", "user" }, { "content", parts[0]["text"] } } );
				}
				parts = json::array();
				has_image = false;
			};
			for ( const auto& b : content ) {
				const std::string type = b.is_object() ? b.value( "type", std::string() ) : std::string( b.is_string() ? "text" : "" );
				if ( type == "text" ) {
					pending += b.is_string() ? b.get<std::string>() : b.value( "text", std::string() );
					has_pending = true;
				}
				else if ( type == "image" ) {
					flush_text();
					json part;
					if ( !image_part( b, &part, images, err ) ) return false;
					parts.push_back( part );
					has_image = true;
				}
				else if ( type == "tool_result" ) {
					flush();
					json tc;
					if ( !tool_content( b.contains( "content" ) ? b["content"] : json(), &tc, images, err ) ) return false;
					if ( b.value( "is_error", false ) ) {
						if ( tc.is_string() && !tc.get<std::string>().empty() ) tc = "Error: " + tc.get<std::string>();
						else if ( tc.is_array() ) tc.insert( tc.begin(), { { "type", "text" }, { "text", "Error:" } } );
					}
					out->push_back( { { "role", "tool" }, { "tool_call_id", b.value( "tool_use_id", std::string() ) }, { "content", tc } } );
				}
				else {
					return fail( err, "unsupported content block type '" + type + "' (text, image and tool_result are supported)" );
				}
			}
			flush();
		}
		else {
			std::string text, thinking;
			bool has_thinking = false;
			json calls = json::array();
			for ( const auto& b : content ) {
				const std::string type = b.is_object() ? b.value( "type", std::string() ) : std::string( b.is_string() ? "text" : "" );
				if ( type == "text" ) {
					text += b.is_string() ? b.get<std::string>() : b.value( "text", std::string() );
				}
				else if ( type == "thinking" ) {
					thinking += b.value( "thinking", std::string() );
					has_thinking = true;
				}
				else if ( type == "redacted_thinking" ) {
					// opaque to us: nothing to render
				}
				else if ( type == "tool_use" ) {
					json input = b.contains( "input" ) && b["input"].is_object() ? b["input"] : json::object();
					json call = json::object();
					call["id"] = b.value( "id", std::string() );
					call["type"] = "function";
					call["function"] = { { "name", b.value( "name", std::string() ) }, { "arguments", input } };
					calls.push_back( call );
				}
				else {
					return fail( err, "unsupported content block type '" + type + "' in an assistant message" );
				}
			}
			json nm = { { "role", "assistant" }, { "content", text } };
			if ( has_thinking ) nm["reasoning_content"] = thinking;
			if ( !calls.empty() ) nm["tool_calls"] = calls;
			out->push_back( nm );
		}
	}
	return true;
}

class AnthropicResponder : public Responder {
public:
	AnthropicResponder( const ChatRequest& req, ConnPtr conn, uint64_t id )
		: conn_( std::move( conn ) ), stream_( req.stream ), model_( req.model ), id_( id ) {
		message_id_ = "msg_" + std::to_string( id );
	}

	void on_prompt( int n ) override { prompt_tokens_ = n; }

	bool on_event( const OutputEvent& ev ) override {
		const Kind kind = ev.type == OutputEvent::Type::Reasoning ? Kind::Thinking : ev.type == OutputEvent::Type::Content ? Kind::Text : Kind::Tool;
		if ( kind == Kind::Tool ) {
			const std::string tid = "toolu_" + std::to_string( id_ ) + "_" + std::to_string( tool_index_++ );
			if ( !stream_ ) {
				blocks_.push_back( { { "type", "tool_use" }, { "id", tid }, { "name", ev.call.name }, { "input", json::parse( ev.call.arguments, nullptr, false ) } } );
				return true;
			}
			if ( !close_block() ) return false;
			if ( !open_block( { { "type", "tool_use" }, { "id", tid }, { "name", ev.call.name }, { "input", json::object() } } ) ) return false;
			if ( !delta( { { "type", "input_json_delta" }, { "partial_json", ev.call.arguments } } ) ) return false;
			return close_block();
		}
		if ( !stream_ ) {
			if ( blocks_.empty() || cur_ != kind ) {
				blocks_.push_back( kind == Kind::Thinking ? json( { { "type", "thinking" }, { "thinking", "" }, { "signature", "" } } ) : json( { { "type", "text" }, { "text", "" } } ) );
				cur_ = kind;
			}
			std::string& field = blocks_.back()[kind == Kind::Thinking ? "thinking" : "text"].get_ref<std::string&>();
			field += ev.text;
			return true;
		}
		if ( cur_ != kind || !block_open_ ) {
			if ( !close_block() ) return false;
			if ( !open_block( kind == Kind::Thinking ? json( { { "type", "thinking" }, { "thinking", "" } } ) : json( { { "type", "text" }, { "text", "" } } ) ) ) return false;
			cur_ = kind;
		}
		return delta( kind == Kind::Thinking ? json( { { "type", "thinking_delta" }, { "thinking", ev.text } } ) : json( { { "type", "text_delta" }, { "text", ev.text } } ) );
	}

	void on_finish( engine::runtime::FinishReason reason, const std::string& matched, bool had_tool_calls, const Usage& usage ) override {
		using engine::runtime::FinishReason;
		if ( reason == FinishReason::Cancelled || conn_->closed() ) {
			conn_->close_now();
			return;
		}
		const char* stop = had_tool_calls ? "tool_use" : reason == FinishReason::Length ? "max_tokens" : reason == FinishReason::StopString ? "stop_sequence" : "end_turn";
		json stop_seq = reason == FinishReason::StopString ? json( matched ) : json( nullptr );
		if ( stream_ ) {
			if ( !ensure_started() || !close_block() ) return;
			if ( !event( "message_delta", { { "type", "message_delta" }, { "delta", { { "stop_reason", stop }, { "stop_sequence", stop_seq } } },
				{ "usage", { { "output_tokens", usage.completion_tokens } } } } ) ) return;
			event( "message_stop", { { "type", "message_stop" } } );
			conn_->finish();
			return;
		}
		json msg = { { "id", message_id_ }, { "type", "message" }, { "role", "assistant" }, { "model", model_ }, { "content", blocks_.empty() ? json::array() : blocks_ },
			{ "stop_reason", stop }, { "stop_sequence", stop_seq },
			{ "usage", { { "input_tokens", usage.prompt_tokens }, { "output_tokens", usage.completion_tokens } } } };
		send_json( conn_, 200, msg.dump() );
	}

private:
	enum class Kind { None, Thinking, Text, Tool };

	bool event( const char* name, const json& data ) {
		return conn_->write_all( std::string( "event: " ) + name + "\ndata: " + data.dump() + "\n\n" );
	}

	bool ensure_started() {
		if ( started_ ) return true;
		started_ = true;
		if ( !start_sse( conn_ ) ) return false;
		return event( "message_start", { { "type", "message_start" }, { "message", { { "id", message_id_ }, { "type", "message" }, { "role", "assistant" }, { "model", model_ },
			{ "content", json::array() }, { "stop_reason", nullptr }, { "stop_sequence", nullptr },
			{ "usage", { { "input_tokens", prompt_tokens_ }, { "output_tokens", 1 } } } } } } );
	}

	bool open_block( const json& block ) {
		if ( !ensure_started() ) return false;
		block_open_ = true;
		return event( "content_block_start", { { "type", "content_block_start" }, { "index", block_index_ }, { "content_block", block } } );
	}

	bool delta( const json& d ) {
		return event( "content_block_delta", { { "type", "content_block_delta" }, { "index", block_index_ }, { "delta", d } } );
	}

	bool close_block() {
		if ( !block_open_ ) return true;
		block_open_ = false;
		if ( cur_ == Kind::Thinking ) {
			if ( !delta( { { "type", "signature_delta" }, { "signature", "" } } ) ) return false;
		}
		const bool ok = event( "content_block_stop", { { "type", "content_block_stop" }, { "index", block_index_ } } );
		++block_index_;
		cur_ = Kind::None;
		return ok;
	}

	ConnPtr conn_;
	bool stream_;
	std::string model_, message_id_;
	uint64_t id_;
	int prompt_tokens_ = 0;
	bool started_ = false, block_open_ = false;
	int block_index_ = 0;
	int tool_index_ = 0;
	Kind cur_ = Kind::None;
	json blocks_ = json::array();
};

class AnthropicProtocol : public ChatProtocol {
public:
	bool parse( const std::string& body, ChatRequest* out, ApiError* err ) override {
		json j = json::parse( body, nullptr, false );
		if ( j.is_discarded() || !j.is_object() ) return fail( err, "the request body is not valid JSON" );
		ChatRequest r;
		r.model = j.value( "model", std::string() );
		json messages;
		if ( !normalize_messages( j, &messages, &r.images, err ) ) return false;
		r.messages_json = messages.dump();

		bool tools_off = j.contains( "tool_choice" ) && j["tool_choice"].is_object() && j["tool_choice"].value( "type", std::string() ) == "none";
		if ( j.contains( "tools" ) && j["tools"].is_array() && !j["tools"].empty() && !tools_off ) {
			bool ok = false;
			std::string tools = normalize_tools( j["tools"].dump(), &ok );
			if ( !ok ) return fail( err, "malformed tools" );
			r.tools_json = tools;
		}
		if ( j.contains( "tool_choice" ) && j["tool_choice"].is_object() ) {
			const std::string t = j["tool_choice"].value( "type", std::string() );
			if ( t == "any" ) r.require_tool = true;
			else if ( t == "tool" ) r.force_tool = j["tool_choice"].value( "name", std::string() );
		}
		r.stream = j.value( "stream", false );
		if ( j.contains( "max_tokens" ) && j["max_tokens"].is_number_integer() ) {
			r.max_tokens = j["max_tokens"].get<int>();
			if ( r.max_tokens <= 0 ) return fail( err, "max_tokens must be positive" );
		}
		if ( j.contains( "temperature" ) && j["temperature"].is_number() ) {
			r.temperature = j["temperature"].get<float>();
			if ( r.temperature < 0.0f || r.temperature > 1.0f ) return fail( err, "temperature must be between 0 and 1" );
		}
		if ( j.contains( "top_p" ) && j["top_p"].is_number() ) r.top_p = j["top_p"].get<float>();
		if ( j.contains( "top_k" ) && j["top_k"].is_number_integer() ) r.top_k = j["top_k"].get<int>();
		if ( j.contains( "stop_sequences" ) && j["stop_sequences"].is_array() ) {
			for ( const auto& s : j["stop_sequences"] ) {
				if ( s.is_string() ) r.stop.push_back( s.get<std::string>() );
			}
		}
		if ( j.contains( "thinking" ) && j["thinking"].is_object() ) {
			const std::string t = j["thinking"].value( "type", std::string() );
			if ( t == "enabled" && j["thinking"].contains( "budget_tokens" ) && j["thinking"]["budget_tokens"].is_number_integer() ) r.thinking_budget = j["thinking"]["budget_tokens"].get<int>();
			if ( t == "enabled" || t == "adaptive" ) r.enable_thinking = 1;
			else if ( t == "disabled" ) r.enable_thinking = 0;
		}
		*out = std::move( r );
		return true;
	}

	std::shared_ptr<Responder> make_responder( const ChatRequest& req, const ConnPtr& conn, uint64_t request_id ) override {
		return std::make_shared<AnthropicResponder>( req, conn, request_id );
	}

	bool anthropic_errors() const override { return true; }
};

} // namespace

std::unique_ptr<ChatProtocol> make_anthropic_protocol() { return std::make_unique<AnthropicProtocol>(); }

} // namespace engine::server
