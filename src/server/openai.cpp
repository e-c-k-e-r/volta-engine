#include "engine/server/chat.h"
#include "engine/loader/image.h"

#define JSON_NOEXCEPTION 1
#include <nlohmann/json.hpp>

#include <ctime>

namespace engine::server {

namespace {

using json = nlohmann::ordered_json;

bool fail( ApiError* err, const std::string& message, const char* type = "invalid_request_error" ) {
	err->status = 400;
	err->type = type;
	err->message = message;
	return false;
}

// OpenAI content (string | null | [parts]) -> template content (string | [text parts])
bool normalize_content( const json& c, json* out, std::vector<std::string>* images, ApiError* err ) {
	if ( c.is_null() ) {
		*out = "";
		return true;
	}
	if ( c.is_string() ) {
		*out = c;
		return true;
	}
	if ( !c.is_array() ) return fail( err, "message content must be a string or an array of content parts" );
	json parts = json::array();
	for ( const auto& p : c ) {
		if ( p.is_string() ) {
			parts.push_back( { { "type", "text" }, { "text", p } } );
		}
		else if ( p.is_object() && p.contains( "type" ) && p["type"] == "text" && p.contains( "text" ) && p["text"].is_string() ) {
			parts.push_back( { { "type", "text" }, { "text", p["text"] } } );
		}
		else if ( p.is_object() && p.value( "type", std::string() ) == "image_url" ) {
			const json& iu = p.contains( "image_url" ) ? p["image_url"] : json();
			const std::string url = iu.is_string() ? iu.get<std::string>() : ( iu.is_object() ? iu.value( "url", std::string() ) : std::string() );
			const size_t comma = url.find( ',' );
			std::string bytes;
			if ( url.compare( 0, 5, "data:" ) != 0 || comma == std::string::npos || url.find( ";base64" ) > comma ) {
				return fail( err, "image_url must be a base64 data: URL (remote URLs are not fetched)" );
			}
			if ( !engine::loader::base64_decode( url.substr( comma + 1 ), &bytes ) || bytes.empty() ) return fail( err, "image_url holds invalid base64" );
			images->push_back( std::move( bytes ) );
			parts.push_back( { { "type", "image" } } );
		}
		else if ( p.is_object() && p.contains( "type" ) && p["type"].is_string() ) {
			return fail( err, "unsupported content part type '" + p["type"].get<std::string>() + "' (text and image_url are supported)" );
		}
		else {
			return fail( err, "malformed content part" );
		}
	}
	*out = parts;
	return true;
}

bool normalize_messages( const json& messages, json* out, std::vector<std::string>* images, ApiError* err ) {
	if ( !messages.is_array() || messages.empty() ) return fail( err, "messages must be a non-empty array" );
	*out = json::array();
	for ( const auto& m : messages ) {
		if ( !m.is_object() || !m.contains( "role" ) || !m["role"].is_string() ) return fail( err, "every message needs a role" );
		std::string role = m["role"].get<std::string>();
		if ( role == "developer" ) role = "system";
		if ( role != "system" && role != "user" && role != "assistant" && role != "tool" ) return fail( err, "unsupported message role '" + role + "'" );
		json nm = json::object();
		nm["role"] = role;
		json content;
		if ( !normalize_content( m.contains( "content" ) ? m["content"] : json(), &content, images, err ) ) return false;
		nm["content"] = content;
		if ( role == "assistant" ) {
			for ( const char* key : { "reasoning_content", "reasoning" } ) {
				if ( m.contains( key ) && m[key].is_string() ) {
					nm["reasoning_content"] = m[key];
					break;
				}
			}
			if ( m.contains( "tool_calls" ) && m["tool_calls"].is_array() && !m["tool_calls"].empty() ) {
				json calls = json::array();
				for ( const auto& tc : m["tool_calls"] ) {
					if ( !tc.is_object() || !tc.contains( "function" ) || !tc["function"].is_object() || !tc["function"].contains( "name" ) ) {
						return fail( err, "malformed tool_calls entry" );
					}
					const json& f = tc["function"];
					json args = json::object();
					if ( f.contains( "arguments" ) ) {
						if ( f["arguments"].is_string() ) {
							const std::string text = f["arguments"].get<std::string>();
							if ( !text.empty() ) {
								args = json::parse( text, nullptr, false );
								if ( args.is_discarded() || !args.is_object() ) return fail( err, "tool call arguments must be a JSON object" );
							}
						}
						else if ( f["arguments"].is_object() ) {
							args = f["arguments"];
						}
						else {
							return fail( err, "tool call arguments must be a JSON object" );
						}
					}
					json call = json::object();
					if ( tc.contains( "id" ) ) call["id"] = tc["id"];
					call["type"] = "function";
					call["function"] = { { "name", f["name"] }, { "arguments", args } };
					calls.push_back( call );
				}
				nm["tool_calls"] = calls;
			}
		}
		else if ( role == "tool" && m.contains( "tool_call_id" ) ) {
			nm["tool_call_id"] = m["tool_call_id"];
		}
		out->push_back( nm );
	}
	return true;
}

class OpenAIResponder : public Responder {
public:
	OpenAIResponder( const ChatRequest& req, ConnPtr conn, uint64_t id )
		: conn_( std::move( conn ) ), stream_( req.stream ), include_usage_( req.include_usage ), model_( req.model ), id_( id ),
		  created_( static_cast<int64_t>( std::time( nullptr ) ) ) {
		completion_id_ = "chatcmpl-" + std::to_string( id );
	}

	bool on_event( const OutputEvent& ev ) override {
		if ( ev.type == OutputEvent::Type::ToolCall ) {
			const size_t index = calls_.size();
			calls_.push_back( ev.call );
			if ( !stream_ ) return true;
			json tc = { { "index", index }, { "id", "call_" + std::to_string( id_ ) + "_" + std::to_string( index ) }, { "type", "function" },
				{ "function", { { "name", ev.call.name }, { "arguments", ev.call.arguments } } } };
			return send_delta( { { "tool_calls", json::array( { tc } ) } } );
		}
		const bool reasoning = ev.type == OutputEvent::Type::Reasoning;
		if ( !stream_ ) {
			( reasoning ? reasoning_ : content_ ) += ev.text;
			return true;
		}
		return send_delta( { { reasoning ? "reasoning_content" : "content", ev.text } } );
	}

	void on_finish( engine::runtime::FinishReason reason, const std::string&, bool had_tool_calls, const Usage& usage ) override {
		using engine::runtime::FinishReason;
		const char* finish = had_tool_calls ? "tool_calls" : ( reason == FinishReason::Length ? "length" : "stop" );
		if ( reason == FinishReason::Cancelled || conn_->closed() ) {
			conn_->close_now();
			return;
		}
		if ( stream_ ) {
			if ( !ensure_started() ) return;
			json last = chunk();
			last["choices"] = json::array( { { { "index", 0 }, { "delta", json::object() }, { "finish_reason", finish } } } );
			if ( !sse( last ) ) return;
			if ( include_usage_ ) {
				json u = chunk();
				u["choices"] = json::array();
				u["usage"] = usage_json( usage );
				if ( !sse( u ) ) return;
			}
			conn_->write_all( "data: [DONE]\n\n" );
			conn_->finish();
			return;
		}
		json msg = { { "role", "assistant" }, { "content", content_.empty() && !calls_.empty() ? json( nullptr ) : json( content_ ) } };
		if ( !reasoning_.empty() ) msg["reasoning_content"] = reasoning_;
		if ( !calls_.empty() ) {
			json tcs = json::array();
			for ( size_t k = 0; k < calls_.size(); ++k ) {
				tcs.push_back( { { "id", "call_" + std::to_string( id_ ) + "_" + std::to_string( k ) }, { "type", "function" },
					{ "function", { { "name", calls_[k].name }, { "arguments", calls_[k].arguments } } } } );
			}
			msg["tool_calls"] = tcs;
		}
		json resp = { { "id", completion_id_ }, { "object", "chat.completion" }, { "created", created_ }, { "model", model_ },
			{ "choices", json::array( { { { "index", 0 }, { "message", msg }, { "finish_reason", finish } } } ) }, { "usage", usage_json( usage ) } };
		send_json( conn_, 200, resp.dump() );
	}

private:
	json chunk() const {
		return { { "id", completion_id_ }, { "object", "chat.completion.chunk" }, { "created", created_ }, { "model", model_ } };
	}

	static json usage_json( const Usage& u ) {
		return { { "prompt_tokens", u.prompt_tokens }, { "completion_tokens", u.completion_tokens }, { "total_tokens", u.prompt_tokens + u.completion_tokens },
			{ "ttft_ms", u.ttft_ms }, { "decode_tok_s", u.decode_tok_s } };
	}

	bool sse( const json& j ) { return conn_->write_all( "data: " + j.dump() + "\n\n" ); }

	bool ensure_started() {
		if ( started_ ) return true;
		started_ = true;
		if ( !start_sse( conn_ ) ) return false;
		json first = chunk();
		first["choices"] = json::array( { { { "index", 0 }, { "delta", { { "role", "assistant" } } }, { "finish_reason", nullptr } } } );
		return sse( first );
	}

	bool send_delta( const json& delta ) {
		if ( !ensure_started() ) return false;
		json c = chunk();
		c["choices"] = json::array( { { { "index", 0 }, { "delta", delta }, { "finish_reason", nullptr } } } );
		return sse( c );
	}

	ConnPtr conn_;
	bool stream_, include_usage_;
	std::string model_, completion_id_;
	uint64_t id_;
	int64_t created_;
	bool started_ = false;
	std::string content_, reasoning_;
	std::vector<ToolCallOut> calls_;
};

class OpenAIProtocol : public ChatProtocol {
public:
	bool parse( const std::string& body, ChatRequest* out, ApiError* err ) override {
		json j = json::parse( body, nullptr, false );
		if ( j.is_discarded() || !j.is_object() ) return fail( err, "the request body is not valid JSON" );
		ChatRequest r;
		r.model = j.value( "model", std::string() );
		if ( !j.contains( "messages" ) ) return fail( err, "messages is required" );
		json messages;
		if ( !normalize_messages( j["messages"], &messages, &r.images, err ) ) return false;
		r.messages_json = messages.dump();

		bool tools_off = j.contains( "tool_choice" ) && j["tool_choice"].is_string() && j["tool_choice"] == "none";
		if ( j.contains( "tools" ) && !j["tools"].is_null() && !tools_off ) {
			bool ok = false;
			std::string tools = normalize_tools( j["tools"].dump(), &ok );
			if ( !ok ) return fail( err, "malformed tools" );
			if ( j["tools"].is_array() && !j["tools"].empty() ) r.tools_json = tools;
		}

		if ( j.contains( "tool_choice" ) ) {
			const json& tc = j["tool_choice"];
			if ( tc.is_string() && tc == "required" ) r.require_tool = true;
			else if ( tc.is_object() && tc.contains( "function" ) && tc["function"].is_object() ) r.force_tool = tc["function"].value( "name", std::string() );
		}
		r.stream = j.value( "stream", false );
		if ( j.contains( "stream_options" ) && j["stream_options"].is_object() ) r.include_usage = j["stream_options"].value( "include_usage", false );
		for ( const char* key : { "max_completion_tokens", "max_tokens" } ) {
			if ( j.contains( key ) && j[key].is_number_integer() ) {
				r.max_tokens = j[key].get<int>();
				if ( r.max_tokens <= 0 ) return fail( err, std::string( key ) + " must be positive" );
				break;
			}
		}
		if ( j.contains( "temperature" ) && j["temperature"].is_number() ) {
			r.temperature = j["temperature"].get<float>();
			if ( r.temperature < 0.0f || r.temperature > 2.0f ) return fail( err, "temperature must be between 0 and 2" );
		}
		if ( j.contains( "top_p" ) && j["top_p"].is_number() ) r.top_p = j["top_p"].get<float>();
		if ( j.contains( "top_k" ) && j["top_k"].is_number_integer() ) r.top_k = j["top_k"].get<int>();
		if ( j.contains( "seed" ) && j["seed"].is_number_integer() ) r.seed = j["seed"].get<uint64_t>();
		if ( j.contains( "stop" ) ) {
			if ( j["stop"].is_string() ) r.stop.push_back( j["stop"].get<std::string>() );
			else if ( j["stop"].is_array() ) {
				for ( const auto& s : j["stop"] ) {
					if ( !s.is_string() ) return fail( err, "stop must be a string or an array of strings" );
					r.stop.push_back( s.get<std::string>() );
				}
			}
			else if ( !j["stop"].is_null() ) return fail( err, "stop must be a string or an array of strings" );
		}
		if ( j.contains( "reasoning_effort" ) && j["reasoning_effort"].is_string() ) r.reasoning_effort = j["reasoning_effort"].get<std::string>();
		for ( const char* key : { "thinking_budget_tokens", "max_reasoning_tokens" } ) {
			if ( j.contains( key ) && j[key].is_number_integer() ) r.thinking_budget = j[key].get<int>();
		}
		if ( j.contains( "enable_thinking" ) && j["enable_thinking"].is_boolean() ) r.enable_thinking = j["enable_thinking"].get<bool>() ? 1 : 0;
		if ( j.contains( "chat_template_kwargs" ) && j["chat_template_kwargs"].is_object() ) {
			const json& k = j["chat_template_kwargs"];
			if ( k.contains( "enable_thinking" ) && k["enable_thinking"].is_boolean() ) r.enable_thinking = k["enable_thinking"].get<bool>() ? 1 : 0;
		}
		*out = std::move( r );
		return true;
	}

	std::shared_ptr<Responder> make_responder( const ChatRequest& req, const ConnPtr& conn, uint64_t request_id ) override {
		return std::make_shared<OpenAIResponder>( req, conn, request_id );
	}
};

} // namespace

std::unique_ptr<ChatProtocol> make_openai_protocol() { return std::make_unique<OpenAIProtocol>(); }

} // namespace engine::server
