#include "engine/server/chat.h"

#define JSON_NOEXCEPTION 1
#include <nlohmann/json.hpp>

namespace engine::server {

namespace {

struct ChatState {
	OutputParser parser;
	std::shared_ptr<Responder> responder;
	ConnPtr conn;
	ChatState( OutputParser p, std::shared_ptr<Responder> r, ConnPtr c ) : parser( std::move( p ) ), responder( std::move( r ) ), conn( std::move( c ) ) {}
};

bool ends_with( const std::string& s, const char* suffix ) {
	const size_t n = std::char_traits<char>::length( suffix );
	return s.size() >= n && s.compare( s.size() - n, n, suffix ) == 0;
}

// Maps the API-level effort names onto what each template accepts; empty leaves the template default.
std::string effort_for( engine::loader::ModelFamily fam, const std::string& e ) {
	if ( e.empty() ) return e;
	if ( fam == engine::loader::ModelFamily::GLM ) {
		if ( e == "minimal" || e == "low" ) return "low";
		if ( e == "medium" || e == "high" ) return "high";
		return std::string(); // xhigh / max: the template default
	}
	if ( e == "minimal" || e == "low" ) return "low";
	if ( e == "medium" ) return "medium";
	return "xhigh";
}

} // namespace

std::string normalize_tools( const std::string& tools_json, bool* ok ) {
	*ok = false;
	auto j = nlohmann::ordered_json::parse( tools_json, nullptr, false );
	if ( j.is_discarded() || !j.is_array() ) return std::string();
	nlohmann::ordered_json out = nlohmann::ordered_json::array();
	for ( const auto& t : j ) {
		if ( !t.is_object() ) return std::string();
		nlohmann::ordered_json fn;
		if ( t.contains( "function" ) && t["function"].is_object() ) {
			fn = t["function"];
		}
		else if ( t.contains( "name" ) ) {
			fn = nlohmann::ordered_json::object();
			fn["name"] = t["name"];
			if ( t.contains( "description" ) ) fn["description"] = t["description"];
			if ( t.contains( "input_schema" ) ) fn["parameters"] = t["input_schema"];
			else if ( t.contains( "parameters" ) ) fn["parameters"] = t["parameters"];
		}
		else {
			return std::string();
		}
		if ( !fn.contains( "name" ) || !fn["name"].is_string() ) return std::string();
		nlohmann::ordered_json tool = nlohmann::ordered_json::object();
		tool["type"] = "function";
		tool["function"] = fn;
		out.push_back( tool );
	}
	*ok = true;
	return out.dump();
}

bool start_chat( engine::runtime::Generator& gen, const ServerConfig& cfg, ChatRequest req, uint64_t request_id,
	std::shared_ptr<Responder> responder, const ConnPtr& conn, ApiError* err ) {
	const auto family = gen.chat_template().model_family();

	bool thinking = req.enable_thinking >= 0 ? req.enable_thinking != 0 : cfg.default_enable_thinking;
	const std::string effort = effort_for( family, req.reasoning_effort );
	if ( req.enable_thinking < 0 && !req.reasoning_effort.empty() ) thinking = true;

	nlohmann::ordered_json vars;
	vars["messages"] = nlohmann::ordered_json::parse( req.messages_json, nullptr, false );
	if ( vars["messages"].is_discarded() ) {
		err->message = "messages could not be normalized";
		return false;
	}
	if ( !req.tools_json.empty() ) vars["tools"] = nlohmann::ordered_json::parse( req.tools_json, nullptr, false );
	vars["add_generation_prompt"] = true;
	vars["enable_thinking"] = thinking;
	if ( !effort.empty() ) vars["reasoning_effort"] = effort;

	std::string prompt, terr;
	if ( !gen.chat_template().render_json( vars.dump(), &prompt, &terr ) ) {
		err->message = "the chat template rejected this conversation: " + terr;
		return false;
	}
	std::vector<int> tokens = gen.tokenizer().encode( prompt, false );
	const int ctx = gen.max_context();
	if ( tokens.empty() || static_cast<int>( tokens.size() ) + 1 >= ctx ) {
		err->message = "this conversation is " + std::to_string( tokens.size() ) + " tokens; the context window is " + std::to_string( ctx );
		err->type = "context_length_exceeded";
		return false;
	}

	const bool in_reasoning = ends_with( prompt, "<think>\n" ) || ends_with( prompt, "<think>" );
	auto state = std::make_shared<ChatState>(
		OutputParser( family, in_reasoning, tool_schemas_from_json( req.tools_json ) ), std::move( responder ), conn );

	engine::runtime::GenerationRequest g;
	g.req_id = request_id;
	g.prompt_tokens = std::move( tokens );
	g.max_tokens = req.max_tokens > 0 ? req.max_tokens : cfg.default_max_tokens;
	g.temperature = req.temperature >= 0.0f ? req.temperature : cfg.default_temperature;
	g.top_p = req.top_p;
	g.top_k = req.top_k;
	g.seed = req.seed;
	g.stop_strings = req.stop;
	g.on_text = [state]( const std::string& delta ) {
		if ( state->conn->closed() ) return false;
		std::vector<OutputEvent> events;
		state->parser.feed( delta, events );
		for ( const OutputEvent& ev : events ) {
			if ( !state->responder->on_event( ev ) ) return false;
		}
		return true;
	};
	g.on_finish = [state]( engine::runtime::FinishReason reason, const std::string& matched, const engine::runtime::SlotMetrics& m ) {
		std::vector<OutputEvent> events;
		state->parser.finish( events );
		for ( const OutputEvent& ev : events ) state->responder->on_event( ev );
		Usage u;
		u.prompt_tokens = m.prompt_tokens;
		u.completion_tokens = m.generated_tokens;
		u.ttft_ms = std::chrono::duration<double, std::milli>( m.first_token_time - m.start_time ).count();
		u.decode_tok_s = m.decode_tok_s;
		state->responder->on_finish( reason, matched, state->parser.saw_tool_call(), u );
	};
	if ( !gen.submit_request( std::move( g ) ) ) {
		err->status = 503;
		err->type = "server_error";
		err->message = "the engine is not accepting requests";
		return false;
	}
	return true;
}

} // namespace engine::server
