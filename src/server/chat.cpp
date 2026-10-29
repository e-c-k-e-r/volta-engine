#include "engine/server/chat.h"
#include "engine/runtime/options.h"

#include <algorithm>

#include <cstdio>

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
	if ( fam == engine::loader::ModelFamily::DEEPSEEK ) {
		// DeepSeek-V4.1 takes a 1-100 budget or low (50) / high (75) / max (100)
		if ( e == "minimal" || e == "low" ) return "low";
		if ( e == "medium" ) return "60";
		if ( e == "high" ) return "high";
		return "max";
	}
	if ( fam == engine::loader::ModelFamily::GLM ) {
		if ( e == "minimal" || e == "low" ) return "low";
		if ( e == "medium" || e == "high" ) return "high";
		return std::string(); // xhigh / max: the template default
	}
	if ( e == "minimal" || e == "low" ) return "low";
	if ( e == "medium" ) return "medium";
	return "xhigh";
}

// Chat templates accept a system message only at the start, but clients (Claude Code) also send system content later in
// the conversation. Leading system messages are folded into one system message; a later one becomes a user message in
// place, so the history stays append-only: the prompt of turn k+1 starts with the prompt of turn k, which is what the
// prefix cache matches on. (Folding them all into the front would rewrite the system prompt every turn.)
void fold_system_messages( nlohmann::ordered_json& messages ) {
	auto text_of = []( const nlohmann::ordered_json& m ) {
		std::string text;
		const auto& c = m["content"];
		if ( c.is_string() ) text = c.get<std::string>();
		else if ( c.is_array() ) for ( const auto& p : c ) text += p.value( "text", std::string() );
		return text;
	};
	std::string system;
	nlohmann::ordered_json out = nlohmann::ordered_json::array();
	bool leading = true;
	for ( const auto& m : messages ) {
		const bool is_system = m.value( "role", std::string() ) == "system";
		if ( is_system && leading ) {
			const std::string text = text_of( m );
			if ( !text.empty() ) system += ( system.empty() ? "" : "\n\n" ) + text;
			continue;
		}
		leading = false;
		if ( is_system ) {
			out.push_back( { { "role", "user" }, { "content", text_of( m ) } } );
			continue;
		}
		out.push_back( m );
	}
	nlohmann::ordered_json result = nlohmann::ordered_json::array();
	if ( !system.empty() ) result.push_back( { { "role", "system" }, { "content", system } } );
	for ( auto& m : out ) result.push_back( m );
	messages = std::move( result );
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

namespace {

using ImageList = std::vector<engine::runtime::GenerationRequest::Image>;

// Replaces each image placeholder token by `count` copies, consuming counts in order. Tokens of a prompt prefix expand the same
// way (the images of the prefix come first), which is how checkpoint positions stay valid.
std::vector<int> expand_image_tokens( const std::vector<int>& tokens, int pad_id, const ImageList& images ) {
	if ( images.empty() || pad_id < 0 ) return tokens;
	std::vector<int> out;
	out.reserve( tokens.size() );
	size_t next = 0;
	for ( int t : tokens ) {
		if ( t == pad_id && next < images.size() ) {
			out.insert( out.end(), images[next++].count, pad_id );
		}
		else out.push_back( t );
	}
	return out;
}

// The image placeholder token of a model family
const char* image_pad_name( engine::loader::ModelFamily fam ) {
	if ( fam == engine::loader::ModelFamily::DEEPSEEK ) return "<｜deepseek_image｜>";
	return fam == engine::loader::ModelFamily::GLM ? "<|image|>" : "<|image_pad|>";
}

// GLM's shipped chat template only knows text (an image part renders as a "cannot process" reminder), so the image markers of
// the model's processor are written into the conversation as text; the tokenizer turns them into the special tokens
void inline_glm_image_markers( nlohmann::ordered_json& messages ) {
	for ( auto& m : messages ) {
		if ( !m.is_object() || !m.contains( "content" ) || !m["content"].is_array() ) continue;
		for ( auto& part : m["content"] ) {
			if ( part.is_object() && part.value( "type", std::string() ) == "image" ) {
				part = { { "type", "text" }, { "text", "<|begin_of_image|><|image|><|end_of_image|>" } };
			}
		}
	}
}

// Chat-control special tokens in message content are text, not control tokens: the escape swaps each such string in the
// conversation for a private-use marker pair around its index before the template renders, and the tokenizer pass turns the
// markers back into the string's ordinary (non-special) pieces. The template's own role / turn markers stay real tokens.
constexpr const char* kEscOpen = "\xEE\x80\x80";
constexpr const char* kEscClose = "\xEE\x80\x81";

void replace_all( std::string& s, const std::string& from, const std::string& to ) {
	for ( size_t p = s.find( from ); p != std::string::npos; p = s.find( from, p + to.size() ) ) s.replace( p, from.size(), to );
}

void escape_control_text( nlohmann::ordered_json& j, const std::vector<std::string>& toks ) {
	if ( j.is_string() ) {
		std::string s = j.get<std::string>();
		replace_all( s, kEscOpen, "\xEF\xBF\xBD" );
		replace_all( s, kEscClose, "\xEF\xBF\xBD" );
		for ( size_t i = 0; i < toks.size(); ++i ) replace_all( s, toks[i], kEscOpen + std::to_string( i ) + kEscClose );
		j = std::move( s );
	} else if ( j.is_array() || j.is_object() ) {
		for ( auto& v : j ) escape_control_text( v, toks );
	}
}

// Visits the escaped prompt as alternating template text and escaped strings
template <typename Text, typename Escaped>
void split_escaped( const std::string& prompt, const std::vector<std::string>& toks, Text text, Escaped escaped ) {
	size_t pos = 0;
	while ( pos < prompt.size() ) {
		const size_t o = prompt.find( kEscOpen, pos );
		if ( o == std::string::npos ) break;
		const size_t c = prompt.find( kEscClose, o );
		if ( c == std::string::npos ) break;
		text( prompt.substr( pos, o - pos ) );
		const size_t idx = static_cast<size_t>( std::atoi( prompt.c_str() + o + 3 ) );
		escaped( idx < toks.size() ? toks[idx] : std::string() );
		pos = c + 3;
	}
	if ( pos < prompt.size() ) text( prompt.substr( pos ) );
}

std::vector<int> encode_prompt( const engine::loader::Tokenizer& tok, const std::string& prompt ) {
	std::vector<int> out;
	split_escaped( prompt, tok.control_tokens(),
		[&]( const std::string& t ) { const auto v = tok.encode( t, false ); out.insert( out.end(), v.begin(), v.end() ); },
		[&]( const std::string& t ) { const auto v = tok.encode_plain( t ); out.insert( out.end(), v.begin(), v.end() ); } );
	return out;
}

std::string unescape_prompt( const engine::loader::Tokenizer& tok, const std::string& prompt ) {
	std::string out;
	split_escaped( prompt, tok.control_tokens(), [&]( const std::string& t ) { out += t; }, [&]( const std::string& t ) { out += t; } );
	return out;
}

bool render_prompt( engine::runtime::Generator& gen, const ServerConfig& cfg, const ChatRequest& req, std::string* prompt, std::vector<int>* tokens, ImageList* images, ApiError* err ) {
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
	if ( vars["messages"].is_array() ) fold_system_messages( vars["messages"] );
	if ( family == engine::loader::ModelFamily::GLM && !req.images.empty() && vars["messages"].is_array() ) inline_glm_image_markers( vars["messages"] );
	// tool_choice required / named: say so on the last user turn (not after a tool result, or the model would loop on it)
	if ( !req.tools_json.empty() && ( req.require_tool || !req.force_tool.empty() ) && vars["messages"].is_array() && !vars["messages"].empty() &&
		vars["messages"].back().value( "role", std::string() ) == "user" ) {
		const std::string hint = req.force_tool.empty() ? "\n\n(Respond by calling one of the available functions.)"
			: "\n\n(Respond by calling the function " + req.force_tool + ".)";
		auto& last = vars["messages"].back();
		if ( last["content"].is_string() ) last["content"] = last["content"].get<std::string>() + hint;
		else if ( last["content"].is_array() ) last["content"].push_back( { { "type", "text" }, { "text", hint } } );
	}
	if ( !req.tools_json.empty() ) vars["tools"] = nlohmann::ordered_json::parse( req.tools_json, nullptr, false );
	vars["add_generation_prompt"] = true;
	vars["enable_thinking"] = thinking;
	if ( !effort.empty() ) vars["reasoning_effort"] = effort;

	{
		const std::vector<std::string> ctl = gen.tokenizer().control_tokens();
		escape_control_text( vars["messages"], ctl );
		if ( vars.contains( "tools" ) ) escape_control_text( vars["tools"], ctl );
	}
	std::string terr;
	if ( !gen.chat_template().render_json( vars.dump(), prompt, &terr ) ) {
		err->message = "the chat template rejected this conversation: " + terr;
		return false;
	}
	// The GLM template always opens <think>; non-thinking mode closes it right away (GLM-4.5 convention)
	if ( !thinking && family == engine::loader::ModelFamily::GLM && ends_with( *prompt, "<think>" ) ) *prompt += "</think>";
	*tokens = encode_prompt( gen.tokenizer(), *prompt );
	if ( !req.images.empty() ) {
		const engine::loader::VisionParams* vp = gen.vision_params();
		const int pad_id = gen.tokenizer().get_special_token( image_pad_name( family ) );
		if ( !vp || pad_id < 0 ) {
			err->message = "this model was loaded without a vision tower, so it cannot take image input";
			return false;
		}
		for ( const std::string& bytes : req.images ) {
			engine::loader::DecodedImage dec;
			auto prep = std::make_shared<engine::loader::PreparedImage>();
			std::string ierr;
			if ( !engine::loader::decode_image( reinterpret_cast<const uint8_t*>( bytes.data() ), bytes.size(), &dec, &ierr ) ||
			     !engine::loader::prepare_image( dec, *vp, prep.get(), &ierr ) ) {
				err->message = ierr + " (JPEG, PNG, BMP and GIF are supported)";
				return false;
			}
			engine::runtime::GenerationRequest::Image im;
			im.grid_h = vp->deepseek ? prep->llm_h : prep->grid_h / vp->merge_size;
			im.grid_w = vp->deepseek ? prep->llm_w : prep->grid_w / vp->merge_size;
			im.count = vp->deepseek ? prep->span_tokens() : im.grid_h * im.grid_w;   // DeepSeek: the delimiters and newlines are part of the span
			im.hash = prep->hash;
			im.prepared = std::move( prep );
			images->push_back( std::move( im ) );
		}
		size_t pads = 0;
		for ( int t : *tokens ) pads += ( t == pad_id ) ? 1 : 0;
		if ( pads != images->size() ) {
			err->message = "the conversation holds " + std::to_string( images->size() ) + " images but the chat template placed " + std::to_string( pads ) + " (images are only allowed in user and tool messages)";
			return false;
		}
		*tokens = expand_image_tokens( *tokens, pad_id, *images );
		int at = 0, k = 0;
		for ( size_t i = 0; i < tokens->size(); ++i ) {
			if ( ( *tokens )[i] != pad_id ) continue;
			if ( at == 0 ) ( *images )[k].start = static_cast<int>( i );
			if ( ++at == ( *images )[k].count ) {
				at = 0;
				++k;
			}
		}
	}
	const int ctx = gen.max_context();
	if ( tokens->empty() || static_cast<int>( tokens->size() ) + 1 >= ctx ) {
		err->message = "this conversation is " + std::to_string( tokens->size() ) + " tokens; the context window is " + std::to_string( ctx );
		err->type = "context_length_exceeded";
		return false;
	}
	return true;
}

} // namespace

bool count_chat_tokens( engine::runtime::Generator& gen, const ServerConfig& cfg, const ChatRequest& req, int* tokens, ApiError* err ) {
	std::string prompt;
	std::vector<int> ids;
	ImageList images;
	if ( !render_prompt( gen, cfg, req, &prompt, &ids, &images, err ) ) return false;
	*tokens = static_cast<int>( ids.size() );
	return true;
}

bool start_chat( engine::runtime::Generator& gen, const ServerConfig& cfg, ChatRequest req, uint64_t request_id,
	std::shared_ptr<Responder> responder, const ConnPtr& conn, ApiError* err ) {
	const auto family = gen.chat_template().model_family();
	if ( req.steer_probe && !gen.steer_enabled() ) {
		err->message = "steer_probe needs the steer option";
		return false;
	}
	if ( std::string why; !gen.check_steer( req.steer_vector, req.steer_scope, &why ) ) {
		err->message = why;
		return false;
	}
	std::string prompt;
	std::vector<int> tokens;
	ImageList images;
	if ( !render_prompt( gen, cfg, req, &prompt, &tokens, &images, err ) ) return false;
	responder->on_prompt( static_cast<int>( tokens.size() ) );
	if ( !runtime::options().dump_requests.empty() ) {
		const std::string path = runtime::options().dump_requests + "/req_" + std::to_string( request_id ) + ".prompt.txt";
		if ( FILE* f = std::fopen( path.c_str(), "wb" ) ) {
			const std::string shown = unescape_prompt( gen.tokenizer(), prompt );
			std::fwrite( shown.data(), 1, shown.size(), f );
			std::fclose( f );
		}
	}

	const bool in_reasoning = ends_with( prompt, "<think>\n" ) || ends_with( prompt, "<think>" );
	auto state = std::make_shared<ChatState>(
		OutputParser( family, in_reasoning, tool_schemas_from_json( req.tools_json ) ), std::move( responder ), conn );

	engine::runtime::GenerationRequest g;
	g.req_id = request_id;
	g.prompt_tokens = std::move( tokens );
	g.images = std::move( images );
	g.max_tokens = req.max_tokens > 0 ? req.max_tokens : cfg.default_max_tokens;
	g.temperature = req.temperature >= 0.0f ? req.temperature : cfg.default_temperature;
	g.top_p = req.top_p >= 0.0f ? req.top_p : cfg.default_top_p;
	g.top_k = req.top_k > 0 ? req.top_k : cfg.default_top_k;
	g.seed = req.seed;
	g.stop_strings = req.stop;
	g.starts_in_reasoning = in_reasoning;
	g.steer_vector = req.steer_vector;
	g.steer_scale = req.steer_scale;
	g.steer_scope = req.steer_scope;
	g.steer_probe = req.steer_probe;
	g.probe_edit_layers = req.probe_edit_layers;
	g.probe_threshold = req.probe_threshold;
	g.probe_edit_strength = req.probe_edit_strength;
	g.probe_hedge_strength = req.probe_hedge_strength;
	g.probe_push = req.probe_push;
	g.think_penalty = req.think_penalty;
	if ( in_reasoning && req.thinking_budget > 0 ) {
		g.reasoning_budget = req.thinking_budget;
		g.reasoning_close = gen.tokenizer().encode( family == engine::loader::ModelFamily::GLM ? "</think>" : "\n</think>\n\n", false );
		g.think_close_id = gen.tokenizer().get_special_token( "</think>" );
	}
	// Prefix cache points. Root: the end of the system block (before the first user turn), shared by every conversation of
	// this client. Turn boundary: the start of the generation prompt; the next turn's prompt re-renders the assistant
	// reply there, so the end of the whole prompt is never a prefix of it but this boundary is.
	{
		const bool glm = family == engine::loader::ModelFamily::GLM;
		const int pad_id = gen.tokenizer().get_special_token( image_pad_name( family ) );
		auto add = [&]( size_t at, bool pinned, size_t min_tokens ) {
			if ( at == std::string::npos || at == 0 ) return;
			const std::vector<int> head = expand_image_tokens( encode_prompt( gen.tokenizer(), prompt.substr( 0, at ) ), pad_id, g.images );
			if ( head.size() < min_tokens || head.size() >= g.prompt_tokens.size() ) return;
			if ( !std::equal( head.begin(), head.end(), g.prompt_tokens.begin() ) ) return;
			if ( !g.checkpoint_at.empty() && static_cast<int>( head.size() ) <= g.checkpoint_at.back().pos ) return;
			g.checkpoint_at.push_back( { static_cast<int>( head.size() ), pinned } );
		};
		const bool ds = family == engine::loader::ModelFamily::DEEPSEEK;
		add( prompt.find( ds ? "<｜User｜>" : glm ? "<|user|>" : "<|im_start|>user" ), true, 1024 );
		add( prompt.rfind( ds ? "<｜Assistant｜>" : glm ? "<|assistant|>" : "<|im_start|>assistant" ), false, 512 );
	}
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
		u.cached_tokens = m.cached_tokens;
		u.completion_tokens = m.generated_tokens;
		u.ttft_ms = std::chrono::duration<double, std::milli>( m.first_token_time - m.start_time ).count();
		u.decode_tok_s = m.decode_tok_s;
		u.gate_score = m.gate_score;
		u.gate_fired = m.gate_fired;
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
