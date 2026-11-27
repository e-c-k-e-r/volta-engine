#include "engine/server/chat.h"
#include "engine/runtime/options.h"
#include "engine/core/batch.h"

#include <algorithm>
#include <cctype>

#include <cstdio>
#include <map>
#include <mutex>

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

// Many chat templates accept a system message only at the start, but clients (Claude Code) also send system content later in
// the conversation (its environment block after the first user message, updates further on). The system messages before the first
// reply join the system prompt, each on a new line (as OpenRouter passes them on); a later one stays a system message where the
// template renders one there, else it becomes a user message in place (later_as_user), so the history stays append-only: the
// prompt of turn k+1 starts with the prompt of turn k, which is what the prefix cache matches on. (Folding them all into the front
// would rewrite the system prompt every turn.)
void fold_system_messages( nlohmann::ordered_json& messages, bool later_as_user ) {
	auto text_of = []( const nlohmann::ordered_json& m ) {
		std::string text;
		const auto& c = m["content"];
		if ( c.is_string() ) text = c.get<std::string>();
		else if ( c.is_array() ) for ( const auto& p : c ) text += p.value( "text", std::string() );
		return text;
	};
	std::string system;
	nlohmann::ordered_json out = nlohmann::ordered_json::array();
	bool replied = false;
	for ( const auto& m : messages ) {
		const std::string role = m.value( "role", std::string() );
		const bool is_system = role == "system";
		if ( is_system && !replied ) {
			const std::string text = text_of( m );
			if ( !text.empty() ) system += ( system.empty() ? "" : "\n" ) + text;
			continue;
		}
		replied = replied || role == "assistant";
		if ( is_system ) {
			out.push_back( { { "role", later_as_user ? "user" : "system" }, { "content", text_of( m ) } } );
			continue;
		}
		out.push_back( m );
	}
	nlohmann::ordered_json result = nlohmann::ordered_json::array();
	if ( !system.empty() ) result.push_back( { { "role", "system" }, { "content", system } } );
	for ( auto& m : out ) result.push_back( m );
	messages = std::move( result );
}

// Whether the template renders a system message once the conversation has started: a probe conversation, rendered once per
// template, with a marker in a later system message (a template may reject one, or drop it)
bool takes_later_system( const engine::loader::ChatTemplate& t ) {
	static std::mutex mu;
	static std::map<const engine::loader::ChatTemplate*, bool> known;
	std::lock_guard<std::mutex> lk( mu );
	if ( const auto it = known.find( &t ); it != known.end() ) return it->second;
	const std::string marker = "later-system-probe-5e1f";
	const std::string out = t.render( { { "system", "s" }, { "user", "u" }, { "assistant", "a" }, { "system", marker }, { "user", "v" } } );
	return known[&t] = out.find( marker ) != std::string::npos;
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

// How a prompt's media placeholders expand. An image pad becomes its `count` pads (between Gemma 4's markers when those are set); a video's
// placeholder becomes its run, the steps' pads among their timestamp text and markers (Qwen3-VL also drops the template's vision_start /
// vision_end around the placeholder, since every step carries its own). Images and videos each expand in their own order; a prompt prefix
// expands the same way (its media come first), which is how checkpoint positions stay valid.
struct MediaTokens {
	int image_pad = -1, image_begin = -1, image_end = -1;
	std::vector<int> image_counts;
	int video_pad = -1, wrap_begin = -1, wrap_end = -1;
	std::vector<std::vector<int>> video_runs;
	std::vector<std::vector<int>> video_starts;   // where each step's pads begin in its video's run
};

// Where an expanded image's or video step's pads begin: video -1 is image `index`, else step `index` of that video
struct MediaSpan {
	int video = -1;
	int index = 0;
	int start = 0;
};

std::vector<int> expand_media( const std::vector<int>& tokens, const MediaTokens& mt, std::vector<MediaSpan>* spans = nullptr ) {
	if ( mt.image_counts.empty() && mt.video_runs.empty() ) return tokens;
	std::vector<int> out;
	out.reserve( tokens.size() );
	size_t img = 0, vid = 0;
	for ( size_t i = 0; i < tokens.size(); ++i ) {
		const int t = tokens[i];
		if ( t == mt.image_pad && mt.image_pad >= 0 && img < mt.image_counts.size() ) {
			if ( mt.image_begin >= 0 ) out.push_back( mt.image_begin );
			if ( spans ) spans->push_back( { -1, static_cast<int>( img ), static_cast<int>( out.size() ) } );
			out.insert( out.end(), mt.image_counts[img++], mt.image_pad );
			if ( mt.image_end >= 0 ) out.push_back( mt.image_end );
		}
		else if ( t == mt.video_pad && mt.video_pad >= 0 && vid < mt.video_runs.size() ) {
			if ( mt.wrap_begin >= 0 && !out.empty() && out.back() == mt.wrap_begin ) out.pop_back();
			const int base = static_cast<int>( out.size() );
			out.insert( out.end(), mt.video_runs[vid].begin(), mt.video_runs[vid].end() );
			if ( spans ) {
				for ( size_t k = 0; k < mt.video_starts[vid].size(); ++k ) spans->push_back( { static_cast<int>( vid ), static_cast<int>( k ), base + mt.video_starts[vid][k] } );
			}
			++vid;
			if ( mt.wrap_end >= 0 && i + 1 < tokens.size() && tokens[i + 1] == mt.wrap_end ) ++i;
		}
		else {
			out.push_back( t );
		}
	}
	return out;
}

// The tokens around an image's placeholders (Gemma 4), -1 for families without them
void image_markers( engine::loader::ModelFamily fam, const engine::loader::Tokenizer& tok, int* begin_id, int* end_id ) {
	const bool gemma = fam == engine::loader::ModelFamily::GEMMA;
	*begin_id = gemma ? tok.get_special_token( "<|image>" ) : -1;
	*end_id = gemma ? tok.get_special_token( "<image|>" ) : -1;
}

// The image placeholder token of a model family
const char* image_pad_name( engine::loader::ModelFamily fam ) {
	if ( fam == engine::loader::ModelFamily::DEEPSEEK ) return "<｜deepseek_image｜>";
	return ( fam == engine::loader::ModelFamily::GLM || fam == engine::loader::ModelFamily::GEMMA ) ? "<|image|>" : "<|image_pad|>";
}

// GLM's shipped chat template only knows text (an image part renders as a "cannot process" reminder), so the image markers of
// the model's processor are written into the conversation as text; the tokenizer turns them into the special tokens
void inline_glm_image_markers( nlohmann::ordered_json& messages, bool images, bool videos ) {
	for ( auto& m : messages ) {
		if ( !m.is_object() || !m.contains( "content" ) || !m["content"].is_array() ) continue;
		for ( auto& part : m["content"] ) {
			const std::string type = part.is_object() ? part.value( "type", std::string() ) : std::string();
			if ( images && type == "image" ) part = { { "type", "text" }, { "text", "<|begin_of_image|><|image|><|end_of_image|>" } };
			if ( videos && type == "video" ) part = { { "type", "text" }, { "text", "<|begin_of_video|><|video|><|end_of_video|>" } };
		}
	}
}

// Image (or video) parts as a text note, for a model loaded without a vision tower (or one that takes no video)
void omit_media( nlohmann::ordered_json& messages, const char* type, const char* note ) {
	for ( auto& m : messages ) {
		if ( !m.is_object() || !m.contains( "content" ) || !m["content"].is_array() ) continue;
		for ( auto& part : m["content"] ) {
			if ( part.is_object() && part.value( "type", std::string() ) == type ) part = { { "type", "text" }, { "text", note } };
		}
	}
}

// A video's placeholder and its steps' markers. Qwen3-VL: the template's vision_start video_pad vision_end becomes, per step,
// "<t seconds>" vision_start video_pad x n vision_end (t: the mean of the step's two frame times); GLM-5: the template's video pad becomes,
// per step, begin_of_image image x n end_of_image "t seconds" (t: the step's first frame time) inside begin_of_video ... end_of_video
struct VideoMarkers {
	int pad = -1, wrap_begin = -1, wrap_end = -1;
	int step_begin = -1, step_pad = -1, step_end = -1;
	bool time_after = false;
};

bool video_markers( engine::loader::ModelFamily fam, const engine::loader::Tokenizer& tok, VideoMarkers* out ) {
	if ( fam == engine::loader::ModelFamily::QWEN ) {
		out->pad = out->step_pad = tok.get_special_token( "<|video_pad|>" );
		out->wrap_begin = out->step_begin = tok.get_special_token( "<|vision_start|>" );
		out->wrap_end = out->step_end = tok.get_special_token( "<|vision_end|>" );
		out->time_after = false;
	}
	else if ( fam == engine::loader::ModelFamily::GLM ) {
		out->pad = tok.get_special_token( "<|video|>" );
		out->step_begin = tok.get_special_token( "<|begin_of_image|>" );
		out->step_pad = tok.get_special_token( "<|image|>" );
		out->step_end = tok.get_special_token( "<|end_of_image|>" );
		out->time_after = true;
		if ( tok.get_special_token( "<|begin_of_video|>" ) < 0 || tok.get_special_token( "<|end_of_video|>" ) < 0 ) return false;
	}
	else {
		return false;
	}
	return out->pad >= 0 && out->step_begin >= 0 && out->step_pad >= 0 && out->step_end >= 0;
}

// A video request part into its prepared steps: a file through FFmpeg, encoded frames, or the host's decoded frames
bool prepare_request_video( const ChatRequest::Video& v, const engine::loader::VisionParams& vp, engine::loader::PreparedVideo* out, std::string* why ) {
	const int budget = runtime::options().max_video_tokens;
	if ( !v.file.empty() ) return engine::loader::decode_video( reinterpret_cast<const uint8_t*>( v.file.data() ), v.file.size(), vp, budget, out, why );
	if ( v.decoded ) return engine::loader::prepare_video( *v.decoded, vp, budget, out, why );
	if ( v.frames.empty() ) {
		*why = "a video part needs video_url, or video with a file's or its frames' data: URLs";
		return false;
	}
	engine::loader::VideoFrames frames;
	frames.frames.resize( v.frames.size() );
	frames.times = v.times;
	for ( size_t i = 0; i < v.frames.size(); ++i ) {
		if ( !engine::loader::decode_image( reinterpret_cast<const uint8_t*>( v.frames[i].data() ), v.frames[i].size(), &frames.frames[i], why ) ) {
			*why += " (a video frame: JPEG, PNG, BMP and GIF are supported)";
			return false;
		}
	}
	return engine::loader::prepare_video( frames, vp, budget, out, why );
}

// Gemma 4's template takes tool results as tool_responses ({ name, response }) on the tool message: the results of consecutive tool
// messages go into one, each named after the call its tool_call_id answers (text parts joined; other parts are dropped)
void gemma_tool_messages( nlohmann::ordered_json& messages ) {
	std::map<std::string, std::string> call_names;
	nlohmann::ordered_json out = nlohmann::ordered_json::array();
	for ( auto& m : messages ) {
		const std::string role = m.value( "role", std::string() );
		if ( role == "assistant" && m.contains( "tool_calls" ) && m["tool_calls"].is_array() ) {
			for ( const auto& tc : m["tool_calls"] ) {
				if ( tc.is_object() && tc.contains( "id" ) && tc["id"].is_string() && tc.contains( "function" ) && tc["function"].is_object() )
					call_names[tc["id"].get<std::string>()] = tc["function"].value( "name", std::string() );
			}
		}
		if ( role != "tool" ) {
			out.push_back( std::move( m ) );
			continue;
		}
		std::string text;
		const auto& c = m["content"];
		if ( c.is_string() ) text = c.get<std::string>();
		else if ( c.is_array() ) for ( const auto& p : c ) if ( p.is_object() && p.value( "type", std::string() ) == "text" ) text += p.value( "text", std::string() );
		const std::string id = m.value( "tool_call_id", std::string() );
		const auto it = call_names.find( id );
		nlohmann::ordered_json resp = { { "name", it != call_names.end() ? it->second : std::string( "unknown" ) }, { "response", text } };
		if ( !out.empty() && out.back().value( "role", std::string() ) == "tool" && out.back().contains( "tool_responses" ) ) out.back()["tool_responses"].push_back( resp );
		else out.push_back( { { "role", "tool" }, { "tool_responses", nlohmann::ordered_json::array( { resp } ) } } );
	}
	messages = std::move( out );
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

// Prompt plus reply tokens a request can hold: generation stops one verify block (MAX_BATCH rows) short of the window
int reply_window( int max_context ) {
	return max_context - pod::MAX_BATCH - 1;
}

PromptModel prompt_model( engine::runtime::Generator& gen ) {
	return { gen.chat_template(), gen.tokenizer(), gen.vision_params(), gen.max_context(),
		[&gen]( const engine::runtime::GenerationRequest::Image& im ) { gen.prefetch_image( im ); } };
}

bool render_prompt( const PromptModel& m, const ServerConfig& cfg, const ChatRequest& req, std::string* prompt, std::vector<int>* tokens, ImageList* images,
	MediaTokens* media, ApiError* err, std::string* vars_json = nullptr ) {
	const auto family = m.chat_template.model_family();

	bool thinking = req.enable_thinking >= 0 ? req.enable_thinking != 0 : cfg.default_enable_thinking;
	const std::string effort = effort_for( family, req.reasoning_effort );
	if ( req.enable_thinking < 0 && !req.reasoning_effort.empty() ) thinking = true;

	nlohmann::ordered_json vars;
	vars["messages"] = nlohmann::ordered_json::parse( req.messages_json, nullptr, false );
	if ( vars["messages"].is_discarded() ) {
		err->message = "messages could not be normalized";
		return false;
	}
	if ( vars["messages"].is_array() ) fold_system_messages( vars["messages"], !takes_later_system( m.chat_template ) );
	// Without a vision tower an image part becomes a note in the text, so an agent that reads an image file carries on instead of failing
	const bool see = !req.images.empty() && m.vision && m.tokenizer.get_special_token( image_pad_name( family ) ) >= 0;
	if ( !req.images.empty() && !see && vars["messages"].is_array() ) omit_media( vars["messages"], "image", "[image omitted: this model cannot view images]" );
	// Videos: a tower whose checkpoint ships a video processor (Qwen3-VL, GLM-5), and the family's markers
	VideoMarkers vm;
	const bool watch = !req.videos.empty() && m.vision && m.vision->video && runtime::options().max_video_tokens > 0 && video_markers( family, m.tokenizer, &vm );
	if ( !req.videos.empty() && !watch && vars["messages"].is_array() ) omit_media( vars["messages"], "video", "[video omitted: this model cannot view videos]" );
	if ( family == engine::loader::ModelFamily::GEMMA ) {
		if ( vars["messages"].is_array() ) gemma_tool_messages( vars["messages"] );
		vars["bos_token"] = m.tokenizer.decode_token( m.tokenizer.get_special_token( "<bos>" ) );
	}
	// tool_choice required / named: say so on the last user turn (not after a tool result, or the model would loop on it)
	if ( !req.tools_json.empty() && ( req.require_tool || !req.force_tool.empty() ) && vars["messages"].is_array() && !vars["messages"].empty() &&
		vars["messages"].back().value( "role", std::string() ) == "user" ) {
		const std::string hint = req.force_tool.empty() ? "\n\n(Respond by calling one of the available functions.)"
			: "\n\n(Respond by calling the function " + req.force_tool + ".)";
		auto& last = vars["messages"].back();
		if ( last["content"].is_string() ) last["content"] = last["content"].get<std::string>() + hint;
		else if ( last["content"].is_array() ) last["content"].push_back( { { "type", "text" }, { "text", hint } } );
	}
	if ( !req.tools_json.empty() ) {
		nlohmann::ordered_json tools = nlohmann::ordered_json::parse( req.tools_json, nullptr, false );
		if ( tools.is_array() && !req.deferred_tools.empty() ) {
			nlohmann::ordered_json listed = nlohmann::ordered_json::array();
			for ( const auto& t : tools ) {
				const std::string name = t.contains( "function" ) ? t["function"].value( "name", std::string() ) : std::string();
				if ( std::find( req.deferred_tools.begin(), req.deferred_tools.end(), name ) == req.deferred_tools.end() ) listed.push_back( t );
			}
			tools = listed;
		}
		if ( !tools.is_array() || !tools.empty() ) vars["tools"] = tools;
	}
	vars["add_generation_prompt"] = true;
	vars["enable_thinking"] = thinking;
	if ( !effort.empty() ) vars["reasoning_effort"] = effort;

	{
		const std::vector<std::string> ctl = m.tokenizer.control_tokens();
		escape_control_text( vars["messages"], ctl );
		if ( vars.contains( "tools" ) ) escape_control_text( vars["tools"], ctl );
	}
	// After the escaping: the markers are the model's own special tokens, not text a user typed
	if ( family == engine::loader::ModelFamily::GLM && ( see || watch ) && vars["messages"].is_array() ) inline_glm_image_markers( vars["messages"], see, watch );
	std::string terr;
	if ( vars_json ) *vars_json = vars.dump();
	if ( !m.chat_template.render_json( vars.dump(), prompt, &terr ) ) {
		err->message = "the chat template rejected this conversation: " + terr;
		return false;
	}
	// The GLM template always opens <think>; non-thinking mode closes it right away (GLM-4.5 convention)
	if ( !thinking && family == engine::loader::ModelFamily::GLM && ends_with( *prompt, "<think>" ) ) *prompt += "</think>";
	*tokens = encode_prompt( m.tokenizer, *prompt );
	MediaTokens local;
	MediaTokens& mt = media ? *media : local;
	mt = MediaTokens();
	ImageList stills;
	std::vector<ImageList> steps;
	if ( see ) {
		const engine::loader::VisionParams* vp = m.vision;
		const int pad_id = m.tokenizer.get_special_token( image_pad_name( family ) );
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
			mt.image_counts.push_back( im.count );
			stills.push_back( std::move( im ) );
		}
		size_t pads = 0;
		for ( int t : *tokens ) pads += ( t == pad_id ) ? 1 : 0;
		if ( pads != stills.size() ) {
			err->message = "the conversation holds " + std::to_string( stills.size() ) + " images but the chat template placed " + std::to_string( pads ) + " (images are only allowed in user and tool messages)";
			return false;
		}
		mt.image_pad = pad_id;
		image_markers( family, m.tokenizer, &mt.image_begin, &mt.image_end );
	}
	if ( watch ) {
		const engine::loader::VisionParams& vp = *m.vision;
		size_t pads = 0;
		for ( int t : *tokens ) pads += ( t == vm.pad ) ? 1 : 0;
		if ( pads != req.videos.size() ) {
			err->message = "the conversation holds " + std::to_string( req.videos.size() ) + " videos but the chat template placed " + std::to_string( pads ) + " (videos are only allowed in user and tool messages)";
			return false;
		}
		mt.video_pad = vm.pad;
		mt.wrap_begin = vm.wrap_begin;
		mt.wrap_end = vm.wrap_end;
		for ( const ChatRequest::Video& v : req.videos ) {
			engine::loader::PreparedVideo pv;
			std::string verr;
			if ( !prepare_request_video( v, vp, &pv, &verr ) ) {
				err->message = verr;
				return false;
			}
			const int n = pv.tokens_per_step( vp.merge_size );
			std::vector<int> run, starts;
			ImageList list;
			for ( size_t k = 0; k < pv.steps.size(); ++k ) {
				char stamp[48];
				std::snprintf( stamp, sizeof( stamp ), vm.time_after ? "%.1f seconds" : "<%.1f seconds>", pv.times[k] );
				const std::vector<int> text = m.tokenizer.encode( stamp, false );
				if ( !vm.time_after ) run.insert( run.end(), text.begin(), text.end() );
				run.push_back( vm.step_begin );
				starts.push_back( static_cast<int>( run.size() ) );
				run.insert( run.end(), n, vm.step_pad );
				run.push_back( vm.step_end );
				if ( vm.time_after ) run.insert( run.end(), text.begin(), text.end() );
				engine::runtime::GenerationRequest::Image im;
				im.grid_h = pv.steps[k].grid_h / vp.merge_size;
				im.grid_w = pv.steps[k].grid_w / vp.merge_size;
				im.count = n;
				im.hash = pv.steps[k].hash;
				im.prepared = std::make_shared<engine::loader::PreparedImage>( std::move( pv.steps[k] ) );
				list.push_back( std::move( im ) );
			}
			mt.video_runs.push_back( std::move( run ) );
			mt.video_starts.push_back( std::move( starts ) );
			steps.push_back( std::move( list ) );
		}
	}
	if ( see || watch ) {
		std::vector<MediaSpan> spans;
		*tokens = expand_media( *tokens, mt, &spans );
		// the images and video steps in prompt order, each where its pads begin
		for ( const MediaSpan& sp : spans ) {
			engine::runtime::GenerationRequest::Image& im = sp.video < 0 ? stills[sp.index] : steps[sp.video][sp.index];
			im.start = sp.start;
			images->push_back( std::move( im ) );
		}
		// vision_cpu encodes here, on the request's own thread, before the request queues for a slot
		if ( m.prefetch_image ) for ( const auto& im : *images ) m.prefetch_image( im );
	}
	const int ctx = m.max_context;
	if ( req.strict_context && static_cast<int>( tokens->size() ) >= reply_window( ctx ) ) {
		err->message = "prompt is too long: " + std::to_string( tokens->size() ) + " tokens > " + std::to_string( reply_window( ctx ) ) + " maximum";
		return false;
	}
	if ( tokens->empty() || static_cast<int>( tokens->size() ) + 1 >= ctx ) {
		err->message = "this conversation is " + std::to_string( tokens->size() ) + " tokens; the context window is " + std::to_string( ctx );
		err->type = "context_length_exceeded";
		return false;
	}
	return true;
}

// dump_requests: the rendered prompt as req_<id>.prompt.txt, special tokens as their text
void dump_prompt( engine::runtime::Generator& gen, const std::string& prompt, uint64_t request_id ) {
	if ( runtime::options().dump_requests.empty() ) return;
	const std::string path = runtime::options().dump_requests + "/req_" + std::to_string( request_id ) + ".prompt.txt";
	if ( FILE* f = std::fopen( path.c_str(), "wb" ) ) {
		const std::string shown = unescape_prompt( gen.tokenizer(), prompt );
		std::fwrite( shown.data(), 1, shown.size(), f );
		std::fclose( f );
	}
}

} // namespace

bool render_chat_prompt( const PromptModel& model, const ServerConfig& cfg, const ChatRequest& req, std::string* prompt, std::vector<int>* tokens,
	ApiError* err, std::string* vars_json ) {
	ImageList images;
	return render_prompt( model, cfg, req, prompt, tokens, &images, nullptr, err, vars_json );
}

std::string shown_prompt( const engine::loader::Tokenizer& tok, const std::string& prompt ) {
	return unescape_prompt( tok, prompt );
}

bool count_chat_tokens( engine::runtime::Generator& gen, const ServerConfig& cfg, const ChatRequest& req, uint64_t request_id, int* tokens,
	ApiError* err ) {
	std::string prompt;
	std::vector<int> ids;
	ImageList images;
	if ( !render_prompt( prompt_model( gen ), cfg, req, &prompt, &ids, &images, nullptr, err ) ) return false;
	dump_prompt( gen, prompt, request_id );
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
	if ( std::string why; !gen.check_probes( req.probes, &why ) ) {
		err->message = why;
		return false;
	}
	std::string prompt;
	std::vector<int> tokens;
	ImageList images;
	MediaTokens media;
	if ( req.raw ) {
		// a text completion: the prompt as given, no chat template
		if ( req.raw_tokens.empty() ) {
			prompt = req.raw_prompt;
			tokens = gen.tokenizer().encode( prompt, false );
		}
		else {
			for ( int t : req.raw_tokens ) {
				if ( t < 0 || t >= static_cast<int>( gen.tokenizer().vocab_size() ) ) {
					err->message = "prompt token id " + std::to_string( t ) + " is outside the vocabulary";
					return false;
				}
			}
			tokens = req.raw_tokens;
			prompt = gen.tokenizer().decode( tokens );
		}
		if ( tokens.empty() ) {
			err->message = "the prompt is empty";
			return false;
		}
		if ( static_cast<int>( tokens.size() ) + 1 >= gen.max_context() ) {
			err->message = "this prompt is " + std::to_string( tokens.size() ) + " tokens; the context window is " + std::to_string( gen.max_context() );
			err->type = "context_length_exceeded";
			return false;
		}
		responder->on_raw_prompt( prompt );
	}
	else if ( !render_prompt( prompt_model( gen ), cfg, req, &prompt, &tokens, &images, &media, err ) ) return false;
	responder->on_prompt( static_cast<int>( tokens.size() ) );
	dump_prompt( gen, prompt, request_id );

	const bool in_reasoning = !req.raw && ( ends_with( prompt, "<think>\n" ) || ends_with( prompt, "<think>" ) );
	auto state = std::make_shared<ChatState>(
		OutputParser( family, in_reasoning, tool_schemas_from_json( req.tools_json ) ), std::move( responder ), conn );

	const int max_tokens = req.max_tokens > 0 ? req.max_tokens : cfg.default_max_tokens;
	if ( req.strict_context && static_cast<int>( tokens.size() ) + max_tokens > reply_window( gen.max_context() ) ) {
		err->message = "input length and `max_tokens` exceed context limit: " + std::to_string( tokens.size() ) + " + " + std::to_string( max_tokens ) + " > " +
			std::to_string( reply_window( gen.max_context() ) ) + ", decrease input length or `max_tokens` and try again";
		return false;
	}

	engine::runtime::GenerationRequest g;
	g.req_id = request_id;
	g.prompt_tokens = std::move( tokens );
	g.images = std::move( images );
	g.max_tokens = max_tokens;
	g.temperature = req.temperature >= 0.0f ? req.temperature : cfg.default_temperature;
	g.top_p = req.top_p >= 0.0f ? req.top_p : cfg.default_top_p;
	g.top_k = req.top_k > 0 ? req.top_k : cfg.default_top_k;
	g.seed = req.seed;
	g.presence_penalty = req.presence_penalty > -1000.0f ? req.presence_penalty : cfg.default_presence_penalty;
	g.frequency_penalty = req.frequency_penalty;
	g.repetition_penalty = req.repetition_penalty > 0.0f ? req.repetition_penalty : cfg.default_repetition_penalty;
	g.logprobs = req.logprobs;
	g.probes = req.probes;
	g.priority = req.priority;
	if ( g.priority == INT_MIN ) {
		// a client's side calls name a smaller model (Claude Code: haiku-class titles and summaries): they queue behind the main conversation
		g.priority = 0;
		const std::string& low = engine::runtime::options().low_priority_models;
		std::string model = req.model;
		for ( char& c : model ) c = static_cast<char>( std::tolower( static_cast<unsigned char>( c ) ) );
		for ( size_t p = 0; p < low.size(); ) {
			const size_t q = std::min( low.find( ',', p ), low.size() );
			if ( q > p && model.find( low.substr( p, q - p ) ) != std::string::npos ) { g.priority = 1; break; }
			p = q + 1;
		}
	}
	g.stop_strings = req.stop;
	if ( req.stop_on_loop >= 0 ) g.loop_detect = req.stop_on_loop ? 2 : 0;
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
	g.probe_directions = req.probe_directions;
	g.probe_keep_norm = req.probe_keep_norm;
	g.think_penalty = req.think_penalty;
	int budget = req.thinking_budget;
	if ( budget <= 0 && engine::runtime::options().thinking_budget != 0 ) {
		const int half = max_tokens / 2;
		budget = engine::runtime::options().thinking_budget < 0 ? half : std::min( engine::runtime::options().thinking_budget, half > 0 ? half : INT_MAX );
	}
	if ( in_reasoning && budget > 0 ) {
		g.reasoning_budget = budget;
		g.reasoning_close = gen.tokenizer().encode( family == engine::loader::ModelFamily::GLM ? "</think>" : "\n</think>\n\n", false );
		g.think_close_id = gen.tokenizer().get_special_token( "</think>" );
	}
	// Prefix cache points. Root: the end of the system block (before the first user turn), shared by every conversation of
	// this client. Turn boundary: the start of the generation prompt; the next turn's prompt re-renders the assistant
	// reply there, so the end of the whole prompt is never a prefix of it but this boundary is.
	{
		const bool glm = family == engine::loader::ModelFamily::GLM;
		auto add = [&]( size_t at, bool pinned, size_t min_tokens ) {
			if ( at == std::string::npos || at == 0 ) return;
			const std::vector<int> head = expand_media( encode_prompt( gen.tokenizer(), prompt.substr( 0, at ) ), media );
			if ( head.size() < min_tokens || head.size() >= g.prompt_tokens.size() ) return;
			if ( !std::equal( head.begin(), head.end(), g.prompt_tokens.begin() ) ) return;
			if ( !g.checkpoint_at.empty() && static_cast<int>( head.size() ) <= g.checkpoint_at.back().pos ) return;
			g.checkpoint_at.push_back( { static_cast<int>( head.size() ), pinned } );
		};
		const bool ds = family == engine::loader::ModelFamily::DEEPSEEK, gemma = family == engine::loader::ModelFamily::GEMMA;
		add( prompt.find( ds ? "<｜User｜>" : glm ? "<|user|>" : gemma ? "<|turn>user" : "<|im_start|>user" ), true, 1024 );
		add( prompt.rfind( ds ? "<｜Assistant｜>" : glm ? "<|assistant|>" : gemma ? "<|turn>model" : "<|im_start|>assistant" ), false, 512 );
	}
	const bool raw = req.raw;
	g.on_text = [state, raw]( const std::string& delta ) {
		if ( state->conn && state->conn->closed() ) return false;
		if ( raw ) return state->responder->on_event( { OutputEvent::Type::Content, delta, {} } );   // plain text: nothing to parse out
		std::vector<OutputEvent> events;
		state->parser.feed( delta, events );
		for ( const OutputEvent& ev : events ) {
			if ( !state->responder->on_event( ev ) ) return false;
		}
		return true;
	};
	g.on_wait = [state]() { return !( state->conn && state->conn->closed() ) && state->responder->on_wait(); };
	g.on_admit = [state]( int cached ) { state->responder->on_admit( cached ); };
	if ( req.logprobs >= 0 ) {
		const engine::loader::Tokenizer* tk = &gen.tokenizer();
		g.on_logprobs = [state, tk]( const std::vector<engine::runtime::TokenLogprob>& entries ) {
			std::vector<LogprobOut> out( entries.size() );
			for ( size_t i = 0; i < entries.size(); ++i ) {
				out[i].token = tk->decode_token( entries[i].id );
				out[i].logprob = entries[i].logprob;
				for ( int k = 0; k < entries[i].n; ++k ) out[i].top.emplace_back( tk->decode_token( entries[i].ids[k] ), entries[i].logprobs[k] );
			}
			state->responder->on_logprobs( out );
		};
	}
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
		u.probes = m.probes;
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

nlohmann::json probes_json( const std::vector<engine::runtime::ProbeSummary>& probes ) {
	nlohmann::json out = nlohmann::json::array();
	auto num = []( float v ) { return std::isfinite( v ) ? nlohmann::json( v ) : nlohmann::json( nullptr ); };
	for ( const auto& p : probes ) {
		out.push_back( { { "name", p.name }, { "prompt", num( p.prompt ) }, { "max", num( p.max ) }, { "mean", num( p.mean ) }, { "last", num( p.last ) },
			{ "above", p.above }, { "tokens", p.n } } );
	}
	return out;
}

} // namespace engine::server
