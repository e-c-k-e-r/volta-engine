#include "engine/loader/chat_template.h"

#define JSON_NOEXCEPTION 1
#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <map>

namespace engine::loader {

ChatTemplate::ChatTemplate() {
}

ChatTemplate::~ChatTemplate() {
}

static std::string trim_string( const std::string& str ) {
	size_t start = 0;
	while ( start < str.size() && ( str[start] == ' ' || str[start] == '\t' || str[start] == '\r' || str[start] == '\n' ) ) {
		start++;
	}
	size_t end = str.size();
	while ( end > start && ( str[end - 1] == ' ' || str[end - 1] == '\t' || str[end - 1] == '\r' || str[end - 1] == '\n' ) ) {
		end--;
	}
	return str.substr( start, end - start );
}

static std::string capitalize_string( const std::string& str ) {
	if ( str.empty() ) return str;
	std::string res = str;
	res[0] = static_cast<char>( std::toupper( static_cast<unsigned char>( res[0] ) ) );
	for ( size_t i = 1; i < res.size(); ++i ) {
		res[i] = static_cast<char>( std::tolower( static_cast<unsigned char>( res[i] ) ) );
	}
	return res;
}

bool ChatTemplate::load( const char* template_path_or_model_dir ) {
	if ( !template_path_or_model_dir ) {
		return false;
	}

	std::string path( template_path_or_model_dir );
	std::string jinja_path;
	std::string cfg_path;

	if ( path.find( ".jinja" ) != std::string::npos ) {
		jinja_path = path;
	} else {
		std::string dir = path;
		if ( !dir.empty() && dir.back() != '/' ) dir.push_back( '/' );
		jinja_path = dir + "chat_template.jinja";
		cfg_path = dir + "tokenizer_config.json";
	}

	std::ifstream jf( jinja_path );
	if ( jf.is_open() ) {
		std::stringstream buffer;
		buffer << jf.rdbuf();
		template_raw_ = buffer.str();
	} else if ( !cfg_path.empty() ) {
		std::ifstream cf( cfg_path );
		if ( cf.is_open() ) {
			nlohmann::json cfg = nlohmann::json::parse( cf, nullptr, false );
			if ( !cfg.is_discarded() && cfg.contains( "chat_template" ) ) {
				if ( cfg["chat_template"].is_string() ) {
					template_raw_ = cfg["chat_template"].get<std::string>();
				}
			}
		}
	}

	// Detect model family
	if ( template_raw_.find( "[gMASK]" ) != std::string::npos || template_raw_.find( "<sop>" ) != std::string::npos || template_raw_.find( "<|observation|>" ) != std::string::npos ) {
		model_family_ = ModelFamily::GLM;
	} else if ( template_raw_.find( "<|im_start|>" ) != std::string::npos || template_raw_.find( "<|im_end|>" ) != std::string::npos ) {
		model_family_ = ModelFamily::QWEN;
	} else {
		if ( path.find( "glm" ) != std::string::npos || path.find( "GLM" ) != std::string::npos ) {
			model_family_ = ModelFamily::GLM;
		} else if ( path.find( "qwen" ) != std::string::npos || path.find( "Qwen" ) != std::string::npos ) {
			model_family_ = ModelFamily::QWEN;
		} else {
			model_family_ = ModelFamily::UNKNOWN;
		}
	}

	return model_family_ != ModelFamily::UNKNOWN;
}

std::string ChatTemplate::render_qwen( const std::vector<ChatMessage>& messages, const ChatTemplateOptions& options ) const {
	std::string out;
	std::string reasoning_instructions;

	if ( options.enable_thinking ) {
		if ( options.reasoning_effort == "xhigh" ) {
			reasoning_instructions = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
		} else if ( options.reasoning_effort == "low" ) {
			reasoning_instructions = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";
		}
	}

	bool has_system = ( !messages.empty() && messages[0].role == "system" );
	if ( has_system ) {
		std::string content = trim_string( messages[0].content );
		if ( !content.empty() ) {
			out += "<|im_start|>system\n";
			if ( !reasoning_instructions.empty() ) {
				out += reasoning_instructions + "\n\n";
			}
			out += content + "<|im_end|>\n";
		} else if ( !reasoning_instructions.empty() ) {
			out += "<|im_start|>system\n" + reasoning_instructions + "<|im_end|>\n";
		}
	} else if ( !reasoning_instructions.empty() ) {
		out += "<|im_start|>system\n" + reasoning_instructions + "<|im_end|>\n";
	}

	for ( size_t i = ( has_system ? 1 : 0 ); i < messages.size(); ++i ) {
		const auto& m = messages[i];
		std::string content = trim_string( m.content );

		if ( m.role == "user" ) {
			out += "<|im_start|>user\n" + content + "<|im_end|>\n";
		} else if ( m.role == "assistant" ) {
			out += "<|im_start|>assistant\n";
			out += content + "<|im_end|>\n";
		}
	}

	if ( options.add_generation_prompt ) {
		out += "<|im_start|>assistant\n";
		if ( !options.enable_thinking ) {
			out += "<think>\n\n</think>\n\n";
		} else {
			out += "<think>\n";
		}
	}

	return out;
}

std::string ChatTemplate::render_glm( const std::vector<ChatMessage>& messages, const ChatTemplateOptions& options ) const {
	std::string out = "[gMASK]<sop>";

	std::string eff = options.reasoning_effort;
	if ( eff != "low" && eff != "high" ) {
		eff = "max";
	}
	out += "<|system|>Reasoning Effort: " + capitalize_string( eff );

	for ( const auto& m : messages ) {
		if ( m.role == "system" ) {
			out += "<|system|>" + m.content;
		} else if ( m.role == "user" ) {
			out += "<|user|>" + m.content;
		} else if ( m.role == "assistant" ) {
			out += "<|assistant|><think></think>" + trim_string( m.content );
		}
	}

	if ( options.add_generation_prompt ) {
		out += "<|assistant|><think>";
	}

	return out;
}

std::string ChatTemplate::render( const std::vector<ChatMessage>& messages, const ChatTemplateOptions& options ) const {
	switch ( model_family_ ) {
		case ModelFamily::QWEN: {
			return render_qwen( messages, options );
		} break;
		case ModelFamily::GLM: {
			return render_glm( messages, options );
		} break;
		default: {
			return render_qwen( messages, options );
		} break;
	}
}

} // namespace engine::loader
