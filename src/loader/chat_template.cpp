#include "engine/loader/chat_template.h"
#include "engine/core/log.h"

#define JSON_NOEXCEPTION 1
#include <nlohmann/json.hpp>

#include <cstdio>
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

	if ( model_family_ == ModelFamily::UNKNOWN || template_raw_.empty() ) return false;
	std::string err;
	if ( !template_.compile( template_raw_, &err ) ) {
		LOG_ERROR( "chat template does not compile: %s\n", err.c_str() );
		return false;
	}
	return true;
}

bool ChatTemplate::render_json( const std::string& vars_json, std::string* out, std::string* error ) const {
	const jinja::Value vars = jinja::from_json( vars_json );
	if ( vars.kind != jinja::Value::Kind::Map ) {
		if ( error ) *error = "chat template variables are not a JSON object";
		return false;
	}
	return template_.render( vars, out, error );
}

std::string ChatTemplate::render( const std::vector<ChatMessage>& messages, const ChatTemplateOptions& options ) const {
	nlohmann::ordered_json vars;
	vars["messages"] = nlohmann::ordered_json::array();
	for ( const auto& m : messages ) vars["messages"].push_back( { { "role", m.role }, { "content", m.content } } );
	vars["add_generation_prompt"] = options.add_generation_prompt;
	vars["enable_thinking"] = options.enable_thinking;
	vars["reasoning_effort"] = options.reasoning_effort;
	std::string out, err;
	if ( !render_json( vars.dump(), &out, &err ) ) return std::string();
	return out;
}

} // namespace engine::loader
