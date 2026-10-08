#pragma once

#include "engine/config.h"
#include "engine/loader/jinja.h"
#include <string>
#include <vector>

namespace engine::loader {

struct ChatMessage {
	std::string role;
	std::string content;
};

struct ChatTemplateOptions {
	bool add_generation_prompt = true;
	bool enable_thinking = true;
	std::string reasoning_effort = "low";
};

enum class ModelFamily {
	UNKNOWN,
	QWEN,
	GLM
};

class ChatTemplate {
public:
	ChatTemplate();
	~ChatTemplate();

	// Load chat template from template file path or model directory
	bool load( const char* template_path_or_model_dir );

	// Renders the checkpoint's own chat template. vars_json is the template context: {"messages": [...], "tools": [...],
	// "add_generation_prompt": bool, "enable_thinking": bool, "reasoning_effort": str, ...} (HF apply_chat_template names).
	bool render_json( const std::string& vars_json, std::string* out, std::string* error ) const;

	// Plain (role, content) messages; returns an empty string if the template rejects them.
	std::string render( const std::vector<ChatMessage>& messages, const ChatTemplateOptions& options = {} ) const;

	ModelFamily model_family() const { return model_family_; }

private:
	ModelFamily model_family_ = ModelFamily::UNKNOWN;
	std::string template_raw_;
	jinja::Template template_;
};

} // namespace engine::loader
