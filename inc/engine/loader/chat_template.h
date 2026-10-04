#pragma once

#include "engine/config.h"
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

	// Render messages into a formatted prompt string matching official Jinja logic
	std::string render( const std::vector<ChatMessage>& messages, const ChatTemplateOptions& options = {} ) const;

	ModelFamily model_family() const { return model_family_; }

private:
	std::string render_qwen( const std::vector<ChatMessage>& messages, const ChatTemplateOptions& options ) const;
	std::string render_glm( const std::vector<ChatMessage>& messages, const ChatTemplateOptions& options ) const;

	ModelFamily model_family_ = ModelFamily::UNKNOWN;
	std::string template_raw_;
};

} // namespace engine::loader
