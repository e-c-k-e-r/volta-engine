#include "engine/models/model.h"
#include "engine/models/qwen38_27b.h"
#include "engine/models/glm53_flash.h"

namespace engine::models {

std::unique_ptr<ModelPipeline> create_model_pipeline( const std::string& model_type ) {
	if ( model_type == "qwen3_5" ) {
		return std::make_unique<Qwen38Pipeline>();
	}
	if ( model_type == "glm5_next" ) {
		return std::make_unique<GLM53Pipeline>();
	}
	return nullptr;
}

bool is_known_model_type( const std::string& model_type ) {
	return model_type == "qwen3_5" || model_type == "glm5_next";
}

} // namespace engine::models
