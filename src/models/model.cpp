#include "engine/models/model.h"
#include "engine/models/qwen3_5.h"
#include "engine/models/glm5_next.h"

namespace engine::models {

std::unique_ptr<ModelPipeline> create_model_pipeline( const std::string& model_type ) {
	if ( model_type == "qwen3_5" ) {
		return std::make_unique<Qwen3_5Pipeline>();
	}
	if ( model_type == "glm5_next" ) {
		return std::make_unique<Glm5NextPipeline>();
	}
	return nullptr;
}

bool is_known_model_type( const std::string& model_type ) {
	return model_type == "qwen3_5" || model_type == "glm5_next";
}

} // namespace engine::models
