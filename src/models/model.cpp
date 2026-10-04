#include "engine/models/model.h"
#include "engine/models/qwen38_27b.h"
#include "engine/models/glm53_flash.h"

namespace engine::models {

std::unique_ptr<ModelPipeline> create_model_pipeline( const std::string& model_type ) {
	if ( model_type == "qwen38" || model_type == "qwen" || model_type == "qwen-27b" ) {
		return std::make_unique<Qwen38Pipeline>();
	}
	if ( model_type == "glm53" || model_type == "glm" || model_type == "glm-flash" ) {
		return std::make_unique<GLM53Pipeline>();
	}
	return nullptr;
}

} // namespace engine::models
