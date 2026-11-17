// The vision tower on the host CPU (see host_vision.h): picks the tower of the checkpoint's architecture
#include "engine/models/host_vision.h"
#include "engine/models/glm5_vision.h"
#include "engine/models/deepseek_v41_vision.h"
#include "engine/models/qwen3_vision.h"
#include "engine/kernels/cpu/cpu_util.h"
#include <cctype>

namespace engine::models {

std::unique_ptr<VisionTower> make_host_vision_tower( const char* model_dir, const char* mmproj_dir, int max_tokens, const std::vector<int>& cpus, std::string* error ) {
	if ( !engine::kernels::cpu::avx2_available() ) {
		*error = "the host vision tower needs AVX2, FMA and F16C";
		return nullptr;
	}
	if ( max_tokens <= 0 ) {
		*error = "the host vision tower needs a positive token limit";
		return nullptr;
	}
	const nlohmann::json cfg = read_json( std::string( model_dir ) + "/config.json" );
	if ( cfg.is_discarded() ) {
		*error = std::string( model_dir ) + "/config.json is missing or not JSON";
		return nullptr;
	}
	std::string arch = cfg.value( "model_type", std::string() );
	if ( cfg.contains( "architectures" ) && cfg["architectures"].is_array() && !cfg["architectures"].empty() && cfg["architectures"][0].is_string() )
		arch += " " + cfg["architectures"][0].get<std::string>();
	for ( char& c : arch ) c = static_cast<char>( std::tolower( static_cast<unsigned char>( c ) ) );
	if ( arch.find( "glm" ) != std::string::npos ) return make_host_glm5_tower( model_dir, mmproj_dir, max_tokens, cpus, error );
	if ( arch.find( "deepseek" ) != std::string::npos ) return make_host_dsv41_tower( model_dir, mmproj_dir, max_tokens, cpus, error );
	if ( arch.find( "qwen" ) != std::string::npos ) return make_host_qwen3_tower( model_dir, mmproj_dir, max_tokens, cpus, error );
	*error = "no host vision tower for the architecture '" + arch + "' yet (GLM-5, DeepSeek-V4.1, Qwen3-VL-style)";
	return nullptr;
}

} // namespace engine::models
