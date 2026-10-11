#include "engine/video/h3_vae.h"
#include "engine/core/log.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <string>

namespace engine::video {

bool h3_read_vae_params( const char* dir, H3VaeParams& out ) {
	std::ifstream f( std::string( dir ) + "/config.json" );
	if ( !f ) {
		LOG_ERROR( "H3 VAE: cannot open %s/config.json", dir );
		return false;
	}
	std::stringstream ss;
	ss << f.rdbuf();
	const auto j = nlohmann::json::parse( ss.str(), nullptr, false );
	if ( j.is_discarded() || !j.contains( "latents_mean" ) || !j.contains( "latents_std" ) ) {
		LOG_ERROR( "H3 VAE: bad config.json in %s", dir );
		return false;
	}
	out.clip_length = j.value( "vae_clip_length", out.clip_length );
	out.token_drop = j.value( "vae_token_drop", out.token_drop );
	out.tile_size = j.value( "vae_tile_size", out.tile_size );
	out.tile_overlap_min = j.value( "vae_tile_overlap_min", out.tile_overlap_min );
	out.mean = j["latents_mean"].get<std::vector<float>>();
	out.stdev = j["latents_std"].get<std::vector<float>>();
	return out.mean.size() == 24 && out.stdev.size() == 24;
}

} // namespace engine::video
