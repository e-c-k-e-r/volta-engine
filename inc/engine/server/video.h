#pragma once

#include "engine/server/config.h"
#include "engine/server/http.h"
#include <memory>

// MiniMax-H3 generation service of volta_server: a job queue with one worker thread that owns the GPUs, the job API and the web page. Started for checkpoint
// directories that are an H3 pipeline (model_type "minimax_h3"); see docs/video.md.
namespace engine::server {

class VideoService {
public:
	VideoService();
	~VideoService();

	// Scans the adapters, loads the job history and starts the worker.
	bool init( const ServerConfig& config );
	// Adds `/`, `/api/info` and `/api/jobs...` to the router.
	void register_routes( Router& router );
	void stop();

private:
	struct Impl;
	std::unique_ptr<Impl> p_;
};

} // namespace engine::server
