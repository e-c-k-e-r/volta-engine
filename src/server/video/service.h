#pragma once

// Internal declarations of the video service (shared by service.cpp and api.cpp).
#include "adapters.h"
#include "engine/server/config.h"
#include "engine/server/system_stats.h"
#include "engine/server/video.h"
#include "jobs.h"
#include <atomic>
#include <thread>

namespace engine::server {

struct VideoService::Impl {
	ServerConfig cfg;
	AdapterCatalog adapters;
	JobStore jobs;
	SystemStats sys;
	double started_at = 0;
	std::unique_ptr<engine::video::H3Engine> engine; // the models, resident between jobs unless video.resident is off
	std::thread worker;

	void run_loop();                 // the worker: takes queued jobs one at a time
	void run_job( const JobPtr& j ); // builds the pipeline request of a job and runs it
	void register_routes( Router& router );

	// handlers (api.cpp)
	void handle_health( const ConnPtr& c, const HttpRequest& r );
	void handle_metrics( const ConnPtr& c, const HttpRequest& r );
	void handle_metrics_prometheus( const ConnPtr& c, const HttpRequest& r );
	void handle_page( const ConnPtr& c, const HttpRequest& r );
	void handle_info( const ConnPtr& c, const HttpRequest& r );
	void handle_list( const ConnPtr& c, const HttpRequest& r );
	void handle_submit( const ConnPtr& c, const HttpRequest& r );
	void handle_job( const ConnPtr& c, const HttpRequest& r ); // /api/jobs/<id>[/video|/thumb|/cancel]
};

} // namespace engine::server
