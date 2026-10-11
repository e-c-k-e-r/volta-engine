// Health and metrics of the video service: queue and job counters, the last job's timings and the VRAM of the GPUs the transformer runs on.
#include "service.h"
#include <cstdio>

namespace engine::server {

using nlohmann::json;

namespace {

const char* kService = "minimax_h3";

} // namespace

void VideoService::Impl::handle_health( const ConnPtr& c, const HttpRequest& ) {
	const JobStore::Counts n = jobs.counts();
	send_json( c, 200, json{ { "status", "ok" }, { "service", kService }, { "queued", n.queued }, { "running", n.running } }.dump() );
}

void VideoService::Impl::handle_metrics( const ConnPtr& c, const HttpRequest& ) {
	const JobStore::Counts n = jobs.counts();
	std::vector<GpuReading> gpus;
	HostReading host;
	sys.read( &gpus, &host );
	json g = json::array();
	for ( int d : cfg.video.devices ) {
		if ( d >= 0 && d < static_cast<int>( gpus.size() ) ) g.push_back( { { "index", d }, { "mem_used", gpus[d].mem_used }, { "mem_total", gpus[d].mem_total }, { "power_w", gpus[d].power_w }, { "temp_c", gpus[d].temp_c } } );
	}
	json m;
	m["uptime_seconds"] = wall_seconds() - started_at;
	m["jobs"] = { { "queued", n.queued }, { "running", n.running }, { "done", n.done }, { "failed", n.failed }, { "cancelled", n.cancelled } };
	m["last_job"] = { { "total_seconds", n.last_total_s }, { "denoise_seconds", n.last_denoise_s } };
	m["gpus"] = g;
	send_json( c, 200, m.dump( 2 ) + "\n" );
}

void VideoService::Impl::handle_metrics_prometheus( const ConnPtr& c, const HttpRequest& ) {
	const JobStore::Counts n = jobs.counts();
	std::vector<GpuReading> gpus;
	HostReading host;
	sys.read( &gpus, &host );
	std::string out;
	char buf[256];
	auto line = [&]( const char* name, const char* type, double v, const char* labels = "" ) {
		std::snprintf( buf, sizeof( buf ), "# TYPE %s %s\n%s%s %.6g\n", name, type, name, labels, v );
		out += buf;
	};
	line( "volta_video_uptime_seconds", "gauge", wall_seconds() - started_at );
	line( "volta_video_jobs_queued", "gauge", n.queued );
	line( "volta_video_jobs_running", "gauge", n.running );
	line( "volta_video_jobs_done_total", "counter", n.done );
	line( "volta_video_jobs_failed_total", "counter", n.failed );
	line( "volta_video_jobs_cancelled_total", "counter", n.cancelled );
	line( "volta_video_last_job_seconds", "gauge", n.last_total_s );
	line( "volta_video_last_denoise_seconds", "gauge", n.last_denoise_s );
	for ( int d : cfg.video.devices ) {
		if ( d < 0 || d >= static_cast<int>( gpus.size() ) ) continue;
		std::snprintf( buf, sizeof( buf ), "{gpu=\"%d\"}", d );
		line( "volta_video_gpu_memory_used_bytes", "gauge", static_cast<double>( gpus[d].mem_used ), buf );
	}
	send_response( c, 200, "text/plain; version=0.0.4", out );
}

} // namespace engine::server
