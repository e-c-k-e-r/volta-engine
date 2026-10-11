#include "service.h"
#include "engine/core/log.h"
#include <cstdlib>
#include <cstring>

namespace engine::server {

VideoService::VideoService() : p_( new Impl ) {}

VideoService::~VideoService() {
	stop();
}

bool VideoService::init( const ServerConfig& config ) {
	Impl& s = *p_;
	s.cfg = config;
	s.started_at = wall_seconds();
	s.adapters.scan( s.cfg.video.lora_dirs, !s.cfg.video.dit_ref2va.empty() );
	s.jobs.init( s.cfg.video.out_dir );
	engine::video::H3Paths paths;
	paths.root = s.cfg.model_dir;
	paths.dit = s.cfg.video.dit;
	paths.dit_ref2va = s.cfg.video.dit_ref2va;
	paths.devices = s.cfg.video.devices;
	paths.device = s.cfg.video.devices.empty() ? 0 : s.cfg.video.devices[0];
	paths.resident_rows = s.cfg.video.max_rows;
	s.engine = std::make_unique<engine::video::H3Engine>( paths, s.cfg.video.resident );
	s.worker = std::thread( [this] { p_->run_loop(); } );
	LOG_INFO( "MiniMax-H3 service: %zu adapter(s), %zu GPU(s), references %s, models %s, jobs in %s\n", s.adapters.all().size(), s.cfg.video.devices.size(),
		s.cfg.video.dit_ref2va.empty() ? "off" : "on", s.cfg.video.resident ? "resident" : "loaded per job", s.cfg.video.out_dir.c_str() );
	return true;
}

void VideoService::register_routes( Router& router ) {
	p_->register_routes( router );
}

void VideoService::stop() {
	if ( !p_ ) return;
	p_->jobs.stop();
	// a running job is not waited for (it can take minutes): the process exits right after the server stops
	if ( p_->worker.joinable() ) p_->worker.detach();
}

void VideoService::Impl::register_routes( Router& router ) {
	router.add( "GET", "/health", [this]( const ConnPtr& c, const HttpRequest& r ) { handle_health( c, r ); } );
	router.add( "GET", "/metrics", [this]( const ConnPtr& c, const HttpRequest& r ) { handle_metrics( c, r ); } );
	router.add( "GET", "/metrics/prometheus", [this]( const ConnPtr& c, const HttpRequest& r ) { handle_metrics_prometheus( c, r ); } );
	router.add( "GET", "/", [this]( const ConnPtr& c, const HttpRequest& r ) { handle_page( c, r ); } );
	router.add( "GET", "/api/info", [this]( const ConnPtr& c, const HttpRequest& r ) { handle_info( c, r ); } );
	router.add( "GET", "/api/jobs", [this]( const ConnPtr& c, const HttpRequest& r ) { handle_list( c, r ); } );
	// uploads are decoded and written to disk, videos are streamed: neither belongs on the epoll thread
	router.add( "POST", "/api/jobs", [this]( const ConnPtr& c, const HttpRequest& r ) { handle_submit( c, r ); }, true );
	router.add_prefix( "*", "/api/jobs/", [this]( const ConnPtr& c, const HttpRequest& r ) { handle_job( c, r ); }, true );
}

void VideoService::Impl::run_loop() {
	const double idle_s = cfg.video.resident ? cfg.video.resident_idle_s : 0;
	double wait_s = idle_s;
	for ( ;; ) {
		bool idle = false;
		JobPtr j = jobs.next( wait_s, &idle );
		if ( idle ) { // nothing queued for resident_idle_s: give the GPUs back, then wait for the next job without a timeout
			LOG_INFO( "MiniMax-H3 service: idle for %d s, freeing the resident transformer\n", cfg.video.resident_idle_s );
			engine->release();
			wait_s = 0;
			continue;
		}
		if ( !j ) break;
		run_job( j );
		wait_s = idle_s;
	}
}

void VideoService::Impl::run_job( const JobPtr& j ) {
	using namespace engine::video;
	const double t0 = wall_seconds();
	H3GenRequest r;
	const nlohmann::json& p = j->params;
	r.prompt = p["prompt"].get<std::string>();
	r.width = p["width"].get<int>();
	r.height = p["height"].get<int>();
	r.frames = p["frames"].get<int>();
	r.steps = j->steps_total;
	r.seed = p["seed"].get<uint64_t>();
	r.loras = j->loras;
	r.images = j->images;
	r.ref_images = j->ref_images;
	r.ref_videos = j->ref_videos;
	r.ref_audios = j->ref_audios;
	r.frame_indices = j->frame_indices;
	r.out = j->dir + "/result.mp4";
	r.cancel = &j->cancel;
	r.progress = [this, j]( const char* stage, int done, int total ) {
		jobs.update( j, [&]( Job& job ) {
			job.stage = stage;
			if ( !std::strcmp( stage, "denoise" ) ) {
				job.step = done;
				job.steps = total;
				job.progress = 0.10 + 0.70 * ( total > 0 ? static_cast<double>( done ) / total : 0.0 );
			} else if ( !std::strcmp( stage, "images" ) ) job.progress = 0.02;
			else if ( !std::strcmp( stage, "text" ) ) job.progress = 0.04;
			else if ( !std::strcmp( stage, "load" ) ) job.progress = 0.08;
			else if ( !std::strcmp( stage, "decode" ) ) job.progress = 0.82;
			else if ( !std::strcmp( stage, "mux" ) ) job.progress = 0.97;
		} );
	};
	H3GenResult res;
	const bool ok = engine->generate( r, &res );
	if ( ok ) {
		const std::string cmd = "ffmpeg -y -loglevel error -i '" + j->dir + "/result.mp4' -vf \"select=eq(n\\,12),scale=640:-1\" -frames:v 1 '" + j->dir + "/thumb.jpg' 2>/dev/null";
		( void )!std::system( cmd.c_str() );
	}
	for ( const auto* files : { &j->images, &j->ref_images, &j->ref_videos, &j->ref_audios } ) {
		for ( const std::string& f : *files ) std::remove( f.c_str() );
	}
	jobs.finish( j, ok, res.cancelled, res, t0 );
}

} // namespace engine::server
