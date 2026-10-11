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
	s.adapters.scan( s.cfg.video.lora_dirs, !s.cfg.video.dit_ref2va.empty() );
	s.jobs.init( s.cfg.video.out_dir );
	s.worker = std::thread( [this] { p_->run_loop(); } );
	LOG_INFO( "MiniMax-H3 service: %zu adapter(s), %zu GPU(s), references %s, jobs in %s\n", s.adapters.all().size(), s.cfg.video.devices.size(),
		s.cfg.video.dit_ref2va.empty() ? "off" : "on", s.cfg.video.out_dir.c_str() );
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

VideoService::Stats VideoService::stats() const {
	const JobStore::Counts c = p_->jobs.counts();
	return { c.queued, c.running, c.done, c.failed, c.cancelled, c.last_total_s, c.last_denoise_s };
}

void VideoService::Impl::register_routes( Router& router ) {
	router.add( "GET", "/", [this]( const ConnPtr& c, const HttpRequest& r ) { handle_page( c, r ); } );
	router.add( "GET", "/api/info", [this]( const ConnPtr& c, const HttpRequest& r ) { handle_info( c, r ); } );
	router.add( "GET", "/api/jobs", [this]( const ConnPtr& c, const HttpRequest& r ) { handle_list( c, r ); } );
	// uploads are decoded and written to disk, videos are streamed: neither belongs on the epoll thread
	router.add( "POST", "/api/jobs", [this]( const ConnPtr& c, const HttpRequest& r ) { handle_submit( c, r ); }, true );
	router.add_prefix( "*", "/api/jobs/", [this]( const ConnPtr& c, const HttpRequest& r ) { handle_job( c, r ); }, true );
}

void VideoService::Impl::run_loop() {
	while ( JobPtr j = jobs.next() ) run_job( j );
}

void VideoService::Impl::run_job( const JobPtr& j ) {
	using namespace engine::video;
	const double t0 = wall_seconds();
	H3Paths paths;
	paths.root = cfg.model_dir;
	paths.dit = cfg.video.dit;
	paths.dit_ref2va = cfg.video.dit_ref2va;
	paths.devices = cfg.video.devices;
	paths.device = cfg.video.devices.empty() ? 0 : cfg.video.devices[0];
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
	const bool ok = h3_generate( paths, r, &res );
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
