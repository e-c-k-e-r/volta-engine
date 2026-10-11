#include "jobs.h"
#include "engine/core/log.h"
#include <sys/stat.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <sstream>

namespace engine::server {

using nlohmann::json;

double wall_seconds() {
	return std::chrono::duration<double>( std::chrono::system_clock::now().time_since_epoch() ).count();
}

namespace {

std::string read_file( const std::string& path ) {
	std::ifstream f( path, std::ios::binary );
	std::stringstream ss;
	ss << f.rdbuf();
	return ss.str();
}

void make_dirs( const std::string& p ) {
	std::string cur;
	for ( size_t i = 0; i <= p.size(); ++i ) {
		if ( ( i == p.size() || p[i] == '/' ) && !cur.empty() ) ::mkdir( cur.c_str(), 0755 );
		if ( i < p.size() ) cur += p[i];
	}
}

double number_or( const json& o, const char* key, double fallback ) {
	return o.contains( key ) && o[key].is_number() ? o[key].get<double>() : fallback;
}

} // namespace

json JobStore::describe_locked( const Job& j ) const {
	json o;
	o["id"] = j.id;
	o["state"] = j.state;
	o["stage"] = j.stage;
	o["step"] = j.step;
	o["steps"] = j.steps;
	o["progress"] = j.progress;
	o["params"] = j.params;
	o["elapsed"] = j.started > 0 ? ( j.finished > 0 ? j.finished : wall_seconds() ) - j.started : 0.0;
	o["created"] = j.created;
	if ( !j.error.empty() ) o["error"] = j.error;
	if ( j.state == "done" ) {
		o["t_total"] = j.t_total;
		o["t_text"] = j.t_text;
		o["t_denoise"] = j.t_denoise;
		o["t_decode"] = j.t_decode;
	}
	return o;
}

void JobStore::save_locked( const Job& j ) const {
	std::ofstream( j.dir + "/meta.json" ) << describe_locked( j ).dump();
}

void JobStore::init( const std::string& out_dir ) {
	out_dir_ = out_dir;
	make_dirs( out_dir_ + "/jobs" );
	DIR* d = ::opendir( ( out_dir_ + "/jobs" ).c_str() );
	if ( !d ) return;
	std::lock_guard<std::mutex> lk( mu_ );
	while ( dirent* e = ::readdir( d ) ) {
		const int id = std::atoi( e->d_name );
		if ( id <= 0 ) continue;
		const std::string dir = out_dir_ + "/jobs/" + e->d_name;
		const std::string meta = read_file( dir + "/meta.json" );
		if ( meta.empty() ) continue;
		const json o = json::parse( meta, nullptr, false );
		if ( o.is_discarded() || !o.is_object() ) continue;
		auto j = std::make_shared<Job>();
		j->id = id;
		j->dir = dir;
		j->params = o.contains( "params" ) ? o["params"] : json::object();
		const std::string st = o.contains( "state" ) && o["state"].is_string() ? o["state"].get<std::string>() : "failed";
		j->state = ( st == "queued" || st == "running" ) ? "failed" : st;
		if ( j->state == "failed" && st != "failed" ) j->error = "server restarted while the job was active";
		if ( o.contains( "error" ) && o["error"].is_string() ) j->error = o["error"].get<std::string>();
		j->progress = j->state == "done" ? 1.0 : 0.0;
		j->t_total = number_or( o, "t_total", 0 );
		j->t_text = number_or( o, "t_text", 0 );
		j->t_denoise = number_or( o, "t_denoise", 0 );
		j->t_decode = number_or( o, "t_decode", 0 );
		j->created = number_or( o, "created", 0 );
		j->finished = j->created; // only used to find the newest finished job
		jobs_[id] = j;
		next_id_ = std::max( next_id_, id + 1 );
	}
	::closedir( d );
	LOG_INFO( "Video jobs: %zu in the history (%s)\n", jobs_.size(), out_dir_.c_str() );
}

JobPtr JobStore::create() {
	auto j = std::make_shared<Job>();
	{
		std::lock_guard<std::mutex> lk( mu_ );
		j->id = next_id_++;
		jobs_[j->id] = j;
	}
	j->dir = out_dir_ + "/jobs/" + std::to_string( j->id );
	make_dirs( j->dir );
	return j;
}

void JobStore::enqueue( const JobPtr& j ) {
	{
		std::lock_guard<std::mutex> lk( mu_ );
		j->created = wall_seconds();
		queue_.push_back( j->id );
		save_locked( *j );
	}
	cv_.notify_one();
}

JobPtr JobStore::find( int id ) const {
	std::lock_guard<std::mutex> lk( mu_ );
	auto it = jobs_.find( id );
	return it == jobs_.end() ? nullptr : it->second;
}

json JobStore::describe( const JobPtr& j ) const {
	std::lock_guard<std::mutex> lk( mu_ );
	return describe_locked( *j );
}

json JobStore::list() const {
	json arr = json::array();
	std::lock_guard<std::mutex> lk( mu_ );
	for ( auto it = jobs_.rbegin(); it != jobs_.rend(); ++it ) arr.push_back( describe_locked( *it->second ) );
	return arr;
}

void JobStore::cancel( const JobPtr& j ) {
	std::lock_guard<std::mutex> lk( mu_ );
	j->cancel = true;
	if ( j->state == "queued" ) {
		j->state = "cancelled";
		save_locked( *j );
	}
}

bool JobStore::remove( const JobPtr& j, std::string* error ) {
	std::lock_guard<std::mutex> lk( mu_ );
	if ( j->state == "running" || j->state == "queued" ) {
		*error = "cancel the job first";
		return false;
	}
	const std::string cmd = "rm -rf '" + j->dir + "'";
	( void )!std::system( cmd.c_str() );
	jobs_.erase( j->id );
	return true;
}

JobPtr JobStore::next() {
	std::unique_lock<std::mutex> lk( mu_ );
	for ( ;; ) {
		cv_.wait( lk, [this] { return stopping_ || !queue_.empty(); } );
		if ( stopping_ ) return nullptr;
		const int id = queue_.front();
		queue_.pop_front();
		auto it = jobs_.find( id );
		if ( it == jobs_.end() || it->second->state != "queued" ) continue; // cancelled or deleted while waiting
		JobPtr j = it->second;
		j->state = "running";
		j->started = wall_seconds();
		j->stage = "images";
		save_locked( *j );
		return j;
	}
}

void JobStore::finish( const JobPtr& j, bool ok, bool cancelled, const engine::video::H3GenResult& res, double started_at ) {
	std::lock_guard<std::mutex> lk( mu_ );
	j->finished = wall_seconds();
	j->t_total = j->finished - started_at;
	j->t_text = res.t_text;
	j->t_denoise = res.t_denoise;
	j->t_decode = res.t_decode;
	if ( ok ) {
		j->state = "done";
		j->progress = 1.0;
	} else if ( cancelled || j->cancel.load() ) {
		j->state = "cancelled";
	} else {
		j->state = "failed";
		j->error = "generation failed; see the server log";
	}
	save_locked( *j );
}

void JobStore::stop() {
	{
		std::lock_guard<std::mutex> lk( mu_ );
		stopping_ = true;
	}
	cv_.notify_all();
}

JobStore::Counts JobStore::counts() const {
	Counts c;
	std::lock_guard<std::mutex> lk( mu_ );
	double newest = 0;
	for ( const auto& [id, j] : jobs_ ) {
		if ( j->state == "queued" ) ++c.queued;
		else if ( j->state == "running" ) ++c.running;
		else if ( j->state == "done" ) {
			++c.done;
			if ( j->finished >= newest ) {
				newest = j->finished;
				c.last_total_s = j->t_total;
				c.last_denoise_s = j->t_denoise;
			}
		}
		else if ( j->state == "failed" ) ++c.failed;
		else if ( j->state == "cancelled" ) ++c.cancelled;
	}
	return c;
}

} // namespace engine::server
