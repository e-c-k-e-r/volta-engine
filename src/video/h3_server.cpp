// h3_server: job server and web UI for MiniMax-H3 generation.
//   h3_server --root <FL2VA dir> --dit <comfy int8 checkpoint> [--lora-dir DIR]... [--out-dir DIR] [--devices 0,1,2,3] [--port 8200] [--host 127.0.0.1]
// One worker thread owns the GPUs and runs the queued jobs one after the other; the API is JSON over HTTP (no authentication: keep it on localhost or a
// trusted network).
#include "engine/core/log.h"
#include "engine/loader/image.h"
#include "engine/video/h3_pipeline.h"
#include "h3_ui.h"
#include <nlohmann/json.hpp>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <dirent.h>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using nlohmann::json;
using namespace engine::video;

namespace {

double now_s() { return std::chrono::duration<double>( std::chrono::system_clock::now().time_since_epoch() ).count(); }

struct LoraInfo {
	std::string file, path, label, format, task;
	bool turbo = false;
	long mb = 0;
};

struct Job {
	int id = 0;
	std::string state = "queued", stage, error, dir;
	json params;
	int step = 0, steps = 0;
	double progress = 0, created = 0, started = 0, finished = 0, t_text = 0, t_load = 0, t_denoise = 0, t_decode = 0, t_total = 0;
	std::atomic<bool> cancel{ false };
	std::vector<H3Lora> loras;
	std::vector<std::string> images;
	std::vector<int> frame_indices;
	int steps_total = 0;
};

struct Config {
	std::string root, dit, out_dir = "h3_jobs", host = "127.0.0.1";
	std::vector<std::string> lora_dirs;
	std::vector<int> devices{ 0 };
	int port = 8200;
	int max_rows = 36000;
};

Config g_cfg;
std::vector<LoraInfo> g_loras;
std::mutex g_mu;
std::condition_variable g_cv;
std::map<int, std::shared_ptr<Job>> g_jobs;
std::deque<int> g_queue;
int g_next_id = 1;

std::string read_file( const std::string& path ) {
	std::ifstream f( path, std::ios::binary );
	std::stringstream ss;
	ss << f.rdbuf();
	return ss.str();
}

bool file_exists( const std::string& p ) { struct stat st; return ::stat( p.c_str(), &st ) == 0; }

void mkdirs( const std::string& p ) {
	std::string cur;
	for ( size_t i = 0; i <= p.size(); ++i ) {
		if ( i == p.size() || p[i] == '/' ) { if ( !cur.empty() ) { ::mkdir( cur.c_str(), 0755 ); } }
		if ( i < p.size() ) { cur += p[i]; }
	}
}

// ---- adapters ----------------------------------------------------------------------------------------------------------------------------------------
void scan_loras() {
	g_loras.clear();
	static const std::regex turbo_re( R"(^minimax_h3_(fl2v|ref2v)_turbo_(\d)step_v([\d.]+)(_768p)?(_bf16)?\.safetensors$)" );
	for ( const std::string& dir : g_cfg.lora_dirs ) {
		DIR* d = ::opendir( dir.c_str() );
		if ( !d ) { continue; }
		while ( dirent* e = ::readdir( d ) ) {
			const std::string name = e->d_name;
			if ( name.size() < 12 || name.substr( name.size() - 12 ) != ".safetensors" ) { continue; }
			LoraInfo li;
			li.file = name;
			li.path = dir + "/" + name;
			struct stat st;
			if ( ::stat( li.path.c_str(), &st ) != 0 ) { continue; }
			li.mb = static_cast<long>( st.st_size >> 20 );
			std::smatch m;
			if ( std::regex_match( name, m, turbo_re ) ) {
				if ( m[1] != "fl2v" ) { continue; } // Ref2V adapters need the Ref2VA transformer
				li.turbo = true;
				li.label = m[2].str() + "-step turbo v" + m[3].str() + ( m[4].matched ? " (768p)" : "" ) + " - fast";
				li.format = "diffusers";
			} else {
				li.label = name.substr( 0, name.size() - 12 );
				FILE* f = std::fopen( li.path.c_str(), "rb" );
				uint64_t len = 0;
				std::string hdr;
				if ( f && std::fread( &len, 8, 1, f ) == 1 && len < ( 64u << 20 ) ) {
					hdr.resize( len );
					if ( std::fread( hdr.data(), 1, len, f ) != len ) { hdr.clear(); }
				}
				if ( f ) { std::fclose( f ); }
				if ( hdr.find( "lora_unet_blocks_" ) != std::string::npos ) { li.format = "kohya"; }
				else if ( hdr.find( "diffusion_model.blocks." ) != std::string::npos ) { li.format = "ai-toolkit"; }
				else if ( hdr.find( "transformer_blocks." ) != std::string::npos ) { li.format = "diffusers"; }
				else { continue; } // not an adapter for this model
				if ( name.find( "ref2va" ) != std::string::npos || hdr.find( "\"ss_h3_training_mode\":\"ref2va\"" ) != std::string::npos ) { li.task = "ref2va"; }
			}
			g_loras.push_back( li );
		}
		::closedir( d );
	}
	std::sort( g_loras.begin(), g_loras.end(), []( const LoraInfo& a, const LoraInfo& b ) { return a.turbo != b.turbo ? a.turbo > b.turbo : a.label < b.label; } );
}

const LoraInfo* find_lora( const std::string& file ) {
	for ( const LoraInfo& l : g_loras ) { if ( l.file == file ) { return &l; } }
	return nullptr;
}

// ---- jobs ------------------------------------------------------------------------------------------------------------------------------------------------
json job_json( const Job& j ) {
	json o;
	o["id"] = j.id;
	o["state"] = j.state;
	o["stage"] = j.stage;
	o["step"] = j.step;
	o["steps"] = j.steps;
	o["progress"] = j.progress;
	o["params"] = j.params;
	o["elapsed"] = j.started > 0 ? ( j.finished > 0 ? j.finished : now_s() ) - j.started : 0.0;
	o["created"] = j.created;
	if ( !j.error.empty() ) { o["error"] = j.error; }
	if ( j.state == "done" ) { o["t_total"] = j.t_total; o["t_text"] = j.t_text; o["t_denoise"] = j.t_denoise; o["t_decode"] = j.t_decode; }
	return o;
}

void save_meta( const Job& j ) {
	json o = job_json( j );
	std::ofstream( j.dir + "/meta.json" ) << o.dump();
}

void load_history() {
	DIR* d = ::opendir( ( g_cfg.out_dir + "/jobs" ).c_str() );
	if ( !d ) { return; }
	while ( dirent* e = ::readdir( d ) ) {
		const int id = std::atoi( e->d_name );
		if ( id <= 0 ) { continue; }
		const std::string dir = g_cfg.out_dir + "/jobs/" + e->d_name;
		const std::string meta = read_file( dir + "/meta.json" );
		if ( meta.empty() ) { continue; }
		const json o = json::parse( meta, nullptr, false );
		if ( o.is_discarded() || !o.is_object() ) { continue; }
		auto j = std::make_shared<Job>();
		j->id = id;
		j->dir = dir;
		j->params = o.contains( "params" ) ? o["params"] : json::object();
		const std::string st = o.contains( "state" ) && o["state"].is_string() ? o["state"].get<std::string>() : "failed";
		j->state = ( st == "queued" || st == "running" ) ? "failed" : st;
		if ( j->state == "failed" && st != "failed" ) { j->error = "server restarted while the job was active"; }
		if ( o.contains( "error" ) && o["error"].is_string() ) { j->error = o["error"].get<std::string>(); }
		j->progress = j->state == "done" ? 1.0 : 0.0;
		for ( const char* k : { "t_total", "t_text", "t_denoise", "t_decode", "created" } ) {
			if ( o.contains( k ) && o[k].is_number() ) {
				const double v = o[k].get<double>();
				if ( !std::strcmp( k, "t_total" ) ) { j->t_total = v; } else if ( !std::strcmp( k, "t_text" ) ) { j->t_text = v; }
				else if ( !std::strcmp( k, "t_denoise" ) ) { j->t_denoise = v; } else if ( !std::strcmp( k, "t_decode" ) ) { j->t_decode = v; } else { j->created = v; }
			}
		}
		g_jobs[id] = j;
		g_next_id = std::max( g_next_id, id + 1 );
	}
	::closedir( d );
}

std::string b64_decode( const std::string& in ) {
	std::string out;
	engine::loader::base64_decode( in, &out );
	return out;
}

void run_job( const std::shared_ptr<Job>& j ) {
	const double t0 = now_s();
	H3Paths paths;
	paths.root = g_cfg.root;
	paths.dit = g_cfg.dit;
	paths.devices = g_cfg.devices;
	paths.device = g_cfg.devices.empty() ? 0 : g_cfg.devices[0];
	H3GenRequest r;
	const json& p = j->params;
	r.prompt = p["prompt"].get<std::string>();
	r.width = p["width"].get<int>();
	r.height = p["height"].get<int>();
	r.frames = p["frames"].get<int>();
	r.steps = j->steps_total;
	r.seed = p["seed"].get<uint64_t>();
	r.loras = j->loras;
	r.images = j->images;
	r.frame_indices = j->frame_indices;
	r.out = j->dir + "/result.mp4";
	r.cancel = &j->cancel;
	r.progress = [j]( const char* stage, int done, int total ) {
		std::lock_guard<std::mutex> lk( g_mu );
		j->stage = stage;
		if ( !std::strcmp( stage, "denoise" ) ) {
			j->step = done;
			j->steps = total;
			j->progress = 0.10 + 0.70 * ( total > 0 ? static_cast<double>( done ) / total : 0.0 );
		} else if ( !std::strcmp( stage, "images" ) ) { j->progress = 0.02; }
		else if ( !std::strcmp( stage, "text" ) ) { j->progress = 0.04; }
		else if ( !std::strcmp( stage, "load" ) ) { j->progress = 0.08; }
		else if ( !std::strcmp( stage, "decode" ) ) { j->progress = 0.82; }
		else if ( !std::strcmp( stage, "mux" ) ) { j->progress = 0.97; }
	};
	H3GenResult res;
	const bool ok = h3_generate( paths, r, &res );
	std::lock_guard<std::mutex> lk( g_mu );
	j->finished = now_s();
	j->t_total = j->finished - t0;
	j->t_text = res.t_text;
	j->t_denoise = res.t_denoise;
	j->t_decode = res.t_decode;
	if ( ok ) {
		j->state = "done";
		j->progress = 1.0;
		const std::string cmd = "ffmpeg -y -loglevel error -i '" + j->dir + "/result.mp4' -vf \"select=eq(n\\,12),scale=640:-1\" -frames:v 1 '" + j->dir + "/thumb.jpg' 2>/dev/null";
		( void )!std::system( cmd.c_str() );
	} else if ( res.cancelled || j->cancel.load() ) {
		j->state = "cancelled";
	} else {
		j->state = "failed";
		j->error = "generation failed; see the server log";
	}
	for ( const std::string& im : j->images ) { std::remove( im.c_str() ); }
	save_meta( *j );
}

void worker() {
	for ( ;; ) {
		std::shared_ptr<Job> j;
		{
			std::unique_lock<std::mutex> lk( g_mu );
			g_cv.wait( lk, [] { return !g_queue.empty(); } );
			const int id = g_queue.front();
			g_queue.pop_front();
			j = g_jobs[id];
			if ( !j || j->state != "queued" ) { continue; }
			j->state = "running";
			j->started = now_s();
			j->stage = "images";
			save_meta( *j );
		}
		run_job( j );
	}
}

// ---- HTTP --------------------------------------------------------------------------------------------------------------------------------------------
struct Request {
	std::string method, path, query, body;
	std::map<std::string, std::string> headers;
};

bool send_all( int fd, const char* d, size_t n ) {
	while ( n > 0 ) {
		const ssize_t w = ::send( fd, d, n, MSG_NOSIGNAL );
		if ( w <= 0 ) { return false; }
		d += w;
		n -= static_cast<size_t>( w );
	}
	return true;
}

void respond( int fd, int status, const std::string& type, const std::string& body, const std::string& extra = "" ) {
	const char* reason = status == 200 ? "OK" : status == 206 ? "Partial Content" : status == 400 ? "Bad Request" : status == 404 ? "Not Found" : status == 413 ? "Payload Too Large" : "Error";
	std::string h = "HTTP/1.1 " + std::to_string( status ) + " " + reason + "\r\nContent-Type: " + type + "\r\nContent-Length: " + std::to_string( body.size() ) + "\r\nConnection: close\r\nCache-Control: no-store\r\n" + extra + "\r\n";
	send_all( fd, h.data(), h.size() );
	send_all( fd, body.data(), body.size() );
}

void respond_json( int fd, int status, const json& j ) { respond( fd, status, "application/json", j.dump() ); }
void respond_error( int fd, int status, const std::string& msg ) { respond_json( fd, status, json{ { "error", msg } } ); }

bool read_request( int fd, Request* rq ) {
	std::string buf;
	char tmp[65536];
	size_t header_end = std::string::npos;
	while ( header_end == std::string::npos ) {
		const ssize_t n = ::recv( fd, tmp, sizeof( tmp ), 0 );
		if ( n <= 0 ) { return false; }
		buf.append( tmp, static_cast<size_t>( n ) );
		header_end = buf.find( "\r\n\r\n" );
		if ( buf.size() > ( 1u << 20 ) && header_end == std::string::npos ) { return false; }
	}
	std::istringstream hs( buf.substr( 0, header_end ) );
	std::string line;
	std::getline( hs, line );
	std::istringstream rl( line );
	std::string target, ver;
	rl >> rq->method >> target >> ver;
	const size_t q = target.find( '?' );
	rq->path = target.substr( 0, q );
	if ( q != std::string::npos ) { rq->query = target.substr( q + 1 ); }
	while ( std::getline( hs, line ) ) {
		if ( !line.empty() && line.back() == '\r' ) { line.pop_back(); }
		const size_t c = line.find( ':' );
		if ( c == std::string::npos ) { continue; }
		std::string k = line.substr( 0, c ), v = line.substr( c + 1 );
		while ( !v.empty() && v.front() == ' ' ) { v.erase( v.begin() ); }
		std::transform( k.begin(), k.end(), k.begin(), []( unsigned char ch ) { return std::tolower( ch ); } );
		rq->headers[k] = v;
	}
	size_t want = 0;
	if ( rq->headers.count( "content-length" ) ) { want = static_cast<size_t>( std::strtoull( rq->headers["content-length"].c_str(), nullptr, 10 ) ); }
	if ( want > ( size_t( 160 ) << 20 ) ) { return false; }
	rq->body = buf.substr( header_end + 4 );
	while ( rq->body.size() < want ) {
		const ssize_t n = ::recv( fd, tmp, sizeof( tmp ), 0 );
		if ( n <= 0 ) { return false; }
		rq->body.append( tmp, static_cast<size_t>( n ) );
	}
	return true;
}

void serve_file( int fd, const Request& rq, const std::string& path, const std::string& type ) {
	std::ifstream f( path, std::ios::binary | std::ios::ate );
	if ( !f ) { respond_error( fd, 404, "not found" ); return; }
	const size_t size = static_cast<size_t>( f.tellg() );
	size_t a = 0, b = size ? size - 1 : 0;
	bool range = false;
	auto it = rq.headers.find( "range" );
	if ( it != rq.headers.end() && it->second.compare( 0, 6, "bytes=" ) == 0 ) {
		const std::string spec = it->second.substr( 6 );
		const size_t dash = spec.find( '-' );
		if ( dash != std::string::npos ) {
			if ( dash > 0 ) { a = std::strtoull( spec.c_str(), nullptr, 10 ); }
			if ( dash + 1 < spec.size() ) { b = std::min<size_t>( std::strtoull( spec.c_str() + dash + 1, nullptr, 10 ), size - 1 ); }
			range = a <= b && a < size;
		}
	}
	if ( !range ) { a = 0; b = size ? size - 1 : 0; }
	std::string body( b - a + 1, '\0' );
	f.seekg( static_cast<std::streamoff>( a ) );
	f.read( body.data(), static_cast<std::streamsize>( body.size() ) );
	std::string extra = "Accept-Ranges: bytes\r\n";
	if ( range ) { extra += "Content-Range: bytes " + std::to_string( a ) + "-" + std::to_string( b ) + "/" + std::to_string( size ) + "\r\n"; }
	respond( fd, range ? 206 : 200, type, body, extra );
}

std::string str_field( const json& j, const char* k, const std::string& def = "" ) { return j.contains( k ) && j[k].is_string() ? j[k].get<std::string>() : def; }
double num_field( const json& j, const char* k, double def ) { return j.contains( k ) && j[k].is_number() ? j[k].get<double>() : def; }

void handle_submit( int fd, const Request& rq ) {
	const json b = json::parse( rq.body, nullptr, false );
	if ( b.is_discarded() || !b.is_object() ) { respond_error( fd, 400, "invalid JSON" ); return; }
	auto j = std::make_shared<Job>();
	const std::string prompt = str_field( b, "prompt" );
	const int width = static_cast<int>( num_field( b, "width", 1280 ) ), height = static_cast<int>( num_field( b, "height", 736 ) ), frames = static_cast<int>( num_field( b, "frames", 124 ) );
	int steps = static_cast<int>( num_field( b, "steps", 4 ) );
	const uint64_t seed = static_cast<uint64_t>( num_field( b, "seed", 1 ) );
	if ( prompt.empty() || prompt.size() > 8000 ) { respond_error( fd, 400, "prompt must be 1 to 8000 characters" ); return; }
	if ( width < 256 || height < 256 || width > 2048 || height > 2048 || width % 32 || height % 32 ) { respond_error( fd, 400, "width and height must be multiples of 32 in [256, 2048]" ); return; }
	if ( frames < 22 || frames > 362 ) { respond_error( fd, 400, "frames must be in [22, 362]" ); return; }
	if ( steps < 1 || steps > 60 ) { respond_error( fd, 400, "steps must be in [1, 60]" ); return; }
	std::vector<H3Lora> loras;
	const std::string turbo = str_field( b, "turbo" );
	if ( !turbo.empty() ) {
		const LoraInfo* t = find_lora( turbo );
		if ( !t || !t->turbo ) { respond_error( fd, 400, "unknown turbo adapter" ); return; }
		loras.push_back( { t->path, 1.0f } );
		const std::regex re( R"(_(\d)step)" );
		std::smatch m;
		if ( std::regex_search( t->file, m, re ) ) { steps = std::atoi( m[1].str().c_str() ); }
	}
	if ( b.contains( "loras" ) && b["loras"].is_array() ) {
		for ( const json& l : b["loras"] ) {
			if ( !l.is_object() ) { continue; }
			const LoraInfo* li = find_lora( str_field( l, "file" ) );
			if ( !li ) { respond_error( fd, 400, "unknown adapter " + str_field( l, "file" ) ); return; }
			loras.push_back( { li->path, static_cast<float>( num_field( l, "scale", 1.0 ) ) } );
		}
	}
	if ( loras.size() > 6 ) { respond_error( fd, 400, "at most six adapters" ); return; }
	std::vector<std::string> image_data;
	std::vector<int> frame_idx;
	if ( b.contains( "images" ) && b["images"].is_array() ) {
		for ( const json& im : b["images"] ) {
			if ( !im.is_object() ) { continue; }
			std::string d = str_field( im, "data" );
			const size_t comma = d.find( "base64," );
			if ( comma != std::string::npos ) { d = d.substr( comma + 7 ); }
			image_data.push_back( b64_decode( d ) );
			frame_idx.push_back( static_cast<int>( num_field( im, "frame_index", 0 ) ) );
		}
	}
	if ( image_data.size() > 2 ) { respond_error( fd, 400, "at most two keyframe images" ); return; }
	// token budget: text + audio + video rows (+ one frame of rows per keyframe image)
	const int lat_t = frames <= 5 ? 2 : ( ( frames - 5 ) / 17 ) * 5 + 2;
	const long rows = 512 + static_cast<long>( lat_t + static_cast<int>( image_data.size() ) ) * ( width / 32 ) * ( height / 32 ) + static_cast<long>( frames / 24.0 * 80 ) + 1400L * static_cast<long>( image_data.size() );
	if ( rows > g_cfg.max_rows ) { respond_error( fd, 400, "this size and duration need about " + std::to_string( rows ) + " tokens; the limit is " + std::to_string( g_cfg.max_rows ) ); return; }
	{
		std::lock_guard<std::mutex> lk( g_mu );
		j->id = g_next_id++;
		g_jobs[j->id] = j;
	}
	j->dir = g_cfg.out_dir + "/jobs/" + std::to_string( j->id );
	mkdirs( j->dir );
	for ( size_t i = 0; i < image_data.size(); ++i ) {
		const std::string path = j->dir + "/key" + std::to_string( i ) + ".img";
		std::ofstream( path, std::ios::binary ) << image_data[i];
		j->images.push_back( path );
		j->frame_indices.push_back( frame_idx[i] == 0 ? 0 : -1 );
	}
	j->loras = loras;
	j->steps_total = steps;
	j->steps = steps;
	j->created = now_s();
	json params = { { "prompt", prompt }, { "width", width }, { "height", height }, { "frames", frames }, { "steps", steps }, { "seed", seed }, { "turbo", turbo }, { "images", image_data.size() } };
	json ls = json::array();
	for ( const json& l : b.value( "loras", json::array() ) ) { if ( l.is_object() ) { ls.push_back( { { "file", str_field( l, "file" ) }, { "scale", num_field( l, "scale", 1.0 ) } } ); } }
	params["loras"] = ls;
	j->params = params;
	{
		std::lock_guard<std::mutex> lk( g_mu );
		g_queue.push_back( j->id );
		save_meta( *j );
	}
	g_cv.notify_one();
	LOG_INFO( "h3_server: job %d queued (%dx%d, %d frames, %d steps, %zu adapter(s), %zu image(s))", j->id, width, height, frames, steps, loras.size(), image_data.size() );
	respond_json( fd, 200, json{ { "id", j->id } } );
}

void handle( int fd ) {
	Request rq;
	if ( !read_request( fd, &rq ) ) { ::close( fd ); return; }
	const std::string& p = rq.path;
	if ( rq.method == "GET" && ( p == "/" || p == "/index.html" ) ) {
		respond( fd, 200, "text/html; charset=utf-8", h3_ui_html() );
	} else if ( rq.method == "GET" && p == "/api/info" ) {
		json o;
		o["model"] = "MiniMax-H3 FL2VA (INT8 ConvRot)";
		o["devices"] = g_cfg.devices;
		o["max_rows"] = g_cfg.max_rows;
		json ls = json::array();
		for ( const LoraInfo& l : g_loras ) { ls.push_back( { { "file", l.file }, { "label", l.label }, { "turbo", l.turbo }, { "format", l.format }, { "mb", l.mb }, { "task", l.task } } ); }
		o["loras"] = ls;
		respond_json( fd, 200, o );
	} else if ( rq.method == "GET" && p == "/api/jobs" ) {
		json arr = json::array();
		{
			std::lock_guard<std::mutex> lk( g_mu );
			for ( auto it = g_jobs.rbegin(); it != g_jobs.rend(); ++it ) { arr.push_back( job_json( *it->second ) ); }
		}
		respond_json( fd, 200, arr );
	} else if ( rq.method == "POST" && p == "/api/jobs" ) {
		handle_submit( fd, rq );
	} else if ( p.compare( 0, 10, "/api/jobs/" ) == 0 ) {
		const std::string rest = p.substr( 10 );
		const size_t slash = rest.find( '/' );
		const int id = std::atoi( rest.substr( 0, slash ).c_str() );
		const std::string sub = slash == std::string::npos ? "" : rest.substr( slash + 1 );
		std::shared_ptr<Job> j;
		{
			std::lock_guard<std::mutex> lk( g_mu );
			auto it = g_jobs.find( id );
			if ( it != g_jobs.end() ) { j = it->second; }
		}
		if ( !j ) { respond_error( fd, 404, "no such job" ); }
		else if ( rq.method == "GET" && sub.empty() ) { std::lock_guard<std::mutex> lk( g_mu ); respond_json( fd, 200, job_json( *j ) ); }
		else if ( rq.method == "GET" && sub == "video" ) { serve_file( fd, rq, j->dir + "/result.mp4", "video/mp4" ); }
		else if ( rq.method == "GET" && sub == "thumb" ) { serve_file( fd, rq, j->dir + "/thumb.jpg", "image/jpeg" ); }
		else if ( rq.method == "POST" && sub == "cancel" ) {
			std::lock_guard<std::mutex> lk( g_mu );
			j->cancel = true;
			if ( j->state == "queued" ) { j->state = "cancelled"; save_meta( *j ); }
			respond_json( fd, 200, json{ { "ok", true } } );
		} else if ( rq.method == "DELETE" && sub.empty() ) {
			std::lock_guard<std::mutex> lk( g_mu );
			if ( j->state == "running" || j->state == "queued" ) { respond_error( fd, 400, "cancel the job first" ); }
			else {
				const std::string cmd = "rm -rf '" + j->dir + "'";
				( void )!std::system( cmd.c_str() );
				g_jobs.erase( id );
				respond_json( fd, 200, json{ { "ok", true } } );
			}
		} else { respond_error( fd, 404, "not found" ); }
	} else {
		respond_error( fd, 404, "not found" );
	}
	::shutdown( fd, SHUT_WR );
	::close( fd );
}

} // namespace

int main( int argc, char** argv ) {
	for ( int i = 1; i < argc; ++i ) {
		const std::string a = argv[i];
		auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
		if ( a == "--root" ) { g_cfg.root = next(); }
		else if ( a == "--dit" ) { g_cfg.dit = next(); }
		else if ( a == "--lora-dir" ) { g_cfg.lora_dirs.push_back( next() ); }
		else if ( a == "--out-dir" ) { g_cfg.out_dir = next(); }
		else if ( a == "--port" ) { g_cfg.port = std::atoi( next().c_str() ); }
		else if ( a == "--host" ) { g_cfg.host = next(); }
		else if ( a == "--max-rows" ) { g_cfg.max_rows = std::atoi( next().c_str() ); }
		else if ( a == "--devices" ) {
			g_cfg.devices.clear();
			const std::string v = next();
			for ( size_t p = 0; p < v.size(); ) { g_cfg.devices.push_back( std::atoi( v.c_str() + p ) ); p = v.find( ',', p ); if ( p == std::string::npos ) { break; } ++p; }
		}
		else { std::fprintf( stderr, "unknown argument %s\n", a.c_str() ); return 1; }
	}
	if ( g_cfg.root.empty() || g_cfg.dit.empty() ) {
		std::fprintf( stderr, "usage: h3_server --root DIR --dit CKPT [--lora-dir DIR]... [--out-dir DIR] [--devices 0,1,2,3] [--port 8200] [--host 127.0.0.1]\n" );
		return 1;
	}
	mkdirs( g_cfg.out_dir + "/jobs" );
	scan_loras();
	load_history();
	std::thread( worker ).detach();
	const int ls = ::socket( AF_INET, SOCK_STREAM, 0 );
	int one = 1;
	::setsockopt( ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof( one ) );
	sockaddr_in addr{};
	addr.sin_family = AF_INET;
	addr.sin_port = htons( static_cast<uint16_t>( g_cfg.port ) );
	::inet_pton( AF_INET, g_cfg.host.c_str(), &addr.sin_addr );
	if ( ::bind( ls, reinterpret_cast<sockaddr*>( &addr ), sizeof( addr ) ) != 0 || ::listen( ls, 64 ) != 0 ) {
		std::fprintf( stderr, "cannot listen on %s:%d\n", g_cfg.host.c_str(), g_cfg.port );
		return 1;
	}
	LOG_INFO( "h3_server: http://%s:%d  (%zu adapter(s), %zu GPU(s), jobs in %s)", g_cfg.host.c_str(), g_cfg.port, g_loras.size(), g_cfg.devices.size(), g_cfg.out_dir.c_str() );
	for ( ;; ) {
		const int fd = ::accept( ls, nullptr, nullptr );
		if ( fd < 0 ) { continue; }
		std::thread( handle, fd ).detach();
	}
}
