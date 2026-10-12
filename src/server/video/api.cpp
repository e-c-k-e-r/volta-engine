// HTTP handlers of the video service: the page, the adapter list, job submission, status, result files, cancel and delete.
#include "engine/core/log.h"
#include "engine/loader/image.h"
#include "page.h"
#include "service.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>

namespace engine::server {

using nlohmann::json;

namespace {

std::string str_field( const json& j, const char* k, const std::string& def = "" ) {
	return j.contains( k ) && j[k].is_string() ? j[k].get<std::string>() : def;
}

double num_field( const json& j, const char* k, double def ) {
	return j.contains( k ) && j[k].is_number() ? j[k].get<double>() : def;
}

void bad_request( const ConnPtr& c, const std::string& message ) {
	send_error( c, 400, "invalid_request_error", message );
}

// base64 payload of an upload ("data:...;base64,<payload>" or the bare payload), decoded
std::string upload_bytes( const json& item ) {
	std::string d = str_field( item, "data" );
	const size_t comma = d.find( "base64," );
	if ( comma != std::string::npos ) d = d.substr( comma + 7 );
	std::string out;
	engine::loader::base64_decode( d, &out );
	return out;
}

void write_file( const std::string& path, const std::string& bytes ) {
	std::ofstream( path, std::ios::binary ) << bytes;
}

int latent_frames( int frames ) {
	return frames <= 5 ? 2 : ( ( frames - 5 ) / 17 ) * 5 + 2;
}

} // namespace

void VideoService::Impl::handle_page( const ConnPtr& c, const HttpRequest& ) {
	send_response( c, 200, "text/html; charset=utf-8", video_page_html() );
}

void VideoService::Impl::handle_info( const ConnPtr& c, const HttpRequest& ) {
	json o;
	o["model"] = "MiniMax-H3 (INT8 ConvRot)";
	o["devices"] = cfg.video.devices;
	o["max_rows"] = cfg.video.max_rows;
	o["ref2va"] = !cfg.video.dit_ref2va.empty();
	json ls = json::array();
	for ( const AdapterInfo& a : adapters.all() ) {
		ls.push_back( { { "file", a.file }, { "label", a.label }, { "turbo", a.turbo }, { "format", a.format }, { "mb", a.mb }, { "task", a.task } } );
	}
	o["loras"] = ls;
	send_json( c, 200, o.dump() );
}

void VideoService::Impl::handle_list( const ConnPtr& c, const HttpRequest& ) {
	send_json( c, 200, jobs.list().dump() );
}

void VideoService::Impl::handle_submit( const ConnPtr& c, const HttpRequest& r ) {
	const json b = json::parse( r.body, nullptr, false );
	if ( b.is_discarded() || !b.is_object() ) return bad_request( c, "invalid JSON" );

	// ---- the generation parameters ----
	const std::string prompt = str_field( b, "prompt" );
	const int width = static_cast<int>( num_field( b, "width", 1280 ) ), height = static_cast<int>( num_field( b, "height", 736 ) ), frames = static_cast<int>( num_field( b, "frames", 124 ) );
	int steps = static_cast<int>( num_field( b, "steps", 4 ) );
	const uint64_t seed = static_cast<uint64_t>( num_field( b, "seed", 1 ) );
	// optional flow shifts of the sigma schedule (0 = the default: video 6 with a 768p adapter else 12, audio 3); a larger audio shift spends more of the few steps at low noise
	const double video_shift = num_field( b, "video_shift", 0 ), audio_shift = num_field( b, "audio_shift", 0 );
	if ( prompt.empty() || prompt.size() > 8000 ) return bad_request( c, "prompt must be 1 to 8000 characters" );
	if ( width < 256 || height < 256 || width > 2048 || height > 2048 || width % 32 || height % 32 ) return bad_request( c, "width and height must be multiples of 32 in [256, 2048]" );
	if ( frames < 22 || frames > 362 ) return bad_request( c, "frames must be in [22, 362]" );
	if ( steps < 1 || steps > 60 ) return bad_request( c, "steps must be in [1, 60]" );
	if ( ( video_shift != 0 && ( video_shift < 0.5 || video_shift > 40 ) ) || ( audio_shift != 0 && ( audio_shift < 0.5 || audio_shift > 40 ) ) ) return bad_request( c, "video_shift and audio_shift must be 0 (default) or in [0.5, 40]" );

	// ---- uploads: keyframes (FL2VA) or references (Ref2VA) ----
	std::vector<std::string> key_data, ref_image_data, ref_video_data, ref_audio_data;
	std::vector<int> key_index;
	if ( b.contains( "images" ) && b["images"].is_array() ) {
		for ( const json& im : b["images"] ) {
			if ( !im.is_object() ) continue;
			key_data.push_back( upload_bytes( im ) );
			key_index.push_back( static_cast<int>( num_field( im, "frame_index", 0 ) ) == 0 ? 0 : -1 );
		}
	}
	for ( const char* key : { "ref_images", "ref_videos", "ref_audios" } ) {
		if ( !b.contains( key ) || !b[key].is_array() ) continue;
		std::vector<std::string>& dst = !std::strcmp( key, "ref_images" ) ? ref_image_data : !std::strcmp( key, "ref_videos" ) ? ref_video_data : ref_audio_data;
		for ( const json& im : b[key] ) {
			if ( im.is_object() ) dst.push_back( upload_bytes( im ) );
		}
	}
	if ( key_data.size() > 2 ) return bad_request( c, "at most two keyframe images" );
	const bool ref_task = !ref_image_data.empty() || !ref_video_data.empty() || !ref_audio_data.empty();
	if ( ref_task ) {
		if ( cfg.video.dit_ref2va.empty() ) return bad_request( c, "this server has no Ref2VA transformer (set video.dit_ref2va)" );
		if ( !key_data.empty() ) return bad_request( c, "keyframes and references are different tasks" );
		if ( ref_image_data.empty() && ref_video_data.empty() ) return bad_request( c, "reference audio needs a reference image or video" );
		if ( ref_image_data.size() > 9 || ref_video_data.size() > 3 || ref_audio_data.size() > 3 || ref_image_data.size() + ref_video_data.size() + ref_audio_data.size() > 12 ) {
			return bad_request( c, "at most 9 reference images, 3 videos and 3 audio clips (12 in all)" );
		}
	}

	// ---- adapters ----
	std::vector<engine::video::H3Lora> loras;
	const std::string turbo = str_field( b, "turbo" );
	if ( !turbo.empty() ) {
		const AdapterInfo* t = adapters.find( turbo );
		if ( !t || !t->turbo ) return bad_request( c, "unknown turbo adapter" );
		if ( ( t->task == "ref2v" ) != ref_task ) return bad_request( c, "the " + t->task + " adapter does not fit this task" );
		loras.push_back( { t->path, 1.0f } );
		static const std::regex step_re( R"(_(\d)step)" );
		std::smatch m;
		if ( std::regex_search( t->file, m, step_re ) ) steps = std::atoi( m[1].str().c_str() );
	}
	if ( b.contains( "loras" ) && b["loras"].is_array() ) {
		for ( const json& l : b["loras"] ) {
			if ( !l.is_object() ) continue;
			const AdapterInfo* a = adapters.find( str_field( l, "file" ) );
			if ( !a ) return bad_request( c, "unknown adapter " + str_field( l, "file" ) );
			loras.push_back( { a->path, static_cast<float>( num_field( l, "scale", 1.0 ) ) } );
		}
	}
	if ( loras.size() > 6 ) return bad_request( c, "at most six adapters" );

	// ---- the packed-sequence budget: text + audio + video rows, plus the rows of keyframes and references ----
	long ref_rows = 0;
	for ( const std::string& bytes : ref_image_data ) { // visual rows of the 2048 px reference plus as many prompt tokens
		engine::loader::DecodedImage di;
		std::string err;
		if ( !engine::loader::decode_image( reinterpret_cast<const uint8_t*>( bytes.data() ), bytes.size(), &di, &err ) ) return bad_request( c, "cannot decode a reference image: " + err );
		const double sc = 2048.0 / std::min( di.width, di.height );
		ref_rows += 2L * std::lround( di.width * sc / 32 ) * std::lround( di.height * sc / 32 );
	}
	// a reference video: latent frames x 1008 rows (768 x 1344 canvas) plus a 1008-token block per two sampled frames
	ref_rows += static_cast<long>( ref_video_data.size() ) * ( latent_frames( frames ) + frames / 24 / 2 + 1 ) * 1008L;
	ref_rows += 1200L * static_cast<long>( ref_audio_data.size() );
	const long rows = 512 + static_cast<long>( latent_frames( frames ) + static_cast<int>( key_data.size() ) ) * ( width / 32 ) * ( height / 32 ) + static_cast<long>( frames / 24.0 * 80 ) +
		1400L * static_cast<long>( key_data.size() ) + ref_rows;
	if ( rows > cfg.video.max_rows ) return bad_request( c, "this size and duration need about " + std::to_string( rows ) + " tokens; the limit is " + std::to_string( cfg.video.max_rows ) );

	// ---- accepted: store the uploads and queue ----
	JobPtr j = jobs.create();
	auto store = [&]( const std::vector<std::string>& data, const char* stem, const char* ext, std::vector<std::string>* paths ) {
		for ( size_t i = 0; i < data.size(); ++i ) {
			const std::string path = j->dir + "/" + stem + std::to_string( i ) + ext;
			write_file( path, data[i] );
			paths->push_back( path );
		}
	};
	store( key_data, "key", ".img", &j->images );
	j->frame_indices = key_index;
	store( ref_image_data, "ref", ".img", &j->ref_images );
	store( ref_video_data, "refvideo", ".mp4", &j->ref_videos );
	store( ref_audio_data, "refaudio", ".bin", &j->ref_audios );
	j->loras = loras;
	j->steps_total = steps;
	j->steps = steps;
	json params = { { "prompt", prompt }, { "width", width }, { "height", height }, { "frames", frames }, { "steps", steps }, { "seed", seed }, { "turbo", turbo }, { "images", key_data.size() },
		{ "ref_images", ref_image_data.size() }, { "ref_videos", ref_video_data.size() }, { "ref_audios", ref_audio_data.size() } };
	json ls = json::array();
	for ( const json& l : b.value( "loras", json::array() ) ) {
		if ( l.is_object() ) ls.push_back( { { "file", str_field( l, "file" ) }, { "scale", num_field( l, "scale", 1.0 ) } } );
	}
	params["loras"] = ls;
	if ( video_shift > 0 ) params["video_shift"] = video_shift;
	if ( audio_shift > 0 ) params["audio_shift"] = audio_shift;
	j->params = params;
	jobs.enqueue( j );
	LOG_INFO( "Video job %d queued (%dx%d, %d frames, %d steps, %zu adapter(s), %zu keyframe(s), %zu reference(s))\n", j->id, width, height, frames, steps, loras.size(), key_data.size(),
		ref_image_data.size() + ref_video_data.size() + ref_audio_data.size() );
	send_json( c, 200, json{ { "id", j->id } }.dump() );
}

void VideoService::Impl::handle_job( const ConnPtr& c, const HttpRequest& r ) {
	const std::string rest = r.path.substr( std::strlen( "/api/jobs/" ) );
	const size_t slash = rest.find( '/' );
	const std::string sub = slash == std::string::npos ? "" : rest.substr( slash + 1 );
	JobPtr j = jobs.find( std::atoi( rest.substr( 0, slash ).c_str() ) );
	if ( !j ) return send_error( c, 404, "not_found_error", "no such job" );
	const bool get = r.method == "GET";
	if ( get && sub.empty() ) return send_json( c, 200, jobs.describe( j ).dump() );
	if ( get && sub == "video" ) return send_file( c, r, j->dir + "/result.mp4", "video/mp4" );
	if ( get && sub == "thumb" ) return send_file( c, r, j->dir + "/thumb.jpg", "image/jpeg" );
	if ( r.method == "POST" && sub == "cancel" ) {
		jobs.cancel( j );
		return send_json( c, 200, "{\"ok\":true}" );
	}
	if ( r.method == "DELETE" && sub.empty() ) {
		std::string err;
		if ( !jobs.remove( j, &err ) ) return bad_request( c, err );
		return send_json( c, 200, "{\"ok\":true}" );
	}
	send_error( c, 404, "not_found_error", "unknown endpoint " + r.method + " " + r.path );
}

} // namespace engine::server
