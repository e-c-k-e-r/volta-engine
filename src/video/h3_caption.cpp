// H3 picture captions: the Qwen3-VL language model describes a picture as a video prompt (H3Engine::caption).
#include "engine/core/log.h"
#include "engine/loader/image.h"
#include "engine/loader/tokenizer.h"
#include "engine/video/h3_text.h"
#include "engine/video/h3_vision.h"
#include "h3_components.h"
#include "h3_run.h"
#include "h3_util.h"
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

namespace engine::video {
namespace {

// What the model is asked when the user wrote no prompt of their own
const char* const kDefaultAsk =
	"Describe this image in detail as a prompt for a video generation model: the subject and its appearance, clothing, pose and expression, the setting and "
	"background, the lighting and colors, and the camera framing and style. Write one flowing paragraph of plain prose, without lists, headings or a preamble.";

// ... and when the prompt box already holds an idea for the video
std::string guided_ask( const std::string& guidance ) {
	return "Describe this image as a prompt for a video generation model. The user's idea for the video is: \"" + guidance + "\"\n"
		"Keep that idea and add what the image shows that a video model needs to follow it: the subject and its appearance, clothing, pose and expression, the "
		"setting and background, the lighting and colors, and the camera framing and style. Write one flowing paragraph of plain prose, without lists, headings or "
		"a preamble.";
}

std::string trim( std::string s ) {
	const char* ws = " \t\r\n";
	s.erase( 0, s.find_first_not_of( ws ) == std::string::npos ? s.size() : s.find_first_not_of( ws ) );
	s.erase( s.find_last_not_of( ws ) == std::string::npos ? 0 : s.find_last_not_of( ws ) + 1 );
	return s;
}

} // namespace

bool h3_caption( const H3Paths& paths, H3Components& comp, const std::string& image_bytes, const std::string& guidance, uint32_t seed, std::string* text ) {
	if ( !paths.captioner || paths.text_encoder.empty() ) { LOG_ERROR( "H3 caption: the server was started without the captioner" ); return false; }
	const auto t0 = std::chrono::steady_clock::now();
	engine::loader::Tokenizer tok;
	if ( !tok.load( ( paths.root + "/tokenizer" ).c_str() ) ) { LOG_ERROR( "H3 caption: tokenizer not found under %s", paths.root.c_str() ); return false; }
	engine::loader::DecodedImage di;
	std::string err;
	if ( !engine::loader::decode_image( reinterpret_cast<const uint8_t*>( image_bytes.data() ), image_bytes.size(), &di, &err ) ) { LOG_ERROR( "H3 caption: %s", err.c_str() ); return false; }
	engine::loader::VisionParams vp; // up to 1 MP: at most 1024 image tokens
	engine::loader::PreparedImage prep;
	if ( !engine::loader::prepare_image( di, vp, &prep, &err ) ) { LOG_ERROR( "H3 caption: %s", err.c_str() ); return false; }
	H3ImageFeatures feat;
	feat.grid_h = prep.grid_h;
	feat.grid_w = prep.grid_w;
	{ // the vision tower is freed before the text tower takes its room
		H3VisionTower tower;
		if ( !tower.load( ( paths.root + "/text_encoder" ).c_str(), paths.device, prep.grid_h * prep.grid_w ) ) { return false; }
		if ( !tower.encode( prep.patches, prep.grid_h, prep.grid_w, &feat.merged, feat.deepstack ) ) { LOG_ERROR( "H3 caption: vision tower failed" ); return false; }
	}
	const int image_pad = tok.get_special_token( "<|image_pad|>" );
	std::vector<int> ids;
	auto push = [&]( const std::vector<int>& v ) { ids.insert( ids.end(), v.begin(), v.end() ); };
	const std::string ask = trim( guidance ).empty() ? std::string( kDefaultAsk ) : guided_ask( trim( guidance ) );
	push( { tok.get_special_token( "<|im_start|>" ) } );
	push( tok.encode( "user\n", false ) );
	push( { tok.get_special_token( "<|vision_start|>" ) } );
	push( std::vector<int>( feat.merged.size() / 5120, image_pad ) );
	push( { tok.get_special_token( "<|vision_end|>" ) } );
	push( tok.encode( ask, false ) );
	push( { tok.get_special_token( "<|im_end|>" ) } );
	push( tok.encode( "\n", false ) );
	push( { tok.get_special_token( "<|im_start|>" ) } );
	push( tok.encode( "assistant\n", false ) );
	std::vector<int> devices = paths.devices;
	if ( devices.empty() ) { devices.push_back( paths.device ); }
	H3Lease<H3TextGroup> group = comp.text_group( devices, static_cast<int>( ids.size() ) + 512 );
	if ( !group ) { return false; }
	H3CaptionOptions opt;
	opt.seed = seed;
	std::vector<int> out;
	const auto tg = std::chrono::steady_clock::now();
	if ( !group->generate( ids, image_pad, { feat }, opt, &out ) ) { LOG_ERROR( "H3 caption: generation failed" ); return false; }
	*text = trim( tok.decode( out ) );
	LOG_INFO( "H3 caption: %zu prompt tokens (%zu image), %zu new tokens in %.1f s (%.1f s in all)", ids.size(), feat.merged.size() / 5120, out.size(), seconds_since( tg ), seconds_since( t0 ) );
	return !text->empty();
}

} // namespace engine::video
