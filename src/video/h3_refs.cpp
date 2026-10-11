#include "engine/video/h3_refs.h"
#include "engine/core/log.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace engine::video {
namespace {

std::string quote( const std::string& s ) {
	std::string o = "'";
	for ( char c : s ) {
		if ( c == '\'' ) { o += "'\\''"; } else { o += c; }
	}
	return o + "'";
}

bool capture( const std::string& cmd, std::vector<uint8_t>* out ) {
	FILE* p = popen( cmd.c_str(), "r" );
	if ( !p ) { return false; }
	uint8_t buf[1 << 16];
	size_t n;
	out->clear();
	while ( ( n = std::fread( buf, 1, sizeof( buf ), p ) ) > 0 ) { out->insert( out->end(), buf, buf + n ); }
	return pclose( p ) == 0;
}

std::string text_of( const std::string& cmd ) {
	std::vector<uint8_t> b;
	capture( cmd, &b );
	return std::string( b.begin(), b.end() );
}

double value_of( const std::string& text, const char* key ) {
	const std::string k = std::string( key ) + "=";
	const size_t p = text.find( k );
	return p == std::string::npos ? -1.0 : std::atof( text.c_str() + p + k.size() );
}

int align32( double v ) { return std::max( 32, static_cast<int>( std::lround( v / 32.0 ) ) * 32 ); }

} // namespace

bool h3_load_ref_video( const std::string& path, int max_frames, double* budget, H3RefVideo* out ) {
	const std::string info = text_of( "ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate:format=duration -of default=nw=1 " + quote( path ) + " 2>/dev/null" );
	const int w = static_cast<int>( value_of( info, "width" ) ), h = static_cast<int>( value_of( info, "height" ) );
	double fps = 0;
	{
		const size_t p = info.find( "r_frame_rate=" );
		if ( p != std::string::npos ) {
			double a = 0, b = 1;
			if ( std::sscanf( info.c_str() + p + 13, "%lf/%lf", &a, &b ) >= 1 && b > 0 ) { fps = a / b; }
		}
	}
	const double duration = value_of( info, "duration" );
	if ( w <= 0 || h <= 0 || duration <= 0 ) { LOG_ERROR( "H3: cannot probe the reference video %s", path.c_str() ); return false; }
	const double ratio = static_cast<double>( w ) / h;
	if ( std::min( w, h ) < 256 || std::max( w, h ) > 5760 || ratio < 0.4 || ratio > 2.5 ) { LOG_ERROR( "H3: reference video %s (%dx%d) must have sides in [256, 5760] px and an aspect ratio in [0.4, 2.5]", path.c_str(), w, h ); return false; }
	if ( fps < 23.976 - 1e-3 || fps > 60.0 + 1e-3 ) { LOG_ERROR( "H3: reference video %s runs at %.3f fps; 23.976 to 60 are supported", path.c_str(), fps ); return false; }
	if ( duration < 2.0 || duration > 15.0 ) { LOG_ERROR( "H3: reference video %s is %.2f s long; 2 to 15 s are supported", path.c_str(), duration ); return false; }
	double use = duration;
	if ( use > *budget ) {
		if ( use - *budget > 1e-2 ) { LOG_ERROR( "H3: reference videos must be at most 15 seconds in total" ); return false; }
		use = *budget;
	}
	if ( use < 2.0 ) { LOG_ERROR( "H3: reference videos must be at most 15 seconds in total" ); return false; }
	*budget -= use;
	// canvas: 768 px short edge, at most 768 x 1344 pixels
	double tw, th;
	if ( ratio >= 1.0 ) { tw = 768.0 * ratio; th = 768.0; } else { tw = 768.0; th = 768.0 / ratio; }
	if ( tw * th > 768.0 * 1344.0 ) { const double s = std::sqrt( 768.0 * 1344.0 / ( tw * th ) ); tw *= s; th *= s; }
	out->width = align32( tw );
	out->height = align32( th );
	out->has_audio = !text_of( "ffprobe -v error -select_streams a -show_entries stream=codec_type -of csv=p=0 " + quote( path ) + " 2>/dev/null" ).empty();
	out->duration = use;
	char cmd[4096];
	std::snprintf( cmd, sizeof( cmd ), "ffmpeg -nostdin -loglevel error -i %s -map 0:v:0 -an -vf \"fps=24,scale=%d:%d:flags=lanczos,setsar=1\" -t %.6f -frames:v %d -f rawvideo -pix_fmt rgb24 - 2>/dev/null",
		quote( path ).c_str(), out->width, out->height, use, max_frames );
	if ( !capture( cmd, &out->rgb ) ) { LOG_ERROR( "H3: ffmpeg failed on %s", path.c_str() ); return false; }
	const size_t frame = static_cast<size_t>( out->width ) * out->height * 3;
	out->frames = static_cast<int>( out->rgb.size() / frame );
	out->rgb.resize( static_cast<size_t>( out->frames ) * frame );
	if ( out->frames < 22 ) { LOG_ERROR( "H3: reference video %s gives %d frames; at least 22 are needed", path.c_str(), out->frames ); return false; }
	return true;
}

bool h3_load_ref_audio( const std::string& path, double max_seconds, std::vector<float>* planar, int* samples, double* full_seconds ) {
	std::vector<uint8_t> raw;
	const std::string cmd = "ffmpeg -nostdin -loglevel error -i " + quote( path ) + " -vn -f f32le -ar 32000 -ac 2 - 2>/dev/null";
	if ( !capture( cmd, &raw ) ) { return false; }
	const size_t n = raw.size() / 8; // interleaved stereo floats
	const float* wave = reinterpret_cast<const float*>( raw.data() );
	*full_seconds = n / 32000.0;
	const size_t keep = max_seconds > 0 ? std::min<size_t>( n, static_cast<size_t>( std::lround( max_seconds * 32000.0 ) ) ) : n;
	planar->assign( keep * 2, 0.0f );
	for ( size_t i = 0; i < keep; ++i ) { ( *planar )[i] = wave[2 * i]; ( *planar )[keep + i] = wave[2 * i + 1]; }
	*samples = static_cast<int>( keep );
	return keep > 0;
}

std::vector<int> h3_video_sample_indices( int frames ) {
	std::vector<int> idx;
	double cursor = 0.0;
	for ( ;; ) {
		const int f = static_cast<int>( std::nearbyint( cursor ) );
		if ( f >= frames ) { break; }
		if ( idx.empty() || f > idx.back() ) { idx.push_back( f ); }
		cursor += 12.0; // 24 fps / 2 fps
	}
	return idx;
}

std::vector<double> h3_video_block_timestamps( int sampled ) {
	std::vector<double> t( sampled );
	for ( int i = 0; i < sampled; ++i ) { t[i] = i / 2.0; }
	while ( t.size() % 2 ) { t.push_back( t.back() ); }
	std::vector<double> blocks;
	for ( size_t i = 0; i < t.size(); i += 2 ) { blocks.push_back( ( t[i] + t[i + 1] ) / 2.0 ); }
	return blocks;
}

} // namespace engine::video
