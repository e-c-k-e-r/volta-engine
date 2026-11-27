// Video files through FFmpeg's libraries (libavformat, libavcodec, libswscale; built with ENGINE_FFMPEG where CMake finds them): the frame
// count and rate from the container, the checkpoint processor's sample of frames, and the sampled frames decoded in order as RGB, two at a
// time into a step (video.cpp)
#include "engine/loader/video.h"

#if ENGINE_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}
#include <algorithm>
#include <cstring>
#endif

namespace engine::loader {

#if ENGINE_FFMPEG

namespace {

struct MemReader {
	const uint8_t* data = nullptr;
	size_t size = 0;
	size_t pos = 0;
};

int mem_read( void* opaque, uint8_t* buf, int n ) {
	auto* r = static_cast<MemReader*>( opaque );
	const size_t k = std::min( r->size - r->pos, static_cast<size_t>( n ) );
	if ( k == 0 ) return AVERROR_EOF;
	std::memcpy( buf, r->data + r->pos, k );
	r->pos += k;
	return static_cast<int>( k );
}

int64_t mem_seek( void* opaque, int64_t offset, int whence ) {
	auto* r = static_cast<MemReader*>( opaque );
	if ( whence & AVSEEK_SIZE ) return static_cast<int64_t>( r->size );
	whence &= ~AVSEEK_FORCE;
	const int64_t base = whence == SEEK_SET ? 0 : whence == SEEK_CUR ? static_cast<int64_t>( r->pos ) : whence == SEEK_END ? static_cast<int64_t>( r->size ) : -1;
	if ( base < 0 || base + offset < 0 || base + offset > static_cast<int64_t>( r->size ) ) return AVERROR( EINVAL );
	r->pos = static_cast<size_t>( base + offset );
	return base + offset;
}

// The file opened from memory with its best video stream's decoder
struct Input {
	MemReader reader;
	AVIOContext* io = nullptr;
	AVFormatContext* fmt = nullptr;
	AVCodecContext* dec = nullptr;
	AVStream* st = nullptr;

	~Input() {
		avcodec_free_context( &dec );
		if ( fmt ) avformat_close_input( &fmt );
		if ( io ) {
			av_freep( &io->buffer );
			avio_context_free( &io );
		}
	}

	bool open( const uint8_t* data, size_t size, std::string* error ) {
		reader = { data, size, 0 };
		constexpr int kBuffer = 1 << 16;
		auto* buf = static_cast<uint8_t*>( av_malloc( kBuffer ) );
		io = buf ? avio_alloc_context( buf, kBuffer, 0, &reader, mem_read, nullptr, mem_seek ) : nullptr;
		fmt = io ? avformat_alloc_context() : nullptr;
		if ( !fmt ) {
			if ( !io ) av_free( buf );
			*error = "out of memory opening the video";
			return false;
		}
		fmt->pb = io;
		fmt->flags |= AVFMT_FLAG_CUSTOM_IO;
		if ( avformat_open_input( &fmt, nullptr, nullptr, nullptr ) < 0 ) {
			*error = "not a video file FFmpeg reads";
			return false;
		}
		if ( avformat_find_stream_info( fmt, nullptr ) < 0 ) {
			*error = "the video's streams cannot be read";
			return false;
		}
		const AVCodec* codec = nullptr;
		const int s = av_find_best_stream( fmt, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0 );
		if ( s < 0 || !codec ) {
			*error = s == AVERROR_DECODER_NOT_FOUND ? "no decoder for the video's codec" : "the file has no video stream";
			return false;
		}
		st = fmt->streams[s];
		dec = avcodec_alloc_context3( codec );
		if ( !dec || avcodec_parameters_to_context( dec, st->codecpar ) < 0 ) {
			*error = "the video's decoder cannot be set up";
			return false;
		}
		dec->thread_count = 4;
		if ( avcodec_open2( dec, codec, nullptr ) < 0 ) {
			*error = "the video's decoder cannot be opened";
			return false;
		}
		return true;
	}
};

// A decoded frame as packed RGB (its colorspace and range as tagged; untagged: BT.601, limited range)
bool to_rgb( const AVFrame* f, SwsContext** sws, DecodedImage* out ) {
	*sws = sws_getCachedContext( *sws, f->width, f->height, static_cast<AVPixelFormat>( f->format ), f->width, f->height, AV_PIX_FMT_RGB24,
		SWS_BICUBIC | SWS_ACCURATE_RND | SWS_FULL_CHR_H_INT, nullptr, nullptr, nullptr );
	if ( !*sws ) return false;
	const int cs = f->colorspace == AVCOL_SPC_UNSPECIFIED || f->colorspace == AVCOL_SPC_RGB ? SWS_CS_ITU601 : static_cast<int>( f->colorspace );
	const int full = f->color_range == AVCOL_RANGE_JPEG ? 1 : 0;
	sws_setColorspaceDetails( *sws, sws_getCoefficients( cs ), full, sws_getCoefficients( SWS_CS_ITU601 ), 1, 0, 1 << 16, 1 << 16 );
	out->width = f->width;
	out->height = f->height;
	out->rgb.resize( static_cast<size_t>( f->width ) * f->height * 3 );
	uint8_t* dst[4] = { out->rgb.data(), nullptr, nullptr, nullptr };
	const int stride[4] = { f->width * 3, 0, 0, 0 };
	return sws_scale( *sws, f->data, f->linesize, 0, f->height, dst, stride ) == f->height;
}

} // namespace

bool video_decoding_available() { return true; }

bool decode_video( const uint8_t* data, size_t size, const VisionParams& p, int max_tokens, PreparedVideo* out, std::string* error ) {
	std::string why;
	auto fail = [&]( const std::string& msg ) {
		if ( error ) *error = msg;
		return false;
	};
	// The frame count: the container's, else the stream's packets (a pass that only demuxes)
	int total = 0;
	double fps = 0.0, duration = 0.0;
	int width = 0, height = 0;
	{
		Input in;
		if ( !in.open( data, size, &why ) ) return fail( why );
		const AVRational rate = in.st->avg_frame_rate.num > 0 ? in.st->avg_frame_rate : in.st->r_frame_rate;
		fps = rate.num > 0 && rate.den > 0 ? av_q2d( rate ) : 24.0;
		if ( in.st->duration != AV_NOPTS_VALUE && in.st->duration > 0 ) duration = in.st->duration * av_q2d( in.st->time_base );
		else if ( in.fmt->duration != AV_NOPTS_VALUE && in.fmt->duration > 0 ) duration = static_cast<double>( in.fmt->duration ) / AV_TIME_BASE;
		width = in.dec->width;
		height = in.dec->height;
		total = in.st->nb_frames > 0 ? static_cast<int>( in.st->nb_frames ) : 0;
		if ( total == 0 ) {
			AVPacket* pkt = av_packet_alloc();
			while ( pkt && av_read_frame( in.fmt, pkt ) >= 0 ) {
				if ( pkt->stream_index == in.st->index ) ++total;
				av_packet_unref( pkt );
			}
			av_packet_free( &pkt );
		}
	}
	if ( total < 1 || width < 1 || height < 1 ) return fail( "the video has no frames" );
	const std::vector<int> sampled = sample_video_frames( p, total, fps, duration );
	VideoPlan plan;
	if ( !plan_video( static_cast<int>( sampled.size() ), height, width, p, max_tokens, &plan, error ) ) return false;
	// The source frames the steps take, in order (repeats where the sampler or an odd count repeats a frame)
	std::vector<int> need( plan.frames.size() );
	for ( size_t k = 0; k < need.size(); ++k ) need[k] = sampled[plan.frames[k]];
	out->steps.assign( plan.steps(), PreparedImage() );
	out->times.assign( plan.steps(), 0.0 );

	Input in;
	if ( !in.open( data, size, &why ) ) return fail( why );
	AVPacket* pkt = av_packet_alloc();
	AVFrame* frm = av_frame_alloc();
	SwsContext* sws = nullptr;
	size_t next = 0;
	int index = 0;
	bool ok = pkt && frm;
	DecodedImage first, last;
	bool have_last = false;
	auto place = [&]( const DecodedImage& img ) {
		const size_t k = next++;
		const double t = need[k] / fps;
		if ( k % 2 == 0 ) {
			first = img;
			out->times[k / 2] = t;
		}
		else {
			prepare_video_step( first, img, plan, p, &out->steps[k / 2] );
			if ( !p.canvas ) out->times[k / 2] = 0.5 * ( out->times[k / 2] + t );
		}
	};
	auto take = [&]( const AVFrame* f ) {
		if ( next < need.size() && need[next] == index ) {
			DecodedImage img;
			if ( !to_rgb( f, &sws, &img ) ) {
				ok = false;
				return;
			}
			while ( next < need.size() && need[next] == index ) place( img );
			last = std::move( img );
			have_last = true;
		}
		++index;
	};
	while ( ok && next < need.size() && av_read_frame( in.fmt, pkt ) >= 0 ) {
		if ( pkt->stream_index == in.st->index && avcodec_send_packet( in.dec, pkt ) >= 0 ) {
			while ( ok && next < need.size() && avcodec_receive_frame( in.dec, frm ) == 0 ) take( frm );
		}
		av_packet_unref( pkt );
	}
	if ( ok && next < need.size() && avcodec_send_packet( in.dec, nullptr ) >= 0 ) {
		while ( ok && next < need.size() && avcodec_receive_frame( in.dec, frm ) == 0 ) take( frm );
	}
	sws_freeContext( sws );
	av_frame_free( &frm );
	av_packet_free( &pkt );
	if ( !ok ) return fail( "a video frame could not be converted to RGB" );
	if ( !have_last ) return fail( "no frame of the video decodes" );
	// the container counted more frames than decode: the rest repeat the last one
	while ( next < need.size() ) place( last );
	return true;
}

#else

bool video_decoding_available() { return false; }

bool decode_video( const uint8_t*, size_t, const VisionParams&, int, PreparedVideo*, std::string* error ) {
	if ( error ) *error = "this build reads no video files (built without FFmpeg's libraries); send the frames instead";
	return false;
}

#endif

} // namespace engine::loader
