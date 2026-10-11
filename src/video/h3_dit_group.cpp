#include "engine/video/h3_dit_group.h"
#include "engine/core/log.h"
#include "engine/video/h3_dit.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <chrono>
#include <thread>

namespace engine::video {
namespace {

class Barrier {
public:
	explicit Barrier( int n ) : n_( n ) {}
	void wait() {
		std::unique_lock<std::mutex> lk( m_ );
		const int gen = gen_;
		if ( ++count_ == n_ ) {
			count_ = 0;
			++gen_;
			cv_.notify_all();
		} else {
			cv_.wait( lk, [&] { return gen != gen_; } );
		}
	}

private:
	int n_, count_ = 0, gen_ = 0;
	std::mutex m_;
	std::condition_variable cv_;
};

template <typename T>
T* up( const std::vector<T>& v ) {
	T* d = nullptr;
	CUDA_CHECK( cudaMalloc( &d, std::max<size_t>( v.size(), 1 ) * sizeof( T ) ) );
	CUDA_CHECK( cudaMemcpy( d, v.data(), v.size() * sizeof( T ), cudaMemcpyHostToDevice ) );
	return d;
}

} // namespace

struct H3DiTGroup::Impl {
	std::vector<int> devices;
	std::vector<std::unique_ptr<H3DiT>> dits;
	std::unique_ptr<Barrier> barrier;
	int max_rows = 0;
	std::vector<std::vector<cudaStream_t>> pull_streams; // [rank][peer]: one copy stream per peer, so a rank pulls from all its peers at once (one NVLink each)
	std::vector<std::vector<cudaEvent_t>> pull_done;
	std::vector<double> t_comp, t_sync, t_xfer; // per rank: host time spent waiting for the slowest rank (before the exchange) and in the exchange itself, summed over a forward
};

H3DiTGroup::H3DiTGroup() : p_( new Impl ) {}
H3DiTGroup::~H3DiTGroup() { delete p_; }
int H3DiTGroup::size() const { return static_cast<int>( p_->dits.size() ); }

bool H3DiTGroup::load( const char* checkpoint, const std::vector<int>& devices, int max_rows ) {
	Impl& s = *p_;
	const int n = static_cast<int>( devices.size() );
	if ( n < 1 ) { return false; }
	int caller_device = 0;
	CUDA_CHECK( cudaGetDevice( &caller_device ) );
	s.devices = devices;
	s.max_rows = max_rows;
	s.barrier.reset( new Barrier( n ) );
	// NVLink peer access in both directions
	for ( int a = 0; a < n; ++a ) {
		CUDA_CHECK( cudaSetDevice( devices[a] ) );
		for ( int b = 0; b < n; ++b ) {
			if ( a == b ) { continue; }
			int can = 0;
			CUDA_CHECK( cudaDeviceCanAccessPeer( &can, devices[a], devices[b] ) );
			if ( !can ) { LOG_ERROR( "H3: GPU %d cannot access GPU %d", devices[a], devices[b] ); return false; }
			const cudaError_t e = cudaDeviceEnablePeerAccess( devices[b], 0 );
			if ( e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled ) { LOG_ERROR( "H3: peer access failed" ); return false; }
			cudaGetLastError();
		}
	}
	s.dits.resize( n );
	std::vector<std::thread> threads;
	std::vector<int> ok( n, 0 );
	for ( int r = 0; r < n; ++r ) {
		threads.emplace_back( [&, r] {
			CUDA_CHECK( cudaSetDevice( devices[r] ) );
			s.dits[r].reset( new H3DiT() );
			H3Config cfg;
			cfg.sp_ranks = n;
			ok[r] = s.dits[r]->load( checkpoint, devices[r], max_rows, cfg ) ? 1 : 0;
		} );
	}
	for ( auto& t : threads ) { t.join(); }
	CUDA_CHECK( cudaSetDevice( caller_device ) ); // the peer-access setup walked over all GPUs; leave the caller where it was
	for ( int r = 0; r < n; ++r ) { if ( !ok[r] ) { return false; } }
	s.pull_streams.assign( n, std::vector<cudaStream_t>( n, nullptr ) );
	s.pull_done.assign( n, std::vector<cudaEvent_t>( n, nullptr ) );
	s.t_comp.assign( n, 0.0 );
	s.t_sync.assign( n, 0.0 );
	s.t_xfer.assign( n, 0.0 );
	for ( int r = 0; r < n; ++r ) {
		Impl* sp = p_;
		s.dits[r]->set_sp( r, n, [sp, r, n]( int total, void* stream_ptr ) {
			// Pull model: each rank copies its peers' K / V slices into its own buffers on its own compute stream (so stream order makes them visible to the
			// attention that follows). Barrier 1: every rank's own slice is final. Barrier 2: every rank has finished pulling, so slices may be rewritten.
			cudaStream_t stream = static_cast<cudaStream_t>( stream_ptr );
			const auto t0 = std::chrono::steady_clock::now();
			CUDA_CHECK( cudaStreamSynchronize( stream ) );
			const auto t05 = std::chrono::steady_clock::now();
			sp->barrier->wait();
			const auto t1 = std::chrono::steady_clock::now();
			const size_t qd = 7168;
			const int chunk = ( total + n - 1 ) / n;
			const half_ptr_pair mine = sp->dits[r]->kv_buffers();
			for ( int q = 0; q < n; ++q ) {
				if ( q == r ) { continue; }
				const int q0 = std::min( total, q * chunk ), q1 = std::min( total, q0 + chunk );
				if ( q1 <= q0 ) { continue; }
				if ( !sp->pull_streams[r][q] ) {
					CUDA_CHECK( cudaStreamCreateWithFlags( &sp->pull_streams[r][q], cudaStreamNonBlocking ) );
					CUDA_CHECK( cudaEventCreateWithFlags( &sp->pull_done[r][q], cudaEventDisableTiming ) );
				}
				const half_ptr_pair theirs = sp->dits[q]->kv_buffers();
				const size_t off = static_cast<size_t>( q0 ) * qd * 2, bytes = static_cast<size_t>( q1 - q0 ) * qd * 2;
				cudaStream_t ps = sp->pull_streams[r][q];
				CUDA_CHECK( cudaMemcpyPeerAsync( static_cast<char*>( mine.k ) + off, sp->devices[r], static_cast<char*>( theirs.k ) + off, sp->devices[q], bytes, ps ) );
				CUDA_CHECK( cudaMemcpyPeerAsync( static_cast<char*>( mine.v ) + off, sp->devices[r], static_cast<char*>( theirs.v ) + off, sp->devices[q], bytes, ps ) );
				CUDA_CHECK( cudaEventRecord( sp->pull_done[r][q], ps ) );
			}
			for ( int q = 0; q < n; ++q ) { // the attention that follows (compute stream) starts when every pull is in
				if ( q != r && sp->pull_streams[r][q] ) { CUDA_CHECK( cudaStreamWaitEvent( stream, sp->pull_done[r][q], 0 ) ); }
			}
			CUDA_CHECK( cudaStreamSynchronize( stream ) );
			sp->barrier->wait();
			sp->t_comp[r] += std::chrono::duration<double>( t05 - t0 ).count();
			sp->t_sync[r] += std::chrono::duration<double>( t1 - t05 ).count();
			sp->t_xfer[r] += std::chrono::duration<double>( std::chrono::steady_clock::now() - t1 ).count();
		} );
	}
	return true;
}

int H3DiTGroup::add_lora( const char* path, float scale ) {
	int id = -1;
	for ( auto& d : p_->dits ) { id = d->add_lora( path, scale ); if ( id < 0 ) { return -1; } }
	return id;
}

void H3DiTGroup::trim() {
	for ( auto& d : p_->dits ) { d->trim(); }
}

bool H3DiTGroup::reserve() {
	for ( auto& d : p_->dits ) { if ( !d->reserve() ) { return false; } }
	return true;
}

void H3DiTGroup::clear_loras() {
	for ( auto& d : p_->dits ) { d->clear_loras(); }
}

void H3DiTGroup::set_lora_scale( int id, float scale ) {
	for ( auto& d : p_->dits ) { d->set_lora_scale( id, scale ); }
}

bool H3DiTGroup::forward( const H3HostRequest& hr, std::vector<float>* video_out, std::vector<float>* audio_out ) {
	Impl& s = *p_;
	const int n = static_cast<int>( s.dits.size() );
	const int U = hr.used;
	std::vector<float> vall( n > 1 ? static_cast<size_t>( U ) * 96 : 0 ), aall( n > 1 ? static_cast<size_t>( U ) * 32 : 0 );
	video_out->assign( static_cast<size_t>( hr.n_img ) * 96, 0.0f );
	audio_out->assign( static_cast<size_t>( hr.n_audio ) * 32, 0.0f );
	std::vector<std::thread> threads;
	for ( int r = 0; r < n; ++r ) {
		threads.emplace_back( [&, r] {
			CUDA_CHECK( cudaSetDevice( s.devices[r] ) );
			// request copy on this device
			float *d_v = up( hr.video_rows ), *d_a = up( hr.audio_rows ), *d_p = up( hr.prompt ), *d_pos = up( hr.position_ids );
			int *d_img = up( hr.img_pos ), *d_aud = up( hr.audio_pos ), *d_txt = up( hr.text_pos ), *d_tags = up( hr.tags ), *d_inv = up( hr.inverse );
			uint8_t* d_mask = up( hr.update_mask );
			CUDA_CHECK( cudaStreamSynchronize( nullptr ) ); // the uploads ran on the default stream; the DiT runs on a non-blocking one that is not ordered against it
			H3Request q;
			q.used = U; q.text_len = hr.text_len; q.n_img = hr.n_img; q.n_audio = hr.n_audio;
			q.video_rows = d_v; q.audio_rows = d_a; q.prompt = d_p; q.position_ids = d_pos;
			q.img_pos = d_img; q.audio_pos = d_aud; q.text_pos = d_txt; q.update_mask = d_mask; q.tags = d_tags; q.inverse = d_inv;
			q.unique_t = hr.unique_t;
			if ( n == 1 ) {
				float *d_vo = nullptr, *d_ao = nullptr;
				CUDA_CHECK( cudaMalloc( &d_vo, video_out->size() * sizeof( float ) ) );
				CUDA_CHECK( cudaMalloc( &d_ao, audio_out->size() * sizeof( float ) ) );
				s.dits[0]->forward( q, d_vo, d_ao );
				CUDA_CHECK( cudaMemcpy( video_out->data(), d_vo, video_out->size() * sizeof( float ), cudaMemcpyDeviceToHost ) );
				CUDA_CHECK( cudaMemcpy( audio_out->data(), d_ao, audio_out->size() * sizeof( float ), cudaMemcpyDeviceToHost ) );
				cudaFree( d_vo ); cudaFree( d_ao );
			} else {
				s.dits[r]->forward_sp( q, vall.data(), aall.data() );
				CUDA_CHECK( cudaDeviceSynchronize() );
			}
			for ( void* ptr : { static_cast<void*>( d_v ), static_cast<void*>( d_a ), static_cast<void*>( d_p ), static_cast<void*>( d_pos ), static_cast<void*>( d_img ),
				static_cast<void*>( d_aud ), static_cast<void*>( d_txt ), static_cast<void*>( d_tags ), static_cast<void*>( d_inv ), static_cast<void*>( d_mask ) } ) { cudaFree( ptr ); }
		} );
	}
	for ( auto& t : threads ) { t.join(); }
	{
		double comp_max = 0, sync_max = 0, xfer_max = 0;
		for ( int r = 0; r < n; ++r ) {
			comp_max = std::max( comp_max, s.t_comp[r] ); sync_max = std::max( sync_max, s.t_sync[r] ); xfer_max = std::max( xfer_max, s.t_xfer[r] );
			s.t_comp[r] = s.t_sync[r] = s.t_xfer[r] = 0;
		}
		LOG_INFO( "H3: sequence-parallel forward (maximum over ranks): %.2f s of GPU compute before the exchanges, %.2f s at the barriers waiting for the slowest rank, %.2f s copying K/V", comp_max, sync_max, xfer_max );
	}

	if ( n > 1 ) {
		for ( int i = 0; i < hr.n_img; ++i ) {
			const float m = hr.update_mask[i] ? 1.0f : 0.0f;
			for ( int c = 0; c < 96; ++c ) { ( *video_out )[static_cast<size_t>( i ) * 96 + c] = vall[static_cast<size_t>( hr.img_pos[i] ) * 96 + c] * m; }
		}
		for ( int i = 0; i < hr.n_audio; ++i ) {
			for ( int c = 0; c < 32; ++c ) { ( *audio_out )[static_cast<size_t>( i ) * 32 + c] = aall[static_cast<size_t>( hr.audio_pos[i] ) * 32 + c]; }
		}
	}
	return true;
}

} // namespace engine::video
