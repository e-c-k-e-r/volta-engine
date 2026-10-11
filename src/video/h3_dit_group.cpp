#include "engine/video/h3_dit_group.h"
#include "engine/core/log.h"
#include "engine/video/h3_dit.h"
#include <cuda_runtime.h>
#if defined( ENGINE_HAVE_NCCL )
#include <nccl.h>
#endif
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
	std::vector<std::vector<cudaEvent_t>> pull_done; // [rank][peer]: the pull of the peer's slice into rank has completed
	std::vector<cudaEvent_t> kv_ready;              // per rank: its K/V slice of the current layer is final
	bool tp = false;                                // tensor parallel instead of sequence parallel
#if defined( ENGINE_HAVE_NCCL )
	std::vector<ncclComm_t> comms;
#endif
	std::vector<double> t_sync;                     // per rank: host time at the two barriers per layer, summed over a forward
};

H3DiTGroup::H3DiTGroup() : p_( new Impl ) {}
H3DiTGroup::~H3DiTGroup() {
#if defined( ENGINE_HAVE_NCCL )
	for ( ncclComm_t c : p_->comms ) { if ( c ) { ncclCommDestroy( c ); } }
#endif
	delete p_;
}
int H3DiTGroup::size() const { return static_cast<int>( p_->dits.size() ); }

bool H3DiTGroup::load( const char* checkpoint, const std::vector<int>& devices, int max_rows, bool tensor_parallel ) {
	Impl& s = *p_;
	s.tp = tensor_parallel && devices.size() > 1;
#if !defined( ENGINE_HAVE_NCCL )
	if ( s.tp ) { LOG_ERROR( "H3: tensor parallelism needs a build with NCCL" ); return false; }
#endif
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
			cfg.sp_ranks = s.tp ? 1 : n;
			cfg.tp_ranks = s.tp ? n : 1;
			cfg.tp_rank = r;
			ok[r] = s.dits[r]->load( checkpoint, devices[r], max_rows, cfg ) ? 1 : 0;
		} );
	}
	for ( auto& t : threads ) { t.join(); }
	CUDA_CHECK( cudaSetDevice( caller_device ) ); // the peer-access setup walked over all GPUs; leave the caller where it was
	for ( int r = 0; r < n; ++r ) { if ( !ok[r] ) { return false; } }
#if defined( ENGINE_HAVE_NCCL )
	if ( s.tp ) {
		s.comms.assign( n, nullptr );
		std::vector<int> devs = devices;
		if ( ncclCommInitAll( s.comms.data(), n, devs.data() ) != ncclSuccess ) { LOG_ERROR( "H3: NCCL init failed" ); return false; }
		for ( int r = 0; r < n; ++r ) {
			ncclComm_t comm = s.comms[r];
			H3TpComm tc;
			tc.sum_f32 = [comm]( float* buf, size_t count, void* stream ) { if ( ncclAllReduce( buf, buf, count, ncclFloat, ncclSum, comm, static_cast<cudaStream_t>( stream ) ) != ncclSuccess ) { LOG_ERROR( "H3: NCCL all-reduce failed" ); } };
			tc.max_f32 = [comm]( float* buf, size_t count, void* stream ) { if ( ncclAllReduce( buf, buf, count, ncclFloat, ncclMax, comm, static_cast<cudaStream_t>( stream ) ) != ncclSuccess ) { LOG_ERROR( "H3: NCCL all-reduce failed" ); } };
			s.dits[r]->set_tp( std::move( tc ) );
		}
		s.t_sync.assign( n, 0.0 );
		return true;
	}
#endif
	s.pull_streams.assign( n, std::vector<cudaStream_t>( n, nullptr ) );
	s.pull_done.assign( n, std::vector<cudaEvent_t>( n, nullptr ) );
	s.t_sync.assign( n, 0.0 );
	s.kv_ready.assign( n, nullptr );
	for ( int r = 0; r < n; ++r ) {
		CUDA_CHECK( cudaSetDevice( devices[r] ) );
		CUDA_CHECK( cudaEventCreateWithFlags( &s.kv_ready[r], cudaEventDisableTiming ) );
	}
	CUDA_CHECK( cudaSetDevice( caller_device ) );
	for ( int r = 0; r < n; ++r ) {
		Impl* sp = p_;
		H3SpExchange ex;
		// Everything is enqueued, nothing waits on the GPU. A rank records "my K/V slice is final" on its compute stream; after a host barrier (so every rank's event is
		// recorded) each pull stream waits for its own rank's event (the attention of the previous layer is done with the buffer) and the peer's event, then copies the
		// peer's slice over NVLink. The compute stream attends to its own keys meanwhile and waits for each peer's pull right before using that slice. Before the next
		// layer rewrites a slice, the compute stream waits for the other ranks' pulls of it (second barrier: their events are recorded).
		ex.begin = [sp, r, n]( int total, void* stream_ptr ) {
			cudaStream_t stream = static_cast<cudaStream_t>( stream_ptr );
			CUDA_CHECK( cudaEventRecord( sp->kv_ready[r], stream ) );
			const auto t0 = std::chrono::steady_clock::now();
			sp->barrier->wait();
			sp->t_sync[r] += std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
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
				CUDA_CHECK( cudaStreamWaitEvent( ps, sp->kv_ready[r], 0 ) );
				CUDA_CHECK( cudaStreamWaitEvent( ps, sp->kv_ready[q], 0 ) );
				CUDA_CHECK( cudaMemcpyPeerAsync( static_cast<char*>( mine.k ) + off, sp->devices[r], static_cast<char*>( theirs.k ) + off, sp->devices[q], bytes, ps ) );
				CUDA_CHECK( cudaMemcpyPeerAsync( static_cast<char*>( mine.v ) + off, sp->devices[r], static_cast<char*>( theirs.v ) + off, sp->devices[q], bytes, ps ) );
				CUDA_CHECK( cudaEventRecord( sp->pull_done[r][q], ps ) );
			}
		};
		ex.wait_peer = [sp, r]( int peer, void* stream_ptr ) { CUDA_CHECK( cudaStreamWaitEvent( static_cast<cudaStream_t>( stream_ptr ), sp->pull_done[r][peer], 0 ) ); };
		ex.end = [sp, r, n]( void* stream_ptr ) {
			const auto t0 = std::chrono::steady_clock::now();
			sp->barrier->wait();
			sp->t_sync[r] += std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
			for ( int q = 0; q < n; ++q ) {
				if ( q != r && sp->pull_done[q][r] ) { CUDA_CHECK( cudaStreamWaitEvent( static_cast<cudaStream_t>( stream_ptr ), sp->pull_done[q][r], 0 ) ); }
			}
		};
		s.dits[r]->set_sp( r, n, std::move( ex ) );
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
	const bool sp = n > 1 && !s.tp;
	std::vector<float> vall( sp ? static_cast<size_t>( U ) * 96 : 0 ), aall( sp ? static_cast<size_t>( U ) * 32 : 0 );
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
			if ( !sp ) { // one GPU, or tensor parallel (every rank computes all rows; rank 0's outputs are returned)
				float *d_vo = nullptr, *d_ao = nullptr;
				CUDA_CHECK( cudaMalloc( &d_vo, video_out->size() * sizeof( float ) ) );
				CUDA_CHECK( cudaMalloc( &d_ao, audio_out->size() * sizeof( float ) ) );
				s.dits[r]->forward( q, d_vo, d_ao );
				CUDA_CHECK( cudaDeviceSynchronize() );
				if ( r == 0 ) {
					CUDA_CHECK( cudaMemcpy( video_out->data(), d_vo, video_out->size() * sizeof( float ), cudaMemcpyDeviceToHost ) );
					CUDA_CHECK( cudaMemcpy( audio_out->data(), d_ao, audio_out->size() * sizeof( float ), cudaMemcpyDeviceToHost ) );
				}
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
		double sync_max = 0;
		for ( int r = 0; r < n; ++r ) {
			sync_max = std::max( sync_max, s.t_sync[r] );
			s.t_sync[r] = 0;
		}
		if ( sp ) { LOG_INFO( "H3: sequence-parallel forward: %.2f s of host time at the layer barriers (maximum over ranks)", sync_max ); }
	}

	if ( sp ) {
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
