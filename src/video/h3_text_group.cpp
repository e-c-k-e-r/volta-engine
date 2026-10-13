#include "engine/video/h3_text_group.h"
#include "engine/core/log.h"
#include <cuda_runtime.h>
#include <atomic>
#include <thread>
#if defined( ENGINE_HAVE_NCCL )
#include <nccl.h>
#endif

namespace engine::video {

struct H3TextGroup::Impl {
	std::vector<int> devices;
	std::vector<std::unique_ptr<H3TextEncoder>> encs;
	int max_tokens = 0;
#if defined( ENGINE_HAVE_NCCL )
	std::vector<ncclComm_t> comms;
#endif
};

H3TextGroup::H3TextGroup() : p_( new Impl ) {}

H3TextGroup::~H3TextGroup() {
#if defined( ENGINE_HAVE_NCCL )
	for ( ncclComm_t c : p_->comms ) { if ( c ) { ncclCommDestroy( c ); } }
#endif
	delete p_;
}

int H3TextGroup::size() const { return static_cast<int>( p_->encs.size() ); }
int H3TextGroup::max_tokens() const { return p_->max_tokens; }

bool H3TextGroup::load( const char* dir, const std::vector<int>& devices, int max_tokens, int layers, bool captioner ) {
	Impl& s = *p_;
	const int n = static_cast<int>( devices.size() );
	if ( n < 1 ) { return false; }
#if !defined( ENGINE_HAVE_NCCL )
	if ( n > 1 ) { LOG_ERROR( "H3 text: tensor parallelism needs a build with NCCL" ); return false; }
#endif
	int caller = 0;
	CUDA_CHECK( cudaGetDevice( &caller ) );
	s.devices = devices;
	s.max_tokens = max_tokens;
	s.encs.resize( n );
	std::atomic<int> ok{ 1 };
	std::vector<std::thread> threads;
	for ( int r = 0; r < n; ++r ) {
		threads.emplace_back( [&, r] {
			CUDA_CHECK( cudaSetDevice( devices[r] ) );
			s.encs[r].reset( new H3TextEncoder() );
			if ( !s.encs[r]->load( dir, devices[r], max_tokens, layers, r, n, captioner ) ) { ok = 0; }
		} );
	}
	for ( auto& t : threads ) { t.join(); }
	CUDA_CHECK( cudaSetDevice( caller ) );
	if ( !ok ) { return false; }
#if defined( ENGINE_HAVE_NCCL )
	if ( n > 1 ) {
		s.comms.assign( n, nullptr );
		std::vector<int> devs = devices;
		if ( ncclCommInitAll( s.comms.data(), n, devs.data() ) != ncclSuccess ) { LOG_ERROR( "H3 text: NCCL init failed" ); return false; }
		for ( int r = 0; r < n; ++r ) {
			ncclComm_t comm = s.comms[r];
			H3TpComm tc;
			tc.sum_f32 = [comm]( float* buf, size_t count, void* stream ) { ncclAllReduce( buf, buf, count, ncclFloat, ncclSum, comm, static_cast<cudaStream_t>( stream ) ); };
			tc.max_f32 = [comm]( float* buf, size_t count, void* stream ) { ncclAllReduce( buf, buf, count, ncclFloat, ncclMax, comm, static_cast<cudaStream_t>( stream ) ); };
			tc.sum_f16 = [comm]( void* buf, size_t count, void* stream ) { ncclAllReduce( buf, buf, count, ncclFloat16, ncclSum, comm, static_cast<cudaStream_t>( stream ) ); };
			s.encs[r]->set_tp( std::move( tc ) );
		}
	}
#endif
	return true;
}

bool H3TextGroup::encode_mm( const std::vector<int>& ids, int image_token_id, const std::vector<H3ImageFeatures>& images, float* d_out ) {
	Impl& s = *p_;
	const int n = static_cast<int>( s.encs.size() );
	if ( n == 0 ) { return false; }
	int caller = 0;
	CUDA_CHECK( cudaGetDevice( &caller ) );
	std::atomic<int> ok{ 1 };
	std::vector<std::thread> threads;
	for ( int r = 0; r < n; ++r ) {
		threads.emplace_back( [&, r] {
			CUDA_CHECK( cudaSetDevice( s.devices[r] ) );
			float* out = d_out;
			if ( r > 0 ) { CUDA_CHECK( cudaMalloc( &out, ids.size() * 5120 * sizeof( float ) ) ); } // the other ranks end with the same hidden states; only rank 0's are returned
			if ( !s.encs[r]->encode_mm( ids, image_token_id, images, out ) ) { ok = 0; }
			if ( r > 0 ) { (void) cudaFree( out ); }
		} );
	}
	for ( auto& t : threads ) { t.join(); }
	CUDA_CHECK( cudaSetDevice( caller ) );
	return ok;
}

bool H3TextGroup::generate( const std::vector<int>& ids, int image_token_id, const std::vector<H3ImageFeatures>& images, const H3CaptionOptions& opt, std::vector<int>* out_tokens ) {
	Impl& s = *p_;
	const int n = static_cast<int>( s.encs.size() );
	if ( n == 0 || !out_tokens ) { return false; }
	int caller = 0;
	CUDA_CHECK( cudaGetDevice( &caller ) );
	std::atomic<int> ok{ 1 };
	std::vector<std::vector<int>> toks( n );
	std::vector<std::thread> threads;
	for ( int r = 0; r < n; ++r ) {
		threads.emplace_back( [&, r] {
			CUDA_CHECK( cudaSetDevice( s.devices[r] ) );
			if ( !s.encs[r]->generate( ids, image_token_id, images, opt, &toks[r] ) ) { ok = 0; }
		} );
	}
	for ( auto& t : threads ) { t.join(); }
	CUDA_CHECK( cudaSetDevice( caller ) );
	*out_tokens = toks[0];
	return ok;
}

} // namespace engine::video
