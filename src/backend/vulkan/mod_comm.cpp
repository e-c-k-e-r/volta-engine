// Vulkan backend: the multi-device collective. There is no peer-to-peer path between Vulkan devices here, so an all-reduce goes through the host:
// every rank thread reads its partial back, the ranks sum a slice each (FP32 accumulation over the ranks in order, one rounding to FP16) into a
// shared buffer, and every rank writes the full result back. Eager execution only: the readback synchronizes the stream, so a graph capture cannot
// contain it (the cluster turns graphs off for tp > 1 on this backend). All ranks must call in lockstep, as with the NVLink ring.
#include "engine/core/comm.h"
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#include "engine/modules/norm.h"
#include "engine/modules/gated_residual.h"
#include "engine/modules/mhc.h"
#include <pthread.h>
#include <cstdlib>
#include <cstring>

namespace engine::comm {

namespace {

constexpr size_t kHostSlotBytes = 32ULL * 1024 * 1024;   // largest all-reduce: rows * hidden FP16

struct HostMesh {
	int ranks = 0;
	uint8_t* bufs[ENGINE_NUM_GPUS] = {};
	uint8_t* res_bufs[ENGINE_NUM_GPUS] = {};   // each rank's copy of the residual (all_reduce_rms_norm)
	uint8_t* result = nullptr;
	pthread_barrier_t barrier;
};

HostMesh* mesh_of( const CollectiveRing& r ) { return static_cast<HostMesh*>( r.peer_ptrs[0] ); }

} // namespace

void CollectiveRing::init( int local_rank, int ranks, void* const peer_staging[ENGINE_NUM_GPUS], void* const peer_flags[ENGINE_NUM_GPUS] ) {
	rank = local_rank;
	nranks = ranks;
	for ( int i = 0; i < ENGINE_NUM_GPUS; ++i ) {
		peer_ptrs[i] = peer_staging[i];
		sync_flags[i] = peer_flags[i];
	}
}

namespace {

// Sums the ranks' partials of x (FP32, rank order, one FP16 rounding). With a residual the sum is added to it before the rounding and the new
// residual is what comes back, in `residual`; without one the sum comes back in x. Collective: every rank calls it.
void reduce_through_host( const CollectiveRing& ring, pod::TensorView& x, pod::TensorView* residual, backend::stream_t stream ) {
	HostMesh* m = mesh_of( ring );
	ENGINE_ASSERT( m != nullptr && x.ndim >= 2, "Vulkan all_reduce: mesh not initialized" );
	const int rank = ring.rank, nranks = ring.nranks;
	const size_t n = static_cast<size_t>( x.shape[0] ) * x.shape[1];
	const size_t bytes = n * sizeof( backend::half_t );
	ENGINE_ASSERT( bytes <= kHostSlotBytes, "Vulkan all_reduce: tensor larger than the host staging slot" );
	backend::memcpy_async( m->bufs[rank], x.data, bytes, backend::Copy::D2H, stream );
	if ( residual != nullptr ) backend::memcpy_async( m->res_bufs[rank], residual->data, bytes, backend::Copy::D2H, stream );
	backend::stream_sync( stream );
	pthread_barrier_wait( &m->barrier );   // every partial is on the host
	const size_t lo = n * rank / nranks, hi = n * ( rank + 1 ) / nranks;
	const backend::half_t* parts[ENGINE_NUM_GPUS];
	for ( int r = 0; r < nranks; ++r ) parts[r] = reinterpret_cast<const backend::half_t*>( m->bufs[r] );
	const backend::half_t* res = residual != nullptr ? reinterpret_cast<const backend::half_t*>( m->res_bufs[rank] ) : nullptr;
	backend::half_t* out = reinterpret_cast<backend::half_t*>( m->result );
	for ( size_t i = lo; i < hi; ++i ) {
		float sum = backend::to_float( parts[0][i] );
		for ( int r = 1; r < nranks; ++r ) sum += backend::to_float( parts[r][i] );
		if ( res != nullptr ) sum += backend::to_float( res[i] );
		out[i] = backend::from_float_half( sum );
	}
	pthread_barrier_wait( &m->barrier );   // the sums are complete
	backend::memcpy_async( residual != nullptr ? residual->data : x.data, m->result, bytes, backend::Copy::H2D, stream );   // copied into staging at record time: the buffer is free to reuse
}

} // namespace

void CollectiveRing::all_reduce( pod::TensorView& x, backend::stream_t stream ) {
	reduce_through_host( *this, x, nullptr, stream );
}

// residual = FP16( sum of the partials + residual ); out = RMSNorm( residual ) (into x when out is null): the fused NVLink kernel's arithmetic
// up to the order of the sum
void CollectiveRing::all_reduce_rms_norm(
	pod::TensorView& x,
	pod::TensorView& residual,
	const pod::TensorView& weight,
	float eps,
	backend::stream_t stream,
	float constant_bias,
	pod::TensorView* out
) {
	reduce_through_host( *this, x, &residual, stream );
	modules::RMSNorm::forward( residual, weight, out != nullptr ? *out : x, eps, constant_bias, stream );
}

// The reduction, then the post-step as the module call it is on the NVLink path's fused kernel
void CollectiveRing::all_reduce_hc_apply( pod::TensorView& y, float* streams, const float* post, const float* comb, backend::stream_t stream ) {
	all_reduce( y, stream );
	modules::HyperConnection::update( streams, streams, y.ptr<const backend::half_t>(), post, comb, y.shape[1], stream, y.shape[0] );
}

// The reduction, then the gated add and the next norm as the two module calls they are on the NVLink path's fused kernel
void CollectiveRing::all_reduce_gated_norm( pod::TensorView& y, float* streams, const float* post, const backend::half_t* norm_w, float* normed, int hc, float eps, backend::stream_t stream, const backend::half_t* addend, const float* ) {
	ENGINE_ASSERT( addend == nullptr, "CollectiveRing::all_reduce_gated_norm: the shared-expert addend is CUDA only" );
	all_reduce( y, stream );
	const int rows = y.shape[0], hidden = y.shape[1];
	modules::GatedResidual::apply( streams, post, y.ptr<const backend::half_t>(), rows, hidden, hc, stream );
	modules::GatedResidual::norm( streams, norm_w, normed, nullptr, nullptr, rows, hidden, hc, eps, nullptr, nullptr, stream );
}

void init_4gpu_mesh( CollectiveRing rings[ENGINE_NUM_GPUS], void* staging_ptrs[ENGINE_NUM_GPUS], void* flag_ptrs[ENGINE_NUM_GPUS], size_t, int ranks ) {
	auto* m = new HostMesh();
	m->ranks = ranks;
	for ( int r = 0; r < ranks; ++r ) {
		m->bufs[r] = static_cast<uint8_t*>( std::malloc( kHostSlotBytes ) );
		m->res_bufs[r] = static_cast<uint8_t*>( std::malloc( kHostSlotBytes ) );
	}
	m->result = static_cast<uint8_t*>( std::malloc( kHostSlotBytes ) );
	pthread_barrier_init( &m->barrier, nullptr, ranks );
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		staging_ptrs[d] = m;
		flag_ptrs[d] = nullptr;
	}
	for ( int r = 0; r < ranks; ++r ) rings[r].init( r, ranks, staging_ptrs, flag_ptrs );
}

void destroy_4gpu_mesh( void* staging_ptrs[ENGINE_NUM_GPUS], void* flag_ptrs[ENGINE_NUM_GPUS] ) {
	auto* m = static_cast<HostMesh*>( staging_ptrs[0] );
	if ( m != nullptr ) {
		pthread_barrier_destroy( &m->barrier );
		for ( int r = 0; r < ENGINE_NUM_GPUS; ++r ) {
			std::free( m->bufs[r] );
			std::free( m->res_bufs[r] );
		}
		std::free( m->result );
		delete m;
	}
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) staging_ptrs[d] = flag_ptrs[d] = nullptr;
}

} // namespace engine::comm
