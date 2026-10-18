// Device half of the CPU expert path: post the routes handed to the CPU, wait for its answer and add it (see engine/modules/cpu_experts.h)
#include "engine/modules/cpu_experts.h"
#include "engine/backend/runtime.h"
#include <cuda_fp16.h>

namespace engine::modules {

using cpu_moe::Mailbox;
using cpu_moe::MailboxRoute;

namespace {

__device__ inline uint64_t global_ns() { uint64_t t; asm volatile( "mov.u64 %0, %%globaltimer;" : "=l"( t ) ); return t; }

__global__ __launch_bounds__( 256 )
void cpu_post_kernel( const half* __restrict__ x, int* __restrict__ sel, const half* __restrict__ rw, int rows, int topk, int hidden, int min_e, int max_e,
	const uint8_t* __restrict__ owner, float frac, uint32_t* __restrict__ heat, Mailbox* mb, uint64_t* ctl, int layer ) {
	__shared__ uint8_t flag[cpu_moe::kMailboxRoutes];
	__shared__ int total;
	const int n = rows * topk;
	if ( threadIdx.x == 0 ) total = 0;
	for ( int i = threadIdx.x; i < n; i += blockDim.x ) {
		const int e = sel[i];
		flag[i] = ( e >= min_e && e < max_e && owner[e - min_e] ) ? 1 : 0;
	}
	__syncthreads();
	// The CPU takes the last floor( frac * d ) of the d distinct cold experts of the layer (one cold expert is cheaper over PCIe than on the CPU)
	__shared__ int chosen[cpu_moe::kMailboxRoutes];
	__shared__ int nchosen;
	if ( threadIdx.x == 0 ) {
		int distinct[cpu_moe::kMailboxRoutes];
		int d = 0;
		for ( int i = 0; i < n; ++i ) {
			if ( !flag[i] ) continue;
			const int e = sel[i];
			bool seen = false;
			for ( int j = 0; j < d; ++j ) seen |= distinct[j] == e;
			if ( !seen ) distinct[d++] = e;
		}
		const int k = static_cast<int>( frac * d + 1e-4f );
		nchosen = k;
		for ( int j = 0; j < k; ++j ) chosen[j] = distinct[d - 1 - j];
		for ( int i = 0; i < n; ++i ) {
			if ( !flag[i] ) continue;
			bool take = false;
			for ( int j = 0; j < k; ++j ) take |= chosen[j] == sel[i];
			flag[i] = take ? 1 : 0;
		}
	}
	__syncthreads();
	for ( int i = threadIdx.x; i < n; i += blockDim.x ) {
		if ( !flag[i] ) continue;
		int pos = 0;
		for ( int j = 0; j < i; ++j ) pos += flag[j];
		const int e = sel[i];
		MailboxRoute r;
		r.row = i / topk; r.expert = e; r.weight = __half2float( rw[i] );
		mb->routes[pos] = r;
		if ( heat != nullptr ) atomicAdd( heat + ( e - min_e ), 1u );
		sel[i] = -1;
		atomicAdd( &total, 1 );
	}
	__syncthreads();
	if ( total == 0 ) {
		if ( threadIdx.x == 0 ) ctl[0] = 0;
		return;
	}
	uint16_t* mx = reinterpret_cast<uint16_t*>( reinterpret_cast<char*>( mb ) + ( ( sizeof( Mailbox ) + 255 ) & ~size_t( 255 ) ) );
	const uint16_t* xs = reinterpret_cast<const uint16_t*>( x );
	for ( int i = threadIdx.x; i < rows * hidden; i += blockDim.x ) mx[i] = xs[i];
	__threadfence_system();
	__syncthreads();
	if ( threadIdx.x == 0 ) {
		mb->layer = layer; mb->rows = rows; mb->nroutes = total; mb->hidden = hidden;
		const uint64_t seq = ctl[1] + 1;
		ctl[1] = seq;
		ctl[0] = seq;
		__threadfence_system();
		mb->req_seq = seq;
		__threadfence_system();
	}
}

__global__ __launch_bounds__( 256 )
void cpu_wait_add_kernel( half* __restrict__ out, int rows, int hidden, Mailbox* mb, const uint64_t* __restrict__ ctl ) {
	__shared__ int ok;
	const uint64_t seq = ctl[0];
	if ( seq == 0 ) return;
	if ( threadIdx.x == 0 ) {
		ok = 1;
		const uint64_t g0 = global_ns();
		const long long t0 = clock64();
		while ( mb->done_seq < seq ) {
			if ( clock64() - t0 > 12000000000LL ) { mb->err = 1; ok = 0; break; }
		}
		mb->wait_ns += global_ns() - g0;
		mb->waits += 1;
	}
	__syncthreads();
	if ( !ok ) return;
	__threadfence_system();
	const float* mo = reinterpret_cast<const float*>( reinterpret_cast<const char*>( mb ) + ( ( sizeof( Mailbox ) + 255 ) & ~size_t( 255 ) ) + size_t( cpu_moe::kMailboxRows ) * hidden * 2 );
	for ( int i = threadIdx.x; i < rows * hidden; i += blockDim.x ) out[i] = __float2half( __half2float( out[i] ) + __ldcv( mo + i ) );
}

}   // namespace

void CpuExperts::post( const backend::half_t* x, int* sel, const backend::half_t* rw, int rows, int topk, int hidden, int min_expert, int max_expert,
	const uint8_t* owner, float frac, uint32_t* heat, Mailbox* mb, uint64_t* ctl, int layer, backend::stream_t stream ) {
	cpu_post_kernel<<<1, 256, 0, stream>>>( reinterpret_cast<const half*>( x ), sel, reinterpret_cast<const half*>( rw ), rows, topk, hidden, min_expert, max_expert, owner, frac, heat, mb, ctl, layer );
}

void CpuExperts::wait_add( backend::half_t* out, int rows, int hidden, Mailbox* mb, const uint64_t* ctl, backend::stream_t stream ) {
	cpu_wait_add_kernel<<<1, 256, 0, stream>>>( reinterpret_cast<half*>( out ), rows, hidden, mb, ctl );
}

}   // namespace engine::modules
