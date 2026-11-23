#pragma once

// The training models' device memory (engine/train/qwen4_exp.h, engine/train/glm5_next.h): a fixed bump pool, the per-call workspace that
// grows to what the calls need, and the pools of trainable vectors with their optimizer state.
#include "engine/train/adam8.h"
#include "engine/train/grad.h"
#include "engine/config.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace engine::train {

// A bump allocator over one device allocation
struct Pool {
	char* base = nullptr;
	size_t cap = 0, off = 0, peak = 0;
	void init( size_t bytes ) {
		CUDA_CHECK( cudaMalloc( &base, bytes ) );
		cap = bytes;
		off = peak = 0;
	}
	void release() {
		if ( base ) cudaFree( base );
		base = nullptr;
		cap = off = peak = 0;
	}
	void* raw( size_t bytes ) {
		const size_t a = ( off + 255 ) / 256 * 256;
		ENGINE_ASSERT( a + bytes <= cap, "training model: device pool exhausted" );
		off = a + bytes;
		peak = std::max( peak, off );
		return base + a;
	}
	float* f( size_t n ) { return static_cast<float*>( raw( n * sizeof( float ) ) ); }
	float* fz( size_t n, cudaStream_t st ) {
		float* p = f( n );
		CUDA_CHECK( cudaMemsetAsync( p, 0, n * sizeof( float ), st ) );
		return p;
	}
	int* i( size_t n ) { return static_cast<int*>( raw( n * sizeof( int ) ) ); }
};

// The per-call workspace: a bump allocator that grows. A call needing more than it holds gets another device block (the pointers handed
// out stay valid); at the next call's start, nothing live, several blocks become one of the peak's size, so it settles at what the calls
// need (the routed experts' share varies with the routing: a later call may grow it once more)
struct Workspace {
	struct Block {
		char* base;
		size_t start, size;   // its range of the offsets
	};
	std::vector<Block> blocks;
	size_t off = 0, peak = 0;
	void add( size_t start, size_t bytes ) {
		Block b{ nullptr, start, ( bytes + 255 ) / 256 * 256 };
		CUDA_CHECK( cudaMalloc( &b.base, b.size ) );
		blocks.push_back( b );
	}
	void init( size_t bytes ) { add( 0, bytes ); }
	void release() {
		for ( Block& b : blocks ) cudaFree( b.base );
		blocks.clear();
		off = peak = 0;
	}
	size_t cap() const { return blocks.empty() ? 0 : blocks.back().start + blocks.back().size; }
	void reset( cudaStream_t st ) {
		off = 0;
		if ( blocks.size() <= 1 ) return;
		CUDA_CHECK( cudaStreamSynchronize( st ) );
		for ( Block& b : blocks ) CUDA_CHECK( cudaFree( b.base ) );
		blocks.clear();
		add( 0, peak );
	}
	void* raw( size_t bytes ) {
		for ( size_t i = 0;; ) {
			const size_t a = ( off + 255 ) / 256 * 256;
			while ( i < blocks.size() && a >= blocks[i].start + blocks[i].size ) ++i;
			if ( i < blocks.size() && a + bytes <= blocks[i].start + blocks[i].size ) {
				off = a + bytes;
				peak = std::max( peak, off );
				return blocks[i].base + ( a - blocks[i].start );
			}
			if ( i + 1 < blocks.size() ) {
				off = blocks[i + 1].start;   // the rest of block i too small: on to the next
				continue;
			}
			const size_t start = cap();
			add( start, std::max( bytes, std::max<size_t>( 64ull << 20, start / 4 ) ) );
			off = start;
		}
	}
	float* f( size_t n ) { return static_cast<float*>( raw( n * sizeof( float ) ) ); }
	float* fz( size_t n, cudaStream_t st ) {
		float* p = f( n );
		CUDA_CHECK( cudaMemsetAsync( p, 0, n * sizeof( float ), st ) );
		return p;
	}
	int* i( size_t n ) { return static_cast<int*>( raw( n * sizeof( int ) ) ); }
};

// Parallel arrays of trainable scale vectors, a tensor's vectors at the same offset in each: on the device the current values and their
// gradients; the optimizer state (the checkpoint's values, the deltas, Adam's moments, the best values kept) on the device too or, with
// host, in pinned host memory, streamed through the device in parts for each step (the same kernels: the same result); its moments FP32
// or 8-bit block-wise (engine/train/adam8.h) either way
struct ParamPool {
	float *s = nullptr, *g = nullptr;                                                 // device
	float *s0 = nullptr, *a = nullptr, *m = nullptr, *v = nullptr, *best = nullptr;   // the state (m, v: FP32 moments)
	uint8_t *mq = nullptr, *vq = nullptr;                                             // 8-bit moments and their blocks' absmax
	float *mmax = nullptr, *vmax = nullptr;
	// host state: a part's device copy of it (s0, a, the moments and their absmax)
	static constexpr size_t kPart = size_t( 1 ) << 22;
	float *p_s0 = nullptr, *p_a = nullptr, *p_m = nullptr, *p_v = nullptr, *p_mmax = nullptr, *p_vmax = nullptr;
	uint8_t *p_mq = nullptr, *p_vq = nullptr;
	size_t cap = 0, used = 0;
	bool host = false;
	int bits = 32;
	void init( size_t n, bool host_state, int moment_bits ) {
		cap = n;
		used = 0;
		host = host_state;
		bits = moment_bits;
		const size_t n4 = std::max<size_t>( n, 4 ), nb = ( n4 + adam8::kBlock - 1 ) / adam8::kBlock;
		auto dev = [&]( void* p, size_t bytes ) {
			CUDA_CHECK( cudaMalloc( static_cast<void**>( p ), bytes ) );
			CUDA_CHECK( cudaMemset( *static_cast<void**>( p ), 0, bytes ) );
		};
		auto state = [&]( void* p, size_t bytes ) {
			if ( !host ) return dev( p, bytes );
			CUDA_CHECK( cudaMallocHost( static_cast<void**>( p ), bytes ) );
			std::memset( *static_cast<void**>( p ), 0, bytes );
		};
		dev( &s, n4 * sizeof( float ) );
		dev( &g, n4 * sizeof( float ) );
		for ( float** p : { &s0, &a, &best } ) state( p, n4 * sizeof( float ) );
		if ( bits == 32 ) {
			state( &m, n4 * sizeof( float ) );
			state( &v, n4 * sizeof( float ) );
		}
		else {
			state( &mq, n4 );
			state( &vq, n4 );
			state( &mmax, nb * sizeof( float ) );
			state( &vmax, nb * sizeof( float ) );
		}
		if ( !host ) return;
		const size_t P = std::min( n4, kPart );
		for ( float** p : { &p_s0, &p_a } ) dev( p, P * sizeof( float ) );
		if ( bits == 32 ) {
			dev( &p_m, P * sizeof( float ) );
			dev( &p_v, P * sizeof( float ) );
		}
		else {
			dev( &p_mq, P );
			dev( &p_vq, P );
			dev( &p_mmax, ( P / adam8::kBlock + 1 ) * sizeof( float ) );
			dev( &p_vmax, ( P / adam8::kBlock + 1 ) * sizeof( float ) );
		}
	}
	void release() {
		void* d[] = { s, g, p_s0, p_a, p_m, p_v, p_mmax, p_vmax, p_mq, p_vq };
		for ( void* p : d ) if ( p ) cudaFree( p );
		void* st[] = { s0, a, m, v, best, mq, vq, mmax, vmax };
		for ( void* p : st ) if ( p ) host ? static_cast<void>( cudaFreeHost( p ) ) : static_cast<void>( cudaFree( p ) );
		s = g = s0 = a = m = v = best = mmax = vmax = p_s0 = p_a = p_m = p_v = p_mmax = p_vmax = nullptr;
		mq = vq = p_mq = p_vq = nullptr;
		cap = used = 0;
	}
	size_t take( size_t n ) {
		const size_t o = used;
		used += ( n + 3 ) / 4 * 4;   // float4 aligned
		ENGINE_ASSERT( used <= cap, "training model: scale pool exhausted" );
		return o;
	}
	bool holds( const float* p ) const { return p >= s && p < s + cap; }
	// After the load (s: the checkpoint's values): the reference values s0
	void start( cudaStream_t st ) {
		if ( used == 0 ) return;
		CUDA_CHECK( cudaMemcpyAsync( s0, s, used * sizeof( float ), host ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice, st ) );
	}
	// One Adam step from the gradients g (log: s = s0 e^a, else s = s0 + a), over [o0, o0 + n) of the state at s0 ... (device pointers)
	void step_part( bool log, const AdamOptions& o, int k, float gs, cudaStream_t st, size_t o0, size_t n, float* ps0, float* pa, float* pm, float* pv,
		uint8_t* pmq, uint8_t* pvq, float* pmm, float* pvm ) {
		if ( bits == 32 && log ) adam_log_scales( ps0, pa, s + o0, g + o0, pm, pv, n, k, gs, o, st );
		else if ( bits == 32 ) adam_deltas( ps0, pa, s + o0, g + o0, pm, pv, n, k, gs, o, st );
		else if ( log ) adam_log_scales8( ps0, pa, s + o0, g + o0, pmq, pvq, pmm, pvm, n, k, gs, o, st, o0 );
		else adam_deltas8( ps0, pa, s + o0, g + o0, pmq, pvq, pmm, pvm, n, k, gs, o, st, o0 );
	}
	void step( bool log, const AdamOptions& o, int k, float gs, cudaStream_t st ) {
		if ( used == 0 ) return;
		if ( !host ) return step_part( log, o, k, gs, st, 0, used, s0, a, m, v, mq, vq, mmax, vmax );
		// the host state part by part through the device: in, the step, out
		const auto h2d = cudaMemcpyHostToDevice, d2h = cudaMemcpyDeviceToHost;
		for ( size_t o0 = 0; o0 < used; o0 += kPart ) {
			const size_t n = std::min( kPart, used - o0 ), b0 = o0 / adam8::kBlock, nb = ( n + adam8::kBlock - 1 ) / adam8::kBlock;
			CUDA_CHECK( cudaMemcpyAsync( p_s0, s0 + o0, n * sizeof( float ), h2d, st ) );
			CUDA_CHECK( cudaMemcpyAsync( p_a, a + o0, n * sizeof( float ), h2d, st ) );
			if ( bits == 32 ) {
				CUDA_CHECK( cudaMemcpyAsync( p_m, m + o0, n * sizeof( float ), h2d, st ) );
				CUDA_CHECK( cudaMemcpyAsync( p_v, v + o0, n * sizeof( float ), h2d, st ) );
			}
			else {
				CUDA_CHECK( cudaMemcpyAsync( p_mq, mq + o0, n, h2d, st ) );
				CUDA_CHECK( cudaMemcpyAsync( p_vq, vq + o0, n, h2d, st ) );
				CUDA_CHECK( cudaMemcpyAsync( p_mmax, mmax + b0, nb * sizeof( float ), h2d, st ) );
				CUDA_CHECK( cudaMemcpyAsync( p_vmax, vmax + b0, nb * sizeof( float ), h2d, st ) );
			}
			step_part( log, o, k, gs, st, o0, n, p_s0, p_a, p_m, p_v, p_mq, p_vq, p_mmax, p_vmax );
			CUDA_CHECK( cudaMemcpyAsync( a + o0, p_a, n * sizeof( float ), d2h, st ) );
			if ( bits == 32 ) {
				CUDA_CHECK( cudaMemcpyAsync( m + o0, p_m, n * sizeof( float ), d2h, st ) );
				CUDA_CHECK( cudaMemcpyAsync( v + o0, p_v, n * sizeof( float ), d2h, st ) );
			}
			else {
				CUDA_CHECK( cudaMemcpyAsync( mq + o0, p_mq, n, d2h, st ) );
				CUDA_CHECK( cudaMemcpyAsync( vq + o0, p_vq, n, d2h, st ) );
				CUDA_CHECK( cudaMemcpyAsync( mmax + b0, p_mmax, nb * sizeof( float ), d2h, st ) );
				CUDA_CHECK( cudaMemcpyAsync( vmax + b0, p_vmax, nb * sizeof( float ), d2h, st ) );
			}
		}
	}
	void keep_best( cudaStream_t st ) {
		if ( used == 0 ) return;
		CUDA_CHECK( cudaMemcpyAsync( best, s, used * sizeof( float ), host ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice, st ) );
	}
	// The kept values on the host
	std::vector<float> best_values() const {
		std::vector<float> h( used );
		if ( used == 0 ) return h;
		if ( host ) std::memcpy( h.data(), best, used * sizeof( float ) );
		else CUDA_CHECK( cudaMemcpy( h.data(), best, used * sizeof( float ), cudaMemcpyDeviceToHost ) );
		return h;
	}
	size_t state_bytes( size_t n ) const {
		const size_t n4 = std::max<size_t>( n, 4 ), nb = ( n4 + adam8::kBlock - 1 ) / adam8::kBlock;
		return n4 * 3 * sizeof( float ) + ( bits == 32 ? n4 * 2 * sizeof( float ) : n4 * 2 + nb * 2 * sizeof( float ) );
	}
	size_t device_bytes() const {
		const size_t P = std::min( std::max<size_t>( cap, 4 ), kPart );
		const size_t part = P * 2 * sizeof( float ) + ( bits == 32 ? P * 2 * sizeof( float ) : P * 2 + ( P / adam8::kBlock + 1 ) * 2 * sizeof( float ) );
		return std::max<size_t>( cap, 4 ) * 2 * sizeof( float ) + ( host ? part : state_bytes( cap ) );
	}
	size_t host_bytes() const { return host ? state_bytes( cap ) : 0; }
};

} // namespace engine::train
