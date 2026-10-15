// Shared token pool (kv_pool): the latent cache and the indexer pools addressed through a page table must give the bytes of the per-slot
// layout. Two slots of different lengths are filled in both layouts, the pool's pages handed out in a shuffled order, then store + attend
// (decode rows of both slots, a verify block, a dense prefill chunk, selected-key lists) and the indexer's pool_update + score are compared
// bit for bit. Needs no model files.
#include "engine/config.h"
#include "engine/core/batch.h"
#include "engine/core/kv_format.h"
#include "engine/modules/dsa.h"
#include "engine/modules/mla.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

using engine::modules::DSA;
using engine::modules::MLALatent;

static std::mt19937 g_rng( 20261014 );
static float rnd() { return std::uniform_real_distribution<float>( -1.0f, 1.0f )( g_rng ); }

template <typename T>
static T* dev( size_t n ) {
	T* p = nullptr;
	CUDA_CHECK( cudaMalloc( &p, n * sizeof( T ) ) );
	CUDA_CHECK( cudaMemset( p, 0, n * sizeof( T ) ) );
	return p;
}

int main() {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	CUDA_CHECK( cudaSetDevice( 0 ) );
	constexpr int LORA = 512, H = 16, S = 2, L = 3, layer = 1, SHIFT = 9, PAGE = 1 << SHIFT;
	const int max_seq = 4096, per_slot = max_seq / PAGE;
	const int pool_tokens = S * max_seq + 2 * PAGE;   // two spare pages
	const int N[S] = { 3000, 2600 };
	bool all_ok = true;

	// Physical pages: a shuffle of the pool, a slot's logical pages taking them in turn
	std::vector<int> perm( pool_tokens / PAGE );
	std::iota( perm.begin(), perm.end(), 0 );
	std::shuffle( perm.begin(), perm.end(), g_rng );
	std::vector<int> table( static_cast<size_t>( S ) * per_slot, 0 );
	for ( int s = 0; s < S; ++s ) {
		for ( int i = 0; i < ( N[s] + PAGE - 1 ) / PAGE; ++i ) table[static_cast<size_t>( s ) * per_slot + i] = perm[s * per_slot + i];
	}
	int* d_table = dev<int>( table.size() );
	CUDA_CHECK( cudaMemcpy( d_table, table.data(), table.size() * sizeof( int ), cudaMemcpyHostToDevice ) );
	pod::KVPages pages;
	pages.table = d_table;
	pages.shift = SHIFT;
	pages.per_slot = per_slot;
	pages.pool_tokens = pool_tokens;

	pod::BatchRows* d_rows = dev<pod::BatchRows>( 1 );
	auto upload = [&]( const pod::BatchRows& br ) { CUDA_CHECK( cudaMemcpy( d_rows, &br, sizeof( br ), cudaMemcpyHostToDevice ) ); };

	// ---------------------------------------------------------------- latent cache
	const pod::KVQ fmts[4] = { pod::KVQ::F16, pod::KVQ::E5M2, pod::KVQ::I8, pod::KVQ::FP4 };
	for ( const pod::KVQ fq : fmts ) {
		const size_t slot_stride = static_cast<size_t>( L ) * max_seq * LORA;
		const size_t pool_elems = static_cast<size_t>( L ) * pool_tokens * LORA;
		void* cache[2] = { dev<char>( pod::kv_data_bytes( fq, S * slot_stride ) ), dev<char>( pod::kv_data_bytes( fq, pool_elems ) ) };
		void* scale[2] = { nullptr, nullptr };
		if ( pod::kv_scale_bytes( fq, pool_elems ) ) {
			scale[0] = dev<char>( pod::kv_scale_bytes( fq, S * slot_stride ) );
			scale[1] = dev<char>( pod::kv_scale_bytes( fq, pool_elems ) );
		}
		pod::KVRef kv[2];
		for ( int v = 0; v < 2; ++v ) {
			kv[v].k = kv[v].v = cache[v];
			kv[v].ks = kv[v].vs = scale[v];
			kv[v].kq = kv[v].vq = fq;
		}
		kv[1].pages = pages;

		// Latents of both slots, stored in chunks of 256 rows
		std::vector<half> h_c( static_cast<size_t>( max_seq ) * LORA );
		half* d_c = dev<half>( h_c.size() );
		for ( int s = 0; s < S; ++s ) {
			for ( auto& x : h_c ) x = __float2half( rnd() * 1.7f );
			CUDA_CHECK( cudaMemcpy( d_c, h_c.data(), h_c.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
			for ( int p0 = 0; p0 < N[s]; p0 += 256 ) {
				const int n = std::min( 256, N[s] - p0 );
				upload( pod::BatchRows::causal( n, p0, s ) );
				pod::SlotRef sr;
				sr.rows = d_rows;
				sr.stride = slot_stride;
				for ( int v = 0; v < 2; ++v ) MLALatent::store( kv[v], pod::make_tensor_2d( d_c + static_cast<size_t>( p0 ) * LORA, pod::DType::FP16, n, LORA ), layer, max_seq, n, sr, nullptr );
			}
		}

		// Query scenarios: { rows, slots, positions }
		struct Case { const char* name; std::vector<int> slots, pos; bool sel; };
		std::vector<Case> cases;
		cases.push_back( { "decode, 2 slots", { 0, 1 }, { N[0] - 1, N[1] - 1 }, false } );
		{
			Case c{ "verify block", {}, {}, false };
			for ( int r = 0; r < 8; ++r ) { c.slots.push_back( 0 ); c.pos.push_back( 1500 + r ); }
			cases.push_back( c );
		}
		{
			Case c{ "prefill chunk", {}, {}, false };
			for ( int r = 0; r < 40; ++r ) { c.slots.push_back( 1 ); c.pos.push_back( 1200 + r ); }
			cases.push_back( c );
		}
		{
			Case c{ "selected keys, decode", {}, {}, true };
			for ( int r = 0; r < 4; ++r ) { c.slots.push_back( r & 1 ); c.pos.push_back( N[r & 1] - 1 - r ); }
			cases.push_back( c );
		}
		{
			Case c{ "selected keys, prefill", {}, {}, true };
			for ( int r = 0; r < 24; ++r ) { c.slots.push_back( 0 ); c.pos.push_back( 2400 + r ); }
			cases.push_back( c );
		}
		for ( const Case& c : cases ) {
			const int R = static_cast<int>( c.pos.size() );
			std::vector<half> h_q( static_cast<size_t>( R ) * H * LORA );
			for ( auto& x : h_q ) x = __float2half( rnd() );
			half* d_q = dev<half>( h_q.size() );
			CUDA_CHECK( cudaMemcpy( d_q, h_q.data(), h_q.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
			std::vector<int> sel( static_cast<size_t>( R ) * DSA::KPAD, 0 ), sel_n( R, -1 );
			int *d_sel = nullptr, *d_sel_n = nullptr;
			if ( c.sel ) {
				for ( int r = 0; r < R; ++r ) {
					sel_n[r] = 700 + 40 * r;
					for ( int j = 0; j < sel_n[r]; ++j ) sel[static_cast<size_t>( r ) * DSA::KPAD + j] = static_cast<int>( g_rng() % static_cast<uint32_t>( c.pos[r] + 1 ) );
				}
				d_sel = dev<int>( sel.size() );
				d_sel_n = dev<int>( R );
				CUDA_CHECK( cudaMemcpy( d_sel, sel.data(), sel.size() * sizeof( int ), cudaMemcpyHostToDevice ) );
				CUDA_CHECK( cudaMemcpy( d_sel_n, sel_n.data(), R * sizeof( int ), cudaMemcpyHostToDevice ) );
			}
			upload( pod::BatchRows::batched( R, c.slots.data(), c.pos.data() ) );
			std::vector<half> out[2];
			for ( int v = 0; v < 2; ++v ) {
				half* d_o = dev<half>( static_cast<size_t>( R ) * H * LORA );
				float* d_ws = dev<float>( MLALatent::workspace_floats( max_seq, H ) );
				pod::SlotRef sr;
				sr.rows = d_rows;
				sr.stride = slot_stride;
				pod::TensorView qt = pod::make_tensor_2d( d_q, pod::DType::FP16, R, H * LORA ), ot = pod::make_tensor_2d( d_o, pod::DType::FP16, R, H * LORA );
				MLALatent::attend( kv[v], qt, ot, layer, max_seq, R, H, sr, d_ws, c.pos.back() + 1, nullptr, d_sel, d_sel_n );
				CUDA_CHECK( cudaDeviceSynchronize() );
				out[v].resize( static_cast<size_t>( R ) * H * LORA );
				CUDA_CHECK( cudaMemcpy( out[v].data(), d_o, out[v].size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
				CUDA_CHECK( cudaFree( d_o ) );
				CUDA_CHECK( cudaFree( d_ws ) );
			}
			const bool same = std::memcmp( out[0].data(), out[1].data(), out[0].size() * sizeof( half ) ) == 0;
			std::printf( "  latent kq=%d %-24s %s\n", static_cast<int>( fq ), c.name, same ? "identical" : "DIFFERENT" );
			all_ok = all_ok && same;
			CUDA_CHECK( cudaFree( d_q ) );
			if ( d_sel ) { CUDA_CHECK( cudaFree( d_sel ) ); CUDA_CHECK( cudaFree( d_sel_n ) ); }
		}
		// The pages the slots do not own stay untouched: the spare pages hold zeros
		CUDA_CHECK( cudaFree( d_c ) );
		for ( int v = 0; v < 2; ++v ) { CUDA_CHECK( cudaFree( cache[v] ) ); if ( scale[v] ) CUDA_CHECK( cudaFree( scale[v] ) ); }
	}

	// ---------------------------------------------------------------- indexer pools
	{
		constexpr int D = DSA::DIM, P = DSA::POOL, HH = DSA::HEADS;
		const int NP = max_seq / P;
		const size_t plane_stride = static_cast<size_t>( L ) * DSA::PLANE_CAP * DSA::PLANE;
		const size_t pool_stride = static_cast<size_t>( L ) * NP * D;
		half* plane[2] = { dev<half>( S * plane_stride ), dev<half>( S * plane_stride ) };
		half* pool[2] = { dev<half>( S * pool_stride ), dev<half>( static_cast<size_t>( L ) * ( pool_tokens / P ) * D ) };
		half *d_proj = dev<half>( static_cast<size_t>( max_seq ) * DSA::PROJ ), *d_lnw = dev<half>( D ), *d_lnb = dev<half>( D ), *d_ape = dev<half>( P * D ), *d_w = dev<half>( 512 * HH );
		std::vector<half> h( static_cast<size_t>( max_seq ) * DSA::PROJ );
		auto fill = [&]( half* d, size_t n, float amp, float off ) {
			std::vector<half> v( n );
			for ( auto& x : v ) x = __float2half( off + amp * rnd() );
			CUDA_CHECK( cudaMemcpy( d, v.data(), n * sizeof( half ), cudaMemcpyHostToDevice ) );
		};
		fill( d_lnw, D, 0.2f, 1.0f );
		fill( d_lnb, D, 0.1f, 0.0f );
		fill( d_ape, P * D, 0.5f, 0.0f );
		for ( int s = 0; s < S; ++s ) {
			fill( d_proj, static_cast<size_t>( max_seq ) * DSA::PROJ, 1.3f, 0.0f );
			for ( int p0 = 0; p0 < N[s]; p0 += 500 ) {
				const int n = std::min( 500, N[s] - p0 );
				upload( pod::BatchRows::causal( n, p0, s ) );
				for ( int v = 0; v < 2; ++v ) {
					DSA::prep( d_proj + static_cast<size_t>( p0 ) * DSA::PROJ, d_lnw, d_lnb, plane[v], d_w, layer, max_seq, n, d_rows, plane_stride, nullptr );
					DSA::pool_update( plane[v], pool[v], d_ape, layer, max_seq, n, d_rows, plane_stride, pool_stride, nullptr, v ? pages : pod::KVPages{} );
				}
			}
		}
		const int R = 4;
		const int slots[R] = { 0, 1, 0, 1 }, poss[R] = { N[0] - 1, N[1] - 1, N[0] - 40, N[1] - 3 };
		upload( pod::BatchRows::batched( R, slots, poss ) );
		half* d_q = dev<half>( static_cast<size_t>( R ) * HH * D );
		fill( d_q, static_cast<size_t>( R ) * HH * D, 0.9f, 0.0f );
		fill( d_w, static_cast<size_t>( R ) * HH, 1.0f, 0.0f );
		std::vector<half> sc[2];
		for ( int v = 0; v < 2; ++v ) {
			half* d_sc = dev<half>( static_cast<size_t>( R ) * NP );
			DSA::score( d_q, d_w, pool[v], d_sc, NP, layer, max_seq, R, d_rows, pool_stride, nullptr, v ? pages : pod::KVPages{} );
			CUDA_CHECK( cudaDeviceSynchronize() );
			sc[v].resize( static_cast<size_t>( R ) * NP );
			CUDA_CHECK( cudaMemcpy( sc[v].data(), d_sc, sc[v].size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
			CUDA_CHECK( cudaFree( d_sc ) );
		}
		bool nonzero = false;
		for ( const half x : sc[0] ) nonzero = nonzero || __half2float( x ) != 0.0f;
		const bool same = std::memcmp( sc[0].data(), sc[1].data(), sc[0].size() * sizeof( half ) ) == 0;
		std::printf( "  indexer pool_update + score: %s%s\n", same ? "identical" : "DIFFERENT", nonzero ? "" : " (all zero: the test scored nothing)" );
		all_ok = all_ok && same && nonzero;
	}

	std::printf( all_ok ? "[+] PASS paged latent cache and indexer pools\n" : "[-] FAIL paged layout differs from the per-slot layout\n" );
	return all_ok ? 0 : 1;
}
