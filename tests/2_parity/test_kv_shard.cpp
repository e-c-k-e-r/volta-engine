// kv_shard: GLM's latent cache spread over the GPUs (logical page lp of a slot on rank lp % ranks, each rank storing the rows of its own pages)
// must give the bytes of the shared pool on one GPU. Two slots are stored on every rank and in a one-GPU pool, then attend runs on the first
// and the last rank, reading the other ranks' pages over peer access (decode rows of both slots, a verify block, a dense prefill chunk,
// selected-key lists), in every latent format, compared bit for bit. Needs 2 or more GPUs (4 ranks with 4), no model files.
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

static std::mt19937 g_rng( 20261124 );
static float rnd() { return std::uniform_real_distribution<float>( -1.0f, 1.0f )( g_rng ); }

template <typename T>
static T* dev( size_t n ) {
	T* p = nullptr;
	CUDA_CHECK( cudaMalloc( &p, n * sizeof( T ) ) );
	CUDA_CHECK( cudaMemset( p, 0, n * sizeof( T ) ) );
	return p;
}

template <typename T>
static T* upload_vec( const std::vector<T>& v ) {
	T* d = dev<T>( v.size() );
	CUDA_CHECK( cudaMemcpy( d, v.data(), v.size() * sizeof( T ), cudaMemcpyHostToDevice ) );
	return d;
}

int main() {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	int devices = 0;
	CUDA_CHECK( cudaGetDeviceCount( &devices ) );
	if ( devices < 2 ) {
		std::printf( "[!] SKIP test_kv_shard: needs 2 GPUs, %d visible\n", devices );
		return 0;
	}
	const int RK = devices >= 4 ? 4 : 2;
	for ( int a = 0; a < RK; ++a ) {
		CUDA_CHECK( cudaSetDevice( a ) );
		for ( int b = 0; b < RK; ++b ) {
			if ( a == b ) continue;
			const cudaError_t e = cudaDeviceEnablePeerAccess( b, 0 );
			if ( e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled ) CUDA_CHECK( e );
			(void) cudaGetLastError();
		}
	}
	constexpr int LORA = 512, H = 16, S = 2, L = 3, layer = 1, SHIFT = 9, PAGE = 1 << SHIFT;
	const int max_seq = 4096, per_slot = max_seq / PAGE;
	const int pool_tokens = S * max_seq + 2 * PAGE;
	const int N[S] = { 3000, 2600 };
	const int shard_pages = ( pool_tokens / PAGE + RK - 1 ) / RK + S;
	bool all_ok = true;

	// The one-GPU pool: a shuffle of its pages. The shards: per owner rank a shuffle of its own pages
	std::vector<int> perm( pool_tokens / PAGE );
	std::iota( perm.begin(), perm.end(), 0 );
	std::shuffle( perm.begin(), perm.end(), g_rng );
	std::vector<std::vector<int>> own( RK, std::vector<int>( shard_pages ) );
	for ( auto& o : own ) {
		std::iota( o.begin(), o.end(), 0 );
		std::shuffle( o.begin(), o.end(), g_rng );
	}
	std::vector<int> table( static_cast<size_t>( S ) * per_slot, 0 ), shard_table( table.size(), 0 );
	std::vector<int> taken( RK, 0 );
	for ( int s = 0; s < S; ++s ) {
		for ( int lp = 0; lp < ( N[s] + PAGE - 1 ) / PAGE; ++lp ) {
			table[static_cast<size_t>( s ) * per_slot + lp] = perm[s * per_slot + lp];
			shard_table[static_cast<size_t>( s ) * per_slot + lp] = own[lp % RK][taken[lp % RK]++];
		}
	}

	// Device state: index 0..RK-1 the shard ranks, RK the one-GPU pool (on GPU 0)
	const int V = RK + 1;
	auto gpu = [&]( int v ) { return v < RK ? v : 0; };
	std::vector<int*> d_table( V ), d_shard( V );
	std::vector<pod::BatchRows*> d_rows( V );
	for ( int v = 0; v < V; ++v ) {
		CUDA_CHECK( cudaSetDevice( gpu( v ) ) );
		d_table[v] = upload_vec( table );
		d_shard[v] = upload_vec( shard_table );
		d_rows[v] = dev<pod::BatchRows>( 1 );
	}
	auto upload_rows = [&]( int v, const pod::BatchRows& br ) {
		CUDA_CHECK( cudaSetDevice( gpu( v ) ) );
		CUDA_CHECK( cudaMemcpy( d_rows[v], &br, sizeof( br ), cudaMemcpyHostToDevice ) );
	};

	const pod::KVQ fmts[4] = { pod::KVQ::F16, pod::KVQ::E5M2, pod::KVQ::I8, pod::KVQ::FP4 };
	for ( const pod::KVQ fq : fmts ) {
		const size_t shard_elems = static_cast<size_t>( L ) * shard_pages * PAGE * LORA;
		const size_t pool_elems = static_cast<size_t>( L ) * pool_tokens * LORA;
		const bool scaled = pod::kv_scale_bytes( fq, pool_elems ) > 0;
		std::vector<void*> data( V, nullptr ), scales( V, nullptr );
		for ( int v = 0; v < V; ++v ) {
			CUDA_CHECK( cudaSetDevice( gpu( v ) ) );
			const size_t elems = v < RK ? shard_elems : pool_elems;
			data[v] = dev<char>( pod::kv_data_bytes( fq, elems ) );
			if ( scaled ) scales[v] = dev<char>( pod::kv_scale_bytes( fq, elems ) );
		}
		std::vector<void*> bases( 2 * RK );
		for ( int r = 0; r < RK; ++r ) {
			bases[r] = data[r];
			bases[RK + r] = scales[r];
		}
		std::vector<void**> d_bases( RK );
		std::vector<pod::KVRef> kv( V );
		for ( int v = 0; v < V; ++v ) {
			kv[v].k = kv[v].v = data[v];
			kv[v].ks = kv[v].vs = scales[v];
			kv[v].kq = kv[v].vq = fq;
			kv[v].pages.table = d_table[v];
			kv[v].pages.shift = SHIFT;
			kv[v].pages.per_slot = per_slot;
			kv[v].pages.pool_tokens = pool_tokens;
			if ( v < RK ) {
				CUDA_CHECK( cudaSetDevice( v ) );
				d_bases[v] = upload_vec( bases );
				kv[v].pages.shard_table = d_shard[v];
				kv[v].pages.shard_bases = d_bases[v];
				kv[v].pages.ranks = RK;
				kv[v].pages.rank = v;
				kv[v].pages.shard_pool_tokens = static_cast<size_t>( shard_pages ) * PAGE;
			}
		}

		// Latents of both slots, stored in chunks of 256 rows on every rank and in the one-GPU pool
		std::vector<half> h_c( static_cast<size_t>( max_seq ) * LORA );
		std::vector<half*> d_c( V );
		for ( int v = 0; v < V; ++v ) {
			CUDA_CHECK( cudaSetDevice( gpu( v ) ) );
			d_c[v] = dev<half>( h_c.size() );
		}
		for ( int s = 0; s < S; ++s ) {
			for ( auto& x : h_c ) x = __float2half( rnd() * 1.7f );
			for ( int v = 0; v < V; ++v ) {
				CUDA_CHECK( cudaSetDevice( gpu( v ) ) );
				CUDA_CHECK( cudaMemcpy( d_c[v], h_c.data(), h_c.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
			}
			for ( int p0 = 0; p0 < N[s]; p0 += 256 ) {
				const int n = std::min( 256, N[s] - p0 );
				for ( int v = 0; v < V; ++v ) {
					upload_rows( v, pod::BatchRows::causal( n, p0, s ) );
					pod::SlotRef sr;
					sr.rows = d_rows[v];
					sr.stride = 0;
					MLALatent::store( kv[v], pod::make_tensor_2d( d_c[v] + static_cast<size_t>( p0 ) * LORA, pod::DType::FP16, n, LORA ), layer, max_seq, n, sr, nullptr );
					CUDA_CHECK( cudaDeviceSynchronize() );
				}
			}
		}

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
			std::vector<int> sel( static_cast<size_t>( R ) * DSA::KPAD, 0 ), sel_n( R, -1 );
			if ( c.sel ) {
				for ( int r = 0; r < R; ++r ) {
					sel_n[r] = 700 + 40 * r;
					for ( int j = 0; j < sel_n[r]; ++j ) sel[static_cast<size_t>( r ) * DSA::KPAD + j] = static_cast<int>( g_rng() % static_cast<uint32_t>( c.pos[r] + 1 ) );
				}
			}
			// The one-GPU pool, then attend on the first and the last rank
			const int runs[3] = { RK, 0, RK - 1 };
			std::vector<half> out[3];
			for ( int i = 0; i < 3; ++i ) {
				const int v = runs[i];
				upload_rows( v, pod::BatchRows::batched( R, c.slots.data(), c.pos.data() ) );
				half* d_q = upload_vec( h_q );
				int* d_sel = c.sel ? upload_vec( sel ) : nullptr;
				int* d_sel_n = c.sel ? upload_vec( sel_n ) : nullptr;
				half* d_o = dev<half>( static_cast<size_t>( R ) * H * LORA );
				float* d_ws = dev<float>( MLALatent::workspace_floats( max_seq, H ) );
				pod::SlotRef sr;
				sr.rows = d_rows[v];
				sr.stride = 0;
				pod::TensorView qt = pod::make_tensor_2d( d_q, pod::DType::FP16, R, H * LORA ), ot = pod::make_tensor_2d( d_o, pod::DType::FP16, R, H * LORA );
				MLALatent::attend( kv[v], qt, ot, layer, max_seq, R, H, sr, d_ws, c.pos.back() + 1, nullptr, d_sel, d_sel_n );
				CUDA_CHECK( cudaDeviceSynchronize() );
				out[i].resize( static_cast<size_t>( R ) * H * LORA );
				CUDA_CHECK( cudaMemcpy( out[i].data(), d_o, out[i].size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
				CUDA_CHECK( cudaFree( d_o ) );
				CUDA_CHECK( cudaFree( d_ws ) );
				CUDA_CHECK( cudaFree( d_q ) );
				if ( d_sel ) { CUDA_CHECK( cudaFree( d_sel ) ); CUDA_CHECK( cudaFree( d_sel_n ) ); }
			}
			bool nonzero = false;
			for ( const half x : out[0] ) nonzero = nonzero || __half2float( x ) != 0.0f;
			for ( int i = 1; i < 3; ++i ) {
				const bool same = std::memcmp( out[0].data(), out[i].data(), out[0].size() * sizeof( half ) ) == 0;
				std::printf( "  latent kq=%d %-24s rank %d of %d: %s%s\n", static_cast<int>( fq ), c.name, runs[i], RK, same ? "identical" : "DIFFERENT",
					nonzero ? "" : " (all zero)" );
				all_ok = all_ok && same && nonzero;
			}
		}
		for ( int v = 0; v < V; ++v ) {
			CUDA_CHECK( cudaSetDevice( gpu( v ) ) );
			CUDA_CHECK( cudaFree( d_c[v] ) );
			CUDA_CHECK( cudaFree( data[v] ) );
			if ( scales[v] ) CUDA_CHECK( cudaFree( scales[v] ) );
			if ( v < RK ) CUDA_CHECK( cudaFree( d_bases[v] ) );
		}
	}

	// ---------------------------------------------------------------- indexer: pools, scores, local candidates and their merge
	{
		constexpr int Dd = DSA::DIM, Pp = DSA::POOL, HH = DSA::HEADS;
		const int mseq = 16384, ps = mseq / PAGE, NP = mseq / Pp;
		const int NI[S] = { 12000, 9000 };   // above 512 pools per rank at 4 ranks: the local selection keeps a top 512, not everything
		const int ptok = S * mseq + 2 * PAGE, spages = ( ptok / PAGE + RK - 1 ) / RK + S;
		std::vector<int> perm2( ptok / PAGE );
		std::iota( perm2.begin(), perm2.end(), 0 );
		std::shuffle( perm2.begin(), perm2.end(), g_rng );
		std::vector<std::vector<int>> own2( RK, std::vector<int>( spages ) );
		for ( auto& o : own2 ) {
			std::iota( o.begin(), o.end(), 0 );
			std::shuffle( o.begin(), o.end(), g_rng );
		}
		std::vector<int> t2( static_cast<size_t>( S ) * ps, 0 ), st2( t2.size(), 0 ), tk( RK, 0 );
		for ( int s = 0; s < S; ++s ) {
			for ( int lp = 0; lp < ( NI[s] + PAGE - 1 ) / PAGE; ++lp ) {
				t2[static_cast<size_t>( s ) * ps + lp] = perm2[s * ps + lp];
				st2[static_cast<size_t>( s ) * ps + lp] = own2[lp % RK][tk[lp % RK]++];
			}
		}
		const size_t plane_stride = static_cast<size_t>( L ) * DSA::PLANE_CAP * DSA::PLANE;
		const int sstride = DSA::shard_score_stride( mseq, RK );
		// Inputs shared by every device
		std::vector<std::vector<half>> proj( S, std::vector<half>( static_cast<size_t>( mseq ) * DSA::PROJ ) );
		for ( auto& v : proj ) for ( auto& x : v ) x = __float2half( 1.3f * rnd() );
		std::vector<half> lnw( Dd ), lnb( Dd ), ape( Pp * Dd );
		for ( auto& x : lnw ) x = __float2half( 1.0f + 0.2f * rnd() );
		for ( auto& x : lnb ) x = __float2half( 0.1f * rnd() );
		for ( auto& x : ape ) x = __float2half( 0.5f * rnd() );
		const int R = 6;
		const int slots[R] = { 0, 1, 0, 1, 0, 0 }, poss[R] = { NI[0] - 1, NI[1] - 1, NI[0] - 40, NI[1] - 3, 2100, 6000 };
		std::vector<half> hq( static_cast<size_t>( R ) * HH * Dd ), hw( static_cast<size_t>( R ) * HH );
		for ( auto& x : hq ) x = __float2half( 0.9f * rnd() );
		for ( auto& x : hw ) x = __float2half( rnd() );
		struct Dev { int *table, *shard; half *plane, *pool, *proj, *lnw, *lnb, *ape, *q, *w, *sc; int *cid, *cn, *idx, *nk; unsigned short* ckey; void** cptr; pod::BatchRows* rows; pod::KVPages pg; };
		std::vector<Dev> dv( V );
		for ( int v = 0; v < V; ++v ) {
			CUDA_CHECK( cudaSetDevice( gpu( v ) ) );
			Dev& d = dv[v];
			d.table = upload_vec( t2 );
			d.shard = upload_vec( st2 );
			d.plane = dev<half>( S * plane_stride );
			d.pool = dev<half>( static_cast<size_t>( L ) * ( ( v < RK ? spages * PAGE : ptok ) / Pp ) * Dd );
			d.proj = dev<half>( static_cast<size_t>( mseq ) * DSA::PROJ );
			d.lnw = upload_vec( lnw );
			d.lnb = upload_vec( lnb );
			d.ape = upload_vec( ape );
			d.q = upload_vec( hq );
			d.w = dev<half>( static_cast<size_t>( 512 ) * HH );   // prep writes a chunk's head weights (500 rows)
			d.sc = dev<half>( static_cast<size_t>( R ) * ( v < RK ? sstride : NP ) );
			d.cid = dev<int>( static_cast<size_t>( R ) * DSA::SEL_POOLS );
			d.ckey = dev<unsigned short>( static_cast<size_t>( R ) * DSA::SEL_POOLS );
			d.cn = dev<int>( R );
			d.idx = dev<int>( static_cast<size_t>( R ) * DSA::KPAD );
			d.nk = dev<int>( R );
			d.rows = dev<pod::BatchRows>( 1 );
			d.pg.table = d.table;
			d.pg.shift = SHIFT;
			d.pg.per_slot = ps;
			d.pg.pool_tokens = ptok;
			if ( v < RK ) {
				d.pg.shard_table = d.shard;
				d.pg.ranks = RK;
				d.pg.rank = v;
				d.pg.shard_pool_tokens = static_cast<size_t>( spages ) * PAGE;
			}
		}
		for ( int v = 0; v < RK; ++v ) {
			std::vector<void*> cp( 3 * RK );
			for ( int o = 0; o < RK; ++o ) {
				cp[o] = dv[o].cid;
				cp[RK + o] = dv[o].ckey;
				cp[2 * RK + o] = dv[o].cn;
			}
			CUDA_CHECK( cudaSetDevice( v ) );
			dv[v].cptr = upload_vec( cp );
		}
		auto put_rows = [&]( int v, const pod::BatchRows& br ) {
			CUDA_CHECK( cudaSetDevice( gpu( v ) ) );
			CUDA_CHECK( cudaMemcpy( dv[v].rows, &br, sizeof( br ), cudaMemcpyHostToDevice ) );
		};
		// The planes and pools of both slots, in chunks of 500 rows, on every rank and the one-GPU pool
		for ( int s = 0; s < S; ++s ) {
			for ( int v = 0; v < V; ++v ) {
				CUDA_CHECK( cudaSetDevice( gpu( v ) ) );
				CUDA_CHECK( cudaMemcpy( dv[v].proj, proj[s].data(), proj[s].size() * sizeof( half ), cudaMemcpyHostToDevice ) );
			}
			for ( int p0 = 0; p0 < NI[s]; p0 += 500 ) {
				const int n = std::min( 500, NI[s] - p0 );
				for ( int v = 0; v < V; ++v ) {
					put_rows( v, pod::BatchRows::causal( n, p0, s ) );
					Dev& d = dv[v];
					DSA::prep( d.proj + static_cast<size_t>( p0 ) * DSA::PROJ, d.lnw, d.lnb, d.plane, d.w, layer, mseq, n, d.rows, plane_stride, nullptr );
					DSA::pool_update( d.plane, d.pool, d.ape, layer, mseq, n, d.rows, plane_stride, 0, nullptr, d.pg );
					CUDA_CHECK( cudaDeviceSynchronize() );
				}
			}
		}
		// prep wrote the head weights of the last chunk into w: put the query's back
		for ( int v = 0; v < V; ++v ) {
			CUDA_CHECK( cudaSetDevice( gpu( v ) ) );
			CUDA_CHECK( cudaMemcpy( dv[v].w, hw.data(), hw.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
			put_rows( v, pod::BatchRows::batched( R, slots, poss ) );
		}
		// One GPU: score + select. The ranks: score + their candidates, then (all done) the merge on the first and the last rank
		{
			Dev& d = dv[RK];
			CUDA_CHECK( cudaSetDevice( 0 ) );
			DSA::score( d.q, d.w, d.pool, d.sc, NP, layer, mseq, R, d.rows, 0, nullptr, d.pg );
			DSA::select( d.sc, NP, d.idx, d.nk, R, d.rows, nullptr );
			CUDA_CHECK( cudaDeviceSynchronize() );
		}
		for ( int v = 0; v < RK; ++v ) {
			Dev& d = dv[v];
			CUDA_CHECK( cudaSetDevice( v ) );
			DSA::score( d.q, d.w, d.pool, d.sc, sstride, layer, mseq, R, d.rows, 0, nullptr, d.pg );
			DSA::select_local( d.sc, sstride, d.cid, d.ckey, d.cn, R, d.rows, d.pg, nullptr );
			CUDA_CHECK( cudaDeviceSynchronize() );
		}
		std::vector<int> ref_idx( static_cast<size_t>( R ) * DSA::KPAD ), ref_nk( R );
		CUDA_CHECK( cudaSetDevice( 0 ) );
		CUDA_CHECK( cudaMemcpy( ref_idx.data(), dv[RK].idx, ref_idx.size() * sizeof( int ), cudaMemcpyDeviceToHost ) );
		CUDA_CHECK( cudaMemcpy( ref_nk.data(), dv[RK].nk, R * sizeof( int ), cudaMemcpyDeviceToHost ) );
		int selected_rows = 0, local_full = 0;
		for ( int r = 0; r < R; ++r ) selected_rows += ref_nk[r] > 0;
		{
			std::vector<int> cn( R );
			CUDA_CHECK( cudaMemcpy( cn.data(), dv[0].cn, R * sizeof( int ), cudaMemcpyDeviceToHost ) );
			for ( int r = 0; r < R; ++r ) local_full += cn[r] == DSA::SEL_POOLS;
		}
		for ( const int v : { 0, RK - 1 } ) {
			Dev& d = dv[v];
			CUDA_CHECK( cudaSetDevice( v ) );
			DSA::select_merge( d.cptr, RK, d.idx, d.nk, R, d.rows, nullptr );
			CUDA_CHECK( cudaDeviceSynchronize() );
			std::vector<int> idx( ref_idx.size() ), nk( R );
			CUDA_CHECK( cudaMemcpy( idx.data(), d.idx, idx.size() * sizeof( int ), cudaMemcpyDeviceToHost ) );
			CUDA_CHECK( cudaMemcpy( nk.data(), d.nk, R * sizeof( int ), cudaMemcpyDeviceToHost ) );
			bool same = nk == ref_nk;
			for ( int r = 0; r < R && same; ++r ) {
				if ( ref_nk[r] > 0 ) same = std::memcmp( idx.data() + static_cast<size_t>( r ) * DSA::KPAD, ref_idx.data() + static_cast<size_t>( r ) * DSA::KPAD, ref_nk[r] * sizeof( int ) ) == 0;
			}
			std::printf( "  indexer pool_update + score + local candidates + merge, rank %d of %d: %s (%d selected rows, %d with a full local top %d on rank 0)\n", v, RK,
				same ? "identical to the one-GPU selection" : "DIFFERENT", selected_rows, local_full, DSA::SEL_POOLS );
			all_ok = all_ok && same && selected_rows > 0 && local_full > 0;
		}
	}

	std::printf( all_ok ? "[+] PASS kv_shard: the latent cache and the indexer over %d GPUs read as the one-GPU pool\n" : "[-] FAIL kv_shard differs from the one-GPU pool (%d GPUs)\n", RK );
	return all_ok ? 0 : 1;
}
