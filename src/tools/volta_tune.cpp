// volta_tune: block-output tuning of routed experts' EXL3 scales. For every expert of the given layers in an EXL3 directory (a checkpoint, or a
// volta_quantize .partial directory), the six scale vectors (gate, up and down suh / svh) are tuned with Adam, multiplicatively, so that the
// quantized expert's output matches the original expert's over the route dump's rows routed to it, each row weighted by its routing weight
// squared (the expert's share of the MoE block's output); every tenth row is held out, and the step with the lowest held-out error is kept. The
// trellis cores stay as they are. The results go to <out>/.partial in volta_quantize's per-tensor form (experts with too few rows unchanged).
//
//   volta_tune --src <original> --exl3 <dir> --route <route dump with its .w> --out <dir> --layers L,L,... [--steps 100] [--lr 0.001]
//              [--max-rows 8192] [--devices 0,1,2,3]
//
// --route-q <dump> switches to block tuning (QTIP's block stage for the routed experts): the dump is of the same text through the quantized
// model, so its rows carry the error of every layer before; all of a layer's experts are tuned together so that the quantized routed MoE
// output on those rows (the quantized model's routing) matches the original's on the reference rows of --route (the reference routing), in
// batches of --batch tokens, every tenth token held out for the step choice; --eval-ref / --eval-q, a reference and a quantized dump of
// held-out text, report the block error there before and after. One layer per device.
#include "engine/config.h"
#include "engine/core/log.h"
#include "engine/train/grad.h"
#include "quant/exl3_read.h"
#include "quant/expert_hessian.h"
#include "quant/plan.h"
#include "quant/source.h"
#include "quant/st_io.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <memory>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

namespace q = engine::tools::quant;
namespace tr = engine::train;

namespace {

struct Args {
	std::string src, exl3, route, out;
	std::vector<int> layers;
	std::vector<int> devices = { 0 };
	int steps = 100;
	float lr = 1e-3f;
	int max_rows = 8192;
	std::string route_q, eval_ref, eval_q;   // block tuning
	int batch = 4096;
};

std::vector<int> int_list( const std::string& s ) {
	std::vector<int> v;
	std::stringstream ss( s );
	for ( std::string item; std::getline( ss, item, ',' ); ) if ( !item.empty() ) v.push_back( std::stoi( item ) );
	return v;
}

bool parse( int argc, char** argv, Args& a ) {
	for ( int i = 1; i + 1 < argc; i += 2 ) {
		const std::string k = argv[i], v = argv[i + 1];
		if ( k == "--src" ) a.src = v;
		else if ( k == "--exl3" ) a.exl3 = v;
		else if ( k == "--route" ) a.route = v;
		else if ( k == "--out" ) a.out = v;
		else if ( k == "--layers" ) a.layers = int_list( v );
		else if ( k == "--devices" ) a.devices = int_list( v );
		else if ( k == "--steps" ) a.steps = std::stoi( v );
		else if ( k == "--lr" ) a.lr = std::stof( v );
		else if ( k == "--max-rows" ) a.max_rows = std::stoi( v );
		else if ( k == "--route-q" ) a.route_q = v;
		else if ( k == "--eval-ref" ) a.eval_ref = v;
		else if ( k == "--eval-q" ) a.eval_q = v;
		else if ( k == "--batch" ) a.batch = std::stoi( v );
		else {
			LOG_ERROR( "[volta_tune] unknown argument %s", k.c_str() );
			return false;
		}
	}
	if ( a.src.empty() || a.exl3.empty() || a.route.empty() || a.out.empty() || a.layers.empty() ) {
		LOG_ERROR( "usage: volta_tune --src <original> --exl3 <dir> --route <route dump> --out <dir> --layers L,L,... [--steps 100] [--lr 0.001] "
				   "[--max-rows 8192] [--devices 0,1,2,3] [--route-q <dump> [--batch 4096] [--eval-ref <dump> --eval-q <dump>]]" );
		return false;
	}
	return true;
}

float h2f( uint16_t b ) {
	half h;
	std::memcpy( &h, &b, 2 );
	return __half2float( h );
}

uint16_t f2h( float f ) {
	const half h = __float2half_rn( f );
	uint16_t b = 0;
	std::memcpy( &b, &h, 2 );
	return b;
}

// Growable device buffer of floats
struct Buf {
	float* p = nullptr;
	size_t n = 0;
	Buf() = default;
	Buf( const Buf& ) = delete;
	Buf& operator=( const Buf& ) = delete;
	~Buf() {
		if ( p ) cudaFree( p );
	}
	float* fit( size_t count ) {
		if ( count > n ) {
			if ( p ) CUDA_CHECK( cudaFree( p ) );
			p = nullptr;
			n = 0;
			CUDA_CHECK( cudaMalloc( &p, count * sizeof( float ) ) );
			n = count;
		}
		return p;
	}
	float* upload( const std::vector<float>& h, cudaStream_t st ) {
		fit( h.size() );
		CUDA_CHECK( cudaMemcpyAsync( p, h.data(), h.size() * sizeof( float ), cudaMemcpyHostToDevice, st ) );
		return p;
	}
};

// One layer's route-dump rows (host) and per expert its routed rows and routing weights
struct LayerRows {
	int layer = -1;
	int64_t rows = 0;
	std::vector<uint16_t> x;
	std::vector<std::vector<int>> e_rows;
	std::vector<std::vector<float>> e_w;
};

struct Shared {
	const Args* a = nullptr;
	const q::Model* model = nullptr;
	const LayerRows* lr = nullptr;
	std::vector<std::pair<int, std::string>> jobs;   // expert, "<...>.mlp.experts.<e>"
	std::atomic<size_t> next{ 0 };
	std::atomic<int> failures{ 0 };
	std::mutex log_mu;
	double sum_before = 0.0, sum_after = 0.0;
	int tuned = 0, kept = 0;
};

// One tensor of the expert: its read EXL3 form, its core on the device, its scales (base and current)
struct Proj {
	q::Exl3Read r;
	Buf core, s0u, s0v, su, sv, gu, gv, au, av, mu, mv, vu, vv, bu, bv, tu, tv;
};

void worker( Shared& sh, int device ) {
	const Args& a = *sh.a;
	CUDA_CHECK( cudaSetDevice( device ) );
	cudaStream_t st = nullptr;
	CUDA_CHECK( cudaStreamCreateWithFlags( &st, cudaStreamNonBlocking ) );
	q::SourceWeights src;
	q::StRepo repo;
	if ( !src.open( a.src ) || !repo.open( a.exl3 ) ) {
		sh.failures++;
		return;
	}
	tr::Blas blas( st );
	const int H = sh.model->hidden;
	const float limit = sh.model->swiglu_limit;
	Buf Xt, Xv, wt, wv, Wg, Wu, Wd, Yrt, Yrv, G, U, A, Zg, Zu, Zd, Y, dY, dA, dG, dU, sn, sk;
	Proj pg, pu, pd;

	for ( size_t j; ( j = sh.next.fetch_add( 1 ) ) < sh.jobs.size(); ) {
		const auto t0 = std::chrono::steady_clock::now();
		const int e = sh.jobs[j].first;
		const std::string& pre = sh.jobs[j].second;
		const std::string names[3] = { pre + ".gate_proj", pre + ".up_proj", pre + ".down_proj" };
		Proj* projs[3] = { &pg, &pu, &pd };
		bool ok = true;
		for ( int i = 0; i < 3 && ok; ++i ) ok = q::exl3_read( repo, names[i], projs[i]->r );
		const int I = ok ? pg.r.t.n : 0;
		ok = ok && pg.r.t.k == H && pu.r.t.k == H && pu.r.t.n == I && pd.r.t.k == I && pd.r.t.n == H;
		if ( !ok ) {
			LOG_ERROR( "[volta_tune] %s: missing or misshaped EXL3 tensors", pre.c_str() );
			sh.failures++;
			continue;
		}
		auto write_all = [&]( const char* note ) {
			bool w = true;
			for ( int i = 0; i < 3; ++i ) {
				w = w && q::exl3_write_partial( a.out + "/.partial/" + names[i] + ".safetensors", names[i], projs[i]->r.t, projs[i]->r.code, projs[i]->r.cb,
					projs[i]->r.volta, { { "tune", note } } );
			}
			return w;
		};

		// The rows: a stride sample down to max_rows, every tenth held out
		const auto& rows = sh.lr->e_rows[e];
		const auto& pw = sh.lr->e_w[e];
		const int n_all = static_cast<int>( rows.size() );
		if ( n_all < 64 ) {
			if ( !write_all( "unchanged (too few rows)" ) ) sh.failures++;
			std::lock_guard<std::mutex> lock( sh.log_mu );
			sh.kept++;
			continue;
		}
		const int stride = std::max( 1, ( n_all + a.max_rows - 1 ) / a.max_rows );
		std::vector<float> xt, xv, wtr, wva;
		int nt = 0, nv = 0, pick = 0;
		for ( int i = 0; i < n_all; i += stride, ++pick ) {
			const bool held = pick % 10 == 9;
			auto& xs = held ? xv : xt;
			const uint16_t* src_row = sh.lr->x.data() + static_cast<size_t>( rows[i] ) * H;
			for ( int c = 0; c < H; ++c ) xs.push_back( h2f( src_row[c] ) );
			( held ? wva : wtr ).push_back( pw[i] * pw[i] );
			( held ? nv : nt )++;
		}
		Xt.upload( xt, st );
		Xv.upload( xv, st );
		wt.upload( wtr, st );
		wv.upload( wva, st );

		// The original expert's outputs, and the quantized one's cores and scales
		ok = src.load( names[0], Wg.fit( static_cast<size_t>( H ) * I ), st ) && src.load( names[1], Wu.fit( static_cast<size_t>( H ) * I ), st ) &&
			src.load( names[2], Wd.fit( static_cast<size_t>( I ) * H ), st );
		if ( !ok ) {
			LOG_ERROR( "[volta_tune] %s: not in the original", pre.c_str() );
			sh.failures++;
			continue;
		}
		const size_t RI = static_cast<size_t>( nt ) * I, RH = static_cast<size_t>( nt ) * H;
		for ( Buf* b : { &G, &U, &A, &Zg, &Zu, &dA, &dG, &dU } ) b->fit( RI );
		for ( Buf* b : { &Zd, &Y, &dY, &sn, &sk } ) b->fit( std::max( RI, RH ) );
		Yrt.fit( RH );
		Yrv.fit( static_cast<size_t>( nv ) * H );
		auto reference = [&]( const float* X, int n, float* Yr ) {
			blas.gemm( false, false, n, I, H, 1.0f, X, Wg.p, 0.0f, G.p );
			blas.gemm( false, false, n, I, H, 1.0f, X, Wu.p, 0.0f, U.p );
			tr::swiglu_forward( G.p, U.p, A.p, static_cast<size_t>( n ) * I, limit, st );
			blas.gemm( false, false, n, H, I, 1.0f, A.p, Wd.p, 0.0f, Yr );
		};
		reference( Xt.p, nt, Yrt.p );
		reference( Xv.p, nv, Yrv.p );
		const double Et = tr::weighted_sq_norm( Yrt.p, wt.p, nt, H, st ), Ev = tr::weighted_sq_norm( Yrv.p, wv.p, nv, H, st );
		for ( Proj* p : projs ) {
			const int k = p->r.t.k, n = p->r.t.n;
			q::exl3_decode( p->r, p->core.fit( static_cast<size_t>( k ) * n ), true );
			std::vector<float> u( k ), v( n );
			for ( int i = 0; i < k; ++i ) u[i] = h2f( p->r.t.suh[i] );
			for ( int i = 0; i < n; ++i ) v[i] = h2f( p->r.t.svh[i] );
			p->s0u.upload( u, st );
			p->s0v.upload( v, st );
			p->su.upload( u, st );
			p->sv.upload( v, st );
			for ( Buf* b : { &p->gu, &p->au, &p->mu, &p->vu, &p->bu, &p->tu } ) CUDA_CHECK( cudaMemsetAsync( b->fit( k ), 0, k * sizeof( float ), st ) );
			for ( Buf* b : { &p->gv, &p->av, &p->mv, &p->vv, &p->bv, &p->tv } ) CUDA_CHECK( cudaMemsetAsync( b->fit( n ), 0, n * sizeof( float ), st ) );
		}
		tr::ScaledLinear lg{ H, I, pg.core.p, pg.su.p, pg.sv.p, pg.gu.p, pg.gv.p }, lu{ H, I, pu.core.p, pu.su.p, pu.sv.p, pu.gu.p, pu.gv.p },
			ld{ I, H, pd.core.p, pd.su.p, pd.sv.p, pd.gu.p, pd.gv.p };

		// The quantized expert's relative error on a row set; with backward the scale gradients of the train set
		auto pass = [&]( const float* X, const float* w, const float* Yr, int n, double E, bool backward ) {
			tr::scaled_linear_forward( blas, lg, X, n, Zg.p, G.p, sk.p );
			tr::scaled_linear_forward( blas, lu, X, n, Zu.p, U.p, sk.p );
			tr::swiglu_forward( G.p, U.p, A.p, static_cast<size_t>( n ) * I, limit, st );
			tr::scaled_linear_forward( blas, ld, A.p, n, Zd.p, Y.p, sk.p );
			const double L = tr::weighted_sq_error( Y.p, Yr, w, n, H, backward ? dY.p : nullptr, static_cast<float>( 1.0 / E ), st );
			if ( backward ) {
				for ( Proj* p : projs ) {
					CUDA_CHECK( cudaMemsetAsync( p->gu.p, 0, p->r.t.k * sizeof( float ), st ) );
					CUDA_CHECK( cudaMemsetAsync( p->gv.p, 0, p->r.t.n * sizeof( float ), st ) );
				}
				tr::scaled_linear_backward( blas, ld, A.p, Zd.p, dY.p, n, dA.p, sn.p, sk.p );
				tr::swiglu_backward( G.p, U.p, dA.p, dG.p, dU.p, static_cast<size_t>( n ) * I, limit, st );
				tr::scaled_linear_backward( blas, lg, X, Zg.p, dG.p, n, nullptr, sn.p, sk.p );
				tr::scaled_linear_backward( blas, lu, X, Zu.p, dU.p, n, nullptr, sn.p, sk.p );
			}
			return L / E;
		};
		auto save = [&]( bool to_best ) {
			for ( Proj* p : projs ) {
				const int k = p->r.t.k, n = p->r.t.n;
				CUDA_CHECK( cudaMemcpyAsync( to_best ? p->bu.p : p->au.p, to_best ? p->au.p : p->bu.p, k * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
				CUDA_CHECK( cudaMemcpyAsync( to_best ? p->bv.p : p->av.p, to_best ? p->av.p : p->bv.p, n * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
			}
		};
		auto apply = [&]() {
			for ( Proj* p : projs ) {
				tr::exp_scale( p->s0u.p, p->au.p, p->su.p, p->r.t.k, st );
				tr::exp_scale( p->s0v.p, p->av.p, p->sv.p, p->r.t.n, st );
			}
		};

		const double train0 = pass( Xt.p, wt.p, Yrt.p, nt, Et, false ), val0 = pass( Xv.p, wv.p, Yrv.p, nv, Ev, false );
		double best = val0, train = train0;
		int best_step = 0;
		tr::AdamOptions o;
		o.lr = a.lr;
		for ( int step = 1; step <= a.steps; ++step ) {
			train = pass( Xt.p, wt.p, Yrt.p, nt, Et, true );
			for ( Proj* p : projs ) {
				// d a = s o d s
				tr::mul( p->su.p, p->gu.p, p->tu.p, p->r.t.k, st );
				tr::mul( p->sv.p, p->gv.p, p->tv.p, p->r.t.n, st );
				tr::adam_step( p->au.p, p->tu.p, p->mu.p, p->vu.p, p->r.t.k, step, o, st );
				tr::adam_step( p->av.p, p->tv.p, p->mv.p, p->vv.p, p->r.t.n, step, o, st );
			}
			apply();
			if ( step % 10 == 0 || step == a.steps ) {
				const double v = pass( Xv.p, wv.p, Yrv.p, nv, Ev, false );
				if ( v < best ) {
					best = v;
					best_step = step;
					save( true );
				}
			}
		}

		// The best step's scales in FP16, as stored, measured again; no gain: the scales stay as they were
		double val1 = val0;
		if ( best_step > 0 ) {
			save( false );
			apply();
			for ( Proj* p : projs ) {
				std::vector<float> u( p->r.t.k ), v( p->r.t.n );
				CUDA_CHECK( cudaMemcpyAsync( u.data(), p->su.p, u.size() * sizeof( float ), cudaMemcpyDeviceToHost, st ) );
				CUDA_CHECK( cudaMemcpyAsync( v.data(), p->sv.p, v.size() * sizeof( float ), cudaMemcpyDeviceToHost, st ) );
				CUDA_CHECK( cudaStreamSynchronize( st ) );
				for ( size_t i = 0; i < u.size(); ++i ) u[i] = h2f( p->r.t.suh[i] = f2h( u[i] ) );
				for ( size_t i = 0; i < v.size(); ++i ) v[i] = h2f( p->r.t.svh[i] = f2h( v[i] ) );
				p->su.upload( u, st );
				p->sv.upload( v, st );
			}
			val1 = pass( Xv.p, wv.p, Yrv.p, nv, Ev, false );
		}
		const bool gain = best_step > 0 && val1 < val0;
		if ( !gain ) {
			for ( Proj* p : projs ) {
				std::vector<float> u( p->r.t.k ), v( p->r.t.n );
				CUDA_CHECK( cudaMemcpyAsync( u.data(), p->s0u.p, u.size() * sizeof( float ), cudaMemcpyDeviceToHost, st ) );
				CUDA_CHECK( cudaMemcpyAsync( v.data(), p->s0v.p, v.size() * sizeof( float ), cudaMemcpyDeviceToHost, st ) );
				CUDA_CHECK( cudaStreamSynchronize( st ) );
				for ( size_t i = 0; i < u.size(); ++i ) p->r.t.suh[i] = f2h( u[i] );
				for ( size_t i = 0; i < v.size(); ++i ) p->r.t.svh[i] = f2h( v[i] );
			}
		}
		char note[160];
		std::snprintf( note, sizeof( note ), "%s: held-out error %.6g -> %.6g at step %d (train %.6g -> %.6g)", gain ? "tuned" : "unchanged", val0,
			gain ? val1 : val0, best_step, train0, train );
		if ( !write_all( note ) ) sh.failures++;
		const double secs = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
		std::lock_guard<std::mutex> lock( sh.log_mu );
		sh.sum_before += val0;
		sh.sum_after += gain ? val1 : val0;
		( gain ? sh.tuned : sh.kept )++;
		LOG_INFO( "[volta_tune] GPU %d: %s, %d rows (%d held out): train %.5f -> %.5f, held out %.5f -> %.5f (step %d, %s), %.1f s", device,
			pre.c_str(), nt + nv, nv, train0, train, val0, gain ? val1 : val0, best_step, gain ? "tuned" : "unchanged", secs );
	}
	CUDA_CHECK( cudaStreamDestroy( st ) );
}

// ---- block tuning -------------------------------------------------------------------------------------------------------------------------

struct IBuf {
	int* p = nullptr;
	size_t n = 0;
	IBuf() = default;
	IBuf( const IBuf& ) = delete;
	IBuf& operator=( const IBuf& ) = delete;
	~IBuf() {
		if ( p ) cudaFree( p );
	}
	int* upload( const std::vector<int>& h, cudaStream_t st ) {
		if ( h.size() > n ) {
			if ( p ) CUDA_CHECK( cudaFree( p ) );
			CUDA_CHECK( cudaMalloc( &p, std::max<size_t>( h.size(), 1 ) * sizeof( int ) ) );
			n = h.size();
		}
		if ( !h.empty() ) CUDA_CHECK( cudaMemcpyAsync( p, h.data(), h.size() * sizeof( int ), cudaMemcpyHostToDevice, st ) );
		return p;
	}
};

// One layer of a route dump on the host: rows (FP16, rows x hidden), expert ids and routing weights (rows x K)
struct Dump {
	int64_t rows = 0;
	int K = 0;
	std::vector<uint16_t> x;
	std::vector<int> ids;
	std::vector<float> w;
	bool read( const std::string& path, int hidden, int layer ) {
		q::RouteRows rr;
		if ( !rr.open( path, hidden ) || !rr.has_weights() ) return false;
		if ( !rr.read( layer, 0, x ) || !rr.routes( layer, 0, ids, w, K ) ) return false;
		rows = static_cast<int64_t>( x.size() / hidden );
		return rows > 0;
	}
};

// The routed pairs of a set of rows, packed by expert: the dump row, the position in the set, the routing weight; off: experts + 1 offsets
struct Routed {
	std::vector<int> src, pos, off;
	std::vector<float> p;
};

Routed route_rows( const Dump& d, const int* rows, int n, int experts ) {
	Routed r;
	r.off.assign( experts + 1, 0 );
	for ( int b = 0; b < n; ++b ) {
		for ( int j = 0; j < d.K; ++j ) {
			const int e = d.ids[static_cast<size_t>( rows[b] ) * d.K + j];
			if ( e >= 0 && e < experts ) r.off[e + 1]++;
		}
	}
	for ( int e = 0; e < experts; ++e ) r.off[e + 1] += r.off[e];
	const int total = r.off[experts];
	r.src.resize( total );
	r.pos.resize( total );
	r.p.resize( total );
	std::vector<int> fill( r.off.begin(), r.off.end() - 1 );
	for ( int b = 0; b < n; ++b ) {
		for ( int j = 0; j < d.K; ++j ) {
			const size_t i = static_cast<size_t>( rows[b] ) * d.K + j;
			const int e = d.ids[i];
			if ( e < 0 || e >= experts ) continue;
			const int at = fill[e]++;
			r.src[at] = rows[b];
			r.pos[at] = b;
			r.p[at] = d.w[i];
		}
	}
	return r;
}

// The original routed MoE output of every row of a dump (rows x hidden, device): its routing over the source experts
bool reference_outputs( tr::Blas& blas, q::SourceWeights& src, const Dump& d, const std::vector<std::string>& prefixes, int H, int I, float limit,
	float* Y, cudaStream_t st ) {
	Buf X, Wg, Wu, Wd, G, U, A, O;
	IBuf dx_src, dx_pos;
	Buf dx_p;
	uint16_t* Xd = nullptr;
	CUDA_CHECK( cudaMalloc( &Xd, d.x.size() * 2 ) );
	CUDA_CHECK( cudaMemcpyAsync( Xd, d.x.data(), d.x.size() * 2, cudaMemcpyHostToDevice, st ) );
	CUDA_CHECK( cudaMemsetAsync( Y, 0, static_cast<size_t>( d.rows ) * H * sizeof( float ), st ) );
	std::vector<int> all( d.rows );
	for ( int64_t r = 0; r < d.rows; ++r ) all[r] = static_cast<int>( r );
	const Routed R = route_rows( d, all.data(), static_cast<int>( d.rows ), static_cast<int>( prefixes.size() ) );
	const int* sp = dx_src.upload( R.src, st );
	const int* pp = dx_pos.upload( R.pos, st );
	const float* wp = dx_p.upload( R.p, st );
	constexpr int CH = 4096;
	bool ok = true;
	for ( size_t e = 0; e < prefixes.size() && ok; ++e ) {
		const int o = R.off[e], n = R.off[e + 1] - R.off[e];
		if ( n == 0 ) continue;
		ok = src.load( prefixes[e] + ".gate_proj", Wg.fit( static_cast<size_t>( H ) * I ), st ) && src.load( prefixes[e] + ".up_proj", Wu.fit( static_cast<size_t>( H ) * I ), st ) &&
			src.load( prefixes[e] + ".down_proj", Wd.fit( static_cast<size_t>( I ) * H ), st );
		for ( int c0 = 0; c0 < n && ok; c0 += CH ) {
			const int c = std::min( CH, n - c0 );
			tr::gather_rows( Xd, true, sp + o + c0, nullptr, c, H, X.fit( static_cast<size_t>( CH ) * H ), st );
			blas.gemm( false, false, c, I, H, 1.0f, X.p, Wg.p, 0.0f, G.fit( static_cast<size_t>( CH ) * I ) );
			blas.gemm( false, false, c, I, H, 1.0f, X.p, Wu.p, 0.0f, U.fit( static_cast<size_t>( CH ) * I ) );
			tr::swiglu_forward( G.p, U.p, A.fit( static_cast<size_t>( CH ) * I ), static_cast<size_t>( c ) * I, limit, st );
			blas.gemm( false, false, c, H, I, 1.0f, A.p, Wd.p, 0.0f, O.fit( static_cast<size_t>( CH ) * H ) );
			tr::scatter_add_rows( O.p, pp + o + c0, wp + o + c0, c, H, Y, st );
		}
	}
	CUDA_CHECK( cudaStreamSynchronize( st ) );
	CUDA_CHECK( cudaFree( Xd ) );
	return ok;
}

// One layer: every routed expert's scales against the original routed MoE output, on rows that carry the quantized model's upstream error
bool block_layer( const Args& a, const q::Model& model, int layer, int device, std::mutex& log_mu ) {
	CUDA_CHECK( cudaSetDevice( device ) );
	cudaStream_t st = nullptr;
	CUDA_CHECK( cudaStreamCreateWithFlags( &st, cudaStreamNonBlocking ) );
	const auto t0 = std::chrono::steady_clock::now();
	const int H = model.hidden, E = model.experts;
	const float limit = model.swiglu_limit;
	q::SourceWeights src;
	q::StRepo repo;
	if ( !src.open( a.src ) || !repo.open( a.exl3 ) ) return false;
	// The layer's experts in the EXL3 directory
	static const std::regex gate_re( R"(^(.*\.layers\.(\d+)\.mlp\.experts)\.(\d+)\.gate_proj\.trellis$)" );
	std::string base;
	for ( const auto& kv : repo.weight_map() ) {
		std::smatch m;
		if ( std::regex_match( kv.first, m, gate_re ) && std::stoi( m[2].str() ) == layer && m[1].str().find( "mtp." ) != 0 ) {
			base = m[1].str();
			break;
		}
	}
	if ( base.empty() ) {
		LOG_ERROR( "[volta_tune] layer %d: no routed experts in %s", layer, a.exl3.c_str() );
		return false;
	}
	std::vector<std::string> prefixes( E );
	for ( int e = 0; e < E; ++e ) prefixes[e] = base + "." + std::to_string( e );
	Dump dref, dq;
	if ( !dref.read( a.route, H, layer ) || !dq.read( a.route_q, H, layer ) || dref.rows != dq.rows ) {
		LOG_ERROR( "[volta_tune] layer %d: the reference and quantized dumps (with routing weights) must hold the same rows", layer );
		return false;
	}
	tr::Blas blas( st );
	// The quantized experts: cores, base scales, parameters
	std::vector<std::unique_ptr<Proj>> projs( static_cast<size_t>( E ) * 3 );
	int I = 0;
	for ( int e = 0; e < E; ++e ) {
		const char* names[3] = { ".gate_proj", ".up_proj", ".down_proj" };
		for ( int k = 0; k < 3; ++k ) {
			auto p = std::make_unique<Proj>();
			if ( !q::exl3_read( repo, prefixes[e] + names[k], p->r ) ) {
				LOG_ERROR( "[volta_tune] %s%s: not in %s", prefixes[e].c_str(), names[k], a.exl3.c_str() );
				return false;
			}
			const int kk = p->r.t.k, nn = p->r.t.n;
			if ( k == 0 ) I = nn;
			q::exl3_decode( p->r, p->core.fit( static_cast<size_t>( kk ) * nn ), true );
			std::vector<float> u( kk ), v( nn );
			for ( int i = 0; i < kk; ++i ) u[i] = h2f( p->r.t.suh[i] );
			for ( int i = 0; i < nn; ++i ) v[i] = h2f( p->r.t.svh[i] );
			p->s0u.upload( u, st );
			p->s0v.upload( v, st );
			p->su.upload( u, st );
			p->sv.upload( v, st );
			for ( Buf* b : { &p->gu, &p->au, &p->mu, &p->vu, &p->bu, &p->tu } ) CUDA_CHECK( cudaMemsetAsync( b->fit( kk ), 0, kk * sizeof( float ), st ) );
			for ( Buf* b : { &p->gv, &p->av, &p->mv, &p->vv, &p->bv, &p->tv } ) CUDA_CHECK( cudaMemsetAsync( b->fit( nn ), 0, nn * sizeof( float ), st ) );
			projs[static_cast<size_t>( e ) * 3 + k] = std::move( p );
		}
	}
	// The original outputs over the reference rows (train / held-out rows of the same dump), and over the held-out text
	Buf Yref;
	if ( !reference_outputs( blas, src, dref, prefixes, H, I, limit, Yref.fit( static_cast<size_t>( dref.rows ) * H ), st ) ) return false;
	Dump eref, eq;
	Buf Yeval;
	const bool eval = !a.eval_ref.empty() && !a.eval_q.empty();
	if ( eval && ( !eref.read( a.eval_ref, H, layer ) || !eq.read( a.eval_q, H, layer ) || eref.rows != eq.rows ||
				   !reference_outputs( blas, src, eref, prefixes, H, I, limit, Yeval.fit( static_cast<size_t>( eref.rows ) * H ), st ) ) ) {
		LOG_ERROR( "[volta_tune] layer %d: the held-out dumps do not pair up", layer );
		return false;
	}
	uint16_t *Xq = nullptr, *Xe = nullptr;
	CUDA_CHECK( cudaMalloc( &Xq, dq.x.size() * 2 ) );
	CUDA_CHECK( cudaMemcpyAsync( Xq, dq.x.data(), dq.x.size() * 2, cudaMemcpyHostToDevice, st ) );
	if ( eval ) {
		CUDA_CHECK( cudaMalloc( &Xe, eq.x.size() * 2 ) );
		CUDA_CHECK( cudaMemcpyAsync( Xe, eq.x.data(), eq.x.size() * 2, cudaMemcpyHostToDevice, st ) );
	}
	std::vector<int> train, val;
	for ( int64_t r = 0; r < dq.rows; ++r ) ( r % 10 == 9 ? val : train ).push_back( static_cast<int>( r ) );
	const int B = std::min<int>( a.batch, static_cast<int>( train.size() ) );
	const size_t P = static_cast<size_t>( B ) * dq.K;   // routed pairs of a batch
	Buf Xp, Zg, G, Zu, U, A, Zd, Yd, dA, dG, dU, dYd, Yb, Yt, dY, Sn, Sk, pw;
	for ( Buf* b : { &Xp, &Zd, &Yd, &dYd } ) b->fit( P * H );
	for ( Buf* b : { &Zg, &G, &Zu, &U, &A, &dA, &dG, &dU } ) b->fit( P * I );
	for ( Buf* b : { &Yb, &Yt, &dY, &Sn, &Sk } ) b->fit( static_cast<size_t>( B ) * H );
	IBuf isrc, ipos, irows;

	// The quantized block over a set of rows (positions 0 .. n of the set); with backward the scale gradients of its relative error
	auto pass = [&]( const uint16_t* X, const Dump& d, const int* rows, int n, const float* Yr, bool backward ) {
		const Routed R = route_rows( d, rows, n, E );
		const int* sp = isrc.upload( R.src, st );
		const int* pp = ipos.upload( R.pos, st );
		const float* wp = pw.upload( R.p, st );
		const std::vector<int> rv( rows, rows + n );
		tr::gather_rows( Yr, false, irows.upload( rv, st ), nullptr, n, H, Yt.p, st );
		CUDA_CHECK( cudaMemsetAsync( Yb.p, 0, static_cast<size_t>( n ) * H * sizeof( float ), st ) );
		for ( int e = 0; e < E; ++e ) {
			const int o = R.off[e], m = R.off[e + 1] - R.off[e];
			if ( !m ) continue;
			Proj &g = *projs[e * 3], &u = *projs[e * 3 + 1], &dn = *projs[e * 3 + 2];
			tr::ScaledLinear lg{ H, I, g.core.p, g.su.p, g.sv.p, g.gu.p, g.gv.p }, lu{ H, I, u.core.p, u.su.p, u.sv.p, u.gu.p, u.gv.p },
				ld{ I, H, dn.core.p, dn.su.p, dn.sv.p, dn.gu.p, dn.gv.p };
			const size_t oh = static_cast<size_t>( o ) * H, oi = static_cast<size_t>( o ) * I;
			tr::gather_rows( X, true, sp + o, nullptr, m, H, Xp.p + oh, st );
			tr::scaled_linear_forward( blas, lg, Xp.p + oh, m, Zg.p + oi, G.p + oi, Sk.p );
			tr::scaled_linear_forward( blas, lu, Xp.p + oh, m, Zu.p + oi, U.p + oi, Sk.p );
			tr::swiglu_forward( G.p + oi, U.p + oi, A.p + oi, static_cast<size_t>( m ) * I, limit, st );
			tr::scaled_linear_forward( blas, ld, A.p + oi, m, Zd.p + oh, Yd.p + oh, Sk.p );
			tr::scatter_add_rows( Yd.p + oh, pp + o, wp + o, m, H, Yb.p, st );
		}
		const double En = tr::weighted_sq_norm( Yt.p, nullptr, n, H, st );
		const double L = tr::weighted_sq_error( Yb.p, Yt.p, nullptr, n, H, backward ? dY.p : nullptr, static_cast<float>( 1.0 / En ), st );
		if ( backward ) {
			for ( auto& p : projs ) {
				CUDA_CHECK( cudaMemsetAsync( p->gu.p, 0, p->r.t.k * sizeof( float ), st ) );
				CUDA_CHECK( cudaMemsetAsync( p->gv.p, 0, p->r.t.n * sizeof( float ), st ) );
			}
			for ( int e = 0; e < E; ++e ) {
				const int o = R.off[e], m = R.off[e + 1] - R.off[e];
				if ( !m ) continue;
				Proj &g = *projs[e * 3], &u = *projs[e * 3 + 1], &dn = *projs[e * 3 + 2];
				tr::ScaledLinear lg{ H, I, g.core.p, g.su.p, g.sv.p, g.gu.p, g.gv.p }, lu{ H, I, u.core.p, u.su.p, u.sv.p, u.gu.p, u.gv.p },
					ld{ I, H, dn.core.p, dn.su.p, dn.sv.p, dn.gu.p, dn.gv.p };
				const size_t oh = static_cast<size_t>( o ) * H, oi = static_cast<size_t>( o ) * I;
				tr::gather_rows( dY.p, false, pp + o, wp + o, m, H, dYd.p + oh, st );
				tr::scaled_linear_backward( blas, ld, A.p + oi, Zd.p + oh, dYd.p + oh, m, dA.p + oi, Sn.p, Sk.p );
				tr::swiglu_backward( G.p + oi, U.p + oi, dA.p + oi, dG.p + oi, dU.p + oi, static_cast<size_t>( m ) * I, limit, st );
				tr::scaled_linear_backward( blas, lg, Xp.p + oh, Zg.p + oi, dG.p + oi, m, nullptr, Sn.p, Sk.p );
				tr::scaled_linear_backward( blas, lu, Xp.p + oh, Zu.p + oi, dU.p + oi, m, nullptr, Sn.p, Sk.p );
			}
		}
		return std::make_pair( L, En );
	};
	// The relative error over a whole row set, in batches
	auto set_error = [&]( const uint16_t* X, const Dump& d, const std::vector<int>& rows, const float* Yr ) {
		double L = 0.0, En = 0.0;
		for ( size_t b0 = 0; b0 < rows.size(); b0 += B ) {
			const int n = static_cast<int>( std::min<size_t>( B, rows.size() - b0 ) );
			const auto r = pass( X, d, rows.data() + b0, n, Yr, false );
			L += r.first;
			En += r.second;
		}
		return L / En;
	};
	auto apply = [&]() {
		for ( auto& p : projs ) {
			tr::exp_scale( p->s0u.p, p->au.p, p->su.p, p->r.t.k, st );
			tr::exp_scale( p->s0v.p, p->av.p, p->sv.p, p->r.t.n, st );
		}
	};
	auto copy = [&]( bool to_best ) {
		for ( auto& p : projs ) {
			CUDA_CHECK( cudaMemcpyAsync( to_best ? p->bu.p : p->au.p, to_best ? p->au.p : p->bu.p, p->r.t.k * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
			CUDA_CHECK( cudaMemcpyAsync( to_best ? p->bv.p : p->av.p, to_best ? p->av.p : p->bv.p, p->r.t.n * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
		}
	};
	std::vector<int> all_eval;
	if ( eval ) {
		for ( int64_t r = 0; r < eq.rows; ++r ) all_eval.push_back( static_cast<int>( r ) );
	}
	const double val0 = set_error( Xq, dq, val, Yref.p );
	const double ev0 = eval ? set_error( Xe, eq, all_eval, Yeval.p ) : 0.0;
	std::mt19937 rng( 1234 + layer );
	std::shuffle( train.begin(), train.end(), rng );
	tr::AdamOptions o;
	o.lr = a.lr;
	double best = val0, train_err = 0.0;
	int best_step = 0;
	size_t at = 0;
	for ( int step = 1; step <= a.steps; ++step ) {
		if ( at + B > train.size() ) {
			std::shuffle( train.begin(), train.end(), rng );
			at = 0;
		}
		const auto r = pass( Xq, dq, train.data() + at, B, Yref.p, true );
		at += B;
		train_err = r.first / r.second;
		for ( auto& p : projs ) {
			tr::mul( p->su.p, p->gu.p, p->tu.p, p->r.t.k, st );
			tr::mul( p->sv.p, p->gv.p, p->tv.p, p->r.t.n, st );
			tr::adam_step( p->au.p, p->tu.p, p->mu.p, p->vu.p, p->r.t.k, step, o, st );
			tr::adam_step( p->av.p, p->tv.p, p->mv.p, p->vv.p, p->r.t.n, step, o, st );
		}
		apply();
		if ( step % 25 == 0 || step == a.steps ) {
			const double v = set_error( Xq, dq, val, Yref.p );
			if ( v < best ) {
				best = v;
				best_step = step;
				copy( true );
			}
		}
	}
	// The best step's scales in FP16 as stored; no gain on the held-out rows: unchanged
	double val1 = val0, ev1 = ev0;
	if ( best_step > 0 ) {
		copy( false );
		apply();
		for ( auto& p : projs ) {
			std::vector<float> u( p->r.t.k ), v( p->r.t.n );
			CUDA_CHECK( cudaMemcpyAsync( u.data(), p->su.p, u.size() * sizeof( float ), cudaMemcpyDeviceToHost, st ) );
			CUDA_CHECK( cudaMemcpyAsync( v.data(), p->sv.p, v.size() * sizeof( float ), cudaMemcpyDeviceToHost, st ) );
			CUDA_CHECK( cudaStreamSynchronize( st ) );
			for ( size_t i = 0; i < u.size(); ++i ) u[i] = h2f( p->r.t.suh[i] = f2h( u[i] ) );
			for ( size_t i = 0; i < v.size(); ++i ) v[i] = h2f( p->r.t.svh[i] = f2h( v[i] ) );
			p->su.upload( u, st );
			p->sv.upload( v, st );
		}
		val1 = set_error( Xq, dq, val, Yref.p );
		if ( eval ) ev1 = set_error( Xe, eq, all_eval, Yeval.p );
	}
	const bool gain = best_step > 0 && val1 < val0;
	char note[200];
	std::snprintf( note, sizeof( note ), "block %s: held-out rows %.6g -> %.6g at step %d", gain ? "tuned" : "unchanged", val0, gain ? val1 : val0, best_step );
	bool ok = true;
	for ( auto& p : projs ) {
		if ( !gain ) {
			std::vector<float> u( p->r.t.k ), v( p->r.t.n );
			CUDA_CHECK( cudaMemcpyAsync( u.data(), p->s0u.p, u.size() * sizeof( float ), cudaMemcpyDeviceToHost, st ) );
			CUDA_CHECK( cudaMemcpyAsync( v.data(), p->s0v.p, v.size() * sizeof( float ), cudaMemcpyDeviceToHost, st ) );
			CUDA_CHECK( cudaStreamSynchronize( st ) );
			for ( size_t i = 0; i < u.size(); ++i ) p->r.t.suh[i] = f2h( u[i] );
			for ( size_t i = 0; i < v.size(); ++i ) p->r.t.svh[i] = f2h( v[i] );
		}
	}
	for ( int e = 0; e < E && ok; ++e ) {
		const char* names[3] = { ".gate_proj", ".up_proj", ".down_proj" };
		for ( int k = 0; k < 3; ++k ) {
			const Proj& p = *projs[e * 3 + k];
			ok = ok && q::exl3_write_partial( a.out + "/.partial/" + prefixes[e] + names[k] + ".safetensors", prefixes[e] + names[k], p.r.t, p.r.code, p.r.cb, p.r.volta,
				{ { "tune", note } } );
		}
	}
	CUDA_CHECK( cudaFree( Xq ) );
	if ( Xe ) CUDA_CHECK( cudaFree( Xe ) );
	CUDA_CHECK( cudaStreamDestroy( st ) );
	const double secs = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
	std::lock_guard<std::mutex> lock( log_mu );
	LOG_INFO( "[volta_tune] GPU %d: layer %d block, %lld rows (%zu held out), %d steps of %d: train %.5f, held-out rows %.5f -> %.5f (step %d, %s)%s, %.1f min",
		device, layer, static_cast<long long>( dq.rows ), val.size(), a.steps, B, train_err, val0, gain ? val1 : val0, best_step, gain ? "tuned" : "unchanged",
		eval ? ( " | held-out text " + std::to_string( ev0 ) + " -> " + std::to_string( gain ? ev1 : ev0 ) ).c_str() : "", secs / 60.0 );
	return ok;
}

} // namespace

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	Args a;
	if ( !parse( argc, argv, a ) ) return 2;
	q::Model model;
	if ( !model.read( a.src ) || model.experts <= 0 || model.hidden <= 0 ) {
		LOG_ERROR( "[volta_tune] %s is not a mixture-of-experts model", a.src.c_str() );
		return 2;
	}
	if ( !a.route_q.empty() ) {
		mkdir( a.out.c_str(), 0755 );
		mkdir( ( a.out + "/.partial" ).c_str(), 0755 );
		std::mutex log_mu;
		std::atomic<size_t> next{ 0 };
		std::atomic<int> failures{ 0 };
		std::vector<std::thread> threads;
		for ( int d : a.devices ) {
			threads.emplace_back( [&, d]() {
				for ( size_t i; ( i = next.fetch_add( 1 ) ) < a.layers.size(); ) {
					if ( !block_layer( a, model, a.layers[i], d, log_mu ) ) failures++;
				}
			} );
		}
		for ( auto& t : threads ) t.join();
		return failures ? 1 : 0;
	}
	q::RouteRows rr;
	if ( !rr.open( a.route, model.hidden ) || !rr.has_weights() ) {
		LOG_ERROR( "[volta_tune] %s: a route dump with its routing weights (.w) is needed", a.route.c_str() );
		return 2;
	}
	q::StRepo repo;
	if ( !repo.open( a.exl3 ) ) return 2;
	mkdir( a.out.c_str(), 0755 );
	mkdir( ( a.out + "/.partial" ).c_str(), 0755 );
	static const std::regex gate_re( R"(^(.*\.layers\.(\d+)\.mlp\.experts\.(\d+))\.gate_proj\.trellis$)" );
	const auto t0 = std::chrono::steady_clock::now();
	Shared total;
	for ( int layer : a.layers ) {
		Shared sh;
		sh.a = &a;
		sh.model = &model;
		for ( const auto& kv : repo.weight_map() ) {
			std::smatch m;
			if ( std::regex_match( kv.first, m, gate_re ) && std::stoi( m[2].str() ) == layer && m[1].str().find( "mtp." ) != 0 ) {
				sh.jobs.emplace_back( std::stoi( m[3].str() ), m[1].str() );
			}
		}
		std::sort( sh.jobs.begin(), sh.jobs.end() );
		if ( sh.jobs.empty() ) {
			LOG_WARN( "[volta_tune] layer %d: no routed experts in %s", layer, a.exl3.c_str() );
			continue;
		}
		LayerRows lr;
		lr.layer = layer;
		std::vector<int> ids;
		std::vector<float> w;
		int K = 0;
		if ( !rr.read( layer, 0, lr.x ) || !rr.routes( layer, 0, ids, w, K ) ) {
			LOG_ERROR( "[volta_tune] the route dump has no rows of layer %d", layer );
			return 1;
		}
		lr.rows = static_cast<int64_t>( lr.x.size() / model.hidden );
		lr.e_rows.assign( model.experts, {} );
		lr.e_w.assign( model.experts, {} );
		for ( int64_t r = 0; r < lr.rows; ++r ) {
			for ( int j = 0; j < K; ++j ) {
				const int e = ids[static_cast<size_t>( r ) * K + j];
				if ( e < 0 || e >= model.experts ) continue;
				lr.e_rows[e].push_back( static_cast<int>( r ) );
				lr.e_w[e].push_back( w[static_cast<size_t>( r ) * K + j] );
			}
		}
		sh.lr = &lr;
		LOG_INFO( "[volta_tune] layer %d: %zu experts, %lld rows", layer, sh.jobs.size(), static_cast<long long>( lr.rows ) );
		std::vector<std::thread> threads;
		for ( int d : a.devices ) threads.emplace_back( worker, std::ref( sh ), d );
		for ( auto& t : threads ) t.join();
		if ( sh.failures ) {
			LOG_ERROR( "[volta_tune] layer %d: %d experts failed", layer, sh.failures.load() );
			return 1;
		}
		const int n = sh.tuned + sh.kept;
		LOG_INFO( "[volta_tune] layer %d: %d tuned, %d unchanged, mean held-out error %.5f -> %.5f", layer, sh.tuned, sh.kept,
			n ? sh.sum_before / n : 0.0, n ? sh.sum_after / n : 0.0 );
		total.sum_before += sh.sum_before;
		total.sum_after += sh.sum_after;
		total.tuned += sh.tuned;
		total.kept += sh.kept;
	}
	const int n = total.tuned + total.kept;
	LOG_INFO( "[volta_tune] done: %d tuned, %d unchanged, mean held-out error %.5f -> %.5f (%.1f min)", total.tuned, total.kept, n ? total.sum_before / n : 0.0,
		n ? total.sum_after / n : 0.0, std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count() / 60.0 );
	return 0;
}
