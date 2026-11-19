// volta_tune: block-output tuning of routed experts' EXL3 scales. For every expert of the given layers in an EXL3 directory (a checkpoint, or a
// volta_quantize .partial directory), the six scale vectors (gate, up and down suh / svh) are tuned with Adam, multiplicatively, so that the
// quantized expert's output matches the original expert's over the route dump's rows routed to it, each row weighted by its routing weight
// squared (the expert's share of the MoE block's output); every tenth row is held out, and the step with the lowest held-out error is kept. The
// trellis cores stay as they are. The results go to <out>/.partial in volta_quantize's per-tensor form (experts with too few rows unchanged).
//
//   volta_tune --src <original> --exl3 <dir> --route <route dump with its .w> --out <dir> --layers L,L,... [--steps 100] [--lr 0.001]
//              [--max-rows 8192] [--devices 0,1,2,3]
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
		else {
			LOG_ERROR( "[volta_tune] unknown argument %s", k.c_str() );
			return false;
		}
	}
	if ( a.src.empty() || a.exl3.empty() || a.route.empty() || a.out.empty() || a.layers.empty() ) {
		LOG_ERROR( "usage: volta_tune --src <original> --exl3 <dir> --route <route dump> --out <dir> --layers L,L,... [--steps 100] [--lr 0.001] "
				   "[--max-rows 8192] [--devices 0,1,2,3]" );
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
