// hess_compare: two Hessian capture directories (engine format files of the same names, e.g. captures of two checkpoints on the same text)
// against each other, and the quantization each leads to.
//   hess_compare stats <dir_a> <dir_b>
//     per file both mean Hessians (sum / rows): |A - B| / |B| (Frobenius), the same for the diagonals, tr(A) / tr(B); by site (s<n>, expert
//     down) the mean and the worst, and the ten worst files
//   hess_compare cross --src <original> --tensor <base> --a <file> --b <file> [--bits 3] [--device 0] [--order volta|stock]
//     the tensor quantized with each Hessian, both results' proxy errors under both mean Hessians: when A's quantization scores under B as
//     B's own does, A is as good a calibration source as B for that tensor
//   hess_compare score --src <original> --tensors A,B,... (--hess-dir <capture dir> [--ranks 4] | --route <route dump>) --exl3 <dir> [--exl3 <dir> ...]
//     each tensor of each EXL3 checkpoint (its codebook and tile order as the checkpoint names them) against the original, the proxy error under
//     one mean Hessian: the capture's file for the tensor (as volta_quantize maps them), or for a routed expert's down projection the Hessian of
//     its activation over the route dump's rows
#include "engine/config.h"
#include "engine/quant/exl3_quantize.h"
#include "engine/modules/trellis.h"
#include "reconstruct.cuh"
#include "trellis_context.h"
#include "exl3_vl.cuh"
#include "engine/modules/trellis_bits.h"
#include "quant/expert_hessian.h"
#include "quant/plan.h"
#include "quant/source.h"
#include "quant/st_io.h"
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include <fstream>
#include <memory>
#include <sstream>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <map>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

struct Hess {
	int k = 0;
	int64_t rows = 0;
	std::vector<float> h;
};

bool read_hess( const std::string& path, Hess& out ) {
	FILE* f = std::fopen( path.c_str(), "rb" );
	if ( !f ) return false;
	int32_t hdr[2];
	bool ok = std::fread( hdr, 4, 2, f ) == 2 && hdr[0] == 0x48455353 && std::fread( &out.rows, 8, 1, f ) == 1 && out.rows > 0;
	if ( ok ) {
		out.k = hdr[1];
		out.h.resize( static_cast<size_t>( out.k ) * out.k );
		ok = std::fread( out.h.data(), 4, out.h.size(), f ) == out.h.size();
	}
	std::fclose( f );
	return ok;
}

// "s1", "s2", "down": the site part of a capture file name
std::string kind_of( const std::string& fn ) {
	int layer = 0, site = 0, rank = 0, expert = 0;
	if ( std::sscanf( fn.c_str(), "L%d_E%d_down.f32", &layer, &expert ) == 2 ) return "expert down";
	if ( std::sscanf( fn.c_str(), "L%d_s%d_r%d.f32", &layer, &site, &rank ) == 3 ) return "s" + std::to_string( site );
	return "other";
}

int stats( const std::string& da, const std::string& db ) {
	std::vector<std::string> names;
	if ( DIR* d = opendir( da.c_str() ) ) {
		while ( dirent* de = readdir( d ) ) {
			const std::string fn = de->d_name;
			if ( fn.size() > 4 && fn.compare( fn.size() - 4, 4, ".f32" ) == 0 ) names.push_back( fn );
		}
		closedir( d );
	}
	std::sort( names.begin(), names.end() );
	struct Row {
		std::string name;
		double fro, diag, trace;
	};
	std::vector<Row> rows;
	std::map<std::string, std::vector<double>> by_kind;
	int missing = 0;
	for ( const auto& fn : names ) {
		Hess a, b;
		if ( !read_hess( da + "/" + fn, a ) ) continue;
		if ( !read_hess( db + "/" + fn, b ) ) {
			++missing;
			continue;
		}
		if ( a.k != b.k ) {
			std::printf( "  %s: k %d against %d\n", fn.c_str(), a.k, b.k );
			continue;
		}
		const double sa = 1.0 / static_cast<double>( a.rows ), sb = 1.0 / static_cast<double>( b.rows );
		double dd = 0.0, bb = 0.0, ddiag = 0.0, bdiag = 0.0, ta = 0.0, tb = 0.0;
		for ( int i = 0; i < a.k; ++i ) {
			for ( int j = 0; j < a.k; ++j ) {
				const size_t e = static_cast<size_t>( i ) * a.k + j;
				const double x = a.h[e] * sa, y = b.h[e] * sb;
				dd += ( x - y ) * ( x - y );
				bb += y * y;
				if ( i == j ) {
					ddiag += ( x - y ) * ( x - y );
					bdiag += y * y;
					ta += x;
					tb += y;
				}
			}
		}
		const Row r{ fn, std::sqrt( dd / std::max( bb, 1e-300 ) ), std::sqrt( ddiag / std::max( bdiag, 1e-300 ) ), ta / std::max( tb, 1e-300 ) };
		rows.push_back( r );
		by_kind[kind_of( fn )].push_back( r.fro );
	}
	std::printf( "[*] hess_compare: %zu files in both, %d only in %s\n", rows.size(), missing, da.c_str() );
	std::printf( "  %-14s %6s %12s %12s   (|A - B| / |B| of the mean Hessians)\n", "site", "files", "mean", "worst" );
	for ( auto& kv : by_kind ) {
		double s = 0.0, w = 0.0;
		for ( double v : kv.second ) {
			s += v;
			w = std::max( w, v );
		}
		std::printf( "  %-14s %6zu %12.5f %12.5f\n", kv.first.c_str(), kv.second.size(), s / kv.second.size(), w );
	}
	std::sort( rows.begin(), rows.end(), []( const Row& x, const Row& y ) { return x.fro > y.fro; } );
	std::printf( "  worst files: name, |A - B| / |B|, the same for the diagonal, tr(A) / tr(B)\n" );
	for ( size_t i = 0; i < rows.size() && i < 10; ++i ) {
		std::printf( "    %-24s %.5f %.5f %.4f\n", rows[i].name.c_str(), rows[i].fro, rows[i].diag, rows[i].trace );
	}
	return 0;
}

// Mean Hessian on the device, and the sum for the quantizer
struct DevHess {
	float* sum = nullptr;
	float* mean = nullptr;
	int k = 0;
	int64_t rows = 0;
	bool load( const std::string& path ) {
		Hess h;
		if ( !read_hess( path, h ) ) return false;
		k = h.k;
		rows = h.rows;
		const size_t bytes = h.h.size() * sizeof( float );
		CUDA_CHECK( cudaMalloc( &sum, bytes ) );
		CUDA_CHECK( cudaMalloc( &mean, bytes ) );
		CUDA_CHECK( cudaMemcpy( sum, h.h.data(), bytes, cudaMemcpyHostToDevice ) );
		for ( auto& v : h.h ) v = static_cast<float>( v / static_cast<double>( rows ) );
		CUDA_CHECK( cudaMemcpy( mean, h.h.data(), bytes, cudaMemcpyHostToDevice ) );
		return true;
	}
	~DevHess() {
		if ( sum ) cudaFree( sum );
		if ( mean ) cudaFree( mean );
	}
};

// The stored tensor back to fp32 k x n in the original basis
bool decode( const engine::quant::Exl3Tensor& t, int code, int cb, bool volta, float* out ) {
	uint16_t *d_trellis = nullptr, *d_suh = nullptr, *d_svh = nullptr;
	half* d_w = nullptr;
	CUDA_CHECK( cudaMalloc( &d_trellis, t.trellis.size() * 2 ) );
	CUDA_CHECK( cudaMalloc( &d_suh, t.k * 2 ) );
	CUDA_CHECK( cudaMalloc( &d_svh, t.n * 2 ) );
	CUDA_CHECK( cudaMalloc( &d_w, static_cast<size_t>( t.k ) * t.n * 2 ) );
	CUDA_CHECK( cudaMemcpy( d_trellis, t.trellis.data(), t.trellis.size() * 2, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( d_suh, t.suh.data(), t.k * 2, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( d_svh, t.svh.data(), t.n * 2, cudaMemcpyHostToDevice ) );
	EXL3_TILES_CALL( volta, reconstruct_had_slice, d_w, d_trellis, reinterpret_cast<const half*>( d_suh ), reinterpret_cast<const half*>( d_svh ), code,
		cb == 1, cb == 2, t.k, t.n, 0, nullptr );
	std::vector<half> h( static_cast<size_t>( t.k ) * t.n );
	CUDA_CHECK( cudaMemcpy( h.data(), d_w, h.size() * 2, cudaMemcpyDeviceToHost ) );
	std::vector<float> f( h.size() );
	for ( size_t i = 0; i < h.size(); ++i ) f[i] = __half2float( h[i] );
	CUDA_CHECK( cudaMemcpy( out, f.data(), f.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaFree( d_trellis ) );
	CUDA_CHECK( cudaFree( d_suh ) );
	CUDA_CHECK( cudaFree( d_svh ) );
	CUDA_CHECK( cudaFree( d_w ) );
	return true;
}

int cross( int argc, char** argv ) {
	std::string src, tensor, fa, fb;
	int bits = 3, device = 0;
	bool volta = true;
	for ( int i = 2; i + 1 < argc; i += 2 ) {
		const std::string k = argv[i], v = argv[i + 1];
		if ( k == "--src" ) src = v;
		else if ( k == "--tensor" ) tensor = v;
		else if ( k == "--a" ) fa = v;
		else if ( k == "--b" ) fb = v;
		else if ( k == "--bits" ) bits = std::stoi( v );
		else if ( k == "--device" ) device = std::stoi( v );
		else if ( k == "--order" ) volta = v != "stock";
	}
	if ( src.empty() || tensor.empty() || fa.empty() || fb.empty() ) {
		std::fprintf( stderr, "usage: hess_compare cross --src <original> --tensor <base> --a <file> --b <file> [--bits 3] [--device 0] [--order volta|stock]\n" );
		return 2;
	}
	CUDA_CHECK( cudaSetDevice( device ) );
	engine::modules::trellis_context_init_all();
#if ENGINE_EXL3_VL
	volta = volta && engine::modules::trellis_volta_tiles_supported();
#else
	volta = false;
#endif
	engine::tools::quant::SourceWeights sw;
	int out = 0, in = 0;
	if ( !sw.open( src ) || !sw.shape( tensor, &out, &in ) ) {
		std::fprintf( stderr, "hess_compare: %s is not in %s\n", tensor.c_str(), src.c_str() );
		return 1;
	}
	DevHess ha, hb;
	if ( !ha.load( fa ) || !hb.load( fb ) || ha.k != in || hb.k != in ) {
		std::fprintf( stderr, "hess_compare: the Hessians do not load or do not match the tensor's %d inputs\n", in );
		return 1;
	}
	float *w = nullptr, *qa = nullptr, *qb = nullptr;
	const size_t bytes = static_cast<size_t>( in ) * out * sizeof( float );
	CUDA_CHECK( cudaMalloc( &w, bytes ) );
	CUDA_CHECK( cudaMalloc( &qa, bytes ) );
	CUDA_CHECK( cudaMalloc( &qb, bytes ) );
	if ( !sw.load( tensor, w, nullptr ) ) return 1;
	engine::quant::Exl3QuantOptions o;
	o.code = bits;
	o.volta = volta;
	const uint64_t seed = 0x5eedu;
	auto quantize = [&]( const DevHess& h, float* q ) {
		engine::quant::Exl3InputFactor f;
		ENGINE_ASSERT( engine::quant::exl3_input_factor( f, h.sum, in, h.rows, o.sigma_reg, seed, nullptr ), "factorization failed" );
		std::vector<engine::quant::Exl3Tensor> res;
		ENGINE_ASSERT( engine::quant::exl3_quantize_group( f, { w }, { out }, { seed + 1 }, o, res, nullptr ), "quantization failed" );
		engine::quant::exl3_input_factor_free( f );
		decode( res[0], o.code, o.cb, volta, q );
		return res[0].proxy_err;
	};
	const double own_a = quantize( ha, qa ), own_b = quantize( hb, qb );
	const double aa = engine::quant::exl3_proxy_error( w, qa, ha.mean, in, out, nullptr );
	const double ab = engine::quant::exl3_proxy_error( w, qa, hb.mean, in, out, nullptr );
	const double ba = engine::quant::exl3_proxy_error( w, qb, ha.mean, in, out, nullptr );
	const double bb = engine::quant::exl3_proxy_error( w, qb, hb.mean, in, out, nullptr );
	std::printf( "[*] hess_compare cross: %s (%d x %d), %d bits, %s order\n", tensor.c_str(), in, out, bits, volta ? "volta" : "stock" );
	std::printf( "  quantized with A: under A %.6f (reported %.6f), under B %.6f\n", aa, own_a, ab );
	std::printf( "  quantized with B: under A %.6f, under B %.6f (reported %.6f)\n", ba, bb, own_b );
	std::printf( "  A as a stand-in for B: %+.2f%% error under B; B for A: %+.2f%% under A\n", 100.0 * ( ab / bb - 1.0 ), 100.0 * ( ba / aa - 1.0 ) );
	CUDA_CHECK( cudaFree( w ) );
	CUDA_CHECK( cudaFree( qa ) );
	CUDA_CHECK( cudaFree( qb ) );
	return 0;
}

// The tensor <base> of an EXL3 checkpoint: width from the trellis shape, codebook from the markers, tile order from quantization_config.json
// ("tile_layout": "volta", with "tile_layout_stock" substrings that keep the stock order)
bool load_exl3( engine::tools::quant::StRepo& repo, const std::string& base, engine::quant::Exl3Tensor& t, int& code, int& cb, bool& volta ) {
	const auto* tr = repo.find( base + ".trellis" );
	const auto* su = repo.find( base + ".suh" );
	const auto* sv = repo.find( base + ".svh" );
	if ( !tr || !su || !sv || tr->shape.size() != 3 ) return false;
	t.k = static_cast<int>( tr->shape[0] ) * 16;
	t.n = static_cast<int>( tr->shape[1] ) * 16;
	code = engine::modules::trellis_code_from_u16( static_cast<int>( tr->shape[2] ) );
	cb = repo.has( base + ".mul1" ) ? 2 : repo.has( base + ".mcg" ) ? 1 : 0;
	t.trellis.assign( reinterpret_cast<const uint16_t*>( tr->data ), reinterpret_cast<const uint16_t*>( tr->data ) + tr->bytes / 2 );
	t.suh.assign( reinterpret_cast<const uint16_t*>( su->data ), reinterpret_cast<const uint16_t*>( su->data ) + su->bytes / 2 );
	t.svh.assign( reinterpret_cast<const uint16_t*>( sv->data ), reinterpret_cast<const uint16_t*>( sv->data ) + sv->bytes / 2 );
	volta = false;
	std::ifstream f( repo.dir() + "/quantization_config.json" );
	if ( f ) {
		std::stringstream ss;
		ss << f.rdbuf();
		const auto j = nlohmann::json::parse( ss.str(), nullptr, false );
		if ( !j.is_discarded() && j.value( "tile_layout", "" ) == "volta" ) {
			volta = true;
			if ( j.contains( "tile_layout_stock" ) ) {
				for ( const auto& s : j["tile_layout_stock"] ) {
					if ( s.is_string() && base.find( s.get<std::string>() ) != std::string::npos ) volta = false;
				}
			}
		}
	}
	return true;
}

int score( int argc, char** argv ) {
	std::string src, hess_dir, route;
	std::vector<std::string> tensors, dirs;
	int ranks = 4, device = 0;
	for ( int i = 2; i + 1 < argc; i += 2 ) {
		const std::string k = argv[i], v = argv[i + 1];
		if ( k == "--src" ) src = v;
		else if ( k == "--tensors" ) {
			std::stringstream ss( v );
			for ( std::string item; std::getline( ss, item, ',' ); ) if ( !item.empty() ) tensors.push_back( item );
		} else if ( k == "--hess-dir" ) hess_dir = v;
		else if ( k == "--route" ) route = v;
		else if ( k == "--ranks" ) ranks = std::stoi( v );
		else if ( k == "--exl3" ) dirs.push_back( v );
		else if ( k == "--device" ) device = std::stoi( v );
	}
	if ( src.empty() || tensors.empty() || dirs.empty() || ( hess_dir.empty() && route.empty() ) ) {
		std::fprintf( stderr, "usage: hess_compare score --src <original> --tensors A,B (--hess-dir <dir> [--ranks 4] | --route <dump>) --exl3 <dir> ...\n" );
		return 2;
	}
	CUDA_CHECK( cudaSetDevice( device ) );
	engine::modules::trellis_context_init_all();
	engine::tools::quant::Model model;
	if ( !model.read( dirs[0] ) ) return 1;
	engine::tools::quant::SourceWeights sw;
	if ( !sw.open( src ) ) return 1;
	std::vector<std::unique_ptr<engine::tools::quant::StRepo>> repos;
	for ( const auto& d : dirs ) {
		repos.push_back( std::make_unique<engine::tools::quant::StRepo>() );
		if ( !repos.back()->open( d ) ) return 1;
	}
	engine::tools::quant::RouteRows rr;
	if ( !route.empty() && !rr.open( route, model.hidden ) ) return 1;
	engine::tools::quant::ExpertHessian eh;
	int x_layer = -1;
	void* d_x = nullptr;
	int64_t x_rows = 0;
	std::vector<double> sum_ratio( dirs.size(), 0.0 );
	int scored = 0;
	std::printf( "[*] hess_compare score: proxy error under the mean Hessian, checkpoints:" );
	for ( size_t c = 0; c < dirs.size(); ++c ) std::printf( " [%zu] %s", c, dirs[c].c_str() );
	std::printf( "\n" );
	for ( const auto& base : tensors ) {
		int out = 0, in = 0;
		if ( !sw.shape( base, &out, &in ) ) {
			std::printf( "  %s: not in the original\n", base.c_str() );
			continue;
		}
		engine::tools::quant::PlanTensor pt;
		pt.base = base;
		engine::tools::quant::hessian_sources( model, hess_dir, ranks, pt );
		// The mean Hessian: the capture's file(s), block diagonal over the ranks, or the expert's activation over the route rows
		std::vector<float> H( static_cast<size_t>( in ) * in, 0.0f );
		bool have = false;
		if ( pt.down_layer >= 0 && !route.empty() ) {
			if ( x_layer != pt.down_layer ) {
				if ( d_x ) CUDA_CHECK( cudaFree( d_x ) );
				std::vector<uint16_t> rows;
				if ( !rr.read( pt.down_layer, 0, rows ) ) continue;
				x_rows = static_cast<int64_t>( rows.size() / model.hidden );
				CUDA_CHECK( cudaMalloc( &d_x, rows.size() * 2 ) );
				CUDA_CHECK( cudaMemcpy( d_x, rows.data(), rows.size() * 2, cudaMemcpyHostToDevice ) );
				x_layer = pt.down_layer;
			}
			const std::string pre = pt.src_layer_prefix + "." + std::to_string( pt.down_expert );
			float *g = nullptr, *u = nullptr, *h = nullptr;
			CUDA_CHECK( cudaMalloc( &g, static_cast<size_t>( out ) * in * 4 ) );
			CUDA_CHECK( cudaMalloc( &u, static_cast<size_t>( out ) * in * 4 ) );
			CUDA_CHECK( cudaMalloc( &h, H.size() * 4 ) );
			have = sw.load( pre + ".gate_proj", g, nullptr ) && sw.load( pre + ".up_proj", u, nullptr ) &&
				eh.compute( d_x, x_rows, out, g, u, in, model.swiglu_limit, h, nullptr );
			if ( have ) {
				CUDA_CHECK( cudaDeviceSynchronize() );
				CUDA_CHECK( cudaMemcpy( H.data(), h, H.size() * 4, cudaMemcpyDeviceToHost ) );
				for ( auto& v : H ) v = static_cast<float>( v / static_cast<double>( x_rows ) );
			}
			CUDA_CHECK( cudaFree( g ) );
			CUDA_CHECK( cudaFree( u ) );
			CUDA_CHECK( cudaFree( h ) );
		} else if ( !pt.hess.empty() ) {
			int off = 0;
			have = true;
			for ( const auto& p : pt.hess ) {
				Hess part;
				if ( !read_hess( p, part ) || off + part.k > in ) {
					have = false;
					break;
				}
				for ( int i = 0; i < part.k; ++i ) {
					for ( int j = 0; j < part.k; ++j ) {
						H[static_cast<size_t>( off + i ) * in + off + j] = static_cast<float>( part.h[static_cast<size_t>( i ) * part.k + j] / static_cast<double>( part.rows ) );
					}
				}
				off += part.k;
			}
			have = have && off == in;
		}
		if ( !have ) {
			std::printf( "  %s: no Hessian\n", base.c_str() );
			continue;
		}
		float *d_h = nullptr, *w = nullptr, *q = nullptr;
		CUDA_CHECK( cudaMalloc( &d_h, H.size() * 4 ) );
		CUDA_CHECK( cudaMemcpy( d_h, H.data(), H.size() * 4, cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMalloc( &w, static_cast<size_t>( in ) * out * 4 ) );
		CUDA_CHECK( cudaMalloc( &q, static_cast<size_t>( in ) * out * 4 ) );
		if ( !sw.load( base, w, nullptr ) ) continue;
		std::printf( "  %-62s", base.c_str() );
		std::vector<double> errs;
		for ( size_t c = 0; c < repos.size(); ++c ) {
			engine::quant::Exl3Tensor t;
			int code = 0, cb = 0;
			bool volta = false;
			if ( !load_exl3( *repos[c], base, t, code, cb, volta ) || t.k != in || t.n != out ) {
				std::printf( "   [%zu] -", c );
				errs.push_back( -1.0 );
				continue;
			}
			decode( t, code, cb, volta, q );
			const double e = engine::quant::exl3_proxy_error( w, q, d_h, in, out, nullptr );
			errs.push_back( e );
			std::printf( "   [%zu] %.6f (%d%s bits)", c, e, engine::modules::trellis_k2( code ) / 2, engine::modules::trellis_k2( code ) % 2 ? ".5" : "" );
		}
		std::printf( "\n" );
		if ( errs.size() > 1 && errs[0] > 0.0 ) {
			bool all = true;
			for ( double e : errs ) all = all && e > 0.0;
			if ( all ) {
				for ( size_t c = 0; c < errs.size(); ++c ) sum_ratio[c] += errs[c] / errs[0];
				++scored;
			}
		}
		CUDA_CHECK( cudaFree( d_h ) );
		CUDA_CHECK( cudaFree( w ) );
		CUDA_CHECK( cudaFree( q ) );
	}
	if ( scored ) {
		std::printf( "  mean error relative to [0] over %d tensors:", scored );
		for ( size_t c = 0; c < dirs.size(); ++c ) std::printf( " [%zu] %.4f", c, sum_ratio[c] / scored );
		std::printf( "\n" );
	}
	if ( d_x ) CUDA_CHECK( cudaFree( d_x ) );
	return 0;
}

} // namespace

int main( int argc, char** argv ) {
	if ( argc >= 4 && !std::strcmp( argv[1], "stats" ) ) return stats( argv[2], argv[3] );
	if ( argc >= 2 && !std::strcmp( argv[1], "cross" ) ) return cross( argc, argv );
	if ( argc >= 2 && !std::strcmp( argv[1], "score" ) ) return score( argc, argv );
	std::fprintf( stderr, "usage: hess_compare stats <dir_a> <dir_b>\n       hess_compare cross --src <original> --tensor <base> --a <file> --b <file> "
						  "[--bits 3] [--device 0] [--order volta|stock]\n" );
	return 2;
}
