// volta_quantize: EXL3 quantization of a checkpoint by the engine. A template EXL3 checkpoint of the same model names the tensors to quantize,
// their widths and codebooks, and supplies everything else; the original weights (BF16 / FP16 / FP32, FP8 E4M3 with block scales, fused experts)
// are quantized with the Hessians of an engine capture (test_ppl --calibrate with hessian_dir), the routed experts' down projections with the
// Hessians of their own activations over the route dump's rows. Work items (the projections of one input, in chunks, or a layer's expert down
// projections) are spread over the devices; every finished tensor is kept in <out>/.partial, so a run continues where it stopped; the shards are
// written at the end in the template's layout (--no-assemble stops before, a later run without it writes them).
//
//   volta_quantize --src <original> --template <exl3 dir> --out <dir> [--hess <capture dir>] [--route <route dump>] [--order volta|stock]
//                  [--devices 0,1,2,3] [--only A,B] [--skip A,B] [--bits B] [--ranks 4] [--route-rows N] [--budget-mb 8192] [--seed N] [--redo]
//                  [--plan] [--no-scaled-hessian] [--affinity-eps E] [--ldl-metric] [--open-trellis] [--tile-gain] [--no-assemble]
//
// --open-trellis stores each tile's start state (<base>.tstart) instead of wrapping its stream, --tile-gain an 8-bit gain per tile (<base>.tgain):
// the extended trellis, which the reconstruct path and the tools decode; until the inference paths do, only with --no-assemble
//
// --affinity-eps weights each routed expert's Hessians by routing (exllamav3 1.6.7's --moe_affinity_eps, see quant/expert_hessian.h): its gate
// and up projections get their own input Hessian from the route dump's rows routed to it, so they are quantized expert by expert; needs the
// dump's routing weights (<route dump>.w)
//
// --bits sets every quantized tensor's width (1 to 8, or 1.5 / 2.5 / 3.5 with the mul1 codebook) instead of the template's: rate ladders on a
// sample (--only) for screening the quantizer
#include "engine/config.h"
#include "engine/core/log.h"
#include "engine/modules/trellis_bits.h"
#include "engine/quant/exl3_quantize.h"
#include "quant/exl3_read.h"
#include "quant/expert_hessian.h"
#include "quant/plan.h"
#include "quant/source.h"
#include "quant/st_io.h"
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace q = engine::tools::quant;
using engine::quant::Exl3InputFactor;
using engine::quant::Exl3QuantOptions;
using engine::quant::Exl3Tensor;

namespace {

struct Args {
	std::string src, tmpl, out, hess, route;
	std::vector<std::string> only, skip;
	int code = 0;   // --bits as a width code, 0: the template's
	std::vector<int> devices = { 0 };
	bool volta = true;
	int ranks = 4;
	int64_t route_rows = 0;
	size_t budget = size_t( 8192 ) << 20;
	uint64_t seed = 0;
	bool redo = false;
	bool plan = false;
	bool assemble = true;
	bool scaled = true;
	float aff_eps = 0.0f;
	bool ldl_metric = false, open_trellis = false, tile_gain = false;
	float sigma_reg = 0.025f;
};

std::vector<std::string> split( const std::string& s, char sep ) {
	std::vector<std::string> out;
	std::stringstream ss( s );
	std::string item;
	while ( std::getline( ss, item, sep ) ) {
		if ( !item.empty() ) out.push_back( item );
	}
	return out;
}

bool parse( int argc, char** argv, Args& a ) {
	for ( int i = 1; i < argc; ++i ) {
		const std::string k = argv[i];
		auto val = [&]() -> std::string {
			if ( i + 1 >= argc ) {
				LOG_ERROR( "[volta_quantize] %s needs a value", k.c_str() );
				std::exit( 2 );
			}
			return argv[++i];
		};
		if ( k == "--src" ) a.src = val();
		else if ( k == "--template" ) a.tmpl = val();
		else if ( k == "--out" ) a.out = val();
		else if ( k == "--hess" ) a.hess = val();
		else if ( k == "--route" ) a.route = val();
		else if ( k == "--only" ) a.only = split( val(), ',' );
		else if ( k == "--bits" ) {
			const float b = std::stof( val() );
			const int K = static_cast<int>( b );
			a.code = b == static_cast<float>( K ) ? K : K + engine::modules::TRELLIS_HALF_CODE;
			if ( K < 1 || K > 8 || ( b != static_cast<float>( K ) && ( b - K != 0.5f || K > 3 ) ) ) {
				LOG_ERROR( "[volta_quantize] --bits %s: 1 to 8, or 1.5, 2.5, 3.5", argv[i] );
				return false;
			}
		}
		else if ( k == "--skip" ) a.skip = split( val(), ',' );
		else if ( k == "--order" ) a.volta = val() != "stock";
		else if ( k == "--ranks" ) a.ranks = std::stoi( val() );
		else if ( k == "--route-rows" ) a.route_rows = std::stoll( val() );
		else if ( k == "--budget-mb" ) a.budget = size_t( std::stoll( val() ) ) << 20;
		else if ( k == "--seed" ) a.seed = std::stoull( val() );
		else if ( k == "--sigma-reg" ) a.sigma_reg = std::stof( val() );
		else if ( k == "--redo" ) a.redo = true;
		else if ( k == "--plan" ) a.plan = true;
		else if ( k == "--no-scaled-hessian" ) a.scaled = false;
		else if ( k == "--no-assemble" ) a.assemble = false;
		else if ( k == "--affinity-eps" ) a.aff_eps = std::stof( val() );
		else if ( k == "--ldl-metric" ) a.ldl_metric = true;
		else if ( k == "--open-trellis" ) a.open_trellis = true;
		else if ( k == "--tile-gain" ) a.tile_gain = true;
		else if ( k == "--devices" ) {
			a.devices.clear();
			for ( const auto& d : split( val(), ',' ) ) a.devices.push_back( std::stoi( d ) );
		} else {
			LOG_ERROR( "[volta_quantize] unknown argument %s", k.c_str() );
			return false;
		}
	}
	if ( a.src.empty() || a.tmpl.empty() || ( a.out.empty() && !a.plan ) ) {
		LOG_ERROR( "usage: volta_quantize --src <original> --template <exl3 dir> --out <dir> [--hess <capture dir>] [--route <route dump>] "
				   "[--order volta|stock] [--devices 0,1] [--only A,B] [--skip A,B] [--bits B] [--ranks 4] [--route-rows N] [--budget-mb 8192] [--seed N] "
				   "[--redo] [--plan] [--no-scaled-hessian] [--affinity-eps E] [--ldl-metric] [--open-trellis] [--tile-gain] [--no-assemble]" );
		return false;
	}
	return true;
}

bool exists( const std::string& p ) {
	struct stat st {};
	return stat( p.c_str(), &st ) == 0;
}

bool ends_with( const std::string& s, const std::string& suf ) { return s.size() >= suf.size() && s.compare( s.size() - suf.size(), suf.size(), suf ) == 0; }

uint64_t hash64( const std::string& s, uint64_t seed ) {
	uint64_t h = 1469598103934665603ull ^ ( seed * 0x9E3779B97F4A7C15ull );
	for ( unsigned char c : s ) {
		h ^= c;
		h *= 1099511628211ull;
	}
	return h;
}

std::string read_text( const std::string& path ) {
	std::ifstream f( path );
	std::stringstream ss;
	ss << f.rdbuf();
	return ss.str();
}

// A capture file of the engine's Hessian format: int32 magic, int32 k, int64 rows, k x k fp32 sums
bool read_hessian( const std::string& path, int& k, int64_t& rows, std::vector<float>& h ) {
	FILE* f = std::fopen( path.c_str(), "rb" );
	if ( !f ) return false;
	int32_t hdr[2];
	bool ok = std::fread( hdr, 4, 2, f ) == 2 && hdr[0] == 0x48455353 && std::fread( &rows, 8, 1, f ) == 1;
	if ( ok ) {
		k = hdr[1];
		h.resize( static_cast<size_t>( k ) * k );
		ok = std::fread( h.data(), 4, h.size(), f ) == h.size();
	}
	std::fclose( f );
	return ok;
}

// A work item: tensors that share one input factor (or one layer's expert down projections, each its own)
struct Item {
	std::vector<int> tensors;       // indices into the plan
	bool expert_down = false;
	bool expert_gate_up = false;    // affinity-weighted: a layer's expert gate / up projections, expert by expert
	int layer = -1;
	std::string key;                // Hessian files joined, or "" (none)
	int64_t tiles = 0;
};

struct Shared {
	const Args* a = nullptr;
	const q::Model* model = nullptr;
	std::vector<q::PlanTensor> plan;
	std::vector<Item> items;
	std::atomic<size_t> next{ 0 };
	std::atomic<int64_t> tiles_done{ 0 };
	int64_t tiles_total = 0;
	std::atomic<int> failures{ 0 };
	std::mutex log_mu;
	std::chrono::steady_clock::time_point t0;
	q::RouteRows route;
};

std::string partial_path( const Args& a, const std::string& base ) { return a.out + "/.partial/" + base + ".safetensors"; }

bool write_partial( const Args& a, const q::PlanTensor& t, const Exl3Tensor& r ) {
	char err[32];
	std::snprintf( err, sizeof( err ), "%.8g", r.proxy_err );
	return q::exl3_write_partial( partial_path( a, t.base ), t.base, r, t.code, t.cb, a.volta, { { "proxy_err", err } } );
}

struct DBuf {
	float* p = nullptr;
	size_t n = 0;
	~DBuf() {
		if ( p ) cudaFree( p );
	}
	bool fit( size_t count ) {
		if ( count <= n ) return true;
		if ( p ) CUDA_CHECK( cudaFree( p ) );
		p = nullptr;
		n = 0;
		if ( cudaMalloc( &p, count * sizeof( float ) ) != cudaSuccess ) {
			(void) cudaGetLastError();
			return false;
		}
		n = count;
		return true;
	}
};

// The capture's Hessian of a tensor's input (k x k) on the device: its file, or the per-rank blocks of a row-parallel input on the diagonal (the
// cross-rank covariance is not captured); the sum over the rows, or with `mean` divided by them
bool read_capture( const q::PlanTensor& t, int k, bool mean, int64_t& rows, float* d_dst, cudaStream_t stream ) {
	std::vector<float> full( static_cast<size_t>( k ) * k, 0.0f ), part;
	int off = 0;
	rows = 0;
	for ( const auto& path : t.hess ) {
		int kr = 0;
		int64_t r = 0;
		if ( !read_hessian( path, kr, r, part ) || off + kr > k ) return false;
		for ( int i = 0; i < kr; ++i ) std::memcpy( &full[static_cast<size_t>( off + i ) * k + off], &part[static_cast<size_t>( i ) * kr], kr * 4 );
		off += kr;
		rows = r;
	}
	if ( off != k ) return false;
	if ( mean && rows > 0 ) {
		const float s = static_cast<float>( 1.0 / static_cast<double>( rows ) );
		for ( auto& v : full ) v *= s;
	}
	CUDA_CHECK( cudaMemcpyAsync( d_dst, full.data(), full.size() * 4, cudaMemcpyHostToDevice, stream ) );
	CUDA_CHECK( cudaStreamSynchronize( stream ) );
	return true;
}

// One device: takes items until none are left
void worker( Shared& sh, int device, const Exl3QuantOptions& base_opts ) {
	const Args& a = *sh.a;
	CUDA_CHECK( cudaSetDevice( device ) );
	cudaStream_t stream = nullptr;
	CUDA_CHECK( cudaStreamCreateWithFlags( &stream, cudaStreamNonBlocking ) );
	q::SourceWeights src;
	if ( !src.open( a.src ) ) {
		sh.failures++;
		return;
	}
	q::ExpertHessian eh;
	Exl3InputFactor factor;
	std::string factor_key;
	int x_layer = -1;
	void* d_x = nullptr;
	int64_t x_rows = 0;
	DBuf hsum, gate, up, hin, row_w, d_rw;
	std::vector<float*> weights;
	// The layer's route-dump rows on the device; with affinity weights, per expert its routed rows and their weights
	std::vector<std::vector<int>> e_rows;
	std::vector<std::vector<float>> e_w;
	auto ensure_layer = [&]( int layer ) {
		if ( x_layer == layer ) return true;
		if ( d_x ) CUDA_CHECK( cudaFree( d_x ) );
		d_x = nullptr;
		x_layer = -1;
		std::vector<uint16_t> rows;
		if ( !sh.route.read( layer, a.route_rows, rows ) ) {
			LOG_ERROR( "[volta_quantize] the route dump has no rows of layer %d", layer );
			return false;
		}
		x_rows = static_cast<int64_t>( rows.size() / sh.model->hidden );
		CUDA_CHECK( cudaMalloc( &d_x, rows.size() * 2 ) );
		CUDA_CHECK( cudaMemcpyAsync( d_x, rows.data(), rows.size() * 2, cudaMemcpyHostToDevice, stream ) );
		CUDA_CHECK( cudaStreamSynchronize( stream ) );
		e_rows.assign( std::max( sh.model->experts, 0 ), {} );
		e_w.assign( e_rows.size(), {} );
		if ( a.aff_eps > 0.0f ) {
			std::vector<int> ids;
			std::vector<float> w;
			int K = 0;
			if ( !sh.route.routes( layer, a.route_rows, ids, w, K ) ) {
				LOG_ERROR( "[volta_quantize] the route dump has no routing weights for layer %d", layer );
				return false;
			}
			for ( int64_t r = 0; r < x_rows; ++r ) {
				for ( int j = 0; j < K; ++j ) {
					const int e = ids[static_cast<size_t>( r ) * K + j];
					if ( e < 0 || e >= static_cast<int>( e_rows.size() ) ) continue;
					e_rows[e].push_back( static_cast<int>( r ) );
					e_w[e].push_back( w[static_cast<size_t>( r ) * K + j] );
				}
			}
		}
		x_layer = layer;
		return true;
	};
	auto w_sum = [&]( int e ) {
		double s = 0.0;
		for ( float v : e_w[e] ) s += v;
		return s;
	};

	for ( size_t idx; ( idx = sh.next.fetch_add( 1 ) ) < sh.items.size(); ) {
		const Item& it = sh.items[idx];
		const auto t_item = std::chrono::steady_clock::now();
		const q::PlanTensor& first = sh.plan[it.tensors[0]];
		Exl3QuantOptions o = base_opts;
		o.code = first.code;
		o.cb = first.cb;
		std::vector<std::pair<int, Exl3Tensor>> done;
		bool ok = true;

		if ( it.expert_down ) {
			// The layer's rows on the device, then per expert: its activation's Hessian, the factor, the down projection
			if ( !ensure_layer( it.layer ) ) {
				sh.failures++;
				continue;
			}
			for ( int ti : it.tensors ) {
				const q::PlanTensor& t = sh.plan[ti];
				const int inter = t.k, hidden = t.n;
				const std::string pre = t.src_layer_prefix + "." + std::to_string( t.down_expert );
				// Affinity: the expert's routing weight per row (0 where it is not routed); never routed: the plain Hessian
				const double ws = a.aff_eps > 0.0f ? w_sum( t.down_expert ) : 0.0;
				const float* rw = nullptr;
				if ( ws > 0.0 && row_w.fit( static_cast<size_t>( x_rows ) ) ) {
					std::vector<float> h( static_cast<size_t>( x_rows ), 0.0f );
					for ( size_t j = 0; j < e_rows[t.down_expert].size(); ++j ) h[e_rows[t.down_expert][j]] = e_w[t.down_expert][j];
					CUDA_CHECK( cudaMemcpyAsync( row_w.p, h.data(), h.size() * 4, cudaMemcpyHostToDevice, stream ) );
					CUDA_CHECK( cudaStreamSynchronize( stream ) );
					rw = row_w.p;
				}
				ok = gate.fit( static_cast<size_t>( hidden ) * inter ) && up.fit( static_cast<size_t>( hidden ) * inter ) &&
					hsum.fit( static_cast<size_t>( inter ) * inter ) && src.load( pre + ".gate_proj", gate.p, stream ) &&
					src.load( pre + ".up_proj", up.p, stream ) &&
					eh.compute( d_x, x_rows, hidden, gate.p, up.p, inter, sh.model->swiglu_limit, hsum.p, stream, rw, ws, a.aff_eps ) &&
					engine::quant::exl3_input_factor( factor, hsum.p, inter, x_rows, a.sigma_reg, hash64( t.base + "#in", a.seed ), stream, !a.scaled );
				factor_key.clear();
				float* w = nullptr;
				ok = ok && cudaMalloc( &w, static_cast<size_t>( inter ) * hidden * sizeof( float ) ) == cudaSuccess && src.load( t.base, w, stream );
				std::vector<Exl3Tensor> out;
				ok = ok && engine::quant::exl3_quantize_group( factor, { w }, { hidden }, { hash64( t.base, a.seed ) }, o, out, stream );
				if ( w ) CUDA_CHECK( cudaFree( w ) );
				if ( !ok ) break;
				o.scale_hint = out[0].g_scale;   // the next expert's scale search starts from this one's
				done.emplace_back( ti, std::move( out[0] ) );
			}
		} else if ( it.expert_gate_up ) {
			// Affinity: the layer input's mean Hessian, then per expert its own Hessian (routed rows weighted, plus eps of the mean), its factor,
			// its gate and up projections
			const int k = first.k;
			int64_t rows = 0;
			ok = ensure_layer( it.layer ) && hin.fit( static_cast<size_t>( k ) * k ) && hsum.fit( static_cast<size_t>( k ) * k ) &&
				read_capture( first, k, true, rows, hin.p, stream );
			std::map<int, std::vector<int>> by_expert;
			for ( int ti : it.tensors ) by_expert[sh.plan[ti].gate_up_expert].push_back( ti );
			for ( const auto& kv : by_expert ) {
				if ( !ok ) break;
				const int e = kv.first;
				const double ws = w_sum( e );
				const int nr = static_cast<int>( e_rows[e].size() );
				int* d_idx = nullptr;
				if ( ws > 0.0 ) {
					ok = cudaMalloc( &d_idx, static_cast<size_t>( nr ) * sizeof( int ) ) == cudaSuccess && d_rw.fit( static_cast<size_t>( nr ) );
					if ( ok ) {
						CUDA_CHECK( cudaMemcpyAsync( d_idx, e_rows[e].data(), static_cast<size_t>( nr ) * sizeof( int ), cudaMemcpyHostToDevice, stream ) );
						CUDA_CHECK( cudaMemcpyAsync( d_rw.p, e_w[e].data(), static_cast<size_t>( nr ) * sizeof( float ), cudaMemcpyHostToDevice, stream ) );
						ok = eh.affinity_input( d_x, x_rows, k, d_idx, d_rw.p, nr, ws, a.aff_eps, hin.p, hsum.p, stream );
						CUDA_CHECK( cudaStreamSynchronize( stream ) );
					}
				} else {
					// Never routed in the calibration: the layer input's Hessian as it is
					CUDA_CHECK( cudaMemcpyAsync( hsum.p, hin.p, static_cast<size_t>( k ) * k * 4, cudaMemcpyDeviceToDevice, stream ) );
				}
				if ( d_idx ) CUDA_CHECK( cudaFree( d_idx ) );
				// The same input sign flips as the shared factor of the plain recipe (one seed per input)
				ok = ok && engine::quant::exl3_input_factor( factor, hsum.p, k, 1, a.sigma_reg, hash64( it.key, a.seed ), stream, !a.scaled );
				factor_key.clear();
				std::vector<const float*> ws_;
				std::vector<int> ns;
				std::vector<uint64_t> seeds;
				for ( int ti : kv.second ) {
					if ( !ok ) break;
					const q::PlanTensor& t = sh.plan[ti];
					float* w = nullptr;
					ok = cudaMalloc( &w, static_cast<size_t>( t.k ) * t.n * sizeof( float ) ) == cudaSuccess && src.load( t.base, w, stream );
					if ( w ) weights.push_back( w );
					ws_.push_back( w );
					ns.push_back( t.n );
					seeds.push_back( hash64( t.base, a.seed ) );
				}
				std::vector<Exl3Tensor> out;
				ok = ok && engine::quant::exl3_quantize_group( factor, ws_, ns, seeds, o, out, stream );
				for ( float* w : weights ) CUDA_CHECK( cudaFree( w ) );
				weights.clear();
				if ( !ok ) break;
				o.scale_hint = out.front().g_scale;
				for ( size_t i = 0; i < kv.second.size(); ++i ) done.emplace_back( kv.second[i], std::move( out[i] ) );
			}
		} else {
			// The shared input factor, kept while consecutive items use it
			if ( it.key != factor_key || it.key.empty() ) {
				factor_key.clear();
				const int k = first.k;
				int64_t rows = 0;
				if ( !it.key.empty() ) ok = hsum.fit( static_cast<size_t>( k ) * k ) && read_capture( first, k, false, rows, hsum.p, stream );
				const uint64_t fseed = hash64( it.key.empty() ? first.base + "#in" : it.key, a.seed );
				ok = ok && engine::quant::exl3_input_factor( factor, it.key.empty() ? nullptr : hsum.p, k, rows, a.sigma_reg, fseed, stream, !a.scaled );
				if ( ok ) factor_key = it.key;
			}
			std::vector<const float*> ws;
			std::vector<int> ns;
			std::vector<uint64_t> seeds;
			for ( float* w : weights ) CUDA_CHECK( cudaFree( w ) );
			weights.clear();
			for ( int ti : it.tensors ) {
				if ( !ok ) break;
				const q::PlanTensor& t = sh.plan[ti];
				float* w = nullptr;
				ok = cudaMalloc( &w, static_cast<size_t>( t.k ) * t.n * sizeof( float ) ) == cudaSuccess && src.load( t.base, w, stream );
				if ( w ) weights.push_back( w );
				ws.push_back( w );
				ns.push_back( t.n );
				seeds.push_back( hash64( t.base, a.seed ) );
			}
			std::vector<Exl3Tensor> out;
			ok = ok && engine::quant::exl3_quantize_group( factor, ws, ns, seeds, o, out, stream );
			for ( float* w : weights ) CUDA_CHECK( cudaFree( w ) );
			weights.clear();
			if ( ok ) {
				for ( size_t i = 0; i < it.tensors.size(); ++i ) done.emplace_back( it.tensors[i], std::move( out[i] ) );
			}
		}

		double err_sum = 0.0;
		for ( const auto& d : done ) {
			if ( !write_partial( a, sh.plan[d.first], d.second ) ) ok = false;
			err_sum += d.second.proxy_err;
		}
		const int64_t tiles = sh.tiles_done += it.tiles;
		const double secs = std::chrono::duration<double>( std::chrono::steady_clock::now() - t_item ).count();
		const double total = std::chrono::duration<double>( std::chrono::steady_clock::now() - sh.t0 ).count();
		std::lock_guard<std::mutex> lock( sh.log_mu );
		if ( !ok ) {
			sh.failures++;
			LOG_ERROR( "[volta_quantize] GPU %d: %s (%zu tensors) failed", device, first.base.c_str(), it.tensors.size() );
			continue;
		}
		const double rate = tiles / std::max( total, 1e-9 );
		LOG_INFO( "[volta_quantize] GPU %d: %s%s, %zu tensors, %d bits%s, mean proxy error %.5f, %.1f s (%.0f%%, %.0f tiles/s, %.0f min left)", device,
			first.base.c_str(), it.tensors.size() > 1 ? " ..." : "", done.size(), engine::modules::trellis_k2( o.code ) / 2,
			engine::modules::trellis_k2( o.code ) % 2 ? ".5" : "", err_sum / std::max<size_t>( done.size(), 1 ), secs,
			100.0 * tiles / std::max<int64_t>( sh.tiles_total, 1 ), rate, ( sh.tiles_total - tiles ) / std::max( rate, 1e-9 ) / 60.0 );
	}
	engine::quant::exl3_input_factor_free( factor );
	if ( d_x ) CUDA_CHECK( cudaFree( d_x ) );
	CUDA_CHECK( cudaStreamDestroy( stream ) );
}

// The output's quantization_config.json: the template's, with the tile order of what was quantized. From a stock template, what stays as the
// template has it (the --skip patterns, the tensors the original lacks or shapes differently, the ones without a result) keeps the stock order:
// "tile_layout_stock" lists them (the loader matches substrings)
std::string layout_config( const Args& a, const std::string& text, const std::vector<std::string>& stock_kept ) {
	auto cfg = nlohmann::json::parse( text.empty() ? std::string( "{\"quant_method\": \"exl3\"}" ) : text, nullptr, false );
	if ( cfg.is_discarded() ) cfg = nlohmann::json::object( { { "quant_method", "exl3" } } );
	const bool template_volta = cfg.value( "tile_layout", "" ) == "volta";
	if ( a.volta ) {
		cfg["tile_layout"] = "volta";
		if ( !template_volta ) {
			std::vector<std::string> stock = a.skip;
			stock.insert( stock.end(), stock_kept.begin(), stock_kept.end() );
			if ( !stock.empty() ) cfg["tile_layout_stock"] = stock;
		}
	} else {
		cfg.erase( "tile_layout" );
		cfg.erase( "tile_layout_stock" );
	}
	return cfg.dump( 4 );
}

bool copy_file( const std::string& from, const std::string& to ) {
	std::ifstream in( from, std::ios::binary );
	std::ofstream out( to, std::ios::binary );
	if ( !in || !out ) return false;
	out << in.rdbuf();
	return static_cast<bool>( out );
}

// The output shards in the template's layout: every quantized tensor from its partial file, everything else as the template has it
bool assemble( const Args& a, const std::vector<q::PlanTensor>& plan, std::vector<std::string> stock_kept, int& kept ) {
	std::set<std::string> quantized;
	for ( const auto& t : plan ) {
		quantized.insert( t.base );
		if ( !exists( partial_path( a, t.base ) ) ) stock_kept.push_back( t.base );
	}
	std::vector<std::string> files;
	if ( DIR* d = opendir( a.tmpl.c_str() ) ) {
		while ( dirent* de = readdir( d ) ) {
			if ( de->d_name[0] != '.' ) files.push_back( de->d_name );
		}
		closedir( d );
	}
	std::sort( files.begin(), files.end() );
	kept = 0;
	for ( const auto& fn : files ) {
		const std::string from = a.tmpl + "/" + fn, to = a.out + "/" + fn;
		if ( fn == "ngram_embedding.safetensors" ) {
			// Flash-Next's n-gram table: its own large file, linked as it is
			if ( !exists( to ) && symlink( from.c_str(), to.c_str() ) != 0 ) return false;
			continue;
		}
		if ( !ends_with( fn, ".safetensors" ) ) {
			struct stat st {};
			if ( stat( from.c_str(), &st ) != 0 || !S_ISREG( st.st_mode ) ) continue;
			if ( fn == "quantization_config.json" ) {
				std::ofstream( to ) << layout_config( a, read_text( from ), stock_kept ) << "\n";
			} else if ( !copy_file( from, to ) ) {
				return false;
			}
			continue;
		}
		if ( exists( to ) && !a.redo ) continue;
		q::StFile tf;
		if ( !tf.open( from ) ) return false;
		std::map<std::string, std::unique_ptr<q::StFile>> parts;
		std::vector<q::StOut> outs;
		for ( const auto& name : tf.names() ) {
			const q::StEntry* e = tf.find( name );
			const size_t dot = name.rfind( '.' );
			const std::string base = name.substr( 0, dot ), suf = name.substr( dot + 1 );
			const q::StEntry* use = e;
			if ( quantized.count( base ) && ( suf == "trellis" || suf == "suh" || suf == "svh" ) ) {
				auto& pf = parts[base];
				if ( !pf && exists( partial_path( a, base ) ) ) {
					pf = std::make_unique<q::StFile>();
					if ( !pf->open( partial_path( a, base ) ) ) pf.reset();
				}
				if ( pf && pf->find( name ) ) use = pf->find( name );
				else if ( suf == "trellis" ) ++kept;
			}
			q::StOut o;
			o.name = name;
			o.dtype = use->dtype;
			o.shape = use->shape;
			o.view = use->data;
			o.view_bytes = use->bytes;
			outs.push_back( std::move( o ) );
		}
		if ( !q::st_write( to, outs, tf.metadata() ) ) return false;
		LOG_INFO( "[volta_quantize] wrote %s", fn.c_str() );
	}
	// A template without a quantization_config.json (the GLM drafter): a minimal one
	if ( !exists( a.tmpl + "/quantization_config.json" ) ) std::ofstream( a.out + "/quantization_config.json" ) << layout_config( a, "", stock_kept ) << "\n";
	return true;
}

} // namespace

int main( int argc, char** argv ) {
	// A run takes hours and is usually redirected to a file: one progress line at a time
	std::setvbuf( stdout, nullptr, _IOLBF, 0 );
	Args a;
	if ( !parse( argc, argv, a ) ) return 2;
	q::Model model;
	if ( !model.read( a.tmpl ) ) return 1;
	q::StRepo tmpl;
	if ( !tmpl.open( a.tmpl ) ) return 1;
	q::SourceWeights src;
	if ( !src.open( a.src ) ) return 1;
	const auto qtext = exists( a.tmpl + "/quantization_config.json" ) ? read_text( a.tmpl + "/quantization_config.json" ) : std::string();
	const auto qcfg = nlohmann::json::parse( qtext.empty() ? std::string( "{}" ) : qtext, nullptr, false );
	Exl3QuantOptions opts;
	opts.volta = a.volta;
	opts.sigma_reg = a.sigma_reg;
	opts.scaled_hessian = a.scaled;
	opts.ldl_metric = a.ldl_metric;
	opts.open_trellis = a.open_trellis;
	opts.tile_gain = a.tile_gain;
	if ( ( a.open_trellis || a.tile_gain ) && a.assemble ) {
		LOG_ERROR( "[volta_quantize] the engine's inference paths do not read tile starts / gains yet: --open-trellis / --tile-gain need --no-assemble" );
		return 2;
	}
	const std::string os = qcfg.is_discarded() ? "auto" : qcfg.value( "out_scales", "auto" );
	opts.out_scales = os == "always" ? 1 : os == "never" ? 0 : -1;
	// Tile orders: a stock output from a Volta template cannot mark what stays in the Volta order, and --only with an order other than the template's
	// would keep most of the model in the other one
	const bool tmpl_volta = !qcfg.is_discarded() && qcfg.value( "tile_layout", "" ) == "volta";
	std::vector<std::string> tmpl_stock;
	if ( tmpl_volta && qcfg.contains( "tile_layout_stock" ) ) {
		for ( const auto& v : qcfg["tile_layout_stock"] ) if ( v.is_string() ) tmpl_stock.push_back( v.get<std::string>() );
	}
	if ( tmpl_volta && !a.volta ) {
		LOG_ERROR( "[volta_quantize] the template is in the Volta tile order: --order stock cannot mark the tensors that stay as the template has them" );
		return 2;
	}
	if ( !a.only.empty() && a.volta != tmpl_volta ) {
		LOG_ERROR( "[volta_quantize] --only with a tile order other than the template's would leave the rest of the model in the other order: use --order %s",
			tmpl_volta ? "volta" : "stock" );
		return 2;
	}
	std::vector<std::string> stock_kept;

	// The plan: every template tensor with a trellis, its source and its Hessian
	Shared sh;
	sh.a = &a;
	sh.model = &model;
	std::map<std::string, int> counts;
	for ( const auto& kv : tmpl.weight_map() ) {
		if ( !ends_with( kv.first, ".trellis" ) ) continue;
		q::PlanTensor t;
		t.base = kv.first.substr( 0, kv.first.size() - 8 );
		if ( t.base.find( "ngram_embedding" ) != std::string::npos ) continue;
		bool skipped = !a.only.empty();
		for ( const auto& s : a.only ) skipped = skipped && t.base.find( s ) == std::string::npos;
		for ( const auto& s : a.skip ) skipped = skipped || t.base.find( s ) != std::string::npos;
		// A Volta template's stock-order exceptions stay as they are (a requantization would be in the Volta order)
		for ( const auto& s : tmpl_stock ) skipped = skipped || t.base.find( s ) != std::string::npos;
		if ( skipped ) {
			counts["kept from the template (--only / --skip)"]++;
			continue;
		}
		const q::StEntry* e = tmpl.find( kv.first );
		if ( !e || e->shape.size() != 3 ) continue;
		t.code = engine::modules::trellis_code_from_u16( static_cast<int>( e->shape[2] ) );
		t.cb = tmpl.has( t.base + ".mul1" ) ? 2 : tmpl.has( t.base + ".mcg" ) ? 1 : 0;
		if ( a.code ) {
			if ( a.code > engine::modules::TRELLIS_HALF_CODE && t.cb != 2 ) {
				LOG_ERROR( "[volta_quantize] %s: a half-integer --bits needs the mul1 codebook", t.base.c_str() );
				return 2;
			}
			t.code = a.code;
		}
		t.k = static_cast<int>( e->shape[0] ) * 16;
		t.n = static_cast<int>( e->shape[1] ) * 16;
		int so = 0, si = 0;
		if ( !src.shape( t.base, &so, &si ) ) {
			counts["kept from the template (no such tensor in the original)"]++;
			stock_kept.push_back( t.base );
			continue;
		}
		if ( so != t.n || si != t.k ) {
			LOG_WARN( "[volta_quantize] %s: %d x %d in the original, %d x %d in the template: kept from the template", t.base.c_str(), so, si, t.n,
				t.k );
			counts["kept from the template (shape differs)"]++;
			stock_kept.push_back( t.base );
			continue;
		}
		q::hessian_sources( model, a.hess, a.ranks, t );
		if ( t.down_layer >= 0 && a.route.empty() ) t.down_layer = t.down_expert = -1;
		counts[t.down_layer >= 0 ? "quantize, Hessian of its own activation (route dump)" : t.hess.empty() ? "quantize without a Hessian"
																												  : "quantize with a captured Hessian"]++;
		sh.plan.push_back( std::move( t ) );
	}
	std::sort( sh.plan.begin(), sh.plan.end(), []( const q::PlanTensor& x, const q::PlanTensor& y ) {
		return x.layer != y.layer ? x.layer < y.layer : x.base < y.base;
	} );
	for ( const auto& kv : counts ) LOG_INFO( "[volta_quantize] %7d  %s", kv.second, kv.first.c_str() );
	if ( a.plan ) return 0;

	if ( !a.route.empty() && !sh.route.open( a.route, model.hidden ) ) return 1;
	if ( a.aff_eps > 0.0f && ( a.route.empty() || !sh.route.has_weights() ) ) {
		LOG_ERROR( "[volta_quantize] --affinity-eps needs a route dump with its routing weights (<route dump>.w)" );
		return 2;
	}
	mkdir( a.out.c_str(), 0755 );
	mkdir( ( a.out + "/.partial" ).c_str(), 0755 );

	// Work items: tensors still to do, grouped by input (Hessian files, rate, codebook), in chunks of the memory budget (the group quantizer
	// holds about five fp32 copies of the columns it works on); a layer's expert down projections in chunks of 32
	std::map<std::string, Item> by_key;
	std::vector<std::string> key_order;
	for ( int i = 0; i < static_cast<int>( sh.plan.size() ); ++i ) {
		const auto& t = sh.plan[i];
		if ( !a.redo && exists( partial_path( a, t.base ) ) ) continue;
		std::string files;
		for ( const auto& h : t.hess ) files += h + ";";
		const bool aff = a.aff_eps > 0.0f && t.gate_up_expert >= 0 && !files.empty();
		const std::string rate = std::to_string( t.code ) + ":" + std::to_string( t.cb );
		const std::string key = t.down_layer >= 0 ? "down:" + std::to_string( t.down_layer ) + ":" + rate
			: aff                 ? "aff:" + files + rate
			: files.empty()       ? "none:" + t.base
								  : files + rate;
		if ( !by_key.count( key ) ) {
			key_order.push_back( key );
			Item it;
			it.expert_down = t.down_layer >= 0;
			it.expert_gate_up = aff;
			it.layer = t.down_layer >= 0 ? t.down_layer : t.layer;
			it.key = files;
			by_key[key] = it;
		}
		by_key[key].tensors.push_back( i );
	}
	for ( const auto& key : key_order ) {
		const Item& g = by_key[key];
		Item cur = g;
		cur.tensors.clear();
		size_t bytes = 0;
		for ( int ti : g.tensors ) {
			const auto& t = sh.plan[ti];
			// about five fp32 copies of its columns, and with scaled Hessians its own k x k factor
			const size_t tb = static_cast<size_t>( t.k ) * t.n * 4 * 5 + ( a.scaled ? static_cast<size_t>( t.k ) * t.k * 4 : 0 );
			// (affinity gate / up: 32 experts' pairs, which the plan's name order keeps together)
			const bool full = g.expert_down ? cur.tensors.size() >= 32 : g.expert_gate_up ? cur.tensors.size() >= 64
															: ( !cur.tensors.empty() && bytes + tb > a.budget );
			if ( full ) {
				sh.items.push_back( cur );
				cur.tensors.clear();
				cur.tiles = 0;
				bytes = 0;
			}
			cur.tensors.push_back( ti );
			cur.tiles += static_cast<int64_t>( t.k / 16 ) * ( t.n / 16 );
			bytes += tb;
		}
		if ( !cur.tensors.empty() ) sh.items.push_back( cur );
	}
	for ( const auto& it : sh.items ) sh.tiles_total += it.tiles;
	LOG_INFO( "[volta_quantize] %zu work items, %lld tiles, on %zu device(s)", sh.items.size(), static_cast<long long>( sh.tiles_total ),
		a.devices.size() );

	sh.t0 = std::chrono::steady_clock::now();
	std::vector<std::thread> threads;
	for ( int d : a.devices ) threads.emplace_back( worker, std::ref( sh ), d, opts );
	for ( auto& t : threads ) t.join();
	if ( sh.failures ) {
		LOG_ERROR( "[volta_quantize] %d work items failed: the shards are not written (a rerun continues with what is missing)", sh.failures.load() );
		return 1;
	}
	if ( !a.assemble ) {
		LOG_INFO( "[volta_quantize] quantized into %s/.partial (%.1f min), shards not written (--no-assemble)", a.out.c_str(),
			std::chrono::duration<double>( std::chrono::steady_clock::now() - sh.t0 ).count() / 60.0 );
		return 0;
	}
	int kept = 0;
	if ( !assemble( a, sh.plan, a.volta && !tmpl_volta ? stock_kept : std::vector<std::string>(), kept ) ) {
		LOG_ERROR( "[volta_quantize] writing the output failed" );
		return 1;
	}
	if ( kept ) LOG_WARN( "[volta_quantize] %d planned tensors had no result and kept the template's version", kept );
	LOG_INFO( "[volta_quantize] done: %s (%.1f min)", a.out.c_str(),
		std::chrono::duration<double>( std::chrono::steady_clock::now() - sh.t0 ).count() / 60.0 );
	return 0;
}
