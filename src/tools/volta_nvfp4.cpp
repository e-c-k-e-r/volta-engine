// volta_nvfp4: calibrated NVFP4 dense projections (engine/quant/nvfp4_quantize.h: block LDLQ with the engine's input Hessians) of a modelopt
// NVFP4 checkpoint's BF16 tensors, for the nvfp4_dense_dir option: Flash-Next's (groups gdn, shared, attn, idx, head) or GLM's (kda, mla, idx,
// shared, head). The Hessians are an engine capture (test_ppl --calibrate with hessian_dir: L<layer>_s<site>_r<rank>.f32); a row-parallel input
// (o_proj, out_proj) takes the ranks' blocks on the diagonal. Tensors go to <out>/dense-L<first>-<last>.safetensors (eight layers a file, the
// loader's shard limit) and dense-head.safetensors in the modelopt layout (.weight, .weight_scale, .weight_scale_2), with an index. The
// tensors are spread over the devices; each prints its proxy error against round to nearest.
//
//   volta_nvfp4 --src <NVFP4 checkpoint> --hess <capture dir> --out <dir> [--model flash|glm] [--groups a,b] [--layers lo-hi] [--tp 4]
//               [--devices 0,1,2,3] [--rtn]
#include "quant/source.h"
#include "quant/st_io.h"
#include "engine/quant/nvfp4_quantize.h"
#include "engine/config.h"
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace q = engine::tools::quant;

namespace {

struct Args {
	std::string src, hess, out, model = "flash", groups, layers;
	int tp = 4;
	std::vector<int> devices = { 0 };
	bool rtn = false;
};

std::vector<std::string> split( const std::string& s, char sep ) {
	std::vector<std::string> out;
	std::stringstream ss( s );
	for ( std::string item; std::getline( ss, item, sep ); ) if ( !item.empty() ) out.push_back( item );
	return out;
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

// One work item: tensors of one input that share a global scale when `shared_scale` (the engine stacks them into one weight)
struct Item {
	std::vector<std::string> bases;
	std::string label;
	int layer = 0, site = 0;
	bool ranks = false;        // row-parallel input: the ranks' Hessian blocks on the diagonal
	bool shared_scale = false;
	std::string file;          // output shard
};

struct Shared {
	const Args* a = nullptr;
	std::vector<Item> items;
	std::atomic<size_t> next{ 0 };
	std::atomic<int> failures{ 0 };
	std::mutex mu;
	std::map<std::string, std::vector<q::StOut>> files;   // shard -> tensors
};

// The mean Hessian of an item's input (k x k): the rank-0 capture, or the ranks' blocks on the diagonal, each over its own rows
bool mean_hessian( const Args& a, const Item& it, int k, std::vector<float>& H ) {
	H.assign( static_cast<size_t>( k ) * k, 0.0f );
	const int parts = it.ranks ? a.tp : 1;
	int off = 0;
	std::vector<float> part;
	for ( int r = 0; r < parts; ++r ) {
		char name[64];
		std::snprintf( name, sizeof( name ), "/L%02d_s%d_r%d.f32", it.layer, it.site, r );
		int kr = 0;
		int64_t rows = 0;
		if ( !read_hessian( a.hess + name, kr, rows, part ) || rows <= 0 || off + kr > k ) return false;
		const float s = static_cast<float>( 1.0 / static_cast<double>( rows ) );
		for ( int i = 0; i < kr; ++i )
			for ( int j = 0; j < kr; ++j ) H[static_cast<size_t>( off + i ) * k + off + j] = part[static_cast<size_t>( i ) * kr + j] * s;
		off += kr;
	}
	return off == k;
}

q::StOut owned( const std::string& name, const char* dtype, std::vector<int64_t> shape, const void* data, size_t bytes ) {
	q::StOut o;
	o.name = name;
	o.dtype = dtype;
	o.shape = std::move( shape );
	o.owned.assign( static_cast<const uint8_t*>( data ), static_cast<const uint8_t*>( data ) + bytes );
	return o;
}

void worker( Shared& sh, int device ) {
	const Args& a = *sh.a;
	CUDA_CHECK( cudaSetDevice( device ) );
	cudaStream_t stream = nullptr;
	CUDA_CHECK( cudaStreamCreateWithFlags( &stream, cudaStreamNonBlocking ) );
	q::SourceWeights src;
	if ( !src.open( a.src ) ) {
		std::fprintf( stderr, "volta_nvfp4: cannot open %s\n", a.src.c_str() );
		sh.failures++;
		return;
	}
	for ( size_t i; ( i = sh.next++ ) < sh.items.size(); ) {
		const Item& it = sh.items[i];
		const auto t0 = std::chrono::steady_clock::now();
		std::vector<int> ns;
		int k = 0;
		bool ok = true;
		for ( const auto& b : it.bases ) {
			int n = 0, kin = 0;
			ok = ok && src.shape( b, &n, &kin );
			ns.push_back( n );
			k = kin;
		}
		std::vector<float> H;
		float* d_w = nullptr;
		float* d_h = nullptr;
		if ( ok && !a.rtn ) ok = mean_hessian( a, it, k, H );
		if ( ok && !H.empty() ) {
			CUDA_CHECK( cudaMalloc( &d_h, H.size() * sizeof( float ) ) );
			CUDA_CHECK( cudaMemcpyAsync( d_h, H.data(), H.size() * sizeof( float ), cudaMemcpyHostToDevice, stream ) );
		}
		// the group's global scale: the largest of its tensors'
		std::vector<float> gs( it.bases.size(), 0.0f );
		float group_gs = 0.0f;
		for ( size_t t = 0; ok && t < it.bases.size(); ++t ) {
			CUDA_CHECK( cudaMalloc( &d_w, static_cast<size_t>( k ) * ns[t] * sizeof( float ) ) );
			ok = src.load( it.bases[t], d_w, stream );
			if ( ok ) gs[t] = engine::quant::nvfp4_global_scale( d_w, static_cast<size_t>( k ) * ns[t], stream );
			group_gs = std::max( group_gs, gs[t] );
			CUDA_CHECK( cudaFree( d_w ) );
			d_w = nullptr;
		}
		std::vector<q::StOut> outs;
		std::string report;
		for ( size_t t = 0; ok && t < it.bases.size(); ++t ) {
			CUDA_CHECK( cudaMalloc( &d_w, static_cast<size_t>( k ) * ns[t] * sizeof( float ) ) );
			ok = src.load( it.bases[t], d_w, stream );
			engine::quant::Nvfp4Tensor qt;
			engine::quant::Nvfp4QuantOptions o;
			ok = ok && engine::quant::nvfp4_quantize( d_w, k, ns[t], d_h, d_h ? 1 : 0, it.shared_scale ? group_gs : gs[t], o, qt, stream );
			CUDA_CHECK( cudaFree( d_w ) );
			d_w = nullptr;
			if ( !ok ) break;
			const std::string& b = it.bases[t];
			outs.push_back( owned( b + ".weight", "U8", { qt.n, qt.k / 2 }, qt.codes.data(), qt.codes.size() ) );
			outs.push_back( owned( b + ".weight_scale", "F8_E4M3", { qt.n, qt.k / 16 }, qt.scales.data(), qt.scales.size() ) );
			outs.push_back( owned( b + ".weight_scale_2", "F32", {}, &qt.global_scale, sizeof( float ) ) );
			char line[160];
			std::snprintf( line, sizeof( line ), "%s%s proxy err rtn %.5f -> %s %.5f", report.empty() ? "" : ", ", it.bases.size() > 1 ? b.substr( b.rfind( '.' ) + 1 ).c_str() : "",
				qt.proxy_rtn, qt.fallback ? "rtn" : "ldlq", qt.proxy_err );
			report += line;
		}
		if ( d_h ) CUDA_CHECK( cudaFree( d_h ) );
		const double s = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
		std::lock_guard<std::mutex> lock( sh.mu );
		if ( !ok ) {
			std::fprintf( stderr, "volta_nvfp4: FAILED layer %d %s (shape, Hessian L%02d_s%d or device memory)\n", it.layer, it.label.c_str(), it.layer, it.site );
			sh.failures++;
			continue;
		}
		std::printf( "  [dev %d] layer %2d %-22s K %5d  %s  (%.1f s)\n", device, it.layer, it.label.c_str(), k, report.c_str(), s );
		auto& f = sh.files[it.file];
		for ( auto& o : outs ) f.push_back( std::move( o ) );
	}
	CUDA_CHECK( cudaStreamDestroy( stream ) );
}

} // namespace

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	Args a;
	for ( int i = 1; i < argc; ++i ) {
		const std::string f = argv[i];
		auto val = [&]() { return i + 1 < argc ? std::string( argv[++i] ) : std::string(); };
		if ( f == "--src" ) a.src = val();
		else if ( f == "--hess" ) a.hess = val();
		else if ( f == "--out" ) a.out = val();
		else if ( f == "--model" ) a.model = val();
		else if ( f == "--groups" ) a.groups = val();
		else if ( f == "--layers" ) a.layers = val();
		else if ( f == "--tp" ) a.tp = std::atoi( val().c_str() );
		else if ( f == "--rtn" ) a.rtn = true;
		else if ( f == "--devices" ) {
			a.devices.clear();
			for ( const auto& d : split( val(), ',' ) ) a.devices.push_back( std::atoi( d.c_str() ) );
		}
		else {
			std::fprintf( stderr, "volta_nvfp4: unknown argument %s\n", f.c_str() );
			return 2;
		}
	}
	const bool glm = a.model == "glm";
	if ( a.src.empty() || a.out.empty() || ( a.hess.empty() && !a.rtn ) || ( !glm && a.model != "flash" ) || a.devices.empty() ) {
		std::fprintf( stderr, "usage: volta_nvfp4 --src <NVFP4 checkpoint> --hess <capture dir> --out <dir> [--model flash|glm] [--groups a,b] [--layers lo-hi] [--tp 4] "
			"[--devices 0,1,2,3] [--rtn]\n" );
		return 2;
	}
	if ( a.groups.empty() ) a.groups = glm ? "kda,mla,idx,shared,head" : "gdn,shared,attn,idx,head";
	if ( a.layers.empty() ) a.layers = glm ? "0-44" : "0-47";
	const auto gl = split( a.groups, ',' );
	const std::set<std::string> groups( gl.begin(), gl.end() );
	int lo = 0, hi = 0;
	if ( std::sscanf( a.layers.c_str(), "%d-%d", &lo, &hi ) == 1 ) hi = lo;
	std::filesystem::create_directories( a.out );

	q::StRepo repo;
	if ( !repo.open( a.src ) ) {
		std::fprintf( stderr, "volta_nvfp4: cannot open %s\n", a.src.c_str() );
		return 1;
	}
	auto has = [&]( const std::string& base ) { return repo.has( base + ".weight" ); };
	Shared sh;
	sh.a = &a;
	const std::string P = "model.language_model";
	int first = lo;
	for ( int L = lo; L <= hi; ++L ) {
		// eight layers a file
		if ( ( L - lo ) % 8 == 0 ) first = L;
		char file[64];
		std::snprintf( file, sizeof( file ), "dense-L%02d-%02d.safetensors", first, std::min( first + 7, hi ) );
		const std::string pre = P + ".layers." + std::to_string( L );
		auto add = [&]( std::vector<std::string> bases, const char* label, int site, bool ranks = false, bool shared_scale = false ) {
			for ( const auto& b : bases ) if ( !has( b ) ) return;
			sh.items.push_back( Item{ bases, label, L, site, ranks, shared_scale, file } );
		};
		if ( glm ) {
			const std::string sa = pre + ".self_attn";
			if ( has( sa + ".q_a_proj" ) ) {
				if ( groups.count( "mla" ) ) {
					add( { sa + ".q_a_proj" }, "q_a_proj", 0 );
					add( { sa + ".kv_a_proj_with_mqa" }, "kv_a_proj_with_mqa", 0 );
					add( { sa + ".q_b_proj" }, "q_b_proj", 4 );
					add( { sa + ".o_proj" }, "o_proj (mla)", 2, true );
				}
				if ( groups.count( "idx" ) ) add( { sa + ".indexer.wq_b" }, "indexer wq_b", 4 );
			}
			else if ( groups.count( "kda" ) ) {
				add( { sa + ".q_proj", sa + ".k_proj", sa + ".v_proj" }, "q / k / v", 0, false, true );
				add( { sa + ".o_proj" }, "o_proj (kda)", 2, true );
			}
			if ( groups.count( "shared" ) && L >= 3 ) {
				add( { pre + ".mlp.shared_experts.gate_proj" }, "shared gate", 1 );
				add( { pre + ".mlp.shared_experts.up_proj" }, "shared up", 1 );
				add( { pre + ".mlp.shared_experts.down_proj" }, "shared down", 3 );
			}
		}
		else {
			if ( has( pre + ".self_attn.q_proj" ) ) {
				if ( groups.count( "attn" ) ) {
					for ( const char* n : { "q_proj", "k_proj", "v_proj" } ) add( { pre + ".self_attn." + n }, n, 0 );
					add( { pre + ".self_attn.o_proj" }, "o_proj", 2, true );
				}
				if ( groups.count( "idx" ) ) add( { pre + ".self_attn.indexer.index_qk_proj" }, "index_qk_proj", 0 );
			}
			else if ( groups.count( "gdn" ) ) {
				add( { pre + ".linear_attn.in_proj_qkv" }, "in_proj_qkv", 0 );
				add( { pre + ".linear_attn.in_proj_z" }, "in_proj_z", 0 );
				add( { pre + ".linear_attn.out_proj" }, "out_proj", 2, true );
			}
			if ( groups.count( "shared" ) ) {
				add( { pre + ".mlp.shared_expert.gate_proj" }, "shared gate", 1 );
				add( { pre + ".mlp.shared_expert.up_proj" }, "shared up", 1 );
				add( { pre + ".mlp.shared_expert.down_proj" }, "shared down", 3 );
			}
		}
	}
	if ( groups.count( "head" ) && lo == 0 && has( "lm_head" ) ) sh.items.push_back( Item{ { "lm_head" }, "lm_head", 0, glm ? 5 : 4, false, false, "dense-head.safetensors" } );
	std::printf( "volta_nvfp4: %zu items over %zu devices%s\n", sh.items.size(), a.devices.size(), a.rtn ? " (round to nearest)" : "" );

	std::vector<std::thread> threads;
	for ( int d : a.devices ) threads.emplace_back( worker, std::ref( sh ), d );
	for ( auto& t : threads ) t.join();
	if ( sh.failures ) {
		std::fprintf( stderr, "volta_nvfp4: %d items failed, nothing written\n", sh.failures.load() );
		return 1;
	}
	nlohmann::json wm = nlohmann::json::object();
	for ( auto& [file, tensors] : sh.files ) {
		for ( const auto& t : tensors ) wm[t.name] = file;
		if ( !q::st_write( a.out + "/" + file, tensors, {} ) ) {
			std::fprintf( stderr, "volta_nvfp4: cannot write %s\n", file.c_str() );
			return 1;
		}
	}
	std::ofstream( a.out + "/model.safetensors.index.json" ) << nlohmann::json{ { "metadata", nlohmann::json::object() }, { "weight_map", wm } }.dump() << "\n";
	std::printf( "volta_nvfp4: %zu files written to %s\n", sh.files.size(), a.out.c_str() );
	return 0;
}
