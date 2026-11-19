#include "quant/exl3_read.h"
#include "engine/config.h"
#include "engine/modules/trellis.h"
#include "engine/modules/trellis_bits.h"
#include "reconstruct.cuh"
#include "trellis_context.h"
#include "exl3_vl.cuh"
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

namespace engine::tools::quant {

bool exl3_read( StRepo& repo, const std::string& base, Exl3Read& out ) {
	auto& t = out.t;
	const auto* tr = repo.find( base + ".trellis" );
	const auto* su = repo.find( base + ".suh" );
	const auto* sv = repo.find( base + ".svh" );
	if ( !tr || !su || !sv || tr->shape.size() != 3 ) return false;
	t.k = static_cast<int>( tr->shape[0] ) * 16;
	t.n = static_cast<int>( tr->shape[1] ) * 16;
	out.code = engine::modules::trellis_code_from_u16( static_cast<int>( tr->shape[2] ) );
	out.cb = repo.has( base + ".mul1" ) ? 2 : repo.has( base + ".mcg" ) ? 1 : 0;
	t.trellis.assign( reinterpret_cast<const uint16_t*>( tr->data ), reinterpret_cast<const uint16_t*>( tr->data ) + tr->bytes / 2 );
	t.suh.assign( reinterpret_cast<const uint16_t*>( su->data ), reinterpret_cast<const uint16_t*>( su->data ) + su->bytes / 2 );
	t.svh.assign( reinterpret_cast<const uint16_t*>( sv->data ), reinterpret_cast<const uint16_t*>( sv->data ) + sv->bytes / 2 );
	t.starts.clear();
	t.gains.clear();
	if ( const auto* e = repo.find( base + ".tstart" ) ) {
		t.starts.assign( reinterpret_cast<const uint16_t*>( e->data ), reinterpret_cast<const uint16_t*>( e->data ) + e->bytes / 2 );
	}
	if ( const auto* e = repo.find( base + ".tgain" ) ) t.gains.assign( e->data, e->data + e->bytes );
	out.volta = false;
	std::ifstream f( repo.dir() + "/quantization_config.json" );
	if ( !f ) {
		const auto it = repo.weight_map().find( base + ".trellis" );
		const auto* file = it == repo.weight_map().end() ? nullptr : repo.shard( it->second );
		if ( file ) {
			const auto& m = file->metadata();
			const auto lay = m.find( "tile_layout" ), cbk = m.find( "codebook" );
			out.volta = lay != m.end() && lay->second == "volta";
			if ( cbk != m.end() ) out.cb = cbk->second == "mul1" ? 2 : cbk->second == "mcg" ? 1 : 0;
		}
		return true;
	}
	std::stringstream ss;
	ss << f.rdbuf();
	const auto j = nlohmann::json::parse( ss.str(), nullptr, false );
	if ( !j.is_discarded() && j.value( "tile_layout", "" ) == "volta" ) {
		out.volta = true;
		if ( j.contains( "tile_layout_stock" ) ) {
			for ( const auto& s : j["tile_layout_stock"] ) {
				if ( s.is_string() && base.find( s.get<std::string>() ) != std::string::npos ) out.volta = false;
			}
		}
	}
	return true;
}

bool exl3_decode( const Exl3Read& r, float* out, bool unit_scales ) {
	const auto& t = r.t;
	uint16_t *d_trellis = nullptr, *d_suh = nullptr, *d_svh = nullptr;
	half* d_w = nullptr;
	CUDA_CHECK( cudaMalloc( &d_trellis, t.trellis.size() * 2 ) );
	CUDA_CHECK( cudaMalloc( &d_suh, t.k * 2 ) );
	CUDA_CHECK( cudaMalloc( &d_svh, t.n * 2 ) );
	CUDA_CHECK( cudaMalloc( &d_w, static_cast<size_t>( t.k ) * t.n * 2 ) );
	CUDA_CHECK( cudaMemcpy( d_trellis, t.trellis.data(), t.trellis.size() * 2, cudaMemcpyHostToDevice ) );
	if ( unit_scales ) {
		const half one = __float2half( 1.0f );
		std::vector<half> ones( std::max( t.k, t.n ), one );
		CUDA_CHECK( cudaMemcpy( d_suh, ones.data(), t.k * 2, cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( d_svh, ones.data(), t.n * 2, cudaMemcpyHostToDevice ) );
	} else {
		CUDA_CHECK( cudaMemcpy( d_suh, t.suh.data(), t.k * 2, cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( d_svh, t.svh.data(), t.n * 2, cudaMemcpyHostToDevice ) );
	}
	uint16_t* d_starts = nullptr;
	uint8_t* d_gains = nullptr;
	if ( !t.starts.empty() ) {
		CUDA_CHECK( cudaMalloc( &d_starts, t.starts.size() * 2 ) );
		CUDA_CHECK( cudaMemcpy( d_starts, t.starts.data(), t.starts.size() * 2, cudaMemcpyHostToDevice ) );
	}
	if ( !t.gains.empty() ) {
		CUDA_CHECK( cudaMalloc( &d_gains, t.gains.size() ) );
		CUDA_CHECK( cudaMemcpy( d_gains, t.gains.data(), t.gains.size(), cudaMemcpyHostToDevice ) );
	}
	EXL3_TILES_CALL( r.volta, reconstruct_had_slice_ext, d_w, d_trellis, d_starts, d_gains, reinterpret_cast<const half*>( d_suh ),
		reinterpret_cast<const half*>( d_svh ), r.code, r.cb == 1, r.cb == 2, t.k, t.n, 0, nullptr );
	if ( d_starts ) CUDA_CHECK( cudaFree( d_starts ) );
	if ( d_gains ) CUDA_CHECK( cudaFree( d_gains ) );
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

bool exl3_write_partial( const std::string& path, const std::string& base, const engine::quant::Exl3Tensor& t, int code, int cb, bool volta,
	const std::map<std::string, std::string>& extra ) {
	std::vector<StOut> outs( 3 );
	if ( !t.starts.empty() ) {
		StOut o;
		o.name = base + ".tstart";
		o.dtype = "U16";
		o.shape = { t.k / 16, t.n / 16 };
		o.owned.resize( t.starts.size() * 2 );
		std::memcpy( o.owned.data(), t.starts.data(), o.owned.size() );
		outs.push_back( std::move( o ) );
	}
	if ( !t.gains.empty() ) {
		StOut o;
		o.name = base + ".tgain";
		o.dtype = "U8";
		o.shape = { t.k / 16, t.n / 16 };
		o.owned.assign( t.gains.begin(), t.gains.end() );
		outs.push_back( std::move( o ) );
	}
	const int words = engine::modules::trellis_tile_u16( code );
	outs[0].name = base + ".trellis";
	outs[0].dtype = "I16";
	outs[0].shape = { t.k / 16, t.n / 16, words };
	outs[0].owned.resize( t.trellis.size() * 2 );
	std::memcpy( outs[0].owned.data(), t.trellis.data(), outs[0].owned.size() );
	outs[1].name = base + ".suh";
	outs[1].dtype = "F16";
	outs[1].shape = { t.k };
	outs[1].owned.resize( t.suh.size() * 2 );
	std::memcpy( outs[1].owned.data(), t.suh.data(), outs[1].owned.size() );
	outs[2].name = base + ".svh";
	outs[2].dtype = "F16";
	outs[2].shape = { t.n };
	outs[2].owned.resize( t.svh.size() * 2 );
	std::memcpy( outs[2].owned.data(), t.svh.data(), outs[2].owned.size() );
	std::map<std::string, std::string> meta = extra;
	meta["tile_layout"] = volta ? "volta" : "stock";
	meta["codebook"] = cb == 2 ? "mul1" : cb == 1 ? "mcg" : "3inst";
	return st_write( path, outs, meta );
}

} // namespace engine::tools::quant
