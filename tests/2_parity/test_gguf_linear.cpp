// GGUF block types through Linear::forward against references made outside the engine (tests/tools/gguf_fixtures.py: blocks from ggml's own
// quantizer, gguf-py's dequantization). Every type for 1 to 8 rows (the q8_1 GEMV of decode and verify blocks), against the product with the
// activations rounded through q8_1 as the kernels do; and 20- and 64-row chunks (the FP16 dequantization and GEMM of prefill on CUDA, the iqk GEMM
// and its repacking prefill path on CPU), against the exact product.
//   test_gguf_linear <fixture dir>
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#include "engine/core/arena.h"
#include "engine/modules/linear.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <string>
#include <vector>

using namespace engine;
using backend::half_t;

namespace {

struct Fixture {
	int type = 0, n = 0, k = 0, m = 0;
	std::vector<uint8_t> blocks;
	std::vector<half_t> x;
	std::vector<float> exact, ref_q;
};

bool read_fixture( const std::string& path, Fixture* f ) {
	FILE* fp = std::fopen( path.c_str(), "rb" );
	if ( !fp ) return false;
	int32_t h[4];
	int64_t bytes = 0;
	bool ok = std::fread( h, sizeof( int32_t ), 4, fp ) == 4 && std::fread( &bytes, sizeof( int64_t ), 1, fp ) == 1;
	if ( ok ) {
		f->type = h[0]; f->n = h[1]; f->k = h[2]; f->m = h[3];
		f->blocks.resize( static_cast<size_t>( bytes ) );
		f->x.resize( static_cast<size_t>( f->m ) * f->k );
		f->exact.resize( static_cast<size_t>( f->m ) * f->n );
		f->ref_q.resize( f->exact.size() );
		ok = std::fread( f->blocks.data(), 1, f->blocks.size(), fp ) == f->blocks.size() && std::fread( f->x.data(), sizeof( half_t ), f->x.size(), fp ) == f->x.size() &&
			std::fread( f->exact.data(), sizeof( float ), f->exact.size(), fp ) == f->exact.size() && std::fread( f->ref_q.data(), sizeof( float ), f->ref_q.size(), fp ) == f->ref_q.size();
	}
	std::fclose( fp );
	return ok;
}

// rms of (got - ref) over rms of ref, the first `rows` rows
double rel_rms( const std::vector<float>& got, const std::vector<float>& ref, int rows, int n ) {
	double e = 0.0, r = 0.0;
	for ( size_t i = 0; i < static_cast<size_t>( rows ) * n; ++i ) {
		e += ( got[i] - ref[i] ) * ( got[i] - ref[i] );
		r += static_cast<double>( ref[i] ) * ref[i];
	}
	return std::sqrt( e / std::max( r, 1e-30 ) );
}

}   // namespace

int main( int argc, char** argv ) {
	if ( argc < 2 ) {
		std::printf( "usage: test_gguf_linear <fixture dir>   (tests/tools/gguf_fixtures.py writes it)\n" );
		return 2;
	}
	std::printf( "=== GGUF block types through Linear::forward ===\n" );
	backend::set_device( 0 );
	std::vector<std::string> files;
	if ( DIR* d = opendir( argv[1] ) ) {
		while ( dirent* e = readdir( d ) ) {
			const std::string name = e->d_name;
			if ( name.size() > 4 && name.compare( name.size() - 4, 4, ".bin" ) == 0 ) files.push_back( std::string( argv[1] ) + "/" + name );
		}
		closedir( d );
	}
	std::sort( files.begin(), files.end() );
	if ( files.empty() ) {
		std::printf( "[-] no fixtures in %s\n", argv[1] );
		return 1;
	}
	pod::MemoryArena arena{};
	ENGINE_ASSERT( core::arena_create( arena, 0, size_t( 256 ) << 20 ), "arena" );
	int fails = 0;
	for ( const std::string& path : files ) {
		Fixture f;
		ENGINE_ASSERT( read_fixture( path, &f ), "fixture unreadable" );
		const auto type = static_cast<modules::GgmlType>( f.type );
		if ( !modules::gguf_linear_supported( type ) ) {
			std::printf( "  %-8s not supported, skipped\n", modules::ggml_type_name( type ) );
			continue;
		}
		const size_t mark = arena.offset_bytes;
		int bdims[2] = { f.n, static_cast<int>( f.blocks.size() / f.n ) };
		pod::TensorView w = core::arena_alloc_tensor( arena, pod::DType::INT8, bdims, 2 );
		backend::memcpy_sync( w.data, f.blocks.data(), f.blocks.size(), backend::Copy::H2D );
		const pod::LinearWeight lw = pod::LinearWeight::make_gguf( w, static_cast<uint32_t>( f.type ), f.k, f.n );
		double worst_gemv = 0.0, worst_tile = 0.0, worst_gemm = 0.0;
		std::vector<half_t> one_row;   // the 1-row product, which the first row of 2 to 8 rows must equal bit for bit on CPU
		int row_mismatch = 0;
		for ( int rows : { 1, 2, 3, 5, 8, 9, 13, 16, 20, 64 } ) {
			if ( rows > f.m ) continue;
			int xdims[2] = { rows, f.k }, odims[2] = { rows, f.n };
			pod::TensorView x = core::arena_alloc_tensor( arena, pod::DType::FP16, xdims, 2 );
			pod::TensorView out = core::arena_alloc_tensor( arena, pod::DType::FP16, odims, 2 );
			backend::memcpy_sync( x.data, f.x.data(), static_cast<size_t>( rows ) * f.k * sizeof( half_t ), backend::Copy::H2D );
			modules::Linear::forward( x, lw, out );
			backend::device_sync();
			std::vector<half_t> oh( static_cast<size_t>( rows ) * f.n );
			backend::memcpy_sync( oh.data(), out.data, oh.size() * sizeof( half_t ), backend::Copy::D2H );
			std::vector<float> o( oh.size() );
			for ( size_t i = 0; i < o.size(); ++i ) o[i] = backend::to_float( oh[i] );
			if ( rows == 1 ) one_row = oh;
			else if ( rows <= 8 ) row_mismatch += std::memcmp( one_row.data(), oh.data(), one_row.size() * sizeof( half_t ) ) != 0;
			if ( rows <= 8 ) worst_gemv = std::max( worst_gemv, rel_rms( o, f.ref_q, rows, f.n ) );
			else if ( rows <= 16 ) worst_tile = std::max( worst_tile, rel_rms( o, f.exact, rows, f.n ) );
			else worst_gemm = std::max( worst_gemm, rel_rms( o, f.exact, rows, f.n ) );
		}
		// The GEMV: FP16 output rounding stays well under 1e-2 against the q8_1 reference. The types whose offset or minimum comes from the q8_1
		// block's half-precision sum of the activations (llama.cpp's formulation: Q4_0, Q4_1, Q5_0, Q5_1, and IQ1_S / IQ1_M for their delta) lose
		// more, because that term nearly cancels the main dot product: a numpy emulation of the same formula gives the same 1.3% on Q4_0, so
		// their bound is 3e-2. The tile kernel and the GEMM read FP16 activations and the exactly dequantized weight: FP16 rounding only
		const bool sum_offset = type == modules::GgmlType::Q4_0 || type == modules::GgmlType::Q4_1 || type == modules::GgmlType::Q5_0 ||
			type == modules::GgmlType::Q5_1 || type == modules::GgmlType::IQ1_S || type == modules::GgmlType::IQ1_M;
		// The CPU backend (llama.cpp's CPU kernels) quantizes the activations at every row count (q8_0, q8_1 or q8_K by type): there the 9-16 and
		// 20-row products meet the exact one at the GEMV's bound; from 32 rows ik_llama.cpp's iqk GEMM also requantizes each weight tile to
		// int8 (its Q8_K_R8 / Q8_0_R8 repack), a little more: 1.5e-2
#if defined(ENGINE_BACKEND_CPU)
		const double full_bound = sum_offset ? 3e-2 : 1e-2;
		const double gemm_bound = sum_offset ? 3e-2 : 1.5e-2;
#elif defined(ENGINE_BACKEND_VULKAN)
		// Vulkan: from 9 rows the activations are quantized to q8 for the int8 dot products and llama.cpp's int8 matmul (on a device that has
		// them), the GEMV's bound (the types without an int8 matmul stay FP16 from 17 rows, well inside it)
		const double full_bound = sum_offset ? 3e-2 : 1e-2;
		const double gemm_bound = full_bound;
#else
		const double full_bound = 5e-3, gemm_bound = full_bound;
#endif
		// On CPU a speculative verify (2 to 8 rows) must give a row exactly what the 1-row decode gives it
#if defined(ENGINE_BACKEND_CPU)
		const bool rows_agree = row_mismatch == 0;
#else
		const bool rows_agree = true;
#endif
		if ( !rows_agree ) std::printf( "  %-8s row 0 of %d of the 2-8 row products differs from the 1-row product\n", modules::ggml_type_name( type ), row_mismatch );
		const bool ok = worst_gemv < ( sum_offset ? 3e-2 : 1e-2 ) && worst_tile < full_bound && worst_gemm < gemm_bound && rows_agree;
		std::printf( "  %-8s n %d k %d: 1-8 rows vs q8_1 reference %.2e, 9-16 rows vs exact %.2e, 20 / 64 rows vs exact %.2e  %s\n",
			modules::ggml_type_name( type ), f.n, f.k, worst_gemv, worst_tile, worst_gemm, ok ? "PASS" : "FAIL" );
		fails += ok ? 0 : 1;
		arena.offset_bytes = mark;
	}
	std::printf( "%s\n", fails ? "[-] GGUF linear FAIL" : "[+] GGUF linear PASS" );
	return fails ? 1 : 0;
}
