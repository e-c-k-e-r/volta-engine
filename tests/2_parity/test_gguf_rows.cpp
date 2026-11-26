// A GGUF checkpoint's own projections through Linear::forward on random activations at the row counts of each path (the 1-8 row GEMV with
// q8_1 activations, the 9-16 row kernels, the 64-row dequantize-and-GEMM): row 0's relative error against the product, in double, of the
// weight as LinearGguf::dequant decodes it with the FP16 activations. Without fixtures, for the shapes of a real model; fails when a
// GEMV row is off by more than the q8_1 bound of the offset types (3e-2).
//   test_gguf_rows <model dir> <tensor base name> [more names...]
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#include "engine/core/arena.h"
#include "engine/loader/safetensors.h"
#include "engine/modules/gguf.h"
#include "engine/modules/linear.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace engine;
using backend::half_t;

int main( int argc, char** argv ) {
	if ( argc < 3 ) {
		std::printf( "usage: test_gguf_rows <model dir> <tensor base name> [...]\n" );
		return 2;
	}
	backend::set_device( 0 );
	loader::SafeTensorsRepo repo;
	ENGINE_ASSERT( repo.open( argv[1] ), "model directory unreadable" );
	pod::MemoryArena arena{};
	ENGINE_ASSERT( core::arena_create( arena, 0, size_t( 2 ) << 30 ), "arena" );
	std::mt19937 rng( 7 );
	std::normal_distribution<float> nd( 0.0f, 1.0f );
	int fails = 0;
	for ( int a = 2; a < argc; ++a ) {
		const std::string base = argv[a];
		const pod::SafeTensorMeta* meta = repo.find( ( base + ".weight" ).c_str() );
		if ( meta == nullptr || meta->dtype != pod::DType::GGUF ) {
			std::printf( "  %s: not a GGUF block tensor, skipped\n", base.c_str() );
			continue;
		}
		const int n = meta->shape[0], k = meta->shape[1];
		const size_t mark = arena.offset_bytes;
		const pod::LinearWeight lw = repo.load_linear( base.c_str(), arena, 4 );
		// the reference: the weight dequantized (FP16) times the activations in FP32
		std::vector<half_t> wf( static_cast<size_t>( n ) * k );
		{
			int dims[2] = { n, static_cast<int>( meta->size_bytes / n ) };
			pod::TensorView raw = core::arena_alloc_tensor( arena, pod::DType::INT8, dims, 2 );
			backend::memcpy_sync( raw.data, repo.cpu_data( ( base + ".weight" ).c_str() ), meta->size_bytes, backend::Copy::H2D );
			int fd[2] = { n, k };
			pod::TensorView w16 = core::arena_alloc_tensor( arena, pod::DType::FP16, fd, 2 );
			modules::LinearGguf::dequant( raw.data, static_cast<modules::GgmlType>( meta->gguf_type ), static_cast<int64_t>( n ) * k, w16.ptr<half_t>() );
			backend::device_sync();
			backend::memcpy_sync( wf.data(), w16.data, wf.size() * sizeof( half_t ), backend::Copy::D2H );
		}
		constexpr int kMaxRows = 64;
		std::vector<half_t> x( static_cast<size_t>( kMaxRows ) * k );
		for ( auto& v : x ) v = backend::from_float_half( nd( rng ) );
		std::vector<double> ref( n );
		for ( int c = 0; c < n; ++c ) {
			double s = 0.0;
			for ( int i = 0; i < k; ++i ) s += static_cast<double>( backend::to_float( wf[static_cast<size_t>( c ) * k + i] ) ) * backend::to_float( x[i] );
			ref[c] = s;
		}
		std::printf( "  %-56s %-6s n %6d k %6d:", base.c_str(), modules::ggml_type_name( static_cast<modules::GgmlType>( meta->gguf_type ) ), n, k );
		double worst = 0.0;
		for ( int rows : { 1, 2, 8, 9, 16, 64 } ) {
			int xd[2] = { rows, k }, od[2] = { rows, n };
			pod::TensorView xt = core::arena_alloc_tensor( arena, pod::DType::FP16, xd, 2 );
			pod::TensorView out = core::arena_alloc_tensor( arena, pod::DType::FP16, od, 2 );
			backend::memcpy_sync( xt.data, x.data(), static_cast<size_t>( rows ) * k * sizeof( half_t ), backend::Copy::H2D );
			modules::Linear::forward( xt, lw, out );
			backend::device_sync();
			std::vector<half_t> o( n );
			backend::memcpy_sync( o.data(), out.data, n * sizeof( half_t ), backend::Copy::D2H );
			double e = 0.0, r = 0.0;
			for ( int c = 0; c < n; ++c ) {
				const double d = backend::to_float( o[c] ) - ref[c];
				e += d * d;
				r += ref[c] * ref[c];
			}
			const double rel = std::sqrt( e / std::max( r, 1e-30 ) );
			if ( rows <= 8 ) worst = std::max( worst, rel );
			std::printf( " %d:%.2e", rows, rel );
		}
		std::printf( "\n" );
		fails += worst > 3e-2;
		arena.offset_bytes = mark;
	}
	std::printf( "[%s] test_gguf_rows\n", fails ? "FAIL" : "PASS" );
	return fails ? 1 : 0;
}
