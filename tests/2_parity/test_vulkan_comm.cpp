// Vulkan multi-device collective (src/backend/vulkan/mod_comm.cpp): one thread per rank, each on its own Vulkan device, runs CollectiveRing::all_reduce
// and all_reduce_rms_norm on seeded FP16 data; the result is compared with an FP32 sum over the ranks (all_reduce: exact, one FP16 rounding) and
// with a host RMSNorm of residual + sum (tolerance: the norm rounds in FP16). Skips (exit 0) with fewer devices than ranks.
//   test_vulkan_comm [ranks ...]   (default 2 4)
#include "engine/backend/numeric.h"
#include "engine/backend/runtime.h"
#include "engine/core/comm.h"
#include "engine/core/tensor.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

using namespace engine;
using backend::half_t;

namespace {

struct Rng {
	uint32_t s;
	float next( float scale ) {
		s = s * 1664525u + 1013904223u;
		return ( static_cast<int>( ( s >> 8 ) % 20001u ) - 10000 ) / 10000.0f * scale;
	}
};

half_t h( float f ) { return backend::from_float_half( f ); }
float f( half_t v ) { return backend::to_float( v ); }

int run( int ranks, int rows, int hidden, bool rms ) {
	comm::CollectiveRing rings[ENGINE_NUM_GPUS];
	void* staging[ENGINE_NUM_GPUS];
	void* flags[ENGINE_NUM_GPUS];
	comm::init_4gpu_mesh( rings, staging, flags, comm::kMeshStagingBytes, ranks );
	const size_t n = static_cast<size_t>( rows ) * hidden;
	std::vector<std::vector<half_t>> part( ranks, std::vector<half_t>( n ) ), residual( n ? 1 : 0, std::vector<half_t>( n ) );
	std::vector<half_t> weight( hidden );
	Rng rng{ 77u + static_cast<uint32_t>( rows * 31 + hidden ) };
	for ( int r = 0; r < ranks; ++r ) for ( auto& v : part[r] ) v = h( rng.next( 2.0f ) );
	for ( auto& v : residual[0] ) v = h( rng.next( 4.0f ) );
	for ( auto& v : weight ) v = h( rng.next( 0.5f ) );
	std::vector<std::vector<half_t>> got( ranks, std::vector<half_t>( n ) ), got_res( ranks, std::vector<half_t>( n ) );
	std::vector<std::thread> th;
	for ( int r = 0; r < ranks; ++r ) {
		th.emplace_back( [&, r]() {
			backend::set_device( r );
			half_t* dx = static_cast<half_t*>( backend::device_malloc( n * sizeof( half_t ) ) );
			half_t* dres = static_cast<half_t*>( backend::device_malloc( n * sizeof( half_t ) ) );
			half_t* dout = static_cast<half_t*>( backend::device_malloc( n * sizeof( half_t ) ) );
			half_t* dw = static_cast<half_t*>( backend::device_malloc( hidden * sizeof( half_t ) ) );
			backend::memcpy_sync( dx, part[r].data(), n * sizeof( half_t ), backend::Copy::H2D );
			backend::memcpy_sync( dres, residual[0].data(), n * sizeof( half_t ), backend::Copy::H2D );
			backend::memcpy_sync( dw, weight.data(), hidden * sizeof( half_t ), backend::Copy::H2D );
			pod::TensorView x = pod::make_tensor_2d( dx, pod::DType::FP16, rows, hidden ), res = pod::make_tensor_2d( dres, pod::DType::FP16, rows, hidden );
			pod::TensorView out = pod::make_tensor_2d( dout, pod::DType::FP16, rows, hidden ), w = pod::make_tensor_2d( dw, pod::DType::FP16, 1, hidden );
			if ( rms ) rings[r].all_reduce_rms_norm( x, res, w, 1e-6f, nullptr, 1.0f, &out );
			else rings[r].all_reduce( x, nullptr );
			backend::stream_sync( nullptr );
			backend::memcpy_sync( got[r].data(), rms ? dout : dx, n * sizeof( half_t ), backend::Copy::D2H );
			backend::memcpy_sync( got_res[r].data(), dres, n * sizeof( half_t ), backend::Copy::D2H );
			backend::device_free( dx );
			backend::device_free( dres );
			backend::device_free( dout );
			backend::device_free( dw );
		} );
	}
	for ( auto& t : th ) t.join();
	comm::destroy_4gpu_mesh( staging, flags );

	int bad = 0;
	double worst = 0.0;
	for ( int row = 0; row < rows; ++row ) {
		std::vector<float> summed( hidden ), res_new( hidden );
		double ss = 0.0;
		for ( int c = 0; c < hidden; ++c ) {
			const size_t i = static_cast<size_t>( row ) * hidden + c;
			float sum = f( part[0][i] );
			for ( int r = 1; r < ranks; ++r ) sum += f( part[r][i] );
			summed[c] = f( h( sum ) );
			res_new[c] = f( h( sum + f( residual[0][i] ) ) );
			ss += static_cast<double>( res_new[c] ) * res_new[c];
		}
		const double inv = 1.0 / std::sqrt( ss / hidden + 1e-6 );
		for ( int r = 0; r < ranks; ++r ) {
			for ( int c = 0; c < hidden; ++c ) {
				const size_t i = static_cast<size_t>( row ) * hidden + c;
				if ( !rms ) {
					if ( f( got[r][i] ) != summed[c] ) ++bad;
				}
				else {
					const double want = res_new[c] * ( f( weight[c] ) + 1.0 ) * inv;
					const double d = std::fabs( f( got[r][i] ) - want );
					worst = std::fmax( worst, d );
					if ( d > 0.02 + 0.01 * std::fabs( want ) || f( got_res[r][i] ) != res_new[c] ) ++bad;
				}
			}
		}
	}
	std::printf( "  %d ranks, %d x %d, %s: %s%s\n", ranks, rows, hidden, rms ? "all_reduce_rms_norm" : "all_reduce", bad ? "FAIL (" : "ok", bad ? std::to_string( bad ).append( " elements)" ).c_str() : "" );
	if ( rms ) std::printf( "    max |norm - host| %.4g\n", worst );
	return bad ? 1 : 0;
}

} // namespace

int main( int argc, char** argv ) {
	std::vector<int> rank_list;
	for ( int i = 1; i < argc; ++i ) rank_list.push_back( std::atoi( argv[i] ) );
	if ( rank_list.empty() ) rank_list = { 2, 4 };
	backend::set_device( 0 );
	const int have = backend::device_count();
	int fails = 0;
	for ( int ranks : rank_list ) {
		if ( ranks > have ) {
			std::printf( "  %d ranks: only %d Vulkan devices, skipped\n", ranks, have );
			continue;
		}
		for ( int rows : { 1, 7, 64 } ) {
			fails += run( ranks, rows, 5120, false );
			fails += run( ranks, rows, 5120, true );
		}
	}
	std::printf( "test_vulkan_comm: %s\n", fails ? "FAILED" : "PASSED" );
	return fails ? 1 : 0;
}
