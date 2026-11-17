// The GLM-5 or DeepSeek-V4.1 vision tower on the host CPU (vision_cpu) against the engine's own tower on the device backend, on one image:
//   test_host_vision <model dir> <image> [max_tokens (576)] [cpus (default plan)] [mmproj dir]
// Both encode the same prepared image; the rows are compared by cosine and the two encode times are printed. The host tower decodes the EXL3
// weights to FP16 once and multiplies densely, the device tower runs the fused kernels, so the rows agree up to FP16 rounding order. With an mmproj
// directory a second host tower takes the original BF16 weights from the vision projector GGUF: its rows are compared with the EXL3 tower's
// (they differ by the quantization of the tower, not by arithmetic).
#include "engine/models/glm5_vision.h"
#include "engine/models/deepseek_v41_vision.h"
#include "engine/models/qwen3_vision.h"
#include "engine/models/host_vision.h"
#include "engine/loader/image.h"
#include "engine/runtime/cpu_expert_attach.h"
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#if ENGINE_BACKEND_CUDA
#include "trellis_context.h"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace engine;

namespace {

double seconds( std::chrono::steady_clock::time_point t0 ) {
	return std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
}

} // namespace

int main( int argc, char** argv ) {
	if ( argc < 3 ) {
		std::fprintf( stderr, "usage: test_host_vision <model dir> <image> [max_tokens] [cpus] [mmproj dir]\n" );
		return 2;
	}
	const std::string model = argv[1], image = argv[2];
	const int max_tokens = argc > 3 ? std::atoi( argv[3] ) : 576;
	const std::vector<int> cpus = cpu_moe::host_worker_cpus( argc > 4 ? argv[4] : "" );
	const std::string mmproj = argc > 5 ? argv[5] : "";

	std::ifstream pf( image, std::ios::binary );
	const std::vector<uint8_t> png( ( std::istreambuf_iterator<char>( pf ) ), std::istreambuf_iterator<char>() );

	backend::set_device( 0 );
#if ENGINE_BACKEND_CUDA
	engine::modules::trellis_context_init_all();
#endif
	pod::MemoryArena arena;
	engine::core::arena_create( arena, 0, 3ULL << 30 );
	auto stream = backend::stream_create();
	// DeepSeek or GLM, by the checkpoint's architecture
	bool ds = false, qwen = false;
	{
		std::ifstream cf( model + "/config.json" );
		const std::string text( ( std::istreambuf_iterator<char>( cf ) ), std::istreambuf_iterator<char>() );
		ds = text.find( "DeepseekV41" ) != std::string::npos || text.find( "deepseek_v41" ) != std::string::npos;
		qwen = !ds && ( text.find( "Qwen" ) != std::string::npos || text.find( "qwen" ) != std::string::npos ) && text.find( "glm" ) == std::string::npos;
	}
	models::Glm5VisionTower glm_tower;
	models::DeepseekV41VisionTower ds_tower;
	models::Qwen3VisionTower qwen_tower;
	models::VisionTower& device = ds ? static_cast<models::VisionTower&>( ds_tower ) : qwen ? static_cast<models::VisionTower&>( qwen_tower ) : static_cast<models::VisionTower&>( glm_tower );
	const bool loaded = ds ? ds_tower.load( model.c_str(), arena, max_tokens, stream ) : qwen ? qwen_tower.load( model.c_str(), arena, max_tokens * 4, stream ) : glm_tower.load( model.c_str(), arena, max_tokens * 4, stream );
	if ( !loaded ) { std::printf( "test_host_vision: FAIL (device tower load)\n" ); return 1; }

	std::string err;
	auto host = models::make_host_vision_tower( model.c_str(), "", max_tokens, cpus, &err );
	if ( !host ) { std::printf( "test_host_vision: FAIL (host tower: %s)\n", err.c_str() ); return 1; }

	loader::DecodedImage dec;
	if ( !loader::decode_image( png.data(), png.size(), &dec, &err ) ) { std::printf( "test_host_vision: FAIL (decode: %s)\n", err.c_str() ); return 1; }
	loader::PreparedImage img;
	if ( !loader::prepare_image( dec, device.params(), &img, &err ) ) { std::printf( "test_host_vision: FAIL (prepare: %s)\n", err.c_str() ); return 1; }
	std::printf( "image %dx%d -> %dx%d patches (%d tokens)\n", dec.width, dec.height, img.grid_h, img.grid_w, ds ? img.span_tokens() : img.num_patches() / 4 );

	std::vector<backend::half_t> a, b, b2;
	auto t0 = std::chrono::steady_clock::now();
	if ( !device.encode( img, &a, stream ) ) { std::printf( "test_host_vision: FAIL (device encode)\n" ); return 1; }
	const double t_device = seconds( t0 );
	t0 = std::chrono::steady_clock::now();
	if ( !host->encode( img, &b, nullptr ) ) { std::printf( "test_host_vision: FAIL (host encode)\n" ); return 1; }
	const double t_host = seconds( t0 );
	t0 = std::chrono::steady_clock::now();
	if ( !host->encode( img, &b2, nullptr ) ) { std::printf( "test_host_vision: FAIL (host encode, second)\n" ); return 1; }
	const double t_host2 = seconds( t0 );
	if ( a.size() != b.size() || a.empty() ) { std::printf( "test_host_vision: FAIL (sizes %zu vs %zu)\n", a.size(), b.size() ); return 1; }

	const int hidden = ds ? 5120 : qwen ? qwen_tower.config().out_hidden : 4096, tokens = static_cast<int>( a.size() / hidden );
	// row cosines of two outputs: minimum, mean, largest |difference|
	auto compare = [&]( const std::vector<backend::half_t>& u, const std::vector<backend::half_t>& v, double* min_cos, double* mean_cos, double* max_abs ) {
		*min_cos = 1.0;
		double sum = 0.0;
		*max_abs = 0.0;
		for ( int t = 0; t < tokens; ++t ) {
			double dot = 0, na = 0, nb = 0;
			for ( int c = 0; c < hidden; ++c ) {
				const double x = backend::to_float( u[static_cast<size_t>( t ) * hidden + c] ), y = backend::to_float( v[static_cast<size_t>( t ) * hidden + c] );
				dot += x * y;
				na += x * x;
				nb += y * y;
				*max_abs = std::fmax( *max_abs, std::fabs( x - y ) );
			}
			const double cs = dot / std::sqrt( na * nb + 1e-30 );
			*min_cos = std::fmin( *min_cos, cs );
			sum += cs;
		}
		*mean_cos = sum / tokens;
	};
	// the rows two outputs agree least on: token (row, column) of the merged grid, the cosine and both row norms
	auto worst = [&]( const char* what, const std::vector<backend::half_t>& u, const std::vector<backend::half_t>& v ) {
		std::vector<std::pair<double, int>> cs;
		std::vector<double> nu( tokens ), nv( tokens );
		for ( int t = 0; t < tokens; ++t ) {
			double dot = 0, na = 0, nb = 0;
			for ( int c = 0; c < hidden; ++c ) {
				const double x = backend::to_float( u[static_cast<size_t>( t ) * hidden + c] ), y = backend::to_float( v[static_cast<size_t>( t ) * hidden + c] );
				dot += x * y;
				na += x * x;
				nb += y * y;
			}
			cs.push_back( { dot / std::sqrt( na * nb + 1e-30 ), t } );
			nu[t] = std::sqrt( na );
			nv[t] = std::sqrt( nb );
		}
		std::sort( cs.begin(), cs.end() );
		const int w = ds ? img.llm_w + 1 : img.grid_w / 2;   // tokens per row of the span (DeepSeek: with the newline)
		std::printf( "lowest rows, %s:", what );
		for ( int i = 0; i < 6 && i < tokens; ++i ) std::printf( "  (%d,%d) cos %.3f norms %.1f/%.1f", cs[i].second / w, cs[i].second % w, cs[i].first, nu[cs[i].second], nv[cs[i].second] );
		std::printf( "\n" );
	};
	double min_cos, mean_cos, max_abs;
	compare( a, b, &min_cos, &mean_cos, &max_abs );
	worst( "device vs host", a, b );
	const double sum_cos = mean_cos * tokens;
	const bool same = b.size() == b2.size() && std::memcmp( b.data(), b2.data(), b.size() * sizeof( backend::half_t ) ) == 0;
	std::printf( "%d rows of %d: cosine min %.5f mean %.6f, max |diff| %.4f; device %.2f s, host %.2f s (second run %.2f s), host runs %s\n", tokens, hidden, min_cos,
		sum_cos / tokens, max_abs, t_device, t_host, t_host2, same ? "identical" : "differ" );
	// The two towers decode the same quantized weights and differ in the FP16 arithmetic order through 24 layers: the mean row cosine is the measure (0.997 to
	// 0.999 on the test images; a few rows of a small norm, the blank canvas, drop to 0.6 to 0.9). A tower read in the wrong tile order is at 0
	bool ok = min_cos > 0.5 && sum_cos / tokens > 0.995 && same;
	if ( !mmproj.empty() ) {
		auto proj = models::make_host_vision_tower( model.c_str(), mmproj.c_str(), max_tokens, cpus, &err );
		if ( !proj ) { std::printf( "test_host_vision: FAIL (mmproj tower: %s)\n", err.c_str() ); return 1; }
		std::vector<backend::half_t> c;
		t0 = std::chrono::steady_clock::now();
		if ( !proj->encode( img, &c, nullptr ) || c.size() != a.size() ) { std::printf( "test_host_vision: FAIL (mmproj encode)\n" ); return 1; }
		const double t_proj = seconds( t0 );
		double mn, me, mx;
		compare( b, c, &mn, &me, &mx );
		std::printf( "mmproj (BF16) tower against the EXL3 host tower: cosine min %.5f mean %.5f, max |diff| %.4f; %.2f s\n", mn, me, mx, t_proj );
		worst( "host vs mmproj", b, c );
		worst( "device vs mmproj", a, c );
		compare( a, c, &mn, &me, &mx );
		std::printf( "mmproj (BF16) tower against the device EXL3 tower: cosine min %.5f mean %.5f, max |diff| %.4f\n", mn, me, mx );
		// the original weights against the quantized ones differ by quantization error; a tower read in the wrong tile order is at 0
		ok = ok && me > 0.985;
	}
	std::printf( "test_host_vision: %s\n", ok ? "PASS" : "FAIL" );
	return ok ? 0 : 1;
}
