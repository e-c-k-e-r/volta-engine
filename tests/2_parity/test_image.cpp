// Image preprocessing against PIL / HF-layout references (tests/tools/gen_image_goldens.py).
#include "engine/loader/image.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

#ifndef TEST_DATA_DIR
#error "TEST_DATA_DIR must point at tests/data"
#endif

using engine::loader::DecodedImage;
using engine::loader::PreparedImage;
using engine::loader::VisionParams;

static std::string read_file( const std::string& path ) {
	std::ifstream f( path, std::ios::binary );
	return std::string( std::istreambuf_iterator<char>( f ), std::istreambuf_iterator<char>() );
}

int main() {
	const std::string dir = TEST_DATA_DIR;
	const auto cases = nlohmann::json::parse( read_file( dir + "/image_cases.json" ) );
	int failures = 0;
	// GLM-5 canvas, expected values from the exllamav3 reference (glm5_vision_canvas; 16..1024 tokens of 28x28)
	const int canvas[][4] = { { 480, 640, 504, 644 }, { 12, 20, 112, 168 }, { 1200, 1600, 756, 1008 }, { 200, 640, 224, 644 }, { 3000, 500, 2184, 364 }, { 28, 28, 112, 112 } };
	for ( const auto& c : canvas ) {
		int h = 0, w = 0;
		engine::loader::glm_canvas( c[0], c[1], 28, 16 * 784, 1024 * 784, &h, &w );
		if ( h != c[2] || w != c[3] ) {
			std::printf( "FAIL glm_canvas %dx%d -> %dx%d, want %dx%d\n", c[0], c[1], h, w, c[2], c[3] );
			++failures;
		}
	}
	for ( const auto& c : cases ) {
		const std::string bytes = read_file( dir + "/" + c["file"].get<std::string>() );
		DecodedImage img;
		PreparedImage prep;
		std::string err;
		VisionParams params;
		if ( !engine::loader::decode_image( reinterpret_cast<const uint8_t*>( bytes.data() ), bytes.size(), &img, &err ) ||
		     !engine::loader::prepare_image( img, params, &prep, &err ) ) {
			std::printf( "FAIL %s: %s\n", c["file"].get<std::string>().c_str(), err.c_str() );
			++failures;
			continue;
		}
		bool ok = prep.grid_h == c["grid_h"].get<int>() && prep.grid_w == c["grid_w"].get<int>();
		const int dim = 1536, stride = c["stride"].get<int>();
		double worst = 0.0, mean = 0.0;
		const auto& sums = c["row_sums"];
		for ( size_t i = 0; ok && i < sums.size(); ++i ) {
			double s = 0.0;
			for ( int k = 0; k < dim; ++k ) s += prep.patches[( i * stride ) * dim + k];
			worst = std::fmax( worst, std::fabs( s - sums[i].get<double>() ) / dim );
		}
		for ( float v : prep.patches ) mean += v;
		mean /= prep.patches.size();
		const double mean_err = std::fabs( mean - c["mean"].get<double>() );
		ok = ok && worst < 0.02 && mean_err < 0.005;
		std::printf( "%s %s %dx%d -> grid %dx%d  row-mean diff %.4f  mean diff %.5f\n", ok ? "PASS" : "FAIL", c["file"].get<std::string>().c_str(),
			img.width, img.height, prep.grid_h, prep.grid_w, worst, mean_err );
		failures += ok ? 0 : 1;
	}
	std::printf( failures ? "test_image: FAIL\n" : "test_image: PASS\n" );
	return failures ? 1 : 0;
}
