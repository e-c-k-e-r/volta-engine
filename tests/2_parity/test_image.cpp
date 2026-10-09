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
