// Up-front validation of an H3 configuration (see h3_pipeline.h).
#include "engine/core/log.h"
#include "engine/loader/safetensors.h"
#include "engine/video/h3_pipeline.h"
#include <cuda_runtime.h>
#include <sys/stat.h>
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace engine::video {
namespace {

double file_gb( const std::string& path ) {
	struct stat st;
	return stat( path.c_str(), &st ) == 0 ? static_cast<double>( st.st_size ) / 1e9 : 0.0;
}

std::string fmt( const char* f, double a, double b = 0, double c = 0 ) {
	char buf[512];
	std::snprintf( buf, sizeof( buf ), f, a, b, c );
	return buf;
}

} // namespace

bool h3_check_paths( const H3Paths& paths, std::string* error, std::vector<std::string>* warnings ) {
	auto fail = [&]( const std::string& msg ) { if ( error ) { *error = msg; } return false; };
	auto warn = [&]( const std::string& msg ) { if ( warnings ) { warnings->push_back( msg ); } };
	std::vector<int> devices = paths.devices;
	if ( devices.empty() ) { devices.push_back( paths.device ); }
	int count = 0;
	if ( cudaGetDeviceCount( &count ) != cudaSuccess || count < 1 ) { return fail( "no CUDA device" ); }
	for ( size_t i = 0; i < devices.size(); ++i ) {
		if ( devices[i] < 0 || devices[i] >= count ) { return fail( "device " + std::to_string( devices[i] ) + " does not exist (this machine has " + std::to_string( count ) + " GPUs)" ); }
		for ( size_t j = 0; j < i; ++j ) { if ( devices[j] == devices[i] ) { return fail( "device " + std::to_string( devices[i] ) + " is listed twice" ); } }
	}
	if ( devices[0] != paths.device ) { return fail( "the first device must be the primary one (encoders and decoders run there)" ); }
	const int n = static_cast<int>( devices.size() );
	const bool tp = paths.tensor_parallel && n > 1;
	if ( paths.tensor_parallel && n == 1 ) { warn( "tensor_parallel with one GPU has nothing to shard; sequence parallelism (a single GPU) is used" ); }
	if ( tp ) {
		// the transformer's shards: 56 heads of 128; the ConvRot rotation works in 256-groups along K, and K is sharded in the attention output (7168) and the MLP down projection (14336)
		if ( 56 % n != 0 || ( 7168 / n ) % 256 != 0 || ( 14336 / n ) % 256 != 0 ) { return fail( std::to_string( n ) + " GPUs cannot shard the transformer (the heads and the 256-wide rotation groups must divide evenly: use 2, 4 or 7)" ); }
	}
	size_t free_b = 0, total_b = 0;
	CUDA_CHECK( cudaSetDevice( devices[0] ) );
	CUDA_CHECK( cudaMemGetInfo( &free_b, &total_b ) );
	const double total_gb = static_cast<double>( total_b ) / 1e9;
	const double dit_gb = std::max( file_gb( paths.dit ), file_gb( paths.dit_ref2va ) );
	if ( !paths.text_encoder.empty() ) {
		const std::string& dir = paths.text_encoder;
		engine::loader::SafeTensorsRepo repo;
		if ( !repo.open( dir.c_str() ) ) { return fail( "text_encoder " + dir + " is not a safetensors model directory" ); }
		const bool exl3 = repo.find( "model.language_model.layers.0.self_attn.q_proj.trellis" ) != nullptr;
		const bool fp4 = repo.find( "model.language_model.layers.0.self_attn.q_proj.weight_packed" ) != nullptr;
		if ( !exl3 && !fp4 ) { return fail( "text_encoder " + dir + " holds neither an EXL3 (trellis) nor an NVFP4 (weight_packed) conversion of the Qwen3-VL language model" ); }
		if ( !repo.find( "model.language_model.layers.49.mlp.down_proj.trellis" ) && !repo.find( "model.language_model.layers.49.mlp.down_proj.weight_packed" ) ) {
			return fail( "text_encoder " + dir + " has fewer than the 50 decoder layers the transformer is conditioned on" );
		}
		const pod::SafeTensorMeta* emb = repo.find( "model.language_model.embed_tokens.weight" );
		if ( !emb || emb->ndim != 2 || emb->shape[1] != 5120 ) { return fail( "text_encoder " + dir + " is not the 5120-wide Qwen3-VL-32B language model MiniMax-H3 uses" ); }
		if ( n > 1 && ( 8 % n != 0 ) ) { return fail( std::to_string( n ) + " GPUs cannot shard the text tower (8 key-value heads: use 2, 4 or 8)" ); }
		// the model's files are 64 layers, the tower runs 50; per GPU: its share of the quantized layers + activations
		double text_gb = 0;
		for ( int i = 0; i < 4; ++i ) {
			char name[64];
			std::snprintf( name, sizeof( name ), "/model-%05d-of-00004.safetensors", i + 1 );
			text_gb += file_gb( dir + name );
		}
		if ( text_gb == 0 ) { // other shard counts: the whole directory's tensors
			for ( int i = 0; i < 32; ++i ) {
				for ( int of = 1; of <= 32; ++of ) {
					char name[96];
					std::snprintf( name, sizeof( name ), "/model-%05d-of-%05d.safetensors", i + 1, of );
					text_gb += file_gb( dir + name );
				}
			}
		}
		const double tower_gb = text_gb * 50.0 / 64.0 / ( n > 1 ? n : 1 ) + 1.5; // 50 of 64 layers, sharded, plus the scratch for 8192 tokens
		const double dit_gpu_gb = ( tp ? dit_gb / n + 9.0 : dit_gb + 6.5 );       // weights + activation scratch (a resident transformer keeps both while denoising)
		if ( !tp && dit_gpu_gb + tower_gb > total_gb * 0.97 ) {
			return fail( fmt( "text_encoder needs tensor_parallel: next to the full-weight sequence-parallel transformer (about %.0f GB per GPU) the text tower (about %.0f GB) does not fit in %.0f GB. Set tensor_parallel (and 2, 4 or 8 devices)", dit_gpu_gb, tower_gb, total_gb ) );
		}
		if ( dit_gpu_gb + tower_gb + 5.8 > total_gb * 0.97 ) { warn( fmt( "the transformer (%.0f GB) and the text tower (%.0f GB) per GPU leave no room for resident decoders (5.8 GB) on a %.0f GB card; they load for each job", dit_gpu_gb, tower_gb, total_gb ) ); }
	}
	return true;
}

} // namespace engine::video
