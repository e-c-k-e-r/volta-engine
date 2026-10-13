// H3 vision tower (see h3_vision.h): the engine's Qwen3-VL vision tower on the BF16 checkpoint, with the DeepStack taps and FP32 host rows the language model stage wants.
#include "engine/video/h3_vision.h"
#include "engine/core/arena.h"
#include "engine/core/log.h"
#include "engine/loader/safetensors.h"
#include "engine/models/qwen3_vision.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace engine::video {

struct H3VisionTower::Impl {
	pod::MemoryArena arena{};
	cudaStream_t stream = nullptr;
	engine::models::Qwen3VisionTower tower;
	int max_patches = 0;
};

H3VisionTower::H3VisionTower() : p_( new Impl ) {}

H3VisionTower::~H3VisionTower() {
	if ( p_ ) {
		if ( p_->stream ) { (void) cudaStreamSynchronize( p_->stream ); }
		if ( p_->arena.base_ptr ) { engine::core::arena_destroy( p_->arena ); }
		if ( p_->stream ) { (void) cudaStreamDestroy( p_->stream ); }
		delete p_;
	}
}

int H3VisionTower::max_patches() const { return p_->max_patches; }

bool H3VisionTower::load( const char* dir, int device, int max_patches ) {
	Impl& s = *p_;
	s.max_patches = max_patches;
	CUDA_CHECK( cudaSetDevice( device ) );
	CUDA_CHECK( cudaStreamCreateWithFlags( &s.stream, cudaStreamNonBlocking ) );
	size_t weight_bytes = 0;
	{
		engine::loader::SafeTensorsRepo repo;
		if ( !repo.open( dir ) ) { return false; }
		weight_bytes = repo.bytes_with_prefix( "model.visual." );
	}
	// weights as FP16 plus the scratch of max_patches rows (about 40 KB per patch)
	if ( !engine::core::arena_create( s.arena, device, weight_bytes + static_cast<size_t>( max_patches ) * 48 * 1024 + ( 64ULL << 20 ) ) ) { return false; }
	if ( !s.tower.load( dir, s.arena, max_patches, s.stream, true ) ) { return false; }
	CUDA_CHECK( cudaStreamSynchronize( s.stream ) );
	return true;
}

bool H3VisionTower::encode( const std::vector<float>& patches, int gh, int gw, std::vector<float>* merged, std::vector<float>* deepstack ) {
	Impl& s = *p_;
	if ( gh * gw < 4 || gh * gw > s.max_patches ) { return false; }
	engine::loader::PreparedImage img;
	img.grid_h = gh;
	img.grid_w = gw;
	img.patches = patches;
	std::vector<half> rows;
	std::vector<std::vector<half>> taps;
	if ( !s.tower.encode( img, &rows, &taps, s.stream ) || taps.size() != 3 ) { return false; }
	auto widen = []( const std::vector<half>& h, std::vector<float>* dst ) {
		dst->resize( h.size() );
		for ( size_t i = 0; i < h.size(); ++i ) { ( *dst )[i] = __half2float( h[i] ); }
	};
	widen( rows, merged );
	for ( int d = 0; d < 3; ++d ) { widen( taps[d], &deepstack[d] ); }
	CUDA_CHECK( cudaGetLastError() );
	return true;
}

} // namespace engine::video
