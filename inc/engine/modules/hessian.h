#pragma once

#include "engine/config.h"
#include "engine/backend/types.h"
#include "engine/backend/runtime.h"
#include "engine/core/arena.h"
#include "engine/core/batch.h"
#include "engine/core/log.h"
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace engine::modules {

// Calibration support: running sum of x^T x over the rows of a projection input (the Hessian of the layer-wise quantization loss).
struct Hessian {
	// h [k, k] fp32 += x^T x for x [rows, k] fp16 (fp32 accumulation)
	static void accumulate( float* h, const backend::half_t* x, int rows, int k, backend::stream_t stream );
};

// The accumulators of one model: `stride` sites per layer, each a k x k fp32 matrix in the arena, summed over the prefill rows that ran.
// One file per site at dump(): int32 magic, int32 k, int64 rows, then k * k fp32; named L<layer>_s<site>_r<rank>.f32 (what tests/tools/nvfp4_ldlq.py and
// exl3_quantize.py read). A model declares the sites that exist on its rank with add(); accumulate() ignores the others and the decode-sized rows.
class HessianSet {
public:
	bool on() const { return !sites_.empty(); }
	void enable( bool on ) { enabled_ = on; }
	void set_min_rows( int rows ) { min_rows_ = rows; }   // rows below this are decode batches and not counted (default: more than one decode batch); the drafter's blocks are 8 rows
	void init( int layers, int stride ) { stride_ = stride; sites_.assign( static_cast<size_t>( layers ) * stride, Site{} ); }
	void add( int layer, int site, int k, pod::MemoryArena& arena, backend::stream_t stream ) {
		Site& hs = sites_[static_cast<size_t>( layer ) * stride_ + site];
		hs.k = k;
		hs.h = static_cast<float*>( core::arena_alloc( arena, static_cast<size_t>( k ) * k * sizeof( float ), 128 ) );
		backend::memset_async( hs.h, 0, static_cast<size_t>( k ) * k * sizeof( float ), stream );
	}
	void accumulate( int layer, int site, const void* x, int k, int m, backend::stream_t stream ) {
		if ( sites_.empty() || !enabled_ || m < min_rows_ || layer < 0 ) return;
		Site& hs = sites_[static_cast<size_t>( layer ) * stride_ + site];
		if ( hs.h == nullptr ) return;
		ENGINE_ASSERT( hs.k == k, "HessianSet::accumulate: site width changed" );
		Hessian::accumulate( hs.h, static_cast<const backend::half_t*>( x ), m, k, stream );
		hs.rows += m;
	}
	void dump( const std::string& dir, int rank ) {
		if ( sites_.empty() ) return;
		backend::device_sync();
		std::vector<float> host;
		for ( size_t i = 0; i < sites_.size(); ++i ) {
			const Site& hs = sites_[i];
			if ( hs.h == nullptr || hs.rows == 0 ) continue;
			host.resize( static_cast<size_t>( hs.k ) * hs.k );
			backend::memcpy_sync( host.data(), hs.h, host.size() * sizeof( float ), backend::Copy::D2H );
			char path[512];
			std::snprintf( path, sizeof( path ), "%s/L%02d_s%d_r%d.f32", dir.c_str(), static_cast<int>( i / stride_ ), static_cast<int>( i % stride_ ), rank );
			if ( FILE* f = std::fopen( path, "wb" ) ) {
				const int32_t hdr[2] = { 0x48455353, hs.k };
				bool ok = std::fwrite( hdr, sizeof( int32_t ), 2, f ) == 2;
				ok = ok && std::fwrite( &hs.rows, sizeof( int64_t ), 1, f ) == 1;
				ok = ok && std::fwrite( host.data(), sizeof( float ), host.size(), f ) == host.size();
				ok = std::fclose( f ) == 0 && ok;
				if ( !ok ) LOG_ERROR( "[RANK %d] hessian: writing %s failed (disk full?)\n", rank, path );
			}
			else LOG_ERROR( "[RANK %d] hessian: cannot write %s\n", rank, path );
		}
		LOG_INFO( "[RANK %d] hessians written to %s\n", rank, dir.c_str() );
	}
private:
	struct Site { float* h = nullptr; int k = 0; int64_t rows = 0; };
	std::vector<Site> sites_;
	int stride_ = 1;
	bool enabled_ = true;
	int min_rows_ = pod::MAX_BATCH + 1;
};

} // namespace engine::modules
