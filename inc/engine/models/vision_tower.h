#pragma once

#include "engine/core/arena.h"
#include "engine/core/tensor.h"
#include "engine/core/log.h"
#include "engine/loader/image.h"
#include "engine/loader/safetensors.h"
#include "engine/backend/types.h"
#include "engine/backend/numeric.h"
#include <nlohmann/json.hpp>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace engine::models {

// An image encoder that turns a prepared image into language-model embedding rows.
class VisionTower {
public:
	virtual ~VisionTower() = default;
	virtual bool loaded() const = 0;
	virtual const engine::loader::VisionParams& params() const = 0;
	// Encodes one prepared image into [tokens, hidden] FP16 rows (host). Blocks until done.
	virtual bool encode( const engine::loader::PreparedImage& img, std::vector<backend::half_t>* out, backend::stream_t stream ) = 0;
};

// Loading helpers the towers share: an FP16 matrix view, a typed arena allocation, and a JSON file that parses to a discarded value on error
inline pod::TensorView view2d( void* p, int rows, int cols ) {
	return pod::make_tensor_2d( p, pod::DType::FP16, rows, cols );
}

template <typename T>
T* take( pod::MemoryArena& arena, size_t count ) {
	return static_cast<T*>( engine::core::arena_alloc( arena, count * sizeof( T ) ) );
}

inline nlohmann::json read_json( const std::string& path ) {
	std::ifstream f( path );
	if ( !f ) return nlohmann::json::value_t::discarded;
	return nlohmann::json::parse( std::string( std::istreambuf_iterator<char>( f ), std::istreambuf_iterator<char>() ), nullptr, false );
}

// A clipped projection's ranges (Gemma 4's Gemma4ClippableLinear): the input and the output clamped, each side on when the checkpoint gives it
struct ClipRange {
	float in_lo = -3.0e38f, in_hi = 3.0e38f, out_lo = -3.0e38f, out_hi = 3.0e38f;
	bool in() const { return in_lo > -3.0e38f || in_hi < 3.0e38f; }
	bool out() const { return out_lo > -3.0e38f || out_hi < 3.0e38f; }
	bool same_input( const ClipRange& o ) const { return in_lo == o.in_lo && in_hi == o.in_hi; }
};

// A scalar tensor of the repo (F32, BF16 or FP16), fallback when it is missing
inline float repo_scalar( const engine::loader::SafeTensorsRepo& repo, const std::string& name, float fallback ) {
	const pod::SafeTensorMeta* m = repo.find( name.c_str() );
	const void* p = m ? repo.cpu_data( name.c_str() ) : nullptr;
	if ( !p ) return fallback;
	if ( m->dtype == pod::DType::FP32 ) {
		float v;
		std::memcpy( &v, p, 4 );
		return v;
	}
	uint16_t h;
	std::memcpy( &h, p, 2 );
	if ( m->dtype == pod::DType::BF16 ) {
		const uint32_t u = static_cast<uint32_t>( h ) << 16;
		float v;
		std::memcpy( &v, &u, 4 );
		return v;
	}
	return m->dtype == pod::DType::FP16 ? backend::to_float( *reinterpret_cast<const backend::half_t*>( p ) ) : fallback;
}

// A tensor of the repo (F32, BF16 or FP16) as FP32 on the host; false when it is missing or of another type
inline bool repo_host_floats( const engine::loader::SafeTensorsRepo& repo, const std::string& name, std::vector<float>* out ) {
	const pod::SafeTensorMeta* m = repo.find( name.c_str() );
	const void* p = m ? repo.cpu_data( name.c_str() ) : nullptr;
	if ( !p ) return false;
	size_t n = 1;
	for ( int d = 0; d < m->ndim; ++d ) n *= static_cast<size_t>( m->shape[d] );
	out->resize( n );
	if ( m->dtype == pod::DType::FP32 ) {
		std::memcpy( out->data(), p, n * 4 );
		return true;
	}
	if ( m->dtype != pod::DType::BF16 && m->dtype != pod::DType::FP16 ) return false;
	const auto* h = static_cast<const uint16_t*>( p );
	for ( size_t i = 0; i < n; ++i ) {
		if ( m->dtype == pod::DType::BF16 ) {
			const uint32_t u = static_cast<uint32_t>( h[i] ) << 16;
			std::memcpy( &( *out )[i], &u, 4 );
		}
		else {
			( *out )[i] = backend::to_float( *reinterpret_cast<const backend::half_t*>( &h[i] ) );
		}
	}
	return true;
}

// The ranges of the clipped projection under `base` (its .input_min / .input_max / .output_min / .output_max)
inline ClipRange repo_clip( const engine::loader::SafeTensorsRepo& repo, const std::string& base ) {
	ClipRange c;
	c.in_lo = repo_scalar( repo, base + ".input_min", c.in_lo );
	c.in_hi = repo_scalar( repo, base + ".input_max", c.in_hi );
	c.out_lo = repo_scalar( repo, base + ".output_min", c.out_lo );
	c.out_hi = repo_scalar( repo, base + ".output_max", c.out_hi );
	return c;
}

// 2-D rotary tables of a patch grid in merge order: [h * w, head_dim] cos / sin (rotate_half layout; half the head dimension
// rotates by the row, the other half by the column).
void vision_rotary_tables( int grid_h, int grid_w, int merge_size, int head_dim, std::vector<float>* cos_t, std::vector<float>* sin_t );

// Bytes the tower's tensors take once loaded: the checkpoint also keeps the unquantized fused qkv weights, which are never loaded
inline size_t vision_weight_bytes( const engine::loader::SafeTensorsRepo& repo, int depth, int hidden ) {
	size_t bytes = repo.bytes_with_prefix( "model.visual." );
	const size_t fused = static_cast<size_t>( 3 ) * hidden * hidden * 2 + 256;
	for ( int i = 0; i < depth; ++i ) {
		const std::string name = "model.visual.blocks." + std::to_string( i ) + ".attn.qkv.weight";
		if ( repo.find( name.c_str() ) && bytes >= fused ) bytes -= fused;
	}
	return bytes;
}

// True when `need` more bytes fit in the arena; otherwise logs why the tower stays off. The GLM arena is sized to the last
// megabyte, so the tower loads after everything else and only into what is left.
inline bool vision_fits( const pod::MemoryArena& arena, size_t need, const char* what ) {
	const size_t left = arena.total_bytes > arena.offset_bytes ? arena.total_bytes - arena.offset_bytes : 0;
	if ( left >= need ) return true;
	LOG_WARN( "[vision] %s needs %.0f MiB but the arena has %.0f MiB left; image input disabled (free memory with a smaller max_seq or max_image_tokens)\n",
		what, need / 1048576.0, left / 1048576.0 );
	return false;
}

} // namespace engine::models
