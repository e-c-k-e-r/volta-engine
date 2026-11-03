#include "engine/models/sparse_head_host.h"
#include "engine/modules/sparse_head.h"
#include "engine/backend/runtime.h"
#include "engine/core/log.h"
#include "engine/runtime/options.h"
#include <algorithm>

namespace engine::models {

void SparseHeadHost::init( pod::MemoryArena& arena, int hidden, int shard_real_cols, int shard_cols, int rows ) {
	const int c = runtime::options().sparse_head;
	if ( c <= 0 ) return;
	// a rank can hold the whole subset (the frequent ids cluster at the low end of the vocabulary, i.e. on the first ranks)
	hidden_ = hidden;
	shard_cols_ = shard_cols;
	rows_ = rows;
	cap_ = std::min( std::max( 256, c ), shard_real_cols );
	const size_t H = hidden, C = cap_, h = sizeof( backend::half_t );
	w_ = static_cast<backend::half_t*>( core::arena_alloc( arena, C * H * h ) );
	cols_ = static_cast<int*>( core::arena_alloc( arena, C * sizeof( int ) ) );
	sub_ = static_cast<backend::half_t*>( core::arena_alloc( arena, static_cast<size_t>( rows ) * C * h ) );
	ident_ = static_cast<backend::half_t*>( core::arena_alloc( arena, static_cast<size_t>( kIdentRows ) * H * h ) );
	chunk_ = static_cast<backend::half_t*>( core::arena_alloc( arena, static_cast<size_t>( kIdentRows ) * shard_cols * h ) );
}

bool SparseHeadHost::set( const pod::LinearWeight& head, const int* cols, int count, int rank, backend::stream_t stream ) {
	if ( cap_ == 0 ) return false;
	ENGINE_ASSERT( count >= 0 && count <= cap_, "SparseHeadHost::set: more columns than the buffer holds" );
	backend::set_device( rank );
	use_ = false;
	count_ = count;
	if ( count == 0 ) return true;
	backend::memcpy_async( cols_, cols, static_cast<size_t>( count ) * sizeof( int ), backend::Copy::H2D, stream );
	modules::SparseHead::extract( head, hidden_, shard_cols_, cols_, count, ident_, chunk_, kIdentRows, w_, stream );
	return true;
}

void SparseHeadHost::logits( const backend::half_t* x, int rows, backend::half_t* logits, backend::stream_t stream ) const {
	ENGINE_ASSERT( rows >= 1 && rows <= rows_, "SparseHeadHost::logits: more rows than the buffers hold" );
	modules::SparseHead::logits( w_, count_, cols_, x, rows, hidden_, sub_, logits, shard_cols_, stream );
}

} // namespace engine::models
