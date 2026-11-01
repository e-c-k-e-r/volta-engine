// Weak stubs: the sparse drafter head exists on the CUDA / ROCm backends only.
#include "engine/modules/sparse_head.h"
#include "engine/backend/tiers.h"

namespace engine::modules {

__attribute__(( weak )) void SparseHead::identity_rows( backend::half_t* ident, int rows, int hidden, int k0, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "SparseHead::identity_rows" );
	__builtin_unreachable();
}

__attribute__(( weak )) void SparseHead::gather( const backend::half_t* chunk, int chunk_ld, int rows, int k0, const int* cols, int count, backend::half_t* w_sub, int hidden,
	backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "SparseHead::gather" );
	__builtin_unreachable();
}

__attribute__(( weak )) void SparseHead::extract( const pod::LinearWeight& head, int hidden, int shard_cols, const int* cols, int count, backend::half_t* ident,
	backend::half_t* chunk, int ident_rows, backend::half_t* w_sub, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "SparseHead::extract" );
	__builtin_unreachable();
}

__attribute__(( weak )) void SparseHead::logits( const backend::half_t* w_sub, int count, const int* cols, const backend::half_t* x, int rows, int hidden,
	backend::half_t* sub, backend::half_t* logits, int logits_ld, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "SparseHead::logits" );
	__builtin_unreachable();
}

__attribute__(( weak )) void SparseHead::scatter( const backend::half_t* sub, int count, const int* cols, backend::half_t* logits, int logits_ld, int rows, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "SparseHead::scatter" );
	__builtin_unreachable();
}

} // namespace engine::modules
