// Weak stubs: runtime steering exists on the CUDA / ROCm backends only.
#include "engine/modules/steer.h"
#include "engine/backend/tiers.h"

namespace engine::modules {

__attribute__(( weak )) void ResidualSteer::apply( const SteerTables& tables, backend::half_t* hidden, const pod::BatchRows* d_rows, int rows, int layer_index, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "ResidualSteer::apply" );
	__builtin_unreachable();
}

__attribute__(( weak )) void ResidualSteer::penalize( const SteerTables& tables, backend::half_t* logits, int logit_cols, const pod::BatchRows* d_rows, int row0, int count,
	int shard_lo, int shard_cols, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "ResidualSteer::penalize" );
	__builtin_unreachable();
}

} // namespace engine::modules
