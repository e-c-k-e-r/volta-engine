// Vulkan backend: Embedding gather through the SPIR-V kernel
#include "mod_common.h"
#include "engine/modules/embedding.h"

namespace engine::modules {

namespace {
namespace vki = ::engine::backend::impl;
}

void Embedding::forward( const pod::TensorView& table, const int* token_ids, int count, pod::TensorView& out, backend::stream_t stream ) {
	ENGINE_ASSERT( table.dtype == pod::DType::BF16, "Embedding::forward expects a BF16 table" );
	const int hidden = table.shape[1];
	vki::EmbedParams p{ vki::dp( table.data ), vki::dp( token_ids ), vki::dp( out.data ), hidden };
	vki::launch( "embed_tokens", p, stream, ( hidden + 255 ) / 256, count );
}

} // namespace engine::modules
