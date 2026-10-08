// CPU backend: Embedding gather through the Slang host kernel
// cpu_kernels.h first: the Slang prelude defines int4, which the backend shim only declares when the prelude is absent
#include "cpu_kernels.h"
#include "engine/modules/embedding.h"

namespace engine::modules {

void Embedding::forward( const pod::TensorView& table, const int* token_ids, int count, pod::TensorView& out, backend::stream_t ) {
	ENGINE_ASSERT( table.dtype == pod::DType::BF16, "Embedding::forward expects a BF16 table" );
	const int hidden = table.shape[1];
	EmbedParams p{ const_cast<uint16_t*>( table.ptr<const uint16_t>() ), const_cast<int*>( token_ids ), out.ptr<half>(), hidden };
	impl::cpu_dispatch( ::embed_tokens, &p, ( hidden + 255 ) / 256, count );
}

} // namespace engine::modules
