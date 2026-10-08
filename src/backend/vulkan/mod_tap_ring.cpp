// Vulkan backend: DFlash tap ring publish and stream reduction through the SPIR-V kernels
#include "mod_common.h"
#include "engine/spec/tap_ring.h"

namespace engine::spec {

void dflash_publish_taps(
	backend::half_t* taps,
	const backend::half_t* hidden,
	const pod::BatchRows* d_rows,
	const int* d_owner,
	int m,
	int tap,
	int H,
	int cap,
	backend::stream_t stream
) {
	namespace vki = ::engine::backend::impl;
	vki::TapStoreParams p{ vki::dp( taps ), vki::dp( hidden ), vki::rows_slot( d_rows ), vki::rows_pos( d_rows ), vki::dp( d_owner ), m, tap, H, cap };
	vki::launch( "dflash_store_taps", p, stream, ( H + 255 ) / 256, m );
}

void dflash_reduce_streams( const float* streams, backend::half_t* out, int m, int H, int mode, backend::stream_t stream ) {
	namespace vki = ::engine::backend::impl;
	vki::ReduceStreamsParams p{ vki::dp( streams ), vki::dp( out ), H, mode };
	vki::launch( "reduce_streams", p, stream, ( H + 255 ) / 256, m );
}

} // namespace engine::spec
