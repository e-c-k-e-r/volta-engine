// CPU backend: DFlash tap ring publish and stream reduction through the Slang host kernels
// cpu_kernels.h first: the Slang prelude defines int4, which the backend shim only declares when the prelude is absent
#include "cpu_kernels.h"
#include "engine/spec/tap_ring.h"
#include "engine/backend/tiers.h"

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
	backend::stream_t
) {
	TapStoreParams p{ taps, const_cast<half*>( hidden ), const_cast<int*>( d_rows->slot ), const_cast<int*>( d_rows->pos ), const_cast<int*>( d_owner ), m, tap, H, cap };
	impl::cpu_dispatch( ::dflash_store_taps, &p, ( H + 255 ) / 256, m );
}

void dflash_publish_taps_slots( backend::half_t*, const backend::half_t*, const pod::BatchRows*, int, int, int, int, size_t, backend::stream_t ) {
	ENGINE_UNSUPPORTED( "dflash_publish_taps_slots" );
}

void dflash_reduce_streams( const float* streams, backend::half_t* out, int m, int H, int mode, backend::stream_t ) {
	ReduceStreamsParams p{ const_cast<float*>( streams ), out, H, mode };
	impl::cpu_dispatch( ::reduce_streams, &p, ( H + 255 ) / 256, m );
}

} // namespace engine::spec
