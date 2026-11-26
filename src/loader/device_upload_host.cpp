// DeviceUploader for the backends without CUDA (engine/loader/device_upload.h): FP32 / FP16 / BF16 converted on the host, then copied
// into the arena.
#include "engine/loader/device_upload.h"
#include "engine/backend/numeric.h"
#include "engine/backend/runtime.h"
#include "engine/core/log.h"
#include <vector>

namespace engine::loader {

const pod::SafeTensorMeta* DeviceUploader::find( const std::string& name ) const {
	const pod::SafeTensorMeta* m = safetensors_find( st, name.c_str() );
	if ( !m ) { LOG_ERROR( "%s: missing tensor %s", tag, name.c_str() ); }
	return m;
}

size_t DeviceUploader::numel( const pod::SafeTensorMeta* m ) {
	size_t n = 1;
	for ( int i = 0; i < m->ndim; ++i ) { n *= static_cast<size_t>( m->shape[i] ); }
	return n;
}

void* DeviceUploader::raw( const pod::SafeTensorMeta* m ) {
	void* d = engine::core::arena_alloc( arena, m->size_bytes );
	backend::memcpy_async( d, safetensors_get_data_cpu( st, m ), m->size_bytes, backend::Copy::H2D, stream );
	backend::stream_sync( stream );
	return d;
}

namespace {

float load( const void* src, pod::DType t, size_t i ) {
	switch ( t ) {
		case pod::DType::FP32: return static_cast<const float*>( src )[i];
		case pod::DType::FP16: return backend::to_float( static_cast<const backend::half_t*>( src )[i] );
		case pod::DType::BF16: return backend::to_float( static_cast<const backend::bf16_t*>( src )[i] );
		default: break;
	}
	ENGINE_ASSERT( false, "dense uploads are FP32, FP16 or BF16" );
	return 0.0f;
}

void store( float v, float* p ) { *p = v; }
void store( float v, backend::half_t* p ) { *p = backend::from_float_half( v ); }

template <typename Tout>
Tout* convert( DeviceUploader& u, const pod::SafeTensorMeta* m ) {
	const size_t n = DeviceUploader::numel( m );
	const void* src = safetensors_get_data_cpu( u.st, m );
	std::vector<Tout> host( n );
	for ( size_t i = 0; i < n; ++i ) { store( load( src, m->dtype, i ), &host[i] ); }
	Tout* d = static_cast<Tout*>( engine::core::arena_alloc( u.arena, n * sizeof( Tout ) ) );
	backend::memcpy_async( d, host.data(), n * sizeof( Tout ), backend::Copy::H2D, u.stream );
	backend::stream_sync( u.stream );
	return d;
}

} // namespace

const float* DeviceUploader::f32( const std::string& name ) {
	const pod::SafeTensorMeta* m = find( name );
	if ( !m ) { return nullptr; }
	if ( m->dtype == pod::DType::FP32 ) { return static_cast<const float*>( raw( m ) ); }
	return convert<float>( *this, m );
}

const backend::half_t* DeviceUploader::f16( const std::string& name, int* rows, int* cols ) {
	const pod::SafeTensorMeta* m = find( name );
	if ( !m ) { return nullptr; }
	if ( rows ) { *rows = m->shape[0]; }
	if ( cols ) { *cols = m->ndim > 1 ? m->shape[1] : 1; }
	if ( m->dtype == pod::DType::FP16 ) { return static_cast<const backend::half_t*>( raw( m ) ); }
	return convert<backend::half_t>( *this, m );
}

} // namespace engine::loader
