#include "engine/loader/device_upload.h"
#include "engine/core/log.h"
#include "engine/kernels/dense.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

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
	CUDA_CHECK( cudaMemcpy( d, safetensors_get_data_cpu( st, m ), m->size_bytes, cudaMemcpyHostToDevice ) );
	return d;
}

template <typename Tout>
static Tout* convert( DeviceUploader& u, const pod::SafeTensorMeta* m ) {
	const size_t n = DeviceUploader::numel( m );
	void* tmp = nullptr;
	CUDA_CHECK( cudaMalloc( &tmp, m->size_bytes ) );
	CUDA_CHECK( cudaMemcpy( tmp, safetensors_get_data_cpu( u.st, m ), m->size_bytes, cudaMemcpyHostToDevice ) );
	Tout* d = static_cast<Tout*>( engine::core::arena_alloc( u.arena, n * sizeof( Tout ) ) );
	if ( m->dtype == pod::DType::BF16 ) {
		engine::dense::k_convert<__nv_bfloat16, Tout><<<1024, 256>>>( static_cast<const __nv_bfloat16*>( tmp ), d, n );
	} else if ( m->dtype == pod::DType::FP16 ) {
		engine::dense::k_convert<half, Tout><<<1024, 256>>>( static_cast<const half*>( tmp ), d, n );
	} else if ( m->dtype == pod::DType::FP32 ) {
		engine::dense::k_convert<float, Tout><<<1024, 256>>>( static_cast<const float*>( tmp ), d, n );
	} else {
		ENGINE_ASSERT( false, "dense uploads are FP32, FP16 or BF16" );
	}
	CUDA_CHECK( cudaDeviceSynchronize() );
	CUDA_CHECK( cudaFree( tmp ) );
	return d;
}

const float* DeviceUploader::f32( const std::string& name ) {
	const pod::SafeTensorMeta* m = find( name );
	if ( !m ) { return nullptr; }
	if ( m->dtype == pod::DType::FP32 ) { return static_cast<const float*>( raw( m ) ); }
	return convert<float>( *this, m );
}

const half* DeviceUploader::f16( const std::string& name, int* rows, int* cols ) {
	const pod::SafeTensorMeta* m = find( name );
	if ( !m ) { return nullptr; }
	if ( rows ) { *rows = m->shape[0]; }
	if ( cols ) { *cols = m->ndim > 1 ? m->shape[1] : 1; }
	if ( m->dtype == pod::DType::FP16 ) { return static_cast<const half*>( raw( m ) ); }
	return convert<half>( *this, m );
}

} // namespace engine::loader
