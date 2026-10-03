#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/arena.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace pod {

struct SafeTensorMeta {
	char        name[256];
	pod::DType  dtype;
	int         ndim;
	int         shape[4];
	size_t      data_offset_bytes;
	size_t      size_bytes;
};

struct SafeTensorsFile {
	int             fd = -1;
	size_t          file_size = 0;
	void*           mmap_addr = nullptr;
	uint64_t        header_len = 0;
	SafeTensorMeta* tensors = nullptr;
	int             tensor_count = 0;
	int             tensor_capacity = 0;
};

} // namespace pod

namespace engine::loader {

bool safetensors_open( pod::SafeTensorsFile& st, const char* filepath );
void safetensors_close( pod::SafeTensorsFile& st );
const pod::SafeTensorMeta* safetensors_find( const pod::SafeTensorsFile& st, const char* name );
const void* safetensors_get_data_cpu( const pod::SafeTensorsFile& st, const pod::SafeTensorMeta* meta );
bool safetensors_load_to_device( const pod::SafeTensorsFile& st, const pod::SafeTensorMeta* meta, void* dst_device_ptr, cudaStream_t stream = nullptr );
pod::TensorView safetensors_alloc_tensor( const pod::SafeTensorsFile& st, const char* name, pod::MemoryArena& arena, cudaStream_t stream = nullptr );

} // namespace engine::loader
