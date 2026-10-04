#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/arena.h"
#include "engine/modules/linear.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
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

class SafeTensorsRepo {
public:
	SafeTensorsRepo();
	~SafeTensorsRepo();

	SafeTensorsRepo( const SafeTensorsRepo& ) = delete;
	SafeTensorsRepo& operator=( const SafeTensorsRepo& ) = delete;

	bool open( const char* dir_path );
	void close();

	const pod::SafeTensorMeta* find( const char* name, int* out_shard_idx = nullptr ) const;

	pod::TensorView load(
		const char* name,
		pod::MemoryArena& arena,
		cudaStream_t stream = nullptr,
		pod::DType target_dtype = pod::DType::AUTO
	);

	pod::TensorView load_slice(
		const char* name,
		pod::MemoryArena& arena,
		int slice_dim,
		int rank,
		int tp_size,
		cudaStream_t stream = nullptr,
		pod::DType target_dtype = pod::DType::AUTO
	);

	pod::LinearWeight load_linear(
		const char* base_name,
		pod::MemoryArena& arena,
		int bits,
		cudaStream_t stream = nullptr
	);

	pod::LinearWeight load_linear_slice(
		const char* base_name,
		pod::MemoryArena& arena,
		int bits,
		int slice_dim,
		int rank,
		int tp_size,
		cudaStream_t stream = nullptr
	);

	pod::LinearWeight load_linear_qkv_qwen(
		const char* base_name,
		pod::MemoryArena& arena,
		int rank,
		int tp_size,
		cudaStream_t stream = nullptr
	);

	pod::TensorView load_conv1d_qwen(
		const char* name,
		pod::MemoryArena& arena,
		int rank,
		int tp_size,
		cudaStream_t stream = nullptr
	);

	const pod::SafeTensorsFile* get_shard( int idx ) const {
		if ( idx >= 0 && idx < num_shards_ ) {
			return &shards_[idx];
		}
		return nullptr;
	}

	int shard_count() const {
		return num_shards_;
	}

	const char* dir_path() const {
		return dir_path_;
	}

private:
	static constexpr int MAX_SHARDS = 32;
	pod::SafeTensorsFile shards_[MAX_SHARDS];
	char shard_names_[MAX_SHARDS][256];
	int num_shards_ = 0;
	char dir_path_[1024];

	std::unordered_map<std::string, std::pair<int, const pod::SafeTensorMeta*>> tensor_map_;
};

} // namespace engine::loader
