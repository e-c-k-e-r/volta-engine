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

	// Vocabulary-parallel shard: this rank's columns of a [K, N] weight, padded up by pad_cols so the
	// shard width is a multiple of 256 and the dp4a int8 GEMV engages. Pad columns decode to exactly
	// 0.0 (zero trellis, zero svh), so the caller must keep them out of the argmax range and offset
	// the returned indices by rank * (N / tp_size).
	// Column range [col_block_begin, col_block_begin + col_blocks) in 16-column blocks (a whole number of 128-column
	// Hadamard spans), zero-padded by pad_cols; the general form of load_linear_slice_padded.
	pod::LinearWeight load_linear_cols_padded(
		const char* base_name,
		pod::MemoryArena& arena,
		int bits,
		int col_block_begin,
		int col_blocks,
		int pad_cols,
		cudaStream_t stream = nullptr
	);

	pod::LinearWeight load_linear_slice_padded(
		const char* base_name,
		pod::MemoryArena& arena,
		int bits,
		int rank,
		int tp_size,
		int pad_cols,
		cudaStream_t stream = nullptr
	);

	pod::LinearWeight load_linear_slice_fused(
		const char* base_name_a,
		const char* base_name_b,
		pod::MemoryArena& arena,
		int bits,
		int slice_dim,
		int rank,
		int tp_size,
		cudaStream_t stream = nullptr
	);

	// Host pointer to a tensor's mmap'd bytes, nullptr if absent.
	const void* cpu_data( const char* name ) const;

	// NVFP4 (compressed-tensors nvfp4-pack-quantized): weight_packed [N, K/2] u8 (low nibble = even k), weight_scale
	// [N, K/16] e4m3, weight_global_scale f32 (weight = e2m1 * scale / global). Loads the output rows
	// [n_begin[i], n_begin[i] + n_count[i]) of each range, concatenated, and K elements [k0, k0 + kcount),
	// prepacked into the skinny kernel's fragment order with N zero-padded to a multiple of 32.
	pod::LinearWeight load_nvfp4(
		const char* base_name,
		pod::MemoryArena& arena,
		const int* n_begin,
		const int* n_count,
		int ranges,
		int k0,
		int kcount,
		cudaStream_t stream = nullptr
	);

	// Tensor-parallel slice: split_n splits the output rows (column parallel), otherwise K (row parallel).
	pod::LinearWeight load_nvfp4_slice(
		const char* base_name,
		pod::MemoryArena& arena,
		bool split_n,
		int rank,
		int tp_size,
		cudaStream_t stream = nullptr
	);

	// The same weights dequantized to dense FP16 [N_local, K] (a .weight tensor), for the few small projections
	// that stay unquantized in the engine.
	pod::TensorView load_nvfp4_dense(
		const char* base_name,
		pod::MemoryArena& arena,
		bool split_n,
		int rank,
		int tp_size,
		cudaStream_t stream = nullptr
	);

	// Dense [rows, K] weight (BF16/FP16 checkpoint tensor) rows [row_begin, row_begin + row_count) as FP16, followed by
	// pad_rows zero rows; the shape is tagged for LinearWeight::make_fp16 (an [n, k] matrix, shape [k, n]).
	pod::LinearWeight load_dense_rows_padded(
		const char* weight_name,
		pod::MemoryArena& arena,
		int row_begin,
		int row_count,
		int pad_rows,
		cudaStream_t stream = nullptr
	);

	pod::LinearWeight load_grouped_trellis_slice(
		const char* base_name,
		pod::MemoryArena& arena,
		const int* group_channel_blocks,
		const int* rank_channel_blocks,
		int num_groups,
		int rank,
		cudaStream_t stream = nullptr
	);

	pod::TensorView load_grouped_slice(
		const char* name,
		pod::MemoryArena& arena,
		const int* group_sizes,
		const int* rank_sizes,
		int num_groups,
		int rank,
		int slice_dim = 0,
		cudaStream_t stream = nullptr,
		pod::DType target_dtype = pod::DType::FP16
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
