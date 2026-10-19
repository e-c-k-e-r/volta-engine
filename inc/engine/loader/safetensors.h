#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/arena.h"
#include "engine/modules/linear.h"
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <vector>
#include "engine/backend/types.h"

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
bool safetensors_load_to_device( const pod::SafeTensorsFile& st, const pod::SafeTensorMeta* meta, void* dst_device_ptr, backend::stream_t stream = nullptr );
pod::TensorView safetensors_alloc_tensor( const pod::SafeTensorsFile& st, const char* name, pod::MemoryArena& arena, backend::stream_t stream = nullptr );

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
		backend::stream_t stream = nullptr,
		pod::DType target_dtype = pod::DType::AUTO
	);

	pod::TensorView load_slice(
		const char* name,
		pod::MemoryArena& arena,
		int slice_dim,
		int rank,
		int tp_size,
		backend::stream_t stream = nullptr,
		pod::DType target_dtype = pod::DType::AUTO
	);

	// code_arena: where the trellis codes (the bulk of the weight) go when they should not share the scale vectors' arena
	// (a slot of the resident-expert pool, or the pinned cold tier); suh / svh always go to `arena`
	pod::LinearWeight load_linear(
		const char* base_name,
		pod::MemoryArena& arena,
		int bits,
		backend::stream_t stream = nullptr,
		pod::MemoryArena* code_arena = nullptr
	);
	// Checkpoint bytes of the tensors whose names start with `prefix` and contain none of the `skip` substrings
	size_t bytes_of( const std::string& prefix, std::initializer_list<const char*> skip = {} ) const;
	// Bytes of the trellis codes of `base_name` (0 when it is not a trellis tensor)
	size_t trellis_bytes( const char* base_name ) const;
	// Where a tensor's bytes are in its shard file (an open descriptor, absolute offset), for reads that bypass the mmap
	bool file_ref( const char* tensor_name, int* fd, size_t* offset, size_t* bytes ) const;
	// ENGINE_DEBUG_EXPERT_SYNTH=2 (timing only): an expert linear that is not in the checkpoint, generated with the shape and bit width the
	// directory's quantization_config.json lists for it (suh / svh of +-1, pseudo-random codes). Both return 0 / an empty weight when that file has no entry
	size_t synth_trellis_bytes( const char* base_name ) const;
	pod::LinearWeight synth_linear( const char* base_name, pod::MemoryArena& arena, backend::stream_t stream, pod::MemoryArena* code_arena = nullptr ) const;

	// A range of a trellis linear [K / 16, N / 16, 16 * bits]: slice_dim 0 = input rows [begin16, begin16 + count16) (in 16-blocks, whole 128-wide
	// Hadamard spans: its suh), slice_dim 1 = output columns (its svh); the other scale vector is loaded whole. The codes go to code_arena when given
	pod::LinearWeight load_linear_range(
		const char* base_name,
		pod::MemoryArena& arena,
		int slice_dim,
		int begin16,
		int count16,
		backend::stream_t stream = nullptr,
		pod::MemoryArena* code_arena = nullptr
	);

	pod::LinearWeight load_linear_slice(
		const char* base_name,
		pod::MemoryArena& arena,
		int bits,
		int slice_dim,
		int rank,
		int tp_size,
		backend::stream_t stream = nullptr
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
		backend::stream_t stream = nullptr
	);

	pod::LinearWeight load_linear_slice_padded(
		const char* base_name,
		pod::MemoryArena& arena,
		int bits,
		int rank,
		int tp_size,
		int pad_cols,
		backend::stream_t stream = nullptr
	);

	pod::LinearWeight load_linear_slice_fused(
		const char* base_name_a,
		const char* base_name_b,
		pod::MemoryArena& arena,
		int bits,
		int slice_dim,
		int rank,
		int tp_size,
		backend::stream_t stream = nullptr
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
		backend::stream_t stream = nullptr,
		int pad_n = 0,
		pod::MemoryArena* codes_arena = nullptr   // the packed codes are placed here (the expert tier's slots), the scales stay in `arena`
	);

	// NVFP4 from a BF16 / FP16 dense `<base_name>.weight` [N, K] (round-to-nearest with a +-2 code E4M3 scale search per 16 values; the global
	// scale comes from the whole tensor). Same row / K ranges and layout as load_nvfp4; N is zero-padded to pad_n when larger than the rows.
	pod::LinearWeight quantize_nvfp4(
		const char* base_name,
		pod::MemoryArena& arena,
		const int* n_begin,
		const int* n_count,
		int ranges,
		int k0,
		int kcount,
		int pad_n = 0,
		backend::stream_t stream = nullptr
	);

	// Prepared (modelopt-named) tensors stacked along the output rows, which must share K and the global scale
	pod::LinearWeight load_nvfp4_concat(
		const char* const* base_names,
		const int* n_begin,
		const int* n_count,
		int parts,
		int k0,
		int kcount,
		int pad_n,
		pod::MemoryArena& arena,
		backend::stream_t stream = nullptr
	);

	// The same for several tensors of one input width stacked along the output rows (a fused q / k / v), one global scale over all of them
	pod::LinearWeight quantize_nvfp4_concat(
		const char* const* base_names,
		const int* n_begin,
		const int* n_count,
		int parts,
		int k0,
		int kcount,
		int pad_n,
		pod::MemoryArena& arena,
		backend::stream_t stream = nullptr
	);

	pod::LinearWeight quantize_nvfp4_slice(
		const char* base_name,
		pod::MemoryArena& arena,
		bool split_n,
		int rank,
		int tp_size,
		int pad_n = 0,
		backend::stream_t stream = nullptr
	);

	// Tensor-parallel slice: split_n splits the output rows (column parallel), otherwise K (row parallel).
	pod::LinearWeight load_nvfp4_slice(
		const char* base_name,
		pod::MemoryArena& arena,
		bool split_n,
		int rank,
		int tp_size,
		backend::stream_t stream = nullptr,
		int pad_n = 0,
		pod::MemoryArena* codes_arena = nullptr
	);

	// The same weights dequantized to dense FP16 [N_local, K] (a .weight tensor), for the few small projections
	// that stay unquantized in the engine.
	pod::TensorView load_nvfp4_dense(
		const char* base_name,
		pod::MemoryArena& arena,
		bool split_n,
		int rank,
		int tp_size,
		backend::stream_t stream = nullptr
	);

	// Dense [rows, K] weight (BF16/FP16 checkpoint tensor) rows [row_begin, row_begin + row_count) as FP16, followed by
	// pad_rows zero rows; the shape is tagged for LinearWeight::make_fp16 (an [n, k] matrix, shape [k, n]).
	pod::LinearWeight load_dense_rows_padded(
		const char* weight_name,
		pod::MemoryArena& arena,
		int row_begin,
		int row_count,
		int pad_rows,
		backend::stream_t stream = nullptr
	);

	pod::LinearWeight load_grouped_trellis_slice(
		const char* base_name,
		pod::MemoryArena& arena,
		const int* group_channel_blocks,
		const int* rank_channel_blocks,
		int num_groups,
		int rank,
		backend::stream_t stream = nullptr
	);

	pod::TensorView load_grouped_slice(
		const char* name,
		pod::MemoryArena& arena,
		const int* group_sizes,
		const int* rank_sizes,
		int num_groups,
		int rank,
		int slice_dim = 0,
		backend::stream_t stream = nullptr,
		pod::DType target_dtype = pod::DType::FP16
	);

	const pod::SafeTensorsFile* get_shard( int idx ) const {
		if ( idx >= 0 && idx < num_shards_ ) {
			return &shards_[idx];
		}
		return nullptr;
	}

	// Total bytes of the tensors whose name starts with `prefix`
	size_t bytes_with_prefix( const char* prefix ) const;
	int shard_count() const {
		return num_shards_;
	}

	const char* dir_path() const {
		return dir_path_;
	}

private:
	static constexpr int MAX_SHARDS = 128;
	pod::SafeTensorsFile shards_[MAX_SHARDS];
	char shard_names_[MAX_SHARDS][256];
	int num_shards_ = 0;
	char dir_path_[1024];

	std::unordered_map<std::string, std::pair<int, const pod::SafeTensorMeta*>> tensor_map_;
};

} // namespace engine::loader
