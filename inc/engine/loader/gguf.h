#pragma once

// GGUF files (llama.cpp's format) as a source of the safetensors repo: a file's tensors join the same index under the Hugging Face names the
// models load, with the transforms llama.cpp's converter applied undone (an architecture adapter per GGUF architecture). Block-quantized
// tensors keep their blocks (DType::GGUF) for LinearGguf; float tensors keep their type.
#include "engine/loader/safetensors.h"
#include <cstdint>
#include <string>

namespace engine::loader {

// SafeTensorMeta::gguf_flags
enum GgufFlags : uint32_t {
	GGUF_COLS_V_TILED = 1,   // input columns in the converter's tiled V-head order (Qwen3.5 linear attention out_proj); the model reorders the activation
};

// Opens a GGUF file into `st`: false when it cannot be read or its architecture has no adapter
bool gguf_open( pod::SafeTensorsFile& st, const char* path );
void gguf_release( pod::SafeTensorsFile& st );

// A block-quantized tensor of `st` as FP16 values (what the dequantizers produce), dequantized on the current device the first time and kept: the norms, embeddings and small
// dense tensors the models read as floats. Returns `meta` itself for float tensors
const pod::SafeTensorMeta* gguf_float_view( pod::SafeTensorsFile& st, const pod::SafeTensorMeta* meta, backend::stream_t stream );

// The linear-attention head layout of the file's model (k_heads 0 when its architecture does not tile V heads)
struct GgufVHeads {
	int k_heads = 0, v_heads = 0, head_v_dim = 0;
};
GgufVHeads gguf_v_heads( const pod::SafeTensorsFile& st );

// The file's general.architecture ("" for safetensors)
std::string gguf_architecture( const pod::SafeTensorsFile& st );

} // namespace engine::loader
