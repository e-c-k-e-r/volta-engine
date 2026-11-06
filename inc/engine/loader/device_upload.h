// Dense weights from an open safetensors file into a device arena: raw bytes, or converted to FP32 / FP16 (from FP32,
// FP16 or BF16). Shared by the dense (non-LLM) models: the H3 video models and the TTS model.
#pragma once

#include "engine/core/arena.h"
#include "engine/loader/safetensors.h"
#include <cuda_fp16.h>
#include <cstddef>
#include <string>

namespace engine::loader {

struct DeviceUploader {
	const pod::SafeTensorsFile& st;
	pod::MemoryArena& arena;
	const char* tag = "upload";  // the log prefix for missing tensors

	DeviceUploader( const pod::SafeTensorsFile& st_, pod::MemoryArena& arena_, const char* tag_ = "upload" ) : st( st_ ), arena( arena_ ), tag( tag_ ) {}

	// The tensor's entry; logs an error and returns nullptr if it's missing
	const pod::SafeTensorMeta* find( const std::string& name ) const;
	static size_t numel( const pod::SafeTensorMeta* m );
	// A device copy of the tensor's bytes as stored
	void* raw( const pod::SafeTensorMeta* m );
	// The tensor converted to FP32 / FP16 (nullptr if missing); f16 also reports a matrix's [rows, cols]
	const float* f32( const std::string& name );
	const half* f16( const std::string& name, int* rows = nullptr, int* cols = nullptr );
};

} // namespace engine::loader
