#pragma once

// EXL3 tensors of a checkpoint or of a volta_quantize .partial directory: reading one (its width, codebook and tile order as the directory names
// them), decoding it to a dense weight, and writing one as a per-tensor result file.
#include "engine/quant/exl3_quantize.h"
#include "quant/st_io.h"
#include <map>
#include <string>

namespace engine::tools::quant {

struct Exl3Read {
	engine::quant::Exl3Tensor t;
	int code = 0;
	int cb = 0;
	bool volta = false;
};

// The tensor <base> (.trellis, .suh, .svh): width from the trellis shape, codebook from the markers, tile order from quantization_config.json
// ("tile_layout": "volta", with "tile_layout_stock" substrings that keep the stock order); in a .partial directory (no quantization_config.json)
// both from the tensor's file metadata
bool exl3_read( StRepo& repo, const std::string& base, Exl3Read& out );

// The dense weight, input-major k x n fp32 on the device (FP16 decode as inference has it); unit_scales: the core alone, with suh / svh taken as 1
bool exl3_decode( const Exl3Read& r, float* out, bool unit_scales = false );

// A per-tensor result file (volta_quantize's <out>/.partial/<base>.safetensors): trellis, suh, svh, and metadata naming the tile order and
// codebook, plus `extra`
bool exl3_write_partial( const std::string& path, const std::string& base, const engine::quant::Exl3Tensor& t, int code, int cb, bool volta,
	const std::map<std::string, std::string>& extra );

} // namespace engine::tools::quant
