#pragma once

// The original weights of a checkpoint's quantized tensors, as the quantizer takes them: fp32 [in, out] on the current device. A template tensor
// <base> is read from <base>.weight in BF16, FP16 or FP32, or in FP8 E4M3 with its block scales (<base>.weight_scale_inv); from the fused
// experts of the original (<prefix>.experts.gate_up_proj [E, 2 I, H], gate rows first, and <prefix>.experts.down_proj [E, H, I]) for
// <prefix>.experts.<e>.<gate|up|down>_proj; and a fused qkv_proj from its q / k / v parts, rows in that order (GLM's linear attention).
#include "st_io.h"
#include "engine/backend/types.h"
#include <string>
#include <vector>

namespace engine::tools::quant {

class SourceWeights {
public:
	bool open( const std::string& dir ) { return repo_.open( dir ); }
	bool has( const std::string& base );
	// [out, in] of <base>; false when the original does not have it
	bool shape( const std::string& base, int* out, int* in );
	// <base> into dst (in x out fp32, device); staging buffers grow as needed
	bool load( const std::string& base, float* dst, backend::stream_t stream );
	StRepo& repo() { return repo_; }

private:
	struct Ref {
		enum Kind { PLAIN, FUSED_GATE, FUSED_UP, FUSED_DOWN, PARTS } kind = PLAIN;
		std::string name;                 // the original's tensor (PLAIN, FUSED_*)
		int expert = 0;
		std::vector<std::string> parts;   // PARTS: the bases of the parts
	};
	bool resolve( const std::string& base, Ref& r );
	bool load_plain( const std::string& name, const StEntry* e, int64_t row0, int rows, int cols, float* dst, int dst_ld, int dst_col0,
		backend::stream_t stream );
	StRepo repo_;
	void* staging_ = nullptr;
	size_t staging_bytes_ = 0;
	void* scales_ = nullptr;
	size_t scales_bytes_ = 0;
};

} // namespace engine::tools::quant
