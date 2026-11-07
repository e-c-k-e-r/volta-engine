#pragma once

// GGUF architecture adapters: how a GGUF tensor name maps to the Hugging Face name the models load, and which of llama.cpp's converter
// transforms to undo (src/loader/gguf.cpp applies them)
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace engine::loader {

// The key-value metadata of a GGUF file (integers, floats and strings; arrays of integers are kept, others skipped)
struct GgufKv {
	std::map<std::string, int64_t> ints;
	std::map<std::string, double> floats;
	std::map<std::string, std::string> strings;
	std::map<std::string, std::vector<int64_t>> int_arrays;
	int64_t get( const std::string& key, int64_t fallback = 0 ) const {
		const auto it = ints.find( key );
		return it == ints.end() ? fallback : it->second;
	}
};

enum class GgufTransform {
	None,
	NormMinusOne,        // the converter stored 1 + w (Qwen3.5 RMSNorm weights): w = stored - 1 (F32)
	ALogFromNegExp,      // the converter stored -exp(A_log) in tiled V-head order: A_log = log(-stored), back to grouped order (F32)
	RowsUntile,          // rows in tiled V-head order (head_dim rows per head): back to grouped order
	RowsUntileVPart,     // the same for the rows after the first `row_offset` (the V part of [q | k | v])
	ColsTiled,           // input columns in tiled order: left as stored, flagged (GGUF_COLS_V_TILED)
	TransposeHeads,      // a stack of per-head matrices stored transposed ([heads][a][b] for [heads][b][a]; F32 / F16 / BF16): each transposed back
};

struct GgufMapped {
	std::string name;                // the Hugging Face name
	GgufTransform transform = GgufTransform::None;
	int head_dim = 1;                // rows (or columns) per V head for the untile transforms
	int row_offset = 0;              // RowsUntileVPart: rows before the V part
	std::vector<int> shape;          // a Hugging Face shape when it differs from the GGUF one beyond the dimension order (empty: reversed ne)
	// A row range of the GGUF tensor (its rows: every dimension but the innermost, flattened): one expert of a stacked expert tensor, one
	// part of a weight the converter concatenated. rows < 0: all of them
	int64_t row_begin = 0, rows = -1;
	// One of `parts` GGUF tensors whose rows, in part order, make up the Hugging Face tensor (the converter split it)
	int part = 0, parts = 1;
	// > 0: the parts are joined head by head (that many heads, each part's rows split evenly) instead of one after the other
	int interleave_heads = 0;
};

struct GgufArch {
	// V heads tiled by the converter: num_k_heads groups of v_per_k heads (0: none)
	int k_heads = 0, v_heads = 0, head_v_dim = 0;
	// Maps one tensor to the Hugging Face tensors it holds (none: the tensor is not used by the engine)
	void ( *map )( const GgufArch& arch, const GgufKv& kv, const std::string& gguf_name, std::vector<GgufMapped>* out ) = nullptr;
};

// The adapter of an architecture (general.architecture), filled from the file's metadata; false when there is none
bool gguf_arch_qwen35( const GgufKv& kv, GgufArch* out );
bool gguf_arch_qwen4exp( const GgufKv& kv, GgufArch* out );
bool gguf_arch_dflash( const GgufKv& kv, GgufArch* out );
bool gguf_arch_glm5next( const GgufKv& kv, GgufArch* out );

} // namespace engine::loader
