#pragma once

// The Qwen3.5-family linear-attention (gated delta net) tensors as llama.cpp's converters write them, shared by the qwen35 and qwen4exp
// adapters: the converter reorders the V heads from grouped (k-head major) to tiled (v-index major) order in in_proj_qkv's V rows,
// in_proj_z, in_proj_a / b, A_log (stored as -exp(A_log)), dt_bias, conv1d's V channels (squeezed) and out_proj's input columns.
#include "gguf_arch.h"
#include <cstring>
#include <string>

namespace engine::loader {

struct Suffix {
	const char* gguf;
	const char* hf;
	GgufTransform transform;
	int head_dim;    // 0: the V head dim; 1: one row per head
	bool v_part;     // RowsUntileVPart
};

inline constexpr Suffix kLinearAttn[] = {
	{ "attn_qkv.weight", "linear_attn.in_proj_qkv.weight", GgufTransform::RowsUntileVPart, 0, true },
	{ "attn_gate.weight", "linear_attn.in_proj_z.weight", GgufTransform::RowsUntile, 0, false },
	{ "ssm_alpha.weight", "linear_attn.in_proj_a.weight", GgufTransform::RowsUntile, 1, false },
	{ "ssm_beta.weight", "linear_attn.in_proj_b.weight", GgufTransform::RowsUntile, 1, false },
	{ "ssm_a", "linear_attn.A_log", GgufTransform::ALogFromNegExp, 1, false },
	{ "ssm_dt.bias", "linear_attn.dt_bias", GgufTransform::RowsUntile, 1, false },
	{ "ssm_conv1d.weight", "linear_attn.conv1d.weight", GgufTransform::RowsUntileVPart, 0, true },
	{ "ssm_norm.weight", "linear_attn.norm.weight", GgufTransform::None, 1, false },
	{ "ssm_out.weight", "linear_attn.out_proj.weight", GgufTransform::ColsTiled, 0, false },
};

// `out` for a suffix entry: its transform, the head geometry of the untile transforms, conv1d's checkpoint shape
inline bool fill_suffix( const GgufArch& arch, const GgufKv& kv, const char* prefix, const std::string& name, const Suffix& s, GgufMapped* out ) {
	out->name = name;
	out->transform = s.transform;
	out->head_dim = s.head_dim == 0 ? arch.head_v_dim : s.head_dim;
	out->row_offset = s.v_part ? 2 * arch.k_heads * static_cast<int>( kv.get( std::string( prefix ) + ".ssm.state_size" ) ) : 0;
	if ( std::strcmp( s.gguf, "ssm_conv1d.weight" ) == 0 ) out->shape = { -1, 1, -1 };   // [channels, 1, kernel] as in the checkpoint
	return true;
}

// The V-head geometry from <prefix>.ssm.*; the converter reorders V heads only when there are more of them than key heads
inline bool linear_attn_heads( const GgufKv& kv, const char* prefix, GgufArch* out ) {
	const std::string p( prefix );
	const int k = static_cast<int>( kv.get( p + ".ssm.group_count" ) ), v = static_cast<int>( kv.get( p + ".ssm.time_step_rank" ) );
	const int inner = static_cast<int>( kv.get( p + ".ssm.inner_size" ) );
	if ( k <= 0 || v <= 0 || inner <= 0 || v % k != 0 ) return false;
	out->k_heads = v != k ? k : 0;
	out->v_heads = v;
	out->head_v_dim = inner / v;
	return true;
}

} // namespace engine::loader
