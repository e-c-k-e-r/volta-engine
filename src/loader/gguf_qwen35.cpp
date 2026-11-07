// GGUF architecture qwen35 (Qwen3.5 / Qwen3.8 dense: llama.cpp's Qwen3_5TextModel converter) -> the Hugging Face names Qwen3_5Pipeline loads.
// The converter (llama.cpp conversion/qwen.py) stores every RMSNorm weight except the linear-attention one as 1 + w, A_log as -exp(A_log),
// renames dt_bias, squeezes conv1d, and reorders the linear-attention V heads from grouped (k-head major) to tiled (v-index major) order in
// in_proj_qkv's V rows, in_proj_z, in_proj_a / b, A_log, dt_bias, conv1d's V channels and out_proj's input columns. Checked against the
// Hugging Face checkpoint tensor by tensor (a BF16 conversion of Qwen3.8-27B, every tensor of a linear-attention, a full-attention and the
// MTP layer).
#include "gguf_arch.h"
#include <cstdio>
#include <cstring>

namespace engine::loader {

namespace {

struct Suffix {
	const char* gguf;
	const char* hf;
	GgufTransform transform;
	int head_dim;    // 0: the V head dim; 1: one row per head
	bool v_part;     // RowsUntileVPart
};

// Per layer: GGUF suffix -> Hugging Face suffix under the layer's prefix
const Suffix kLayer[] = {
	{ "attn_norm.weight", "input_layernorm.weight", GgufTransform::NormMinusOne, 1, false },
	{ "post_attention_norm.weight", "post_attention_layernorm.weight", GgufTransform::NormMinusOne, 1, false },
	{ "attn_q.weight", "self_attn.q_proj.weight", GgufTransform::None, 1, false },
	{ "attn_k.weight", "self_attn.k_proj.weight", GgufTransform::None, 1, false },
	{ "attn_v.weight", "self_attn.v_proj.weight", GgufTransform::None, 1, false },
	{ "attn_output.weight", "self_attn.o_proj.weight", GgufTransform::None, 1, false },
	{ "attn_q_norm.weight", "self_attn.q_norm.weight", GgufTransform::NormMinusOne, 1, false },
	{ "attn_k_norm.weight", "self_attn.k_norm.weight", GgufTransform::NormMinusOne, 1, false },
	{ "ffn_gate.weight", "mlp.gate_proj.weight", GgufTransform::None, 1, false },
	{ "ffn_up.weight", "mlp.up_proj.weight", GgufTransform::None, 1, false },
	{ "ffn_down.weight", "mlp.down_proj.weight", GgufTransform::None, 1, false },
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

// The MTP block's own tensors (the converter's nextn layer, the last block)
const Suffix kNextn[] = {
	{ "nextn.eh_proj.weight", "mtp.fc.weight", GgufTransform::None, 1, false },
	{ "nextn.enorm.weight", "mtp.pre_fc_norm_embedding.weight", GgufTransform::NormMinusOne, 1, false },
	{ "nextn.hnorm.weight", "mtp.pre_fc_norm_hidden.weight", GgufTransform::NormMinusOne, 1, false },
	{ "nextn.shared_head_norm.weight", "mtp.norm.weight", GgufTransform::NormMinusOne, 1, false },
};

bool map_qwen35( const GgufArch& arch, const GgufKv& kv, const std::string& g, GgufMapped* out ) {
	auto fill = [&]( const std::string& name, const Suffix& s ) {
		out->name = name;
		out->transform = s.transform;
		out->head_dim = s.head_dim == 0 ? arch.head_v_dim : s.head_dim;
		out->row_offset = s.v_part ? 2 * arch.k_heads * static_cast<int>( kv.get( "qwen35.ssm.state_size" ) ) : 0;
		if ( std::strcmp( s.gguf, "ssm_conv1d.weight" ) == 0 ) out->shape = { -1, 1, -1 };   // [channels, 1, kernel] as in the checkpoint
		return true;
	};
	if ( g == "token_embd.weight" ) { out->name = "model.language_model.embed_tokens.weight"; return true; }
	if ( g == "output.weight" ) { out->name = "lm_head.weight"; return true; }
	if ( g == "output_norm.weight" ) { out->name = "model.language_model.norm.weight"; out->transform = GgufTransform::NormMinusOne; return true; }
	int layer = -1, used = 0;
	if ( std::sscanf( g.c_str(), "blk.%d.%n", &layer, &used ) != 1 || used == 0 ) return false;
	const std::string rest = g.substr( static_cast<size_t>( used ) );
	const int nextn = static_cast<int>( kv.get( "qwen35.nextn_predict_layers" ) );
	const int first_nextn = static_cast<int>( kv.get( "qwen35.block_count" ) ) - nextn;
	if ( nextn > 0 && layer >= first_nextn ) {
		if ( layer != first_nextn ) return false;   // one MTP block
		for ( const Suffix& s : kNextn ) if ( rest == s.gguf ) return fill( s.hf, s );
		for ( const Suffix& s : kLayer ) if ( rest == s.gguf ) return fill( std::string( "mtp.layers.0." ) + s.hf, s );
		return false;
	}
	for ( const Suffix& s : kLayer ) {
		if ( rest == s.gguf ) return fill( "model.language_model.layers." + std::to_string( layer ) + "." + s.hf, s );
	}
	return false;
}

} // namespace

bool gguf_arch_qwen35( const GgufKv& kv, GgufArch* out ) {
	const int k = static_cast<int>( kv.get( "qwen35.ssm.group_count" ) ), v = static_cast<int>( kv.get( "qwen35.ssm.time_step_rank" ) );
	const int inner = static_cast<int>( kv.get( "qwen35.ssm.inner_size" ) );
	if ( k <= 0 || v <= 0 || inner <= 0 || v % k != 0 ) return false;
	// the converter reorders V heads only when there are more of them than key heads
	out->k_heads = v != k ? k : 0;
	out->v_heads = v;
	out->head_v_dim = inner / v;
	out->map = map_qwen35;
	return true;
}

} // namespace engine::loader
