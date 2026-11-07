// GGUF architecture qwen35 (Qwen3.5 / Qwen3.8 dense: llama.cpp's Qwen3_5TextModel converter) -> the Hugging Face names Qwen3_5Pipeline loads.
// The converter (llama.cpp conversion/qwen.py) stores every RMSNorm weight except the linear-attention one as 1 + w, A_log as -exp(A_log),
// renames dt_bias, squeezes conv1d, and reorders the linear-attention V heads from grouped (k-head major) to tiled (v-index major) order in
// in_proj_qkv's V rows, in_proj_z, in_proj_a / b, A_log, dt_bias, conv1d's V channels and out_proj's input columns. Checked against the
// Hugging Face checkpoint tensor by tensor (a BF16 conversion of Qwen3.8-27B, every tensor of a linear-attention, a full-attention and the
// MTP layer).
#include "gguf_qwen_linear.h"
#include <cstdio>

namespace engine::loader {

namespace {

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
};

// The MTP block's own tensors (the converter's nextn layer, the last block)
const Suffix kNextn[] = {
	{ "nextn.eh_proj.weight", "mtp.fc.weight", GgufTransform::None, 1, false },
	{ "nextn.enorm.weight", "mtp.pre_fc_norm_embedding.weight", GgufTransform::NormMinusOne, 1, false },
	{ "nextn.hnorm.weight", "mtp.pre_fc_norm_hidden.weight", GgufTransform::NormMinusOne, 1, false },
	{ "nextn.shared_head_norm.weight", "mtp.norm.weight", GgufTransform::NormMinusOne, 1, false },
};

bool map_one( const GgufArch& arch, const GgufKv& kv, const std::string& g, GgufMapped* out ) {
	auto fill = [&]( const std::string& name, const Suffix& s ) { return fill_suffix( arch, kv, "qwen35", name, s, out ); };
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
		for ( const Suffix& s : kLinearAttn ) if ( rest == s.gguf ) return fill( std::string( "mtp.layers.0." ) + s.hf, s );
		return false;
	}
	const std::string pre = "model.language_model.layers." + std::to_string( layer ) + ".";
	for ( const Suffix& s : kLayer ) if ( rest == s.gguf ) return fill( pre + s.hf, s );
	for ( const Suffix& s : kLinearAttn ) if ( rest == s.gguf ) return fill( pre + s.hf, s );
	return false;
}

void map_qwen35( const GgufArch& arch, const GgufKv& kv, const std::string& g, std::vector<GgufMapped>* out ) {
	GgufMapped m;
	if ( map_one( arch, kv, g, &m ) ) out->push_back( std::move( m ) );
}

} // namespace

bool gguf_arch_qwen35( const GgufKv& kv, GgufArch* out ) {
	if ( !linear_attn_heads( kv, "qwen35", out ) ) return false;
	out->map = map_qwen35;
	return true;
}

} // namespace engine::loader
