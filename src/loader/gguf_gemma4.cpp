// GGUF architecture gemma4 (Gemma 4 dense text models: llama.cpp's Gemma4Model converter) -> the Hugging Face names Gemma4Pipeline loads.
// Every tensor is stored as it is in the checkpoint (the RMSNorm weights without an offset); the tied head is the token embedding, so
// token_embd also maps to lm_head. rope_freqs (the proportional RoPE's frequency mask) is not used: the pipeline rotates the first pairs only.
#include "gguf_arch.h"
#include <cstdio>

namespace engine::loader {

namespace {

struct Name {
	const char* gguf;
	const char* hf;
};

// Per layer, under model.language_model.layers.N.
const Name kLayer[] = {
	{ "attn_norm.weight", "input_layernorm.weight" },
	{ "post_attention_norm.weight", "post_attention_layernorm.weight" },
	{ "ffn_norm.weight", "pre_feedforward_layernorm.weight" },
	{ "post_ffw_norm.weight", "post_feedforward_layernorm.weight" },
	{ "layer_output_scale.weight", "layer_scalar" },
	{ "attn_q.weight", "self_attn.q_proj.weight" },
	{ "attn_k.weight", "self_attn.k_proj.weight" },
	{ "attn_v.weight", "self_attn.v_proj.weight" },
	{ "attn_output.weight", "self_attn.o_proj.weight" },
	{ "attn_q_norm.weight", "self_attn.q_norm.weight" },
	{ "attn_k_norm.weight", "self_attn.k_norm.weight" },
	{ "ffn_gate.weight", "mlp.gate_proj.weight" },
	{ "ffn_up.weight", "mlp.up_proj.weight" },
	{ "ffn_down.weight", "mlp.down_proj.weight" },
};

void map_gemma4( const GgufArch&, const GgufKv&, const std::string& g, std::vector<GgufMapped>* out ) {
	auto emit = [&]( const std::string& name ) {
		GgufMapped m;
		m.name = name;
		out->push_back( std::move( m ) );
	};
	if ( g == "token_embd.weight" ) {
		emit( "model.language_model.embed_tokens.weight" );
		emit( "lm_head.weight" );
		return;
	}
	if ( g == "output.weight" ) { emit( "lm_head.weight" ); return; }
	if ( g == "output_norm.weight" ) { emit( "model.language_model.norm.weight" ); return; }
	int layer = -1, used = 0;
	if ( std::sscanf( g.c_str(), "blk.%d.%n", &layer, &used ) != 1 || used == 0 ) return;
	const std::string rest = g.substr( static_cast<size_t>( used ) );
	for ( const Name& n : kLayer ) {
		if ( rest == n.gguf ) { emit( "model.language_model.layers." + std::to_string( layer ) + "." + n.hf ); return; }
	}
}

} // namespace

bool gguf_arch_gemma4( const GgufKv&, GgufArch* out ) {
	out->map = map_gemma4;
	return true;
}

} // namespace engine::loader
