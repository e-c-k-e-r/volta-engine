// GGUF architecture dflash (DFlash 2 block-diffusion drafters, e.g. incoai's Qwen3.8-27B-DFlash2 in the z-lab / incoai GGUF releases) ->
// the tensor names of the drafter checkpoint DFlashSpeculator loads. Every tensor is stored as it is in the checkpoint, under a llama.cpp
// name: checked against the BF16 release tensor by tensor (a Q8_0 conversion, the global tensors and layers 0 and 4).
#include "gguf_arch.h"
#include <cstdio>

namespace engine::loader {

namespace {

struct Name {
	const char* gguf;
	const char* hf;
};

const Name kGlobal[] = {
	{ "fc.weight", "fc.weight" },
	{ "enc.output_norm.weight", "hidden_norm.weight" },
	{ "output_norm.weight", "norm.weight" },
	{ "selector_hidden.weight", "candidate_selector.hidden_projection.weight" },
	{ "selector_predecessor.weight", "candidate_selector.predecessor_codebook" },
	{ "selector_successor.weight", "candidate_selector.successor_codebook" },
};

// Per layer, under layers.N.
const Name kLayer[] = {
	{ "attn_norm.weight", "input_layernorm.weight" },
	{ "ffn_norm.weight", "post_attention_layernorm.weight" },
	{ "attn_q.weight", "self_attn.q_proj.weight" },
	{ "attn_k.weight", "self_attn.k_proj.weight" },
	{ "attn_v.weight", "self_attn.v_proj.weight" },
	{ "attn_output.weight", "self_attn.o_proj.weight" },
	{ "attn_q_norm.weight", "self_attn.q_norm.weight" },
	{ "attn_k_norm.weight", "self_attn.k_norm.weight" },
	{ "ffn_gate.weight", "mlp.gate_proj.weight" },
	{ "ffn_up.weight", "mlp.up_proj.weight" },
	{ "ffn_down.weight", "mlp.down_proj.weight" },
	{ "attn_conv_base", "attention_conv.base_kernel" },
	{ "attn_conv_proj.weight", "attention_conv.kernel_projection.weight" },
	{ "ffn_conv_base", "mlp_conv.base_kernel" },
	{ "ffn_conv_proj.weight", "mlp_conv.kernel_projection.weight" },
};

void map_dflash( const GgufArch&, const GgufKv&, const std::string& g, std::vector<GgufMapped>* out ) {
	auto emit = [&]( const std::string& name ) {
		GgufMapped m;
		m.name = name;
		out->push_back( std::move( m ) );
	};
	for ( const Name& n : kGlobal ) {
		if ( g == n.gguf ) { emit( n.hf ); return; }
	}
	int layer = -1, used = 0;
	if ( std::sscanf( g.c_str(), "blk.%d.%n", &layer, &used ) != 1 || used == 0 ) return;
	const std::string rest = g.substr( static_cast<size_t>( used ) );
	for ( const Name& n : kLayer ) {
		if ( rest == n.gguf ) { emit( "layers." + std::to_string( layer ) + "." + n.hf ); return; }
	}
}

} // namespace

bool gguf_arch_dflash( const GgufKv&, GgufArch* out ) {
	out->map = map_dflash;
	return true;
}

} // namespace engine::loader
