// GGUF architecture clip (llama.cpp's mmproj files) for the projector type gemma4v: Gemma 4's vision tower and embedding projection
// under the Hugging Face names Gemma4VisionTower loads. The converter keeps every tensor as it is (norms F32, projections BF16) except the
// patch embedding, which it stores as a conv kernel [n, channel, y, x]: the PatchChwToHwc transform restores [n, y * x * channel].
#include "gguf_arch.h"
#include <cstdio>

namespace engine::loader {

namespace {

struct Name {
	const char* gguf;
	const char* hf;
};

// Per block, under model.vision_tower.encoder.layers.N.
const Name kBlock[] = {
	{ "ln1.weight", "input_layernorm.weight" },
	{ "ln2.weight", "pre_feedforward_layernorm.weight" },
	{ "attn_post_norm.weight", "post_attention_layernorm.weight" },
	{ "ffn_post_norm.weight", "post_feedforward_layernorm.weight" },
	{ "attn_q.weight", "self_attn.q_proj.linear.weight" },
	{ "attn_k.weight", "self_attn.k_proj.linear.weight" },
	{ "attn_v.weight", "self_attn.v_proj.linear.weight" },
	{ "attn_out.weight", "self_attn.o_proj.linear.weight" },
	{ "attn_q_norm.weight", "self_attn.q_norm.weight" },
	{ "attn_k_norm.weight", "self_attn.k_norm.weight" },
	{ "ffn_gate.weight", "mlp.gate_proj.linear.weight" },
	{ "ffn_up.weight", "mlp.up_proj.linear.weight" },
	{ "ffn_down.weight", "mlp.down_proj.linear.weight" },
};

void map_gemma4v( const GgufArch&, const GgufKv&, const std::string& g, std::vector<GgufMapped>* out ) {
	auto emit = [&]( const std::string& name, GgufTransform tr = GgufTransform::None ) {
		GgufMapped m;
		m.name = name;
		m.transform = tr;
		out->push_back( std::move( m ) );
	};
	if ( g == "v.patch_embd.weight" ) { emit( "model.vision_tower.patch_embedder.input_proj.weight", GgufTransform::PatchChwToHwc ); return; }
	if ( g == "v.position_embd.weight" ) { emit( "model.vision_tower.patch_embedder.position_embedding_table" ); return; }
	if ( g == "v.std_bias" ) { emit( "model.vision_tower.std_bias" ); return; }
	if ( g == "v.std_scale" ) { emit( "model.vision_tower.std_scale" ); return; }
	if ( g == "mm.input_projection.weight" ) { emit( "model.embed_vision.embedding_projection.weight" ); return; }
	int layer = -1, used = 0;
	if ( std::sscanf( g.c_str(), "v.blk.%d.%n", &layer, &used ) != 1 || used == 0 ) return;
	const std::string rest = g.substr( static_cast<size_t>( used ) );
	for ( const Name& n : kBlock ) {
		if ( rest == n.gguf ) { emit( "model.vision_tower.encoder.layers." + std::to_string( layer ) + "." + n.hf ); return; }
	}
}

} // namespace

bool gguf_arch_clip( const GgufKv& kv, GgufArch* out ) {
	const auto it = kv.strings.find( "clip.vision.projector_type" );
	if ( it == kv.strings.end() || it->second != "gemma4v" ) return false;
	out->map = map_gemma4v;
	return true;
}

} // namespace engine::loader
