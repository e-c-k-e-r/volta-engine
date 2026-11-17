// GGUF architecture clip (llama.cpp's mmproj files) for the projector types gemma4v (Gemma 4's vision tower and embedding projection under the
// Hugging Face names Gemma4VisionTower loads), glm5v (GLM-5's tower and merger under the checkpoint's model.visual.* names) and deepseek41v
// (DeepSeek-V4.1's tower, aligner and delimiter embeddings under the checkpoint's vision.* / aligner.* names) and qwen3vl_merger (the Qwen3-VL tower
// of Flash-Next under model.visual.*). The converter keeps every tensor as it is (norms F32, projections BF16) except the
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

// GLM-5 (glm5v): the converter keeps the tower as the checkpoint has it (norms and biases F32, projections BF16) except the fused qkv, which stays
// fused (the weight is cut into q / k / v row ranges here; the tower slices the fused bias), and the patch embedding, which it cuts along the
// temporal axis into two conv kernels [n, channel, y, x] (.t0 and .t1 here; the tower joins them into [n, channel, t, y, x])
const Name kGlmBlock[] = {
	{ "ln1.weight", "norm1.weight" },
	{ "ln2.weight", "norm2.weight" },
	{ "attn_q_norm.weight", "attn.q_norm.weight" },
	{ "attn_k_norm.weight", "attn.k_norm.weight" },
	{ "attn_qkv.bias", "attn.qkv.bias" },
	{ "attn_out.weight", "attn.proj.weight" },
	{ "attn_out.bias", "attn.proj.bias" },
	{ "ffn_gate.weight", "mlp.gate_proj.weight" },
	{ "ffn_gate.bias", "mlp.gate_proj.bias" },
	{ "ffn_up.weight", "mlp.up_proj.weight" },
	{ "ffn_up.bias", "mlp.up_proj.bias" },
	{ "ffn_down.weight", "mlp.down_proj.weight" },
	{ "ffn_down.bias", "mlp.down_proj.bias" },
};

const Name kGlmTop[] = {
	{ "v.patch_embd.weight", "model.visual.patch_embed.proj.weight.t0" },
	{ "v.patch_embd.weight.1", "model.visual.patch_embed.proj.weight.t1" },
	{ "v.patch_embd.bias", "model.visual.patch_embed.proj.bias" },
	{ "v.post_ln.weight", "model.visual.post_layernorm.weight" },
	{ "mm.patch_merger.weight", "model.visual.downsample.weight" },
	{ "mm.patch_merger.bias", "model.visual.downsample.bias" },
	{ "mm.model.fc.weight", "model.visual.merger.proj.weight" },
	{ "mm.post_norm.weight", "model.visual.merger.post_projection_norm.weight" },
	{ "mm.post_norm.bias", "model.visual.merger.post_projection_norm.bias" },
	{ "mm.gate.weight", "model.visual.merger.gate_proj.weight" },
	{ "mm.up.weight", "model.visual.merger.up_proj.weight" },
	{ "mm.down.weight", "model.visual.merger.down_proj.weight" },
};

void map_glm5v( const GgufArch&, const GgufKv& kv, const std::string& g, std::vector<GgufMapped>* out ) {
	auto emit = [&]( const std::string& name, int64_t row_begin = 0, int64_t rows = -1 ) {
		GgufMapped m;
		m.name = name;
		m.row_begin = row_begin;
		m.rows = rows;
		out->push_back( std::move( m ) );
	};
	for ( const Name& n : kGlmTop ) {
		if ( g == n.gguf ) { emit( n.hf ); return; }
	}
	int layer = -1, used = 0;
	if ( std::sscanf( g.c_str(), "v.blk.%d.%n", &layer, &used ) != 1 || used == 0 ) return;
	const std::string rest = g.substr( static_cast<size_t>( used ) ), base = "model.visual.blocks." + std::to_string( layer ) + ".";
	if ( rest == "attn_qkv.weight" ) {
		const int64_t hidden = kv.get( "clip.vision.embedding_length" );
		if ( hidden <= 0 ) return;
		emit( base + "attn.q_proj.weight", 0, hidden );
		emit( base + "attn.k_proj.weight", hidden, hidden );
		emit( base + "attn.v_proj.weight", 2 * hidden, hidden );
		return;
	}
	for ( const Name& n : kGlmBlock ) {
		if ( rest == n.gguf ) { emit( base + n.hf ); return; }
	}
}

// DeepSeek-V4.1 (deepseek41v): the tower as the checkpoint has it: a fused qkv (weight cut into q / k / v row ranges, the fused bias kept and sliced by
// the tower), the attention output with its bias, the gate / up / down projections without biases, the aligner's two linears (mm.1, mm.2) and the
// three delimiter embeddings of an image span
const Name kDsBlock[] = {
	{ "ln1.weight", "norm1.weight" },
	{ "ln2.weight", "norm2.weight" },
	{ "attn_qkv.bias", "attn.qkv.bias" },
	{ "attn_out.weight", "attn.wo.weight" },
	{ "attn_out.bias", "attn.wo.bias" },
	{ "ffn_gate.weight", "mlp.gate_proj.weight" },
	{ "ffn_up.weight", "mlp.up_proj.weight" },
	{ "ffn_down.weight", "mlp.w2.weight" },
};

const Name kDsTop[] = {
	{ "v.patch_embd.weight", "vision.patch_embed.proj.weight" },
	{ "v.patch_embd.bias", "vision.patch_embed.proj.bias" },
	{ "v.post_ln.weight", "vision.norm.weight" },
	{ "mm.1.weight", "aligner.w1.weight" },
	{ "mm.1.bias", "aligner.w1.bias" },
	{ "mm.2.weight", "aligner.w2.weight" },
	{ "mm.2.bias", "aligner.w2.bias" },
	{ "v.token_embd.img_start", "image_start" },
	{ "v.token_embd.img_end", "image_end" },
	{ "v.image_newline", "image_newline" },
};

void map_deepseek41v( const GgufArch&, const GgufKv& kv, const std::string& g, std::vector<GgufMapped>* out ) {
	auto emit = [&]( const std::string& name, int64_t row_begin = 0, int64_t rows = -1 ) {
		GgufMapped m;
		m.name = name;
		m.row_begin = row_begin;
		m.rows = rows;
		out->push_back( std::move( m ) );
	};
	for ( const Name& n : kDsTop ) {
		if ( g == n.gguf ) { emit( n.hf ); return; }
	}
	int layer = -1, used = 0;
	if ( std::sscanf( g.c_str(), "v.blk.%d.%n", &layer, &used ) != 1 || used == 0 ) return;
	const std::string rest = g.substr( static_cast<size_t>( used ) ), base = "vision.blocks." + std::to_string( layer ) + ".";
	if ( rest == "attn_qkv.weight" ) {
		const int64_t hidden = kv.get( "clip.vision.embedding_length" );
		if ( hidden <= 0 ) return;
		emit( base + "attn.q_proj.weight", 0, hidden );
		emit( base + "attn.k_proj.weight", hidden, hidden );
		emit( base + "attn.v_proj.weight", 2 * hidden, hidden );
		return;
	}
	for ( const Name& n : kDsBlock ) {
		if ( rest == n.gguf ) { emit( base + n.hf ); return; }
	}
}

// Qwen3-VL tower (qwen3vl_merger): the tower as the checkpoint has it, with the fused qkv (weight cut into q / k / v row ranges, the fused bias
// kept and sliced by the tower), the patch embedding cut along the temporal axis into two conv kernels (.t0 and .t1 here; the tower joins them),
// the position table, and the merger's norm and two linears (mm.0, mm.2)
const Name kQwenBlock[] = {
	{ "ln1.weight", "norm1.weight" },
	{ "ln1.bias", "norm1.bias" },
	{ "ln2.weight", "norm2.weight" },
	{ "ln2.bias", "norm2.bias" },
	{ "attn_qkv.bias", "attn.qkv.bias" },
	{ "attn_out.weight", "attn.proj.weight" },
	{ "attn_out.bias", "attn.proj.bias" },
	{ "ffn_up.weight", "mlp.linear_fc1.weight" },
	{ "ffn_up.bias", "mlp.linear_fc1.bias" },
	{ "ffn_down.weight", "mlp.linear_fc2.weight" },
	{ "ffn_down.bias", "mlp.linear_fc2.bias" },
};

const Name kQwenTop[] = {
	{ "v.patch_embd.weight", "model.visual.patch_embed.proj.weight.t0" },
	{ "v.patch_embd.weight.1", "model.visual.patch_embed.proj.weight.t1" },
	{ "v.patch_embd.bias", "model.visual.patch_embed.proj.bias" },
	{ "v.position_embd.weight", "model.visual.pos_embed.weight" },
	{ "v.post_ln.weight", "model.visual.merger.norm.weight" },
	{ "v.post_ln.bias", "model.visual.merger.norm.bias" },
	{ "mm.0.weight", "model.visual.merger.linear_fc1.weight" },
	{ "mm.0.bias", "model.visual.merger.linear_fc1.bias" },
	{ "mm.2.weight", "model.visual.merger.linear_fc2.weight" },
	{ "mm.2.bias", "model.visual.merger.linear_fc2.bias" },
};

void map_qwen3vl( const GgufArch&, const GgufKv& kv, const std::string& g, std::vector<GgufMapped>* out ) {
	auto emit = [&]( const std::string& name, int64_t row_begin = 0, int64_t rows = -1 ) {
		GgufMapped m;
		m.name = name;
		m.row_begin = row_begin;
		m.rows = rows;
		out->push_back( std::move( m ) );
	};
	for ( const Name& n : kQwenTop ) {
		if ( g == n.gguf ) { emit( n.hf ); return; }
	}
	int layer = -1, used = 0;
	if ( std::sscanf( g.c_str(), "v.blk.%d.%n", &layer, &used ) != 1 || used == 0 ) return;
	const std::string rest = g.substr( static_cast<size_t>( used ) ), base = "model.visual.blocks." + std::to_string( layer ) + ".";
	if ( rest == "attn_qkv.weight" ) {
		const int64_t hidden = kv.get( "clip.vision.embedding_length" );
		if ( hidden <= 0 ) return;
		emit( base + "attn.q_proj.weight", 0, hidden );
		emit( base + "attn.k_proj.weight", hidden, hidden );
		emit( base + "attn.v_proj.weight", 2 * hidden, hidden );
		return;
	}
	for ( const Name& n : kQwenBlock ) {
		if ( rest == n.gguf ) { emit( base + n.hf ); return; }
	}
}

} // namespace

bool gguf_arch_clip( const GgufKv& kv, GgufArch* out ) {
	std::string type;
	for ( const char* key : { "clip.vision.projector_type", "clip.projector_type" } ) {
		const auto it = kv.strings.find( key );
		if ( it != kv.strings.end() ) { type = it->second; break; }
	}
	if ( type == "gemma4v" ) out->map = map_gemma4v;
	else if ( type == "glm5v" ) out->map = map_glm5v;
	else if ( type == "deepseek41v" ) out->map = map_deepseek41v;
	else if ( type == "qwen3vl_merger" ) out->map = map_qwen3vl;
	else return false;
	return true;
}

} // namespace engine::loader
