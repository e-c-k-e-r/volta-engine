// GGUF architecture clip (llama.cpp's mmproj files) for the projector types gemma4v (Gemma 4's vision tower and embedding projection, and in the
// same file the gemma4a audio tower and its projection, under the Hugging Face names Gemma4VisionTower and Gemma4AudioTower load), glm5v (GLM-5's tower and merger under the checkpoint's model.visual.* names) and deepseek41v
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

// Gemma 4's audio tower (gemma4a, in the same file as the vision tower), per block under model.audio_tower.layers.N.: the clipped projections
// (their weights under .linear., their ranges beside) and the rest. The converter stores softplus( per_dim_scale ), named apart here
const Name kAudioLinear[] = {
	{ "ffn_up", "feed_forward1.ffw_layer_1" },
	{ "ffn_down", "feed_forward1.ffw_layer_2" },
	{ "ffn_up_1", "feed_forward2.ffw_layer_1" },
	{ "ffn_down_1", "feed_forward2.ffw_layer_2" },
	{ "attn_q", "self_attn.q_proj" },
	{ "attn_k", "self_attn.k_proj" },
	{ "attn_v", "self_attn.v_proj" },
	{ "attn_out", "self_attn.post" },
	{ "conv_pw1", "lconv1d.linear_start" },
	{ "conv_pw2", "lconv1d.linear_end" },
};
const Name kAudioBlock[] = {
	{ "ffn_norm.weight", "feed_forward1.pre_layer_norm.weight" },
	{ "ffn_post_norm.weight", "feed_forward1.post_layer_norm.weight" },
	{ "ffn_norm_1.weight", "feed_forward2.pre_layer_norm.weight" },
	{ "ffn_post_norm_1.weight", "feed_forward2.post_layer_norm.weight" },
	{ "attn_pre_norm.weight", "norm_pre_attn.weight" },
	{ "attn_post_norm.weight", "norm_post_attn.weight" },
	{ "attn_k_rel.weight", "self_attn.relative_k_proj.weight" },
	{ "per_dim_scale.weight", "self_attn.per_dim_scale_softplus" },
	{ "conv_norm.weight", "lconv1d.pre_layer_norm.weight" },
	{ "conv_dw.weight", "lconv1d.depthwise_conv1d.weight" },
	{ "norm_conv.weight", "lconv1d.conv_norm.weight" },
	{ "ln2.weight", "norm_out.weight" },
};
const Name kAudioTop[] = {
	{ "a.conv1d.0.weight", "model.audio_tower.subsample_conv_projection.layer0.conv.weight" },
	{ "a.conv1d.0.norm.weight", "model.audio_tower.subsample_conv_projection.layer0.norm.weight" },
	{ "a.conv1d.1.weight", "model.audio_tower.subsample_conv_projection.layer1.conv.weight" },
	{ "a.conv1d.1.norm.weight", "model.audio_tower.subsample_conv_projection.layer1.norm.weight" },
	{ "a.input_projection.weight", "model.audio_tower.subsample_conv_projection.input_proj_linear.weight" },
	{ "a.pre_encode.out.weight", "model.audio_tower.output_proj.weight" },
	{ "a.pre_encode.out.bias", "model.audio_tower.output_proj.bias" },
	{ "mm.a.input_projection.weight", "model.embed_audio.embedding_projection.weight" },
};
const char* const kClipSuffix[] = { "input_min", "input_max", "output_min", "output_max" };

// A clipped projection's tensor: <gguf>.weight is <hf>.linear.weight, <gguf>.<range> is <hf>.<range>
bool map_clipped( const std::string& rest, const std::string& gguf, const std::string& hf, std::string* name ) {
	if ( rest == gguf + ".weight" ) {
		*name = hf + ".linear.weight";
		return true;
	}
	for ( const char* sfx : kClipSuffix ) {
		if ( rest == gguf + "." + sfx ) {
			*name = hf + "." + sfx;
			return true;
		}
	}
	return false;
}

void map_gemma4v( const GgufArch&, const GgufKv&, const std::string& g, std::vector<GgufMapped>* out ) {
	auto emit = [&]( const std::string& name, GgufTransform tr = GgufTransform::None ) {
		GgufMapped m;
		m.name = name;
		m.transform = tr;
		out->push_back( std::move( m ) );
	};
	for ( const Name& n : kAudioTop ) {
		if ( g == n.gguf ) { emit( n.hf ); return; }
	}
	{
		int layer = -1, used = 0;
		if ( std::sscanf( g.c_str(), "a.blk.%d.%n", &layer, &used ) == 1 && used > 0 ) {
			const std::string rest = g.substr( static_cast<size_t>( used ) ), base = "model.audio_tower.layers." + std::to_string( layer ) + ".";
			std::string name;
			for ( const Name& n : kAudioLinear ) {
				if ( map_clipped( rest, n.gguf, n.hf, &name ) ) { emit( base + name ); return; }
			}
			for ( const Name& n : kAudioBlock ) {
				if ( rest == n.gguf ) { emit( base + n.hf ); return; }
			}
			return;
		}
	}
	if ( g == "v.patch_embd.weight" ) { emit( "model.vision_tower.patch_embedder.input_proj.weight", GgufTransform::PatchChwToHwc ); return; }
	if ( g == "v.position_embd.weight" ) { emit( "model.vision_tower.patch_embedder.position_embedding_table" ); return; }
	if ( g == "v.std_bias" ) { emit( "model.vision_tower.std_bias" ); return; }
	if ( g == "v.std_scale" ) { emit( "model.vision_tower.std_scale" ); return; }
	if ( g == "mm.input_projection.weight" ) { emit( "model.embed_vision.embedding_projection.weight" ); return; }
	int layer = -1, used = 0;
	if ( std::sscanf( g.c_str(), "v.blk.%d.%n", &layer, &used ) != 1 || used == 0 ) return;
	const std::string rest = g.substr( static_cast<size_t>( used ) ), base = "model.vision_tower.encoder.layers." + std::to_string( layer ) + ".";
	for ( const Name& n : kBlock ) {
		if ( rest == n.gguf ) { emit( base + n.hf ); return; }
	}
	// the clipped projections' ranges (the small models)
	for ( const Name& n : kBlock ) {
		const std::string gw = n.gguf, hw = n.hf;
		const size_t gs = gw.rfind( ".weight" ), hs = hw.rfind( ".linear.weight" );
		if ( gs == std::string::npos || hs == std::string::npos ) continue;
		for ( const char* sfx : kClipSuffix ) {
			if ( rest == gw.substr( 0, gs ) + "." + sfx ) { emit( base + hw.substr( 0, hs ) + "." + sfx ); return; }
		}
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
