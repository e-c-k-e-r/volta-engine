// GGUF architecture glm5-next (GLM-5.3-Flash: llama.cpp's Glm5NextModel converter, conversion/glm.py) -> the Hugging Face names
// Glm5NextPipeline loads, in the layout of the engine's EXL3 conversion: the KDA layers' q / k / v projections and their three short
// convolutions are one qkv_proj and one conv1d (q, k, v in that order). The converter stacks the routed experts into one tensor per
// projection ([experts, n, k]), splits every MLA layer's kv_b_proj into per-head k_b (transposed) and v_b, stores A_log as -exp(A_log),
// renames dt_bias to dt_proj.bias and gives the hyper-connection and k-pool tensors a .weight suffix; the norms are stored as they are.
// No multi-token-prediction block is read.
#include "gguf_arch.h"
#include <cstdio>

namespace engine::loader {

namespace {

const char* const P = "model.language_model";

struct Simple {
	const char* gguf;
	const char* hf;
	GgufTransform transform;
};

// Per layer, under the layer's prefix
const Simple kLayer[] = {
	{ "attn_norm.weight", "input_layernorm.weight", GgufTransform::None },
	{ "ffn_norm.weight", "post_attention_layernorm.weight", GgufTransform::None },
	{ "hc_attn_base.weight", "hc_attn_base", GgufTransform::None },
	{ "hc_attn_fn.weight", "hc_attn_fn", GgufTransform::None },
	{ "hc_attn_scale.weight", "hc_attn_scale", GgufTransform::None },
	{ "hc_ffn_base.weight", "hc_ffn_base", GgufTransform::None },
	{ "hc_ffn_fn.weight", "hc_ffn_fn", GgufTransform::None },
	{ "hc_ffn_scale.weight", "hc_ffn_scale", GgufTransform::None },
	{ "ffn_gate.weight", "mlp.gate_proj.weight", GgufTransform::None },
	{ "ffn_up.weight", "mlp.up_proj.weight", GgufTransform::None },
	{ "ffn_down.weight", "mlp.down_proj.weight", GgufTransform::None },
	// KDA
	{ "ssm_a", "self_attn.A_log", GgufTransform::ALogFromNegExp },
	{ "ssm_dt.bias", "self_attn.dt_bias", GgufTransform::None },
	{ "ssm_beta.weight", "self_attn.b_proj.weight", GgufTransform::None },
	{ "ssm_f_a.weight", "self_attn.f_a_proj.weight", GgufTransform::None },
	{ "ssm_f_b.weight", "self_attn.f_b_proj.weight", GgufTransform::None },
	{ "ssm_g_a.weight", "self_attn.g_a_proj.weight", GgufTransform::None },
	{ "ssm_g_b.weight", "self_attn.g_b_proj.weight", GgufTransform::None },
	{ "ssm_norm.weight", "self_attn.o_norm.weight", GgufTransform::None },
	{ "attn_output.weight", "self_attn.o_proj.weight", GgufTransform::None },
	// MLA
	{ "attn_q_a.weight", "self_attn.q_a_proj.weight", GgufTransform::None },
	{ "attn_q_a_norm.weight", "self_attn.q_a_layernorm.weight", GgufTransform::None },
	{ "attn_q_b.weight", "self_attn.q_b_proj.weight", GgufTransform::None },
	{ "attn_kv_a_mqa.weight", "self_attn.kv_a_proj_with_mqa.weight", GgufTransform::None },
	{ "attn_kv_a_norm.weight", "self_attn.kv_a_layernorm.weight", GgufTransform::None },
	// DSA indexer
	{ "indexer.attn_q_b.weight", "self_attn.indexer.wq_b.weight", GgufTransform::None },
	{ "indexer.attn_k.weight", "self_attn.indexer.wk.weight", GgufTransform::None },
	{ "indexer.proj.weight", "self_attn.indexer.weights_proj.weight", GgufTransform::None },
	{ "indexer.k_norm.weight", "self_attn.indexer.k_norm.weight", GgufTransform::None },
	{ "indexer.k_norm.bias", "self_attn.indexer.k_norm.bias", GgufTransform::None },
	{ "indexer_compressor_ape.weight", "self_attn.indexer.index_kpool_compress_ape", GgufTransform::None },
	{ "indexer_compressor_gate.weight", "self_attn.indexer.index_kpool_compress_gate", GgufTransform::None },
	// MoE
	{ "ffn_gate_inp.weight", "mlp.gate.weight", GgufTransform::None },
	{ "exp_probs_b.bias", "mlp.gate.e_score_correction_bias", GgufTransform::None },
	{ "ffn_gate_shexp.weight", "mlp.shared_experts.gate_proj.weight", GgufTransform::None },
	{ "ffn_up_shexp.weight", "mlp.shared_experts.up_proj.weight", GgufTransform::None },
	{ "ffn_down_shexp.weight", "mlp.shared_experts.down_proj.weight", GgufTransform::None },
};

GgufMapped simple( const std::string& name, GgufTransform t ) {
	GgufMapped m;
	m.name = name;
	m.transform = t;
	return m;
}

void map_glm5next( const GgufArch&, const GgufKv& kv, const std::string& g, std::vector<GgufMapped>* out ) {
	const std::string p( P );
	if ( g == "token_embd.weight" ) { out->push_back( simple( p + ".embed_tokens.weight", GgufTransform::None ) ); return; }
	if ( g == "output.weight" ) { out->push_back( simple( "lm_head.weight", GgufTransform::None ) ); return; }
	if ( g == "output_norm.weight" ) { out->push_back( simple( p + ".norm.weight", GgufTransform::None ) ); return; }
	int layer = -1, used = 0;
	if ( std::sscanf( g.c_str(), "blk.%d.%n", &layer, &used ) != 1 || used == 0 ) return;
	if ( layer >= kv.get( "glm5-next.block_count" ) - kv.get( "glm5-next.nextn_predict_layers" ) ) return;   // an MTP block: not read
	const std::string rest = g.substr( static_cast<size_t>( used ) ), pre = p + ".layers." + std::to_string( layer ) + ".";
	for ( const Simple& s : kLayer ) {
		if ( rest == s.gguf ) { out->push_back( simple( pre + s.hf, s.transform ) ); return; }
	}
	// KDA: q, k and v (and their convolutions) are the parts of one tensor, in that order
	const char* const qkv[3] = { "q", "k", "v" };
	for ( int i = 0; i < 3; ++i ) {
		if ( rest == std::string( "attn_" ) + qkv[i] + ".weight" ) {
			GgufMapped m = simple( pre + "self_attn.qkv_proj.weight", GgufTransform::None );
			m.part = i;
			m.parts = 3;
			out->push_back( std::move( m ) );
			return;
		}
		if ( rest == std::string( "ssm_conv1d_" ) + qkv[i] + ".weight" ) {
			GgufMapped m = simple( pre + "self_attn.conv1d.weight", GgufTransform::None );
			// [channels, 1, kernel] (the converter stored [1, channels, 1, kernel])
			m.shape = { static_cast<int>( kv.get( "glm5-next.attention.head_count" ) * kv.get( "glm5-next.kda.head_dim" ) ), 1, static_cast<int>( kv.get( "glm5-next.ssm.conv_kernel" ) ) };
			m.part = i;
			m.parts = 3;
			out->push_back( std::move( m ) );
			return;
		}
	}
	// MLA: kv_b_proj is head by head [k_b^T | v_b] ([qk_nope + v, kv_lora] rows per head)
	if ( rest == "attn_k_b.weight" || rest == "attn_v_b.weight" ) {
		const bool k = rest == "attn_k_b.weight";
		GgufMapped m = simple( pre + "self_attn.kv_b_proj.weight", k ? GgufTransform::TransposeHeads : GgufTransform::None );
		m.part = k ? 0 : 1;
		m.parts = 2;
		const int heads = static_cast<int>( kv.get( "glm5-next.attention.head_count" ) );
		m.interleave_heads = heads;
		m.shape = { heads * static_cast<int>( kv.get( k ? "glm5-next.attention.key_length_mla" : "glm5-next.attention.value_length_mla" ) ),
			static_cast<int>( kv.get( "glm5-next.attention.kv_lora_rank" ) ) };
		out->push_back( std::move( m ) );
		return;
	}
	// routed experts: expert e is rows [e n, (e + 1) n) of the stacked tensor, n its projection's output width
	const int64_t experts = kv.get( "glm5-next.expert_count" ), inter = kv.get( "glm5-next.expert_feed_forward_length" ), hidden = kv.get( "glm5-next.embedding_length" );
	const struct { const char* gguf; const char* hf; int64_t n; } kExperts[] = {
		{ "ffn_gate_exps.weight", "gate_proj", inter }, { "ffn_up_exps.weight", "up_proj", inter }, { "ffn_down_exps.weight", "down_proj", hidden } };
	for ( const auto& x : kExperts ) {
		if ( rest != x.gguf ) continue;
		for ( int64_t e = 0; e < experts; ++e ) {
			GgufMapped m = simple( pre + "mlp.experts." + std::to_string( e ) + "." + x.hf + ".weight", GgufTransform::None );
			m.row_begin = e * x.n;
			m.rows = x.n;
			out->push_back( std::move( m ) );
		}
		return;
	}
}

} // namespace

bool gguf_arch_glm5next( const GgufKv&, GgufArch* out ) {
	out->map = map_glm5next;
	return true;
}

} // namespace engine::loader
