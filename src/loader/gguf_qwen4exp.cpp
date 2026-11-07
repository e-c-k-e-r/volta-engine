// GGUF architecture qwen4exp (Qwen3.8-Flash-Next: llama.cpp's Qwen4ExpTextModel converter, conversion/qwen4exp.py) -> the Hugging Face
// names Qwen4ExpPipeline loads. On top of the Qwen3.5 linear-attention transforms (gguf_qwen_linear.h) the converter stores the
// hyper-connection, attention q / k, indexer and PLE norms as 1 + w, splits the indexer's index_qk_proj into q and k rows, squeezes the
// PLE conv1d, stacks the routed experts into one tensor per projection ([experts, n, k]; gate and up out of the fused gate_up_proj),
// joins the n-gram table's shards into one table, and may merge weights that read the same input ([down | inject] of a hyper-connection
// site, [qkv | z] and [beta | alpha] of a linear-attention layer). Checked against the Hugging Face checkpoint tensor by tensor (an
// IQ3_S conversion of Qwen3.8-Flash-Next, every tensor of a linear-attention, the PLE and a full-attention layer and experts 0 and 511 of
// two layers).
#include "gguf_qwen_linear.h"
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
	{ "hc_attn_norm.weight", "attn_hyper_connection.hc_norm.weight", GgufTransform::NormMinusOne },
	{ "hc_attn_down.weight", "attn_hyper_connection.input_mix_weight_down.weight", GgufTransform::None },
	{ "hc_attn_up.weight", "attn_hyper_connection.input_mix_weight_up.weight", GgufTransform::None },
	{ "hc_attn_inject.weight", "attn_hyper_connection.block_inject_weight.weight", GgufTransform::None },
	{ "hc_ffn_norm.weight", "mlp_hyper_connection.hc_norm.weight", GgufTransform::NormMinusOne },
	{ "hc_ffn_down.weight", "mlp_hyper_connection.input_mix_weight_down.weight", GgufTransform::None },
	{ "hc_ffn_up.weight", "mlp_hyper_connection.input_mix_weight_up.weight", GgufTransform::None },
	{ "hc_ffn_inject.weight", "mlp_hyper_connection.block_inject_weight.weight", GgufTransform::None },
	{ "attn_q.weight", "self_attn.q_proj.weight", GgufTransform::None },
	{ "attn_k.weight", "self_attn.k_proj.weight", GgufTransform::None },
	{ "attn_v.weight", "self_attn.v_proj.weight", GgufTransform::None },
	{ "attn_output.weight", "self_attn.o_proj.weight", GgufTransform::None },
	{ "attn_q_norm.weight", "self_attn.q_norm.weight", GgufTransform::NormMinusOne },
	{ "attn_k_norm.weight", "self_attn.k_norm.weight", GgufTransform::NormMinusOne },
	{ "indexer.q_norm.weight", "self_attn.indexer.q_layernorm.weight", GgufTransform::NormMinusOne },
	{ "indexer.k_norm.weight", "self_attn.indexer.k_layernorm.weight", GgufTransform::NormMinusOne },
	{ "ffn_gate_inp.weight", "mlp.gate.weight", GgufTransform::None },
	{ "ffn_gate_shexp.weight", "mlp.shared_expert.gate_proj.weight", GgufTransform::None },
	{ "ffn_up_shexp.weight", "mlp.shared_expert.up_proj.weight", GgufTransform::None },
	{ "ffn_down_shexp.weight", "mlp.shared_expert.down_proj.weight", GgufTransform::None },
	{ "ple_key.weight", "ple.key_proj.weight", GgufTransform::None },
	{ "ple_value.weight", "ple.value_proj.weight", GgufTransform::None },
	{ "ple_norm_key.weight", "ple.norm_key.weight", GgufTransform::NormMinusOne },
	{ "ple_norm_query.weight", "ple.norm_query.weight", GgufTransform::NormMinusOne },
	{ "ple_norm_conv.weight", "ple.norm_conv.weight", GgufTransform::NormMinusOne },
};

GgufMapped simple( const std::string& name, GgufTransform t ) {
	GgufMapped m;
	m.name = name;
	m.transform = t;
	return m;
}

GgufMapped rows( GgufMapped m, int64_t begin, int64_t count ) {
	m.row_begin = begin;
	m.rows = count;
	return m;
}

void map_qwen4exp( const GgufArch& arch, const GgufKv& kv, const std::string& g, std::vector<GgufMapped>* out ) {
	const std::string p( P );
	if ( g == "token_embd.weight" ) { out->push_back( simple( p + ".embed_tokens.weight", GgufTransform::None ) ); return; }
	if ( g == "output.weight" ) { out->push_back( simple( "lm_head.weight", GgufTransform::None ) ); return; }
	if ( g == "output_hc_norm.weight" ) { out->push_back( simple( p + ".hyper_connection_mixer.hc_norm.weight", GgufTransform::NormMinusOne ) ); return; }
	if ( g == "output_hc_down.weight" ) { out->push_back( simple( p + ".hyper_connection_mixer.input_mix_weight_down.weight", GgufTransform::None ) ); return; }
	if ( g == "output_hc_up.weight" ) { out->push_back( simple( p + ".hyper_connection_mixer.input_mix_weight_up.weight", GgufTransform::None ) ); return; }
	if ( g == "per_layer_token_embd.weight" ) {
		// the n-gram table, read row by row on the host (NgramTable::open_gguf)
		const auto it = kv.int_arrays.find( "qwen4exp.ple.layers" );
		if ( it != kv.int_arrays.end() && !it->second.empty() ) out->push_back( simple( p + ".layers." + std::to_string( it->second[0] ) + ".ple.ngram_table", GgufTransform::None ) );
		return;
	}
	int layer = -1, used = 0;
	if ( std::sscanf( g.c_str(), "blk.%d.%n", &layer, &used ) != 1 || used == 0 ) return;
	if ( layer >= kv.get( "qwen4exp.block_count" ) - kv.get( "qwen4exp.nextn_predict_layers" ) ) return;   // the MTP block: not read from GGUF yet
	const std::string rest = g.substr( static_cast<size_t>( used ) ), pre = p + ".layers." + std::to_string( layer ) + ".";
	for ( const Simple& s : kLayer ) {
		if ( rest == s.gguf ) { out->push_back( simple( pre + s.hf, s.transform ) ); return; }
	}
	for ( const Suffix& s : kLinearAttn ) {
		if ( rest == s.gguf ) {
			GgufMapped m;
			fill_suffix( arch, kv, "qwen4exp", pre + s.hf, s, &m );
			out->push_back( std::move( m ) );
			return;
		}
	}
	if ( rest == "ffn_gate_inp_shexp.weight" ) {
		GgufMapped m = simple( pre + "mlp.shared_expert_gate.weight", GgufTransform::None );
		m.shape = { 1, -1 };
		out->push_back( std::move( m ) );
		return;
	}
	if ( rest == "ple_conv1d.weight" ) {
		GgufMapped m = simple( pre + "ple.conv1d.weight", GgufTransform::None );
		m.shape = { -1, 1, -1 };
		out->push_back( std::move( m ) );
		return;
	}
	// the indexer's q and k rows, in that order, are index_qk_proj
	if ( rest == "indexer.q_proj.weight" || rest == "indexer.k_proj.weight" ) {
		GgufMapped m = simple( pre + "self_attn.indexer.index_qk_proj.weight", GgufTransform::None );
		m.part = rest == "indexer.k_proj.weight" ? 1 : 0;
		m.parts = 2;
		out->push_back( std::move( m ) );
		return;
	}
	// routed experts: expert e is rows [e n, (e + 1) n) of the stacked tensor, n its projection's output width
	const int64_t experts = kv.get( "qwen4exp.expert_count" ), inter = kv.get( "qwen4exp.expert_feed_forward_length" ), hidden = kv.get( "qwen4exp.embedding_length" );
	const struct { const char* gguf; const char* hf; int64_t n; } kExperts[] = {
		{ "ffn_gate_exps.weight", "gate_proj", inter }, { "ffn_up_exps.weight", "up_proj", inter }, { "ffn_down_exps.weight", "down_proj", hidden } };
	for ( const auto& x : kExperts ) {
		if ( rest != x.gguf ) continue;
		for ( int64_t e = 0; e < experts; ++e ) out->push_back( rows( simple( pre + "mlp.experts." + std::to_string( e ) + "." + x.hf + ".weight", GgufTransform::None ), e * x.n, x.n ) );
		return;
	}
	// merged weights: the row ranges of their parts, each with its own transform
	const int64_t hc_rank = kv.get( "qwen4exp.hyper_connection.low_rank" ), hc = kv.get( "qwen4exp.hyper_connection.count" );
	for ( const char* site : { "attn", "ffn" } ) {
		if ( rest != std::string( "hc_" ) + site + "_down_inject.weight" ) continue;
		const std::string hs = pre + ( site[0] == 'a' ? "attn_hyper_connection." : "mlp_hyper_connection." );
		out->push_back( rows( simple( hs + "input_mix_weight_down.weight", GgufTransform::None ), 0, hc_rank ) );
		out->push_back( rows( simple( hs + "block_inject_weight.weight", GgufTransform::None ), hc_rank, hc ) );
		return;
	}
	if ( rest == "attn_qkv_gate.weight" ) {
		const int64_t qkv = 2 * arch.k_heads * kv.get( "qwen4exp.ssm.state_size" ) + kv.get( "qwen4exp.ssm.inner_size" );
		GgufMapped a, b;
		fill_suffix( arch, kv, "qwen4exp", pre + kLinearAttn[0].hf, kLinearAttn[0], &a );
		fill_suffix( arch, kv, "qwen4exp", pre + kLinearAttn[1].hf, kLinearAttn[1], &b );
		out->push_back( rows( a, 0, qkv ) );
		out->push_back( rows( b, qkv, kv.get( "qwen4exp.ssm.inner_size" ) ) );
		return;
	}
	if ( rest == "ssm_beta_alpha.weight" ) {
		GgufMapped b, a;
		fill_suffix( arch, kv, "qwen4exp", pre + kLinearAttn[3].hf, kLinearAttn[3], &b );
		fill_suffix( arch, kv, "qwen4exp", pre + kLinearAttn[2].hf, kLinearAttn[2], &a );
		out->push_back( rows( b, 0, arch.v_heads ) );
		out->push_back( rows( a, arch.v_heads, arch.v_heads ) );
		return;
	}
}

} // namespace

bool gguf_arch_qwen4exp( const GgufKv& kv, GgufArch* out ) {
	if ( !linear_attn_heads( kv, "qwen4exp", out ) ) return false;
	out->map = map_qwen4exp;
	return true;
}

} // namespace engine::loader
