// GGUF architecture deepseek41 (DeepSeek-V4.1-Flash: llama.cpp's converter, conversion/deepseek41.py) -> the checkpoint's own names, which
// DeepseekV41Pipeline loads. The converter stacks the routed experts into one tensor per projection ([experts, n, k]), stores the Engram
// tables of all Engram layers as one tensor (per_layer_token_embd: the rows of layer 1's heads, then layer 14's), renames engram_wkv to
// engram_kv and folds engram_q * engram_k into one F32 engram_gate; everything else keeps its shape. No multi-token-prediction block is in
// the file (the DSpark drafter is a checkpoint of its own).
#include "gguf_arch.h"
#include <cstdio>

namespace engine::loader {

namespace {

struct Simple {
	const char* gguf;
	const char* name;
};

// Per layer, under "layers.<i>."
const Simple kLayer[] = {
	{ "attn_norm.weight", "attn_norm.weight" },
	{ "ffn_norm.weight", "ffn_norm.weight" },
	{ "attn_q_a.weight", "attn.wq_a.weight" },
	{ "attn_q_a_norm.weight", "attn.q_norm.weight" },
	{ "attn_q_b.weight", "attn.wq_b.weight" },
	{ "attn_kv.weight", "attn.wkv.weight" },
	{ "attn_kv_a_norm.weight", "attn.kv_norm.weight" },
	{ "attn_output_a.weight", "attn.wo_a.weight" },
	{ "attn_output_b.weight", "attn.wo_b.weight" },
	{ "attn_sinks.weight", "attn.attn_sink" },
	{ "attn_compressor_gate.weight", "attn.compressor.wgate.weight" },
	{ "attn_compressor_kv.weight", "attn.compressor.wkv.weight" },
	{ "attn_compressor_norm.weight", "attn.compressor.norm.weight" },
	{ "indexer.attn_q_b.weight", "attn.indexer.wq_b.weight" },
	{ "indexer.attn_k.weight", "attn.indexer.wk.weight" },
	{ "indexer.proj.weight", "attn.indexer.weights_proj.weight" },
	{ "indexer.k_norm.weight", "attn.indexer.k_norm.weight" },
	{ "ffn_gate_inp.weight", "ffn.gate.weight" },
	{ "exp_probs_b.bias", "ffn.gate.bias" },
	{ "exp_probs_b_vl.bias", "ffn.gate.bias_vl" },
	{ "ffn_gate_shexp.weight", "ffn.shared_experts.w1.weight" },
	{ "ffn_up_shexp.weight", "ffn.shared_experts.w3.weight" },
	{ "ffn_down_shexp.weight", "ffn.shared_experts.w2.weight" },
	{ "hc_attn_base.weight", "hc_attn_base" },
	{ "hc_attn_fn.weight", "hc_attn_fn" },
	{ "hc_attn_scale.weight", "hc_attn_scale" },
	{ "hc_ffn_base.weight", "hc_ffn_base" },
	{ "hc_ffn_fn.weight", "hc_ffn_fn" },
	{ "hc_ffn_scale.weight", "hc_ffn_scale" },
	{ "engram_kv.weight", "engram.wkv.weight" },
	{ "engram_gate.weight", "engram.gate" },   // q * k, [4, hidden] F32
};

GgufMapped simple( const std::string& name ) {
	GgufMapped m;
	m.name = name;
	return m;
}

const std::vector<int64_t>* int_array( const GgufKv& kv, const char* key ) {
	const auto it = kv.int_arrays.find( key );
	return it == kv.int_arrays.end() ? nullptr : &it->second;
}

void map_dsv41( const GgufArch&, const GgufKv& kv, const std::string& g, std::vector<GgufMapped>* out ) {
	if ( g == "token_embd.weight" ) { out->push_back( simple( "embed.weight" ) ); return; }
	if ( g == "output.weight" ) { out->push_back( simple( "head.weight" ) ); return; }
	if ( g == "output_norm.weight" ) { out->push_back( simple( "norm.weight" ) ); return; }
	if ( g == "per_layer_token_embd.weight" ) {
		// one table per Engram layer: the rows of its (n-gram size, head) buckets, ple.head_vocab_sizes in layer order
		const std::vector<int64_t>* layers = int_array( kv, "deepseek41.ple.layers" );
		const std::vector<int64_t>* sizes = int_array( kv, "deepseek41.ple.head_vocab_sizes" );
		if ( layers == nullptr || sizes == nullptr || layers->empty() || sizes->size() % layers->size() != 0 ) return;
		const size_t per = sizes->size() / layers->size();
		int64_t begin = 0;
		for ( size_t e = 0; e < layers->size(); ++e ) {
			int64_t rows = 0;
			for ( size_t c = 0; c < per; ++c ) rows += ( *sizes )[e * per + c];
			GgufMapped m = simple( "layers." + std::to_string( ( *layers )[e] ) + ".engram.embed.weight" );
			m.row_begin = begin;
			m.rows = rows;
			out->push_back( std::move( m ) );
			begin += rows;
		}
		return;
	}
	int layer = -1, used = 0;
	if ( std::sscanf( g.c_str(), "blk.%d.%n", &layer, &used ) != 1 || used == 0 ) return;
	const std::string rest = g.substr( static_cast<size_t>( used ) ), pre = "layers." + std::to_string( layer ) + ".";
	for ( const Simple& s : kLayer ) {
		if ( rest == s.gguf ) { out->push_back( simple( pre + s.name ) ); return; }
	}
	// routed experts: expert e is rows [e n, (e + 1) n) of the stacked tensor, n its projection's output width
	const int64_t experts = kv.get( "deepseek41.expert_count" ), inter = kv.get( "deepseek41.expert_feed_forward_length" ), hidden = kv.get( "deepseek41.embedding_length" );
	const struct { const char* gguf; const char* name; int64_t n; } kExperts[] = {
		{ "ffn_gate_exps.weight", "w1", inter }, { "ffn_up_exps.weight", "w3", inter }, { "ffn_down_exps.weight", "w2", hidden } };
	for ( const auto& x : kExperts ) {
		if ( rest != x.gguf ) continue;
		for ( int64_t e = 0; e < experts; ++e ) {
			GgufMapped m = simple( pre + "ffn.experts." + std::to_string( e ) + "." + x.name + ".weight" );
			m.row_begin = e * x.n;
			m.rows = x.n;
			out->push_back( std::move( m ) );
		}
		return;
	}
}

} // namespace

bool gguf_arch_dsv41( const GgufKv&, GgufArch* out ) {
	out->map = map_dsv41;
	return true;
}

} // namespace engine::loader
