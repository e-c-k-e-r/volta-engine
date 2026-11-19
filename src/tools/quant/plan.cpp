#include "plan.h"
#include "engine/core/log.h"
#include <nlohmann/json.hpp>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <sys/stat.h>

namespace engine::tools::quant {

namespace {

bool exists( const std::string& p ) {
	struct stat st {};
	return stat( p.c_str(), &st ) == 0;
}

std::string site_file( const std::string& dir, int layer, int site, int rank ) {
	char buf[64];
	std::snprintf( buf, sizeof( buf ), "/L%02d_s%d_r%d.f32", layer, site, rank );
	return dir + buf;
}

// One file (rank 0), or every rank's (a row-parallel input); none when any is missing
std::vector<std::string> site_files( const std::string& dir, int layer, int site, bool per_rank, int ranks ) {
	std::vector<std::string> out;
	for ( int r = 0; r < ( per_rank ? ranks : 1 ); ++r ) {
		const std::string p = site_file( dir, layer, site, r );
		if ( !exists( p ) ) return {};
		out.push_back( p );
	}
	return out;
}

struct Site {
	int site = -1;
	bool per_rank = false;
	bool expert_down = false;
};

bool ends( const std::string& s, const char* suffix ) {
	const size_t n = std::strlen( suffix );
	return s.size() >= n && s.compare( s.size() - n, n, suffix ) == 0;
}

// The capture site of a decoder layer's projection (rest = the name after "<layers>.<L>.")
Site layer_site( const std::string& type, const std::string& rest ) {
	static const std::regex expert( R"(^mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$)" );
	std::smatch m;
	const bool is_expert = std::regex_match( rest, m, expert );
	if ( is_expert && m[1].str() == "down_proj" ) return { -1, false, true };
	const bool attn_in = ends( rest, "self_attn.q_proj" ) || ends( rest, "self_attn.k_proj" ) || ends( rest, "self_attn.v_proj" ) ||
		rest == "self_attn.indexer.index_qk_proj" || rest == "linear_attn.in_proj_qkv" || rest == "linear_attn.in_proj_z" ||
		rest == "self_attn.qkv_proj" || rest == "self_attn.q_a_proj" || rest == "self_attn.kv_a_proj_with_mqa";
	if ( attn_in ) return { 0, false, false };
	if ( rest == "self_attn.o_proj" || rest == "linear_attn.out_proj" ) return { 2, true, false };
	const bool mlp_in = is_expert || rest == "mlp.gate_proj" || rest == "mlp.up_proj" || rest == "mlp.shared_expert.gate_proj" ||
		rest == "mlp.shared_expert.up_proj" || rest == "mlp.shared_experts.gate_proj" || rest == "mlp.shared_experts.up_proj";
	if ( mlp_in ) return { 1, false, false };
	if ( type == "glm5_next" ) {
		if ( rest == "self_attn.q_b_proj" || rest == "self_attn.indexer.wq_b" ) return { 4, false, false };
		if ( rest == "mlp.shared_experts.down_proj" || rest == "mlp.down_proj" ) return { 3, false, false };
	} else if ( type == "qwen4_exp" ) {
		if ( rest == "mlp.shared_expert.down_proj" ) return { 3, false, false };
	} else {
		// Qwen3.5-family dense: down_proj's input channels are split over the ranks
		if ( rest == "mlp.down_proj" ) return { 3, true, false };
	}
	return {};
}

} // namespace

bool Model::read( const std::string& dir ) {
	std::ifstream f( dir + "/config.json" );
	if ( !f ) {
		LOG_ERROR( "[volta_quantize] %s has no config.json", dir.c_str() );
		return false;
	}
	std::stringstream ss;
	ss << f.rdbuf();
	const auto c = nlohmann::json::parse( ss.str(), nullptr, false );
	if ( c.is_discarded() ) return false;
	type = c.value( "model_type", "" );
	const auto& t = c.contains( "text_config" ) ? c["text_config"] : c;
	layers = t.value( "num_hidden_layers", 0 );
	hidden = t.value( "hidden_size", 0 );
	experts = t.value( "num_experts", t.value( "n_routed_experts", 0 ) );
	swiglu_limit = t.value( "swiglu_limit", 0.0f );
	return !type.empty();
}

void hessian_sources( const Model& m, const std::string& dir, int ranks, PlanTensor& t ) {
	t.hess.clear();
	t.down_layer = t.down_expert = -1;
	if ( dir.empty() ) return;
	// The LM head's input: site 5 on GLM, site 4 otherwise, recorded under layer 0
	if ( t.base == "lm_head" ) {
		t.hess = site_files( dir, 0, m.type == "glm5_next" ? 5 : 4, false, ranks );
		return;
	}
	// Flash-Next's MTP head: its block is layer N (the decoder's layer count), its input projections layer N + 1
	if ( m.type == "qwen4_exp" && t.base == "mtp.fc_embedding" ) {
		t.hess = site_files( dir, m.layers + 1, 0, false, ranks );
		return;
	}
	if ( m.type == "qwen4_exp" && t.base == "mtp.fc_hidden" ) {
		t.hess = site_files( dir, m.layers + 1, 1, false, ranks );
		return;
	}
	static const std::regex layer_re( R"(^(model\.language_model\.layers|mtp\.layers)\.(\d+)\.(.+)$)" );
	std::smatch mm;
	if ( !std::regex_match( t.base, mm, layer_re ) ) return;
	const bool mtp = mm[1].str() == "mtp.layers";
	int layer = std::stoi( mm[2].str() );
	if ( mtp ) {
		if ( m.type != "qwen4_exp" ) return;
		layer = m.layers + layer;
	} else if ( m.type == "glm5_next" && layer >= m.layers ) {
		return;   // GLM's NVFP4-only MTP layer
	}
	t.layer = layer;
	const std::string rest = mm[3].str();
	const Site s = layer_site( m.type, rest );
	if ( s.expert_down ) {
		static const std::regex e_re( R"(^mlp\.experts\.(\d+)\.down_proj$)" );
		std::smatch em;
		std::regex_match( rest, em, e_re );
		t.down_layer = layer;
		t.down_expert = std::stoi( em[1].str() );
		t.src_layer_prefix = mm[1].str() + "." + mm[2].str() + ".mlp.experts";
		return;
	}
	if ( s.site < 0 ) return;
	t.hess = site_files( dir, layer, s.site, s.per_rank, ranks );
}

} // namespace engine::tools::quant
