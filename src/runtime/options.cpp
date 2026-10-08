#include "engine/runtime/options.h"

#include <cstdlib>
#include <cstring>

namespace engine::runtime {

namespace {

bool env_int( const char* name, int& out ) {
	const char* e = std::getenv( name );
	if ( !e ) return false;
	out = std::atoi( e );
	return true;
}

bool env_flag( const char* name, bool& out ) {
	int v = 0;
	if ( !env_int( name, v ) ) return false;
	out = ( v != 0 );
	return true;
}

Options from_env() {
	Options o;
	env_int( "EXL3_TP", o.tp );
	env_int( "EXL3_ARENA_GIB", o.arena_gib );
	env_int( "EXL3_VRAM_RESERVE_MIB", o.vram_reserve_mib );
	env_int( "EXL3_MAX_SEQ", o.max_seq );
	env_int( "EXL3_MAX_SLOTS", o.max_slots );
	env_int( "ENGINE_PREFIX_CACHE_GIB", o.prefix_cache_gib );
	if ( const char* e = std::getenv( "EXL3_KV" ) ) o.kv_format = e;
	if ( const char* e = std::getenv( "EXL3_EXPERT_CACHE" ) ) {
		const double f = std::atof( e );
		if ( f > 0.0 && f < 1.0 ) o.expert_cache = f;
	}
	env_int( "EXL3_EXPERT_ADAPT_EVERY", o.expert_adapt_every );
	env_int( "EXL3_EXPERT_SWAPS", o.expert_swaps );
	if ( const char* e = std::getenv( "EXL3_DRAFT_TP" ) ) o.draft_tp = ( e[0] != '0' );
	env_int( "EXL3_LM_SHARD", o.lm_shard );
	env_flag( "EXL3_MOE_TP", o.moe_tp );
	env_flag( "EXL3_DSA", o.dsa );
	env_flag( "EXL3_MLA_LATENT", o.mla_latent );
	env_flag( "VOLTA_SPEC_CTRL", o.spec_ctrl );
	o.expert_log = std::getenv( "ENGINE_DEBUG_EXPERT_LOG" ) != nullptr;
	o.dsa_diag = std::getenv( "ENGINE_DEBUG_DSA" ) != nullptr;
	o.spec_ctrl_log = std::getenv( "ENGINE_DEBUG_SPEC_CTRL" ) != nullptr;
	o.no_graphs = std::getenv( "ENGINE_DEBUG_NO_GRAPHS" ) != nullptr;
	env_int( "ENGINE_DEBUG_DUMP_DRAFT", o.dump_draft_round );
	if ( const char* e = std::getenv( "ENGINE_DEBUG_DUMP_REQUESTS" ) ) o.dump_requests = e;
	return o;
}

} // namespace

Options& options() {
	static Options o = from_env();
	return o;
}

bool set_option( const std::string& key, const std::string& value ) {
	Options& o = options();
	const char* v = value.c_str();
	char* end = nullptr;
	auto as_int = [&]( int& out ) {
		const long x = std::strtol( v, &end, 10 );
		if ( end == v || *end ) return false;
		out = static_cast<int>( x );
		return true;
	};
	auto as_bool = [&]( bool& out ) {
		if ( value == "true" || value == "1" ) out = true;
		else if ( value == "false" || value == "0" ) out = false;
		else return false;
		return true;
	};
	if ( key == "tp" ) return as_int( o.tp );
	if ( key == "arena_gib" ) return as_int( o.arena_gib );
	if ( key == "vram_reserve_mib" ) return as_int( o.vram_reserve_mib );
	if ( key == "max_seq" ) return as_int( o.max_seq );
	if ( key == "max_slots" ) return as_int( o.max_slots );
	if ( key == "prefix_cache_gib" ) return as_int( o.prefix_cache_gib );
	if ( key == "kv_format" ) {
		if ( !value.empty() && value != "fp16" && value != "fp8" && value != "k8v4" && value != "nvfp4" ) return false;
		o.kv_format = value;
		return true;
	}
	if ( key == "expert_cache" ) {
		const double f = std::strtod( v, &end );
		if ( end == v || *end || f <= 0.0 || f > 1.0 ) return false;
		o.expert_cache = f;
		return true;
	}
	if ( key == "expert_adapt_every" ) return as_int( o.expert_adapt_every );
	if ( key == "expert_swaps" ) return as_int( o.expert_swaps );
	if ( key == "draft_tp" ) return as_bool( o.draft_tp );
	if ( key == "lm_shard" ) return as_int( o.lm_shard );
	if ( key == "moe_tp" ) return as_bool( o.moe_tp );
	if ( key == "dsa" ) return as_bool( o.dsa );
	if ( key == "mla_latent" ) return as_bool( o.mla_latent );
	if ( key == "spec_ctrl" ) return as_bool( o.spec_ctrl );
	if ( key == "expert_log" ) return as_bool( o.expert_log );
	if ( key == "dsa_diag" ) return as_bool( o.dsa_diag );
	if ( key == "spec_ctrl_log" ) return as_bool( o.spec_ctrl_log );
	if ( key == "no_graphs" ) return as_bool( o.no_graphs );
	if ( key == "dump_requests" ) {
		o.dump_requests = value;
		return true;
	}
	if ( key == "dump_draft_round" ) return as_int( o.dump_draft_round );
	return false;
}

} // namespace engine::runtime
