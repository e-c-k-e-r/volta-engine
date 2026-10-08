#include "engine/runtime/options.h"

#include <cstdlib>

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
	env_flag( "EXL3_ATTN_TC", o.attn_tc );
	o.expert_log = std::getenv( "EXL3_EXPERT_LOG" ) != nullptr;
	o.dsa_diag = std::getenv( "EXL3_DSA_DIAG" ) != nullptr;
	return o;
}

} // namespace

Options& options() {
	static Options o = from_env();
	return o;
}

} // namespace engine::runtime
