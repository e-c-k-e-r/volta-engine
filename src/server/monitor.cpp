#include "engine/server/server.h"
#include "engine/server/monitor.h"
#include "engine/runtime/options.h"
#include <nlohmann/json.hpp>
#include <cmath>

namespace engine::server {

namespace {

const char* reason_name( engine::runtime::FinishReason r ) {
	switch ( r ) {
		case engine::runtime::FinishReason::Stop: return "stop";
		case engine::runtime::FinishReason::Length: return "length";
		case engine::runtime::FinishReason::StopString: return "stop_string";
		case engine::runtime::FinishReason::Cancelled: return "cancelled";
	}
	return "stop";
}

} // namespace

void VoltaServer::handle_monitor( const ConnPtr& conn ) {
	send_response( conn, 200, "text/html; charset=utf-8", monitor_html() );
}

void VoltaServer::handle_api_stats( const ConnPtr& conn ) {
	using nlohmann::json;
	const auto mon = generator_->get_monitor();
	const auto met = generator_->get_metrics();
	const auto& opt = runtime::options();
	json j;
	j["server"] = { { "model", model_name_ }, { "model_type", config_.model_type }, { "uptime_s", mon.uptime_seconds }, { "slots", mon.num_slots },
		{ "max_context", mon.max_context }, { "tp", mon.tp }, { "queued", mon.queued }, { "speculation", config_.use_speculation },
		{ "speculator", config_.use_speculation ? opt.speculator : std::string() }, { "kv_format", opt.kv_format.empty() ? std::string( "default" ) : opt.kv_format } };
	j["totals"] = { { "requests", mon.requests }, { "completed", mon.completed }, { "tokens_out", mon.tokens_out }, { "tokens_prefilled", mon.tokens_prefilled },
		{ "prompt_tokens", mon.prompt_tokens }, { "cached_tokens", mon.cached_tokens },
		{ "finished", { { "stop", mon.finished[0] }, { "length", mon.finished[1] }, { "stop_string", mon.finished[2] }, { "cancelled", mon.finished[3] } } } };
	j["averages"] = { { "ttft_ms", met.avg_ttft_ms }, { "prefill_tok_s", met.avg_prefill_tok_s }, { "decode_tok_s", met.avg_decode_tok_s } };
	j["latency"] = { { "gap_ms", { mon.gap_ms[0], mon.gap_ms[1], mon.gap_ms[2], mon.gap_ms[3], mon.gap_ms[4] } }, { "gap_samples", mon.gap_samples },
		{ "ttft_ms", { mon.ttft_ms[0], mon.ttft_ms[1], mon.ttft_ms[2] } }, { "gap_hist", mon.gap_hist } };
	j["speculation"] = { { "rounds", mon.spec_rounds }, { "proposed", mon.spec_proposed }, { "accepted", mon.spec_accepted },
		{ "accept_rate", mon.spec_proposed ? static_cast<double>( mon.spec_accepted ) / mon.spec_proposed : 0.0 },
		{ "tokens_per_round", mon.spec_rounds ? 1.0 + static_cast<double>( mon.spec_accepted ) / mon.spec_rounds : 0.0 },
		{ "suffix", { { "rounds", mon.suffix_rounds }, { "proposed", mon.suffix_proposed }, { "accepted", mon.suffix_accepted } } } };
	json slots = json::array();
	for ( size_t i = 0; i < mon.slots.size(); ++i ) {
		const auto& s = mon.slots[i];
		slots.push_back( { { "slot", i }, { "active", s.active }, { "phase", !s.active ? "idle" : s.prefilling ? "prefill" : s.reasoning ? "thinking" : "decode" },
			{ "req_id", s.req_id }, { "pos", s.pos }, { "prompt_tokens", s.prompt_tokens }, { "cached_tokens", s.cached_tokens }, { "generated", s.generated },
			{ "max_tokens", s.max_tokens }, { "age_s", s.age_s }, { "decode_tok_s", s.decode_tok_s } } );
	}
	j["slots"] = slots;
	json recent = json::array();
	for ( auto it = mon.recent.rbegin(); it != mon.recent.rend(); ++it ) {
		recent.push_back( { { "req_id", it->req_id }, { "age_s", mon.uptime_seconds - it->end_s }, { "prompt_tokens", it->prompt_tokens }, { "cached_tokens", it->cached_tokens },
			{ "generated", it->generated }, { "queue_ms", it->queue_ms }, { "ttft_ms", it->ttft_ms }, { "restore_ms", it->restore_ms }, { "prefill_ms", it->prefill_ms }, { "store_ms", it->store_ms }, { "total_ms", it->total_ms }, { "decode_tok_s", it->decode_tok_s }, { "reason", reason_name( it->reason ) } } );
	}
	j["recent"] = recent;
	json t = json::array(), dec = json::array(), pf = json::array(), act = json::array();
	for ( const auto& h : mon.history ) {
		t.push_back( std::round( h.t ) );
		dec.push_back( std::round( h.decode_tok_s * 10.0 ) / 10.0 );
		pf.push_back( std::round( h.prefill_tok_s * 10.0 ) / 10.0 );
		act.push_back( h.active );
	}
	j["history"] = { { "t", t }, { "decode", dec }, { "prefill", pf }, { "active", act } };
	if ( met.prefix_enabled ) {
		j["prefix_cache"] = { { "lookups", met.prefix.lookups }, { "hits", met.prefix.hits }, { "hit_tokens", met.prefix.hit_tokens }, { "inserts", met.prefix.inserts },
			{ "evictions", met.prefix.evictions }, { "entries", met.prefix.entries }, { "bytes", met.prefix.bytes }, { "budget_bytes", static_cast<uint64_t>( std::max( 0, opt.prefix_cache_gib ) ) << 30 } };
	}
	std::vector<GpuReading> gpus;
	HostReading host;
	sys_.read( &gpus, &host );
	json g = json::array();
	for ( size_t i = 0; i < gpus.size(); ++i ) {
		const auto& r = gpus[i];
		g.push_back( { { "index", i }, { "name", r.name }, { "power_w", r.power_w }, { "power_limit_w", r.power_limit_w }, { "energy_kj", r.energy_kj }, { "temp_c", r.temp_c },
			{ "util_gpu", r.util_gpu }, { "util_mem", r.util_mem }, { "sm_mhz", r.sm_mhz }, { "mem_mhz", r.mem_mhz }, { "mem_used", r.mem_used }, { "mem_total", r.mem_total },
			{ "arena_bytes", i < met.gpu_vram_used_bytes.size() ? met.gpu_vram_used_bytes[i] : 0 }, { "throttle", throttle_names( r.throttle ) } } );
	}
	j["gpus"] = g;
	j["host"] = { { "cpu_pct", host.cpu_pct }, { "mem_used", host.mem_used }, { "mem_total", host.mem_total }, { "rss", host.rss }, { "load1", host.load1 } };
	send_json( conn, 200, j.dump() );
}

void VoltaServer::handle_api_experts( const ConnPtr& conn ) {
	using nlohmann::json;
	const auto e = generator_->get_expert_stats();
	json j;
	j["ranks"] = e.ranks;
	j["layers"] = e.layers;
	j["experts"] = e.experts;
	json total = json::array(), recent = json::array();
	for ( uint64_t v : e.total ) total.push_back( v );
	for ( float v : e.recent ) recent.push_back( static_cast<int>( v + 0.5f ) );
	j["total"] = total;
	j["recent"] = recent;
	send_json( conn, 200, j.dump() );
}

} // namespace engine::server
