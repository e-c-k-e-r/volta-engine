// Gives the cold experts of an expert tier their CPU-layout copies (see cpu_expert_attach.h)
#include "engine/runtime/cpu_expert_attach.h"
#include "engine/runtime/cpu_expert_service.h"
#include "engine/runtime/options.h"
#include "engine/kernels/cpu/nvfp4_linear.h"
#include "engine/kernels/cpu/trellis_linear.h"
#include "engine/backend/runtime.h"
#include "engine/core/log.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <memory>
#include <vector>

namespace engine::cpu_moe {

namespace {

using kernels::cpu::Linear;

// The expert threads and the node of the CPU-side copies: the options when given, otherwise one thread per physical core of a NUMA node. EXL3
// experts are compute-bound (measured: the same speed from local or remote memory), so they take the last node, on two sockets the one away from
// the GPU rank threads; NVFP4 experts read their codes in place from the expert tier's pinned buffers on node 0 (remote reads cost 19-86%), so
// they take node 0's cores from 4 on (0 to 3 drive the GPUs)
struct CpuPlan { std::vector<int> cpus, cpus2; int node = -1; };

std::vector<int> read_cpulist( const char* path ) {
	std::vector<int> out;
	if ( std::FILE* f = std::fopen( path, "r" ) ) {
		char buf[4096] = {};
		if ( std::fgets( buf, sizeof buf, f ) ) out = Service::parse_cpus( std::string( buf, std::strcspn( buf, "\n" ) ) );
		std::fclose( f );
	}
	return out;
}

constexpr int kRankCores = 4;   // cores 0 to 3 of node 0 drive the GPU ranks

// The first SMT thread of every core in a node's cpulist, from core skip_below on
std::vector<int> first_threads( int node, int skip_below ) {
	std::vector<int> out;
	char path[96];
	std::snprintf( path, sizeof path, "/sys/devices/system/node/node%d/cpulist", node );
	for ( int c : read_cpulist( path ) ) {
		if ( c < skip_below ) continue;
		std::snprintf( path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c );
		const std::vector<int> sib = read_cpulist( path );
		if ( sib.empty() || sib[0] == c ) out.push_back( c );
	}
	return out;
}

// The plan is made once, by the first tier attached (its expert format picks the default node); later calls read it
CpuPlan g_plan;
bool g_plan_made = false;

const CpuPlan& cpu_plan() {
	ENGINE_ASSERT( g_plan_made, "cpu_experts: the CPU plan is read before it was made" );
	return g_plan;
}

const CpuPlan& make_cpu_plan( bool nvfp4 ) {
	if ( g_plan_made ) return g_plan;
	g_plan_made = true;
	g_plan = [nvfp4] {
		const auto& opt = runtime::options();
		CpuPlan p;
		p.node = opt.cpu_expert_node;
		if ( !opt.cpu_expert_cpus.empty() ) { p.cpus = Service::parse_cpus( opt.cpu_expert_cpus ); return p; }
		int nodes = 0;
		while ( nodes < 64 ) {
			char path[96];
			std::snprintf( path, sizeof path, "/sys/devices/system/node/node%d/cpulist", nodes );
			if ( std::FILE* f = std::fopen( path, "r" ) ) { std::fclose( f ); ++nodes; } else break;
		}
		if ( nodes == 0 ) return p;   // no NUMA information: the service's default (4 unpinned threads)
		const int node = p.node >= 0 && p.node < nodes ? p.node : nvfp4 ? 0 : nodes - 1;
		p.cpus = first_threads( node, node == 0 ? kRankCores : 0 );
		p.node = node;
		// EXL3 on a two-socket host: a second pool on node 0 (from core 4) works on the other half of every expert's output columns; the copies
		// stay on the first node (EXL3 reads cost the same from either)
		if ( !nvfp4 && nodes > 1 && p.node != 0 && opt.cpu_expert_node < 0 ) p.cpus2 = first_threads( 0, kRankCores );
		return p;
	}();
	return g_plan;
}

std::unique_ptr<Linear> make_linear( const pod::LinearWeight& lw, Service& svc ) {
	const int node = cpu_plan().node;
	if ( lw.type == pod::QuantType::NVFP4 ) {
		// w0: packed codes [N, K / 2] bytes (host copy, in the prepacked tile order), w1: E4M3 block scales [N * K / 16] (device), alpha: the global scale
		const int n = lw.w0.shape[0], k = lw.w0.shape[1] * 2;
		std::vector<uint8_t> scales( static_cast<size_t>( n ) * k / 16 );
		backend::memcpy_sync( scales.data(), lw.w1.data, scales.size(), backend::Copy::D2H );
		svc.add_host_bytes( scales.size() );   // the codes are read in place from the tier's pinned copy
		return std::unique_ptr<Linear>( new kernels::cpu::Nvfp4Linear( static_cast<const uint8_t*>( lw.w0.data ), scales.data(), lw.alpha, k, n ) );
	}
	ENGINE_ASSERT( lw.type == pod::QuantType::TRELLIS && lw.mul1 && kernels::cpu::TrellisLinear::supported( lw.bits ), "cpu_experts: experts must be NVFP4 or mul1 trellis" );
	ENGINE_ASSERT( !lw.volta_tiles, "cpu_experts: the CPU expert kernel reads the stock EXL3 tile order, not the Volta one (tile_layout volta): use cpu_experts=0 or a stock-order checkpoint" );
	const int kt = lw.w0.shape[0], nt = lw.w0.shape[1], bits = lw.bits;
	std::vector<uint16_t> suh( static_cast<size_t>( kt ) * 16 ), svh( static_cast<size_t>( nt ) * 16 );
	backend::memcpy_sync( suh.data(), lw.w1.data, suh.size() * 2, backend::Copy::D2H );
	backend::memcpy_sync( svh.data(), lw.w2.data, svh.size() * 2, backend::Copy::D2H );
	const size_t nb = kernels::cpu::TrellisLinear::repacked_bytes( kt, nt, bits );
	void* mem = svc.numa_alloc( nb, node );
	ENGINE_ASSERT( mem != nullptr, "cpu_experts: host allocation failed" );
	svc.add_host_bytes( nb );
	return std::unique_ptr<Linear>( new kernels::cpu::TrellisLinear( static_cast<const uint16_t*>( lw.w0.data ), kt, nt, bits, suh.data(), svh.data(), mem ) );
}

void build( modules::ExpertTier& tier, int rank, int first_expert, int li, int e ) {
	Service& svc = Service::get();
	auto& l = tier.layers[li];
	Linear* have[3];
	if ( svc.find_expert( rank, li, first_expert + e, have ) ) {
		// The expert was cold before: its codes moved to another pinned buffer with the swaps; the in-place linears follow
		pod::LinearWeight* lws[3] = { &l.mw->gate[e], &l.mw->up[e], &l.mw->down[e] };
		if ( lws[0]->type == pod::QuantType::NVFP4 )
			for ( int p = 0; p < 3; ++p ) static_cast<kernels::cpu::Nvfp4Linear*>( have[p] )->rebind( static_cast<const uint8_t*>( lws[p]->w0.data ) );
		return;
	}
	svc.add_expert( rank, li, first_expert + e, make_linear( l.mw->gate[e], svc ), make_linear( l.mw->up[e], svc ), make_linear( l.mw->down[e], svc ) );
}

}   // namespace

CpuExpertHandle attach_cpu_experts( modules::ExpertTier& tier, int rank, int first_expert, int hidden, float act_limit, pod::MemoryArena& device, backend::stream_t stream ) {
	const auto& opt = runtime::options();
	Service& svc = Service::get();
	if ( !svc.started() ) {
		// the format of this tier's experts decides the default node (the plan is fixed by the first tier)
		bool nvfp4 = false;
		for ( size_t li = 0; li < tier.layers.size() && !nvfp4; ++li ) nvfp4 = tier.layers[li].mw && tier.layers[li].mw->type == pod::QuantType::NVFP4;
		const CpuPlan& plan = make_cpu_plan( nvfp4 );
		svc.start( plan.cpus, opt.cpu_expert_spin_us, plan.cpus2 );
		if ( rank == 0 ) LOG_INFO( "[cpu-experts] %zu expert threads%s%s, host-side copies on node %d\n", plan.cpus.size() + plan.cpus2.size(),
			opt.cpu_expert_cpus.empty() ? " (one per core, cpu_expert_cpus unset)" : "", plan.cpus2.empty() ? "" : ", a second pool on node 0", plan.node );
	}
	CpuExpertHandle h;
	h.mb = svc.mailbox( rank, hidden, act_limit );
	h.ctl = static_cast<uint64_t*>( engine::core::arena_alloc( device, 2 * sizeof( uint64_t ), 256 ) );
	h.frac = static_cast<float>( opt.cpu_experts );
	backend::memset_async( h.ctl, 0, 2 * sizeof( uint64_t ), stream );
	tier.assign_cpu( device );
	backend::stream_sync( stream );
	int built = 0;
	for ( size_t li = 0; li < tier.layers.size(); ++li )
		for ( int e = 0; e < tier.experts; ++e )
			if ( tier.layers[li].cpu_owned[e] ) { build( tier, rank, first_expert, static_cast<int>( li ), e ); ++built; }
	tier.on_cold = [&tier, rank, first_expert]( int li, int e ) { build( tier, rank, first_expert, li, e ); };
	if ( rank == 0 ) LOG_INFO( "[cpu-experts] rank 0: %d cold experts on the CPU path, %.2f GiB of host-side copies (all ranks so far); up to %.0f%% of a layer's cold experts go to the CPU; node %d\n", built, svc.host_bytes() / 1073741824.0, 100.0 * opt.cpu_experts, cpu_plan().node );
	return h;
}

void detach_cpu_experts() { Service::get().quiesce(); }

}   // namespace engine::cpu_moe
