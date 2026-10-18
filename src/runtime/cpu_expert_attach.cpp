// Gives the cold experts of an expert tier their CPU-layout copies (see cpu_expert_attach.h)
#include "engine/runtime/cpu_expert_attach.h"
#include "engine/runtime/cpu_expert_service.h"
#include "engine/runtime/options.h"
#include "engine/kernels/cpu/nvfp4_linear.h"
#include "engine/kernels/cpu/trellis_linear.h"
#include "engine/backend/runtime.h"
#include "engine/core/log.h"
#include <memory>
#include <vector>

namespace engine::cpu_moe {

namespace {

using kernels::cpu::Linear;

std::unique_ptr<Linear> make_linear( const pod::LinearWeight& lw, Service& svc ) {
	const int node = runtime::options().cpu_expert_node;
	if ( lw.type == pod::QuantType::NVFP4 ) {
		// w0: packed codes [N, K / 2] bytes (host copy, in the prepacked tile order), w1: E4M3 block scales [N * K / 16] (device), alpha: the global scale
		const int n = lw.w0.shape[0], k = lw.w0.shape[1] * 2;
		std::vector<uint8_t> scales( static_cast<size_t>( n ) * k / 16 );
		backend::memcpy_sync( scales.data(), lw.w1.data, scales.size(), backend::Copy::D2H );
		svc.add_host_bytes( scales.size() );   // the codes are read in place from the tier's pinned copy
		return std::unique_ptr<Linear>( new kernels::cpu::Nvfp4Linear( static_cast<const uint8_t*>( lw.w0.data ), scales.data(), lw.alpha, k, n ) );
	}
	ENGINE_ASSERT( lw.type == pod::QuantType::TRELLIS && lw.mul1 && kernels::cpu::TrellisLinear::supported( lw.bits ), "cpu_experts: experts must be NVFP4 or mul1 trellis" );
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
	if ( !svc.started() ) svc.start( Service::parse_cpus( opt.cpu_expert_cpus ), opt.cpu_expert_spin_us );
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
	if ( rank == 0 ) LOG_INFO( "[cpu-experts] rank 0: %d cold experts on the CPU path, %.2f GiB of host-side copies (all ranks so far); up to %.0f%% of a layer's cold experts go to the CPU; node %d\n", built, svc.host_bytes() / 1073741824.0, 100.0 * opt.cpu_experts, opt.cpu_expert_node );
	return h;
}

}   // namespace engine::cpu_moe
