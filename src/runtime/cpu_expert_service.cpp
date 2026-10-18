// The CPU expert service (see cpu_expert_service.h)
#include "engine/runtime/cpu_expert_service.h"
#include "engine/backend/runtime.h"
#include "engine/core/log.h"
#include "engine/kernels/cpu/cpu_util.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <sstream>
#include <thread>
#if defined( __linux__ )
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#endif
#if defined( __x86_64__ )
#include <immintrin.h>
#endif

namespace engine::cpu_moe {

Service& Service::get() {
	static Service s;
	return s;
}

Service::~Service() {
	stop_ = true;
	if ( th_.joinable() ) th_.join();
	for ( auto& m : maps_ ) munmap( m.first, m.second );
	for ( auto& r : ranks_ ) if ( r && r->mb ) backend::host_free_pinned( r->mb );
}

void Service::start( const std::vector<int>& cpus, int spin_us ) {
	std::lock_guard<std::mutex> l( mu_ );
	if ( started_ ) return;
	started_ = true;
	const int n = cpus.empty() ? 4 : static_cast<int>( cpus.size() );
	std::vector<int> rest;
	for ( size_t i = 1; i < cpus.size(); ++i ) rest.push_back( cpus[i] );
	pool_.reset( new kernels::cpu::Pool( n, rest, spin_us ) );
	const int first = cpus.empty() ? -1 : cpus[0];
	th_ = std::thread( [this, first] {
#if defined( __linux__ )
		if ( first >= 0 ) { cpu_set_t cs; CPU_ZERO( &cs ); CPU_SET( first, &cs ); pthread_setaffinity_np( pthread_self(), sizeof cs, &cs ); }
#else
		(void) first;
#endif
		loop();
	} );
	LOG_INFO( "[cpu-experts] service started: %d threads%s\n", n, cpus.empty() ? "" : " pinned" );
}

Mailbox* Service::mailbox( int rank, int hidden, float act_limit ) {
	std::lock_guard<std::mutex> l( mu_ );
	if ( rank >= static_cast<int>( ranks_.size() ) ) ranks_.resize( rank + 1 );
	if ( !ranks_[rank] ) ranks_[rank].reset( new Rank );
	Rank& r = *ranks_[rank];
	if ( !r.mb ) {
		void* p = backend::host_alloc_pinned( mailbox_bytes( hidden ), backend::Pin::MappedPortable );
		ENGINE_ASSERT( p != nullptr, "cpu-experts: mailbox allocation failed" );
		std::memset( p, 0, mailbox_bytes( hidden ) );
		r.mb = static_cast<Mailbox*>( p );
		r.hidden = hidden;
		r.act_limit = act_limit;
		++ranks_version_;
	}
	return r.mb;
}

void Service::add_expert( int rank, int layer, int expert, std::unique_ptr<kernels::cpu::Linear> gate, std::unique_ptr<kernels::cpu::Linear> up, std::unique_ptr<kernels::cpu::Linear> down ) {
	std::lock_guard<std::mutex> l( mu_ );
	ENGINE_ASSERT( rank < static_cast<int>( ranks_.size() ) && ranks_[rank], "cpu-experts: add_expert before mailbox" );
	Rank& r = *ranks_[rank];
	if ( layer >= static_cast<int>( r.layers.size() ) ) r.layers.resize( layer + 1 );
	Owned o;
	o.gate = std::move( gate ); o.up = std::move( up ); o.down = std::move( down );
	if ( !r.layers[layer].count( expert ) ) r.layers[layer][expert] = std::move( o );   // a demoted expert that was cold before keeps its copy
}

void* Service::numa_alloc( size_t bytes, int node ) {
#if defined( __linux__ )
	void* p = mmap( nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0 );
	if ( p == MAP_FAILED ) return nullptr;
	if ( node >= 0 && node < 64 ) {
		unsigned long mask = 1ul << node;
		syscall( SYS_mbind, p, bytes, 2 /* MPOL_BIND */, &mask, 64ul, 0 );   // best effort: a failure leaves first-touch placement
	}
	std::lock_guard<std::mutex> l( mu_ );
	maps_.emplace_back( p, bytes );
	return p;
#else
	(void) node;
	return nullptr;
#endif
}

std::vector<int> Service::parse_cpus( const std::string& list ) {
	std::vector<int> out;
	std::stringstream ss( list );
	std::string tok;
	while ( std::getline( ss, tok, ',' ) ) {
		if ( tok.empty() ) continue;
		const size_t dash = tok.find( '-' );
		const int a = std::atoi( tok.substr( 0, dash ).c_str() ), b = dash == std::string::npos ? a : std::atoi( tok.substr( dash + 1 ).c_str() );
		for ( int c = a; c <= b; ++c ) out.push_back( c );
	}
	return out;
}

bool Service::find_expert( int rank, int layer, int expert, kernels::cpu::Linear** lin3 ) const {
	std::lock_guard<std::mutex> l( mu_ );
	if ( rank >= static_cast<int>( ranks_.size() ) || !ranks_[rank] || layer >= static_cast<int>( ranks_[rank]->layers.size() ) ) return false;
	auto it = ranks_[rank]->layers[layer].find( expert );
	if ( it == ranks_[rank]->layers[layer].end() ) return false;
	lin3[0] = it->second.gate.get(); lin3[1] = it->second.up.get(); lin3[2] = it->second.down.get();
	return true;
}

void Service::loop() {
	using clk = std::chrono::steady_clock;
	auto last = clk::now();
	std::vector<Rank*> active;
	uint32_t version = ~0u;
	while ( !stop_.load( std::memory_order_relaxed ) ) {
		if ( version != ranks_version_.load( std::memory_order_acquire ) ) {
			std::lock_guard<std::mutex> l( mu_ );
			version = ranks_version_.load( std::memory_order_acquire );
			active.clear();
			for ( auto& r : ranks_ ) if ( r && r->mb ) active.push_back( r.get() );
		}
		bool any = false;
		for ( Rank* r : active ) {
			if ( r->mb->req_seq != r->seen ) { handle( *r ); any = true; last = clk::now(); }
		}
		if ( any ) continue;
		// idle: spin a few ms after the last request (layers of one token come within a millisecond), then poll gently
		if ( clk::now() - last > std::chrono::milliseconds( 5 ) ) {
			for ( Rank* r : active ) {
				if ( r->mb->err && !r->err_reported ) {
					r->err_reported = true;
					LOG_ERROR( "[cpu-experts] a GPU gave up waiting for the service (the layer's CPU experts were left out)\n" );
				}
			}
			std::this_thread::sleep_for( std::chrono::microseconds( 100 ) );
		}
		else {
#if defined( __x86_64__ )
			_mm_pause();
#endif
		}
	}
}

void Service::handle( Rank& r ) {
	const auto t0 = std::chrono::steady_clock::now();
	Mailbox* mb = r.mb;
	const uint64_t seq = mb->req_seq;
	std::atomic_thread_fence( std::memory_order_acquire );
	const int rows = mb->rows, layer = mb->layer, n = mb->nroutes, H = r.hidden;
	const uint16_t* x = mailbox_x( mb );
	float* out = mailbox_out( mb, H );
	std::memset( out, 0, static_cast<size_t>( rows ) * H * sizeof( float ) );
	std::vector<int> order( n );
	for ( int i = 0; i < n; ++i ) order[i] = i;
	std::stable_sort( order.begin(), order.end(), [&]( int a, int b ) { return mb->routes[a].expert < mb->routes[b].expert; } );
	// one group per expert: the rows routed to it, gathered; all groups are evaluated together
	int groups = 0;
	for ( int i = 0; i < n; ) {
		int j = i;
		const int e = mb->routes[order[i]].expert;
		while ( j < n && mb->routes[order[j]].expert == e ) ++j;
		const int m = j - i;
		const Owned* o = nullptr;
		{
			std::lock_guard<std::mutex> l( mu_ );   // add_expert may run (a swap demoted an expert); entries are never erased, so the pointer stays valid
			if ( layer < static_cast<int>( r.layers.size() ) ) {
				auto it = r.layers[layer].find( e );
				if ( it != r.layers[layer].end() ) o = &it->second;
			}
		}
		if ( o ) {
			if ( groups == static_cast<int>( groups_.size() ) ) groups_.emplace_back();
			Group& g = groups_[groups++];
			g.o = o; g.first = i; g.m = m;
			g.xg.resize( static_cast<size_t>( m ) * H );
			g.wts.resize( m );
			g.tmp.assign( static_cast<size_t>( m ) * H, 0.0f );
			for ( int k = 0; k < m; ++k ) {
				const MailboxRoute& rt = mb->routes[order[i + k]];
				std::memcpy( &g.xg[static_cast<size_t>( k ) * H], x + static_cast<size_t>( rt.row ) * H, static_cast<size_t>( H ) * 2 );
				g.wts[k] = rt.weight;
			}
		}
		i = j;
	}
	calls_.resize( groups );
	for ( int gi = 0; gi < groups; ++gi ) {
		const Group& g = groups_[gi];
		Expert ex; ex.gate = g.o->gate.get(); ex.up = g.o->up.get(); ex.down = g.o->down.get();
		calls_[gi] = ExpertCall{ ex, g.m, g.xg.data(), g.wts.data(), groups_[gi].tmp.data() };
	}
	if ( groups ) expert_forward_many( *pool_, calls_.data(), groups, scratch_, r.act_limit );
	for ( int gi = 0; gi < groups; ++gi ) {
		const Group& g = groups_[gi];
		for ( int k = 0; k < g.m; ++k ) {
			float* dst = out + static_cast<size_t>( mb->routes[order[g.first + k]].row ) * H;
			const float* src = &g.tmp[static_cast<size_t>( k ) * H];
			for ( int c = 0; c < H; ++c ) dst[c] += src[c];
		}
	}
	const int experts = groups;
	std::atomic_thread_fence( std::memory_order_release );
	r.seen = seq;
	mb->done_seq = seq;
	++requests_; experts_ += experts; rows_ += rows;
	busy_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>( std::chrono::steady_clock::now() - t0 ).count();
	if ( requests_ % 1000 == 0 ) log_stats();
}

void Service::log_stats() {
	// the window since the last line, not the whole run
	uint64_t wait_ns = 0, waits = 0;
	{
		std::lock_guard<std::mutex> l( mu_ );
		for ( auto& rk : ranks_ ) if ( rk && rk->mb ) { wait_ns += rk->mb->wait_ns; waits += rk->mb->waits; }
	}
	const uint64_t req = requests_.load(), exp = experts_.load(), busy = busy_ns_.load();
	const uint64_t dreq = req - last_.requests, dexp = exp - last_.experts, dbusy = busy - last_.busy_ns, dwn = wait_ns - last_.wait_ns, dw = waits - last_.waits;
	last_ = { req, exp, busy, wait_ns, waits };
	LOG_INFO( "[cpu-experts] %llu requests so far; last %llu: %.2f ms per request, %.2f ms per expert, the GPUs waited %.2f ms per request\n", static_cast<unsigned long long>( req ),
		static_cast<unsigned long long>( dreq ), dreq ? dbusy * 1e-6 / dreq : 0.0, dexp ? dbusy * 1e-6 / dexp : 0.0, dw ? dwn * 1e-6 / dw : 0.0 );
}

}   // namespace engine::cpu_moe
