// Routed expert on the host CPU (see cpu_experts.h)
#include "engine/runtime/cpu_experts.h"
#include "engine/kernels/cpu/cpu_util.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace engine::cpu_moe {

using namespace engine::kernels::cpu;

namespace {

constexpr int kUnitsPerBlock = kBlock / 16;
constexpr int kJobs = 16;      // experts evaluated at once
constexpr int kManyTasks = 4;
constexpr int kPieceWeights = 40000;
// A piece of work (a unit of 16 columns takes about 10 us, a block's finishing step a few) that has been held longer than this by a thread is taken
// over: that thread was most likely descheduled
constexpr uint64_t kUnitStaleNs = 80000, kFinishStaleNs = 40000;

inline float silu( float x ) { return x / ( 1.0f + std::exp( -x ) ); }

inline uint64_t now_ns() { return static_cast<uint64_t>( std::chrono::duration_cast<std::chrono::nanoseconds>( std::chrono::steady_clock::now().time_since_epoch() ).count() ); }

// A piece of work: 0 free, 1 claimed (at ts), 2 done. Whoever moves it from claimed to done counts it; a takeover redoes it, which rewrites the
// same values
struct Piece {
	std::atomic<uint8_t> state{ 0 };
	std::atomic<uint64_t> ts{ 0 };
};

}   // namespace

struct ExpertJob : kernels::cpu::Task {
	// inputs
	Expert e;
	int rows = 0, H = 0, I = 0, nu = 0, nb_gu = 0, nu_d = 0, nb_d = 0;
	int cg_gu = 1, cg_d = 1, nch_gu = 0, nch_d = 0;   // units per piece (a piece is about 40k weights), pieces per matrix
	float act_limit = 0.0f, wts[kMaxRows] = {};
	// work memory (resized by prepare())
	std::vector<float> xf, xg, xu, raw_g, raw_u, act_in, raw_d, bsum, out_tmp;
	float xsum[2 * kMaxRows] = {}, dsum[kMaxRows] = {};
	// progress
	std::vector<Piece> unit;                     // [gate units | up units | down units]
	std::vector<std::atomic<int>> unit_done_b;   // done units per block: [gu blocks | down blocks]
	std::vector<Piece> fin;                      // finishing steps: [gu blocks | down blocks]
	std::atomic<int> next_unit{ 0 }, gu_blocks_done{ 0 }, d_blocks_done{ 0 };
	std::atomic<bool> done{ false };

	bool finished() const override { return done.load( std::memory_order_acquire ); }

	void prepare( const Expert& ex, int r, const uint16_t* x, const float* w, float limit ) {
		e = ex; rows = r; act_limit = limit;
		H = e.hidden(); I = e.inter();
		nu = e.gate->units(); nb_gu = I / kBlock; nu_d = e.down->units(); nb_d = H / kBlock;
		{
			// pieces of about the same work whatever the matrix: 16 columns of K = 2560 are 41k weights
			auto chunk = [&]( int k ) { int c = 1; while ( c < kUnitsPerBlock && c * 16 * k < kPieceWeights ) c *= 2; return c; };
			cg_gu = chunk( H ); cg_d = chunk( I );
			nch_gu = nu / cg_gu; nch_d = ( nu_d + cg_d - 1 ) / cg_d;
		}
		if ( rows < 1 || rows > kMaxRows || e.up->out() != I || e.down->in() != I || e.down->out() != H ) std::abort();
		for ( int i = 0; i < rows; ++i ) wts[i] = w[i];
		const size_t rH = static_cast<size_t>( rows ) * H, rI = static_cast<size_t>( rows ) * I;
		xf.resize( rH ); xg.resize( rH ); xu.resize( rH ); raw_d.resize( rH ); out_tmp.assign( rH, 0.0f );
		raw_g.resize( rI ); raw_u.resize( rI ); act_in.resize( rI ); bsum.assign( static_cast<size_t>( rows ) * nb_gu, 0.0f );
		half_to_float( x, xf.data(), rH );
		for ( int i = 0; i < 2 * rows; ++i ) xsum[i] = 0.0f;
		for ( int rr = 0; rr < rows; ++rr ) {
			double a = 0, b = 0;
			for ( int blk = 0; blk < H / kBlock; ++blk ) {
				const size_t o = static_cast<size_t>( rr ) * H + static_cast<size_t>( blk ) * kBlock;
				e.gate->prepare_block( &xf[o], blk, &xg[o] );
				e.up->prepare_block( &xf[o], blk, &xu[o] );
				if ( e.gate->needs_row_sum() ) for ( int i = 0; i < kBlock; ++i ) a += xg[o + i];
				if ( e.up->needs_row_sum() ) for ( int i = 0; i < kBlock; ++i ) b += xu[o + i];
			}
			xsum[rr] = static_cast<float>( a ); xsum[rows + rr] = static_cast<float>( b );
		}
		const size_t total_units = static_cast<size_t>( 2 * nch_gu + nch_d ), total_fin = static_cast<size_t>( nb_gu + nb_d );
		if ( unit.size() != total_units ) { std::vector<Piece> u( total_units ); unit.swap( u ); }
		if ( fin.size() != total_fin ) { std::vector<Piece> f( total_fin ); fin.swap( f ); }
		if ( unit_done_b.size() != total_fin ) { std::vector<std::atomic<int>> c( total_fin ); unit_done_b.swap( c ); }
		for ( auto& p : unit ) { p.state.store( 0, std::memory_order_relaxed ); p.ts.store( 0, std::memory_order_relaxed ); }
		for ( auto& p : fin ) { p.state.store( 0, std::memory_order_relaxed ); p.ts.store( 0, std::memory_order_relaxed ); }
		for ( auto& c : unit_done_b ) c.store( 0, std::memory_order_relaxed );
		next_unit.store( 0, std::memory_order_relaxed ); gu_blocks_done.store( 0, std::memory_order_relaxed ); d_blocks_done.store( 0, std::memory_order_relaxed );
		dsum_ready.store( false, std::memory_order_relaxed );
		done.store( false, std::memory_order_release );
	}

	// Gate/up block b for every row: output transforms, FP16 rounding, SwiGLU, then the down input's transform (block b of the intermediate is block b
	// of the down input); bsum gets the block's sum of that input
	void finish_gu( int b ) {
		float g[kBlock], u[kBlock], a[kBlock];
		for ( int r = 0; r < rows; ++r ) {
			const size_t base = static_cast<size_t>( r ) * I + static_cast<size_t>( b ) * kBlock;
			std::memcpy( g, &raw_g[base], sizeof g );
			std::memcpy( u, &raw_u[base], sizeof u );
			e.gate->finish_block( g, b );
			e.up->finish_block( u, b );
			round_to_half( g, kBlock ); round_to_half( u, kBlock );
			for ( int i = 0; i < kBlock; ++i ) {
				const float gc = act_limit > 0 ? std::fmin( g[i], act_limit ) : g[i], uc = act_limit > 0 ? std::fmax( -act_limit, std::fmin( u[i], act_limit ) ) : u[i];
				a[i] = silu( gc ) * uc;
			}
			round_to_half( a, kBlock );
			e.down->prepare_block( a, b, &act_in[base] );
			float sum = 0;
			if ( e.down->needs_row_sum() ) for ( int i = 0; i < kBlock; ++i ) sum += act_in[base + i];
			bsum[static_cast<size_t>( r ) * nb_gu + b] = sum;
		}
	}

	// Down block b: output transform and the route weight, into out_tmp
	void finish_d( int b ) {
		float d[kBlock];
		for ( int r = 0; r < rows; ++r ) {
			std::memcpy( d, &raw_d[static_cast<size_t>( r ) * H + static_cast<size_t>( b ) * kBlock], sizeof d );
			e.down->finish_block( d, b );
			round_to_half( d, kBlock );
			for ( int i = 0; i < kBlock; ++i ) out_tmp[static_cast<size_t>( r ) * H + b * kBlock + i] = wts[r] * d[i];
		}
	}

	// Claims piece p (free, or claimed too long ago) for the caller
	static bool claim( Piece& p, uint64_t stale_ns, uint64_t now ) {
		uint8_t st = p.state.load( std::memory_order_acquire );
		if ( st == 0 ) {
			if ( !p.state.compare_exchange_strong( st, 1, std::memory_order_acq_rel ) ) return false;
		}
		else if ( st == 1 ) {
			const uint64_t t = p.ts.load( std::memory_order_acquire );
			if ( now - t < stale_ns ) return false;
		}
		else return false;
		p.ts.store( now, std::memory_order_release );
		return true;
	}

	// 1 -> 2; true for the thread that got there first
	static bool complete( Piece& p ) {
		uint8_t st = 1;
		return p.state.compare_exchange_strong( st, 2, std::memory_order_acq_rel );
	}

	void run_unit( int u ) {
		if ( u >= 2 * nch_gu ) {
			const int c = u - 2 * nch_gu;
			e.down->run( rows, act_in.data(), dsum, c * cg_d, std::min( nu_d, ( c + 1 ) * cg_d ), raw_d.data() );
		}
		else if ( u >= nch_gu ) e.up->run( rows, xu.data(), &xsum[rows], ( u - nch_gu ) * cg_gu, ( u - nch_gu + 1 ) * cg_gu, raw_u.data() );
		else e.gate->run( rows, xg.data(), xsum, u * cg_gu, ( u + 1 ) * cg_gu, raw_g.data() );
	}

	// The finishing step of block `blk` (indices as unit_done_b: gate/up blocks, then down blocks), if its units are all done and nobody holds it
	bool try_finish( int blk, uint64_t now ) {
		const bool down = blk >= nb_gu;
		const int b = down ? blk - nb_gu : blk;
		if ( unit_done_b[blk].load( std::memory_order_acquire ) < ( down ? 1 : 2 ) * kUnitsPerBlock || !claim( fin[blk], kFinishStaleNs, now ) ) return false;
		if ( !down ) {
			finish_gu( b );
			if ( complete( fin[blk] ) && gu_blocks_done.fetch_add( 1, std::memory_order_acq_rel ) + 1 == nb_gu ) {
				// the down input's row sums (a few adds), then the down units become available
				for ( int r = 0; r < rows; ++r ) { float t = 0; for ( int bb = 0; bb < nb_gu; ++bb ) t += bsum[static_cast<size_t>( r ) * nb_gu + bb]; dsum[r] = t; }
				dsum_ready.store( true, std::memory_order_release );
			}
		}
		else {
			finish_d( b );
			if ( complete( fin[blk] ) && d_blocks_done.fetch_add( 1, std::memory_order_acq_rel ) + 1 == nb_d ) done.store( true, std::memory_order_release );
		}
		return true;
	}

	bool work() override {
		if ( finished() ) return false;
		const uint64_t now = now_ns();
		const int total = 2 * nch_gu + ( dsum_ready.load( std::memory_order_acquire ) ? nch_d : 0 );
		// a fresh unit in order (the common case: no scanning)
		int u = next_unit.load( std::memory_order_relaxed );
		while ( u < total ) {
			if ( next_unit.compare_exchange_weak( u, u + 1, std::memory_order_acq_rel ) ) {
				if ( claim( unit[u], kUnitStaleNs, now ) ) { do_unit( u ); return true; }
				u = next_unit.load( std::memory_order_relaxed );
			}
		}
		// nothing fresh: finishing steps that nobody took (or took too long ago), then units held too long
		for ( int blk = 0; blk < nb_gu + nb_d; ++blk )
			if ( try_finish( blk, now ) ) return true;
		for ( int i = 0; i < total; ++i )
			if ( unit[i].state.load( std::memory_order_acquire ) == 1 && claim( unit[i], kUnitStaleNs, now ) ) { do_unit( i ); return true; }
		return false;
	}

	void do_unit( int u ) {
		run_unit( u );
		if ( complete( unit[u] ) ) {
			// the block of this piece and how many 16-column units the piece holds
			int blk, units;
			if ( u >= 2 * nch_gu ) { const int c = u - 2 * nch_gu; blk = nb_gu + ( c * cg_d ) / kUnitsPerBlock; units = std::min( nu_d, ( c + 1 ) * cg_d ) - c * cg_d; }
			else { const int c = u >= nch_gu ? u - nch_gu : u; blk = ( c * cg_gu ) / kUnitsPerBlock; units = cg_gu; }
			// the thread that completes a block's last unit finishes the block at once
			const int need = ( blk >= nb_gu ? 1 : 2 ) * kUnitsPerBlock;
			if ( unit_done_b[blk].fetch_add( units, std::memory_order_acq_rel ) + units == need ) try_finish( blk, now_ns() );
		}
	}

	std::atomic<bool> dsum_ready{ false };
};

// Several experts' jobs as one task: a thread takes a piece from the first job that has one, starting at a different job each time, so the tail of one
// expert (its last pieces, the barrier before its down projection) is filled with the pieces of the others
struct ManyTask : kernels::cpu::Task {
	ExpertJob* jobs[kJobs] = {};
	int n = 0;
	std::atomic<uint32_t> rr{ 0 };
	bool finished() const override {
		for ( int i = 0; i < n; ++i ) if ( !jobs[i]->finished() ) return false;
		return true;
	}
	bool work() override {
		const uint32_t start = rr.fetch_add( 1, std::memory_order_relaxed );
		for ( int i = 0; i < n; ++i ) {
			ExpertJob* j = jobs[( start + i ) % n];
			if ( j->finished() ) continue;
			j->refs.fetch_add( 1, std::memory_order_acq_rel );   // the job may not be reused while this thread is inside it
			const bool did = j->work();
			j->refs.fetch_sub( 1, std::memory_order_acq_rel );
			if ( did ) return true;
		}
		return false;
	}
};

Scratch::Scratch() {
	for ( int i = 0; i < kJobs; ++i ) jobs.emplace_back( new ExpertJob );
	for ( int i = 0; i < kManyTasks; ++i ) many.emplace_back( new ManyTask );
}
Scratch::~Scratch() {}

namespace {
// The next object of a ring that nobody holds (a descheduled pool thread may still hold an old one)
template <class T>
T* acquire( std::vector<std::unique_ptr<T>>& ring, size_t& next ) {
	for ( ;; ) {
		for ( size_t tries = 0; tries < ring.size(); ++tries ) {
			T* c = ring[next++ % ring.size()].get();
			if ( c->refs.load( std::memory_order_acquire ) == 0 ) return c;
		}
		std::this_thread::yield();
	}
}
}   // namespace

void expert_forward_many( Pool& pool, const ExpertCall* calls, int count, Scratch& s, float act_limit ) {
	for ( int base = 0; base < count; base += kJobs ) {
		const int n = std::min( kJobs, count - base );
		ManyTask* task = acquire( s.many, s.next_many );
		task->n = n;
		for ( int i = 0; i < n; ++i ) {
			// the jobs of a wave stay distinct: take them in order, not holding two of the same
			ExpertJob* j = acquire( s.jobs, s.next );
			for ( int k = 0; k < i && task->jobs[k] == j; ++k ) j = acquire( s.jobs, s.next );
			const ExpertCall& c = calls[base + i];
			j->prepare( c.e, c.rows, c.x, c.wts, act_limit );
			task->jobs[i] = j;
		}
		pool.run( *task );
		for ( int i = 0; i < n; ++i ) {
			const ExpertCall& c = calls[base + i];
			const size_t m = static_cast<size_t>( c.rows ) * task->jobs[i]->H;
			for ( size_t q = 0; q < m; ++q ) c.out[q] += task->jobs[i]->out_tmp[q];
		}
	}
}

void expert_forward( Pool& pool, const Expert& e, int rows, const uint16_t* x, const float* wts, float* out, Scratch& s, float act_limit ) {
	const ExpertCall c = { e, rows, x, wts, out };
	expert_forward_many( pool, &c, 1, s, act_limit );
}

}   // namespace engine::cpu_moe
