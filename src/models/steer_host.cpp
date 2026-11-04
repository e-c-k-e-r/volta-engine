#include "engine/models/steer_host.h"
#include "engine/backend/runtime.h"
#include "engine/core/log.h"
#include "engine/runtime/options.h"
#include "engine/spec/tap_ring.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>

namespace engine::models {

using namespace engine::modules;

void SteerHost::init( pod::MemoryArena& arena, int hidden, int num_layers, int slots, int rank, int mean_rows, int route_experts ) {
	if ( runtime::options().steer == "off" ) return;
	int first = 0, last = 0;
	ENGINE_ASSERT( parse_steer_layers( runtime::options().steer_layers, num_layers, &first, &last ), "steer_layers is malformed or out of range" );
	const size_t H = hidden, nl = last - first + 1, K = kSteerEntries, S = slots;
	rank_ = rank;
	t_.hidden = hidden;
	t_.layers = static_cast<int>( nl );
	t_.slots = slots;
	first_ = first;
	t_.vec = static_cast<float*>( core::arena_alloc( arena, S * nl * K * H * sizeof( float ) ) );
	t_.par = static_cast<float*>( core::arena_alloc( arena, S * nl * K * sizeof( float ) ) );
	t_.kind = static_cast<int32_t*>( core::arena_alloc( arena, S * nl * K * sizeof( int32_t ) ) );
	t_.ctl = static_cast<SteerSlotCtl*>( core::arena_alloc( arena, S * sizeof( SteerSlotCtl ) ) );
	t_.penalty_ids = static_cast<int32_t*>( core::arena_alloc( arena, S * kSteerPenaltyIds * sizeof( int32_t ) ) );
	t_.probe = static_cast<float*>( core::arena_alloc( arena, S * kProbes * kProbeRing * sizeof( float ) ) );
	cap_ = static_cast<backend::half_t*>( core::arena_alloc( arena, static_cast<size_t>( num_layers ) * H * sizeof( backend::half_t ) ) );
	if ( mean_rows > 0 ) mean_ = static_cast<backend::half_t*>( core::arena_alloc( arena, static_cast<size_t>( mean_rows ) * H * sizeof( backend::half_t ) ) );
	backend::memset_sync( t_.vec, 0, S * nl * K * H * sizeof( float ) );
	backend::memset_sync( t_.par, 0, S * nl * K * sizeof( float ) );
	backend::memset_sync( t_.kind, 0, S * nl * K * sizeof( int32_t ) );
	backend::memset_sync( t_.ctl, 0, S * sizeof( SteerSlotCtl ) );
	backend::memset_sync( t_.penalty_ids, 0, S * kSteerPenaltyIds * sizeof( int32_t ) );
	backend::memset_sync( t_.probe, 0, S * kProbes * kProbeRing * sizeof( float ) );
	if ( route_experts > 0 ) {
		t_.route_layers = num_layers;
		t_.route_experts = route_experts;
		const size_t rb = S * static_cast<size_t>( num_layers ) * route_experts * sizeof( float );
		t_.route = static_cast<float*>( core::arena_alloc( arena, rb ) );
		backend::memset_sync( t_.route, 0, rb );
	}
	ctl_host_.assign( S, SteerSlotCtl{} );
	cap_layer_.assign( num_layers, -1 );
	if ( rank == 0 ) LOG_INFO( "[steer] layers %d-%d of %d, %zu KiB per slot\n", first, last, num_layers, nl * K * H * sizeof( float ) / 1024 );
}

void SteerHost::set( int slot, const std::vector<SteerEntry>& entries, backend::stream_t stream ) {
	ENGINE_ASSERT( on() && slot >= 0 && slot < t_.slots, "SteerHost::set: steering is off or the slot is out of range" );
	backend::set_device( rank_ );
	constexpr int K = kSteerEntries;
	const size_t H = t_.hidden, per_slot = static_cast<size_t>( t_.layers ) * K;
	std::vector<float> vec( per_slot * H, 0.0f ), par( per_slot, 0.0f );
	std::vector<int32_t> kind( per_slot, STEER_OFF );
	// a probe's first layer writes its ring cell, the later ones add to it
	int probe_first[kProbes];
	for ( int& f : probe_first ) f = INT32_MAX;
	int probes = 0;
	// router bias entries: the slot's whole table (zero where no entry sets it)
	std::vector<float> route;
	if ( t_.route ) route.assign( static_cast<size_t>( route_stride() ), 0.0f );
	for ( const auto& e : entries ) {
		if ( e.kind != STEER_ROUTE ) continue;
		ENGINE_ASSERT( t_.route && e.layer >= 0 && e.layer < t_.route_layers && e.expert >= 0 && e.expert < t_.route_experts, "SteerHost::set: router entry out of range (or the model has no router steering)" );
		route[static_cast<size_t>( e.layer ) * t_.route_experts + e.expert] += e.par;
	}
	if ( t_.route ) backend::memcpy_async( t_.route + static_cast<size_t>( slot ) * route_stride(), route.data(), route.size() * sizeof( float ), backend::Copy::H2D, stream );
	for ( const auto& e : entries ) {
		if ( e.kind != STEER_PROBE ) continue;
		ENGINE_ASSERT( e.probe >= 0 && e.probe < kProbes, "SteerHost::set: probe index out of range" );
		probe_first[e.probe] = std::min( probe_first[e.probe], e.layer );
		probes = std::max( probes, e.probe + 1 );
	}
	// probe entries first on their layer, so they read the state before the edits
	for ( int pass = 0; pass < 2; ++pass ) {
		for ( const auto& e : entries ) {
			if ( e.kind == STEER_ROUTE || ( e.kind == STEER_PROBE ) != ( pass == 0 ) ) continue;
			const int li = e.layer - first_;
			ENGINE_ASSERT( li >= 0 && li < t_.layers, "SteerHost::set: entry layer outside the steered range" );
			ENGINE_ASSERT( e.vec.size() == H && ( e.kind == STEER_ADD || e.kind == STEER_CLAMP || e.kind == STEER_CLAMP_KEEP || e.kind == STEER_PROBE ), "SteerHost::set: bad entry" );
			int k = 0;
			while ( k < K && kind[static_cast<size_t>( li ) * K + k] != STEER_OFF ) ++k;
			ENGINE_ASSERT( k < K, "SteerHost::set: more than kSteerEntries entries on a layer" );
			const size_t idx = static_cast<size_t>( li ) * K + k;
			kind[idx] = e.kind == STEER_PROBE ? kSteerProbeKind + 2 * e.probe + ( e.layer == probe_first[e.probe] ? 1 : 0 ) : e.kind;
			par[idx] = e.par;
			std::memcpy( vec.data() + idx * H, e.vec.data(), H * sizeof( float ) );
		}
	}
	SteerSlotCtl& c = ctl_host_[slot];
	c.probe_n = probes;
	backend::memcpy_async( t_.ctl + slot, &c, sizeof( c ), backend::Copy::H2D, stream );
	if ( !entries.empty() ) {   // without entries every kind is off and the vectors are never read
		backend::memcpy_async( t_.vec + slot * per_slot * H, vec.data(), vec.size() * sizeof( float ), backend::Copy::H2D, stream );
		backend::memcpy_async( t_.par + slot * per_slot, par.data(), par.size() * sizeof( float ), backend::Copy::H2D, stream );
	}
	backend::memcpy_async( t_.kind + slot * per_slot, kind.data(), kind.size() * sizeof( int32_t ), backend::Copy::H2D, stream );
}

void SteerHost::control( int slot, int scope, int prompt_len, int think_end, backend::stream_t stream ) {
	ENGINE_ASSERT( on() && slot >= 0 && slot < t_.slots, "SteerHost::control: steering is off or the slot is out of range" );
	backend::set_device( rank_ );
	SteerSlotCtl& c = ctl_host_[slot];
	c.scope = scope;
	c.prompt_len = prompt_len;
	c.think_end = think_end;
	backend::memcpy_async( t_.ctl + slot, &c, sizeof( c ), backend::Copy::H2D, stream );
}

void SteerHost::penalty( int slot, const std::vector<int>& ids, float penalty, backend::stream_t stream ) {
	ENGINE_ASSERT( on() && slot >= 0 && slot < t_.slots, "SteerHost::penalty: steering is off or the slot is out of range" );
	ENGINE_ASSERT( ids.size() <= static_cast<size_t>( kSteerPenaltyIds ), "SteerHost::penalty: too many ids" );
	backend::set_device( rank_ );
	SteerSlotCtl& c = ctl_host_[slot];
	c.penalty_n = static_cast<int>( ids.size() );
	c.penalty = penalty;
	if ( !ids.empty() ) backend::memcpy_async( t_.penalty_ids + static_cast<size_t>( slot ) * kSteerPenaltyIds, ids.data(), ids.size() * sizeof( int ), backend::Copy::H2D, stream );
	backend::memcpy_async( t_.ctl + slot, &c, sizeof( c ), backend::Copy::H2D, stream );
}

void SteerHost::capture( const std::vector<int>& layers, bool on ) {
	ENGINE_ASSERT( this->on(), "SteerHost::capture: steering is off" );
	cap_layer_.assign( cap_layer_.size(), -1 );
	cap_order_.clear();
	if ( on ) {
		for ( int l : layers ) {
			ENGINE_ASSERT( l >= 0 && l < static_cast<int>( cap_layer_.size() ) && cap_layer_[l] < 0, "SteerHost::capture: bad or repeated layer" );
			cap_layer_[l] = static_cast<int>( cap_order_.size() );
			cap_order_.push_back( l );
		}
	}
	cap_on_ = on;
}

namespace {
float fp16_bits_to_float( uint16_t h ) {
	const uint32_t sign = ( h >> 15 ) & 1u, exp = ( h >> 10 ) & 31u, man = h & 1023u;
	float v;
	if ( exp == 0 ) v = std::ldexp( static_cast<float>( man ), -24 );
	else if ( exp == 31 ) v = man ? NAN : INFINITY;
	else v = std::ldexp( static_cast<float>( man | 1024u ), static_cast<int>( exp ) - 25 );
	return sign ? -v : v;
}
}   // namespace

bool SteerHost::read_probes( int slot, int first_pos, int count, float* out, backend::stream_t stream ) {
	if ( !on() || slot < 0 || slot >= t_.slots || first_pos < 0 || count <= 0 || count > kProbeRing ) return false;
	backend::set_device( rank_ );
	for ( int p = 0; p < kProbes; ++p ) {
		const float* ring = t_.probe + ( static_cast<size_t>( slot ) * kProbes + p ) * kProbeRing;
		// position q lives in ring cell q % kProbeRing: at most two contiguous runs
		for ( int done = 0; done < count; ) {
			const int r = ( first_pos + done ) % kProbeRing;
			const int n = std::min( count - done, kProbeRing - r );
			backend::memcpy_async( out + static_cast<size_t>( p ) * count + done, ring + r, static_cast<size_t>( n ) * sizeof( float ), backend::Copy::D2H, stream );
			done += n;
		}
	}
	backend::stream_sync( stream );
	return true;
}

bool SteerHost::read( std::vector<float>& out, backend::stream_t stream ) {
	if ( !cap_on_ || cap_order_.empty() ) return false;
	backend::set_device( rank_ );
	const size_t H = t_.hidden, n = cap_order_.size();
	std::vector<uint16_t> raw( n * H );
	backend::memcpy_async( raw.data(), cap_, raw.size() * sizeof( uint16_t ), backend::Copy::D2H, stream );
	backend::stream_sync( stream );
	out.resize( n * H );
	for ( size_t i = 0; i < out.size(); ++i ) out[i] = fp16_bits_to_float( raw[i] );
	return true;
}

void SteerHost::layer_rows( backend::half_t* hidden, const pod::BatchRows* d_rows, int m, int layer, backend::stream_t stream ) const {
	if ( !on() ) return;
	if ( const int li = index( layer ); li >= 0 ) ResidualSteer::apply( t_, hidden, d_rows, m, li, stream );
	if ( backend::half_t* dst = capture_dst( layer ) )
		backend::memcpy_async( dst, hidden + static_cast<size_t>( m - 1 ) * t_.hidden, t_.hidden * sizeof( backend::half_t ), backend::Copy::D2D, stream );
}

void SteerHost::layer_streams( float* streams, const pod::BatchRows* d_rows, int m, int layer, backend::stream_t stream ) const {
	if ( !on() ) return;
	if ( const int li = index( layer ); li >= 0 ) ResidualSteer::apply_streams( t_, streams, 4, d_rows, m, li, stream );
	if ( backend::half_t* dst = capture_dst( layer ) ) {
		ENGINE_ASSERT( mean_ != nullptr, "SteerHost::layer_streams: init without mean_rows" );
		engine::spec::dflash_reduce_streams( streams, mean_, m, t_.hidden, 0, stream );
		backend::memcpy_async( dst, mean_ + static_cast<size_t>( m - 1 ) * t_.hidden, t_.hidden * sizeof( backend::half_t ), backend::Copy::D2D, stream );
	}
}

void SteerHost::penalize( backend::half_t* logits, int logit_cols, const pod::BatchRows* d_rows, int row0, int n, int shard_lo, int shard_cols, backend::stream_t stream ) const {
	if ( on() ) ResidualSteer::penalize( t_, logits, logit_cols, d_rows, row0, n, shard_lo, shard_cols, stream );
}

} // namespace engine::models
