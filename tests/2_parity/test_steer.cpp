#include "engine/modules/steer.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "test_util.h"
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// ResidualSteer against an fp32 host reference: a batch whose rows belong to slots with different edits and scopes. Rows the scope does not
// cover must come back bit-identical; the edited rows may differ from the reference by fp16 rounding only.
using namespace engine::modules;

static float rnd( uint32_t& s ) { s = s * 1664525u + 1013904223u; return ( static_cast<float>( s >> 8 ) / 16777216.0f ) - 0.5f; }

int main() {
	std::printf( "========================================================\n" );
	std::printf( "   volta-engine parity: ResidualSteer (edits, penalty)  \n" );
	std::printf( "========================================================\n" );
	CUDA_CHECK( cudaSetDevice( 0 ) );
	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );

	const int H = 5120, slots = 4, layers = 3, rows = 12;
	constexpr int K = kSteerEntries;
	uint32_t seed = 7;
	std::vector<float> vec( static_cast<size_t>( slots ) * layers * K * H );
	std::vector<float> par( static_cast<size_t>( slots ) * layers * K, 0.0f );
	std::vector<int32_t> kind( par.size(), STEER_OFF );
	for ( size_t e = 0; e < par.size(); ++e ) {
		double n = 0;
		for ( int i = 0; i < H; ++i ) { vec[e * H + i] = rnd( seed ); n += vec[e * H + i] * vec[e * H + i]; }
		for ( int i = 0; i < H; ++i ) vec[e * H + i] /= static_cast<float>( std::sqrt( n ) );
	}
	auto set = [&]( int slot, int layer, int k, int kd, float p ) { const size_t i = ( static_cast<size_t>( slot ) * layers + layer ) * K + k; kind[i] = kd; par[i] = p; };
	// slot 1: add then clamp on layer 1; slot 2: clamp only (reference level below the projections, so it always fires); slot 3: add on layer 0
	set( 1, 1, 0, STEER_ADD, 3.0f ); set( 1, 1, 1, STEER_CLAMP, 0.5f );
	set( 2, 1, 0, STEER_CLAMP, -2.0f );
	set( 2, 1, 1, STEER_CLAMP_KEEP, -1.0f ); set( 2, 1, 2, STEER_CLAMP, -0.5f );   // a norm-keeping clamp, and a third edit on the layer
	set( 3, 0, 0, STEER_ADD, -1.5f );
	std::vector<SteerSlotCtl> ctl( slots, SteerSlotCtl{} );
	ctl[1].scope = STEER_ALL;                     ctl[1].prompt_len = 4; ctl[1].think_end = INT_MAX;
	ctl[2].scope = STEER_GENERATED;               ctl[2].prompt_len = 4; ctl[2].think_end = 7;
	ctl[3].scope = STEER_PROMPT;                  ctl[3].prompt_len = 4; ctl[3].think_end = INT_MAX;
	// slot 0 stays unsteered (scope 0); slot 3's penalty tests below
	std::vector<int32_t> pen_ids( static_cast<size_t>( slots ) * kSteerPenaltyIds, 0 );
	const int vocab = 1000, shard_lo = 400, shard_cols = 300, logit_cols = 320;
	ctl[2].penalty_n = 3; ctl[2].penalty = 4.0f;
	pen_ids[2 * kSteerPenaltyIds + 0] = 410; pen_ids[2 * kSteerPenaltyIds + 1] = 699; pen_ids[2 * kSteerPenaltyIds + 2] = 900;   // 900 lies outside this shard
	(void) vocab;

	pod::BatchRows br;
	br.count = rows;
	const int row_slot[rows] = { 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3 };
	const int row_pos[rows]  = { 5, 2, 3, 4, 9, 3, 4, 5, 6, 8, 2, 6 };
	for ( int r = 0; r < rows; ++r ) { br.slot[r] = row_slot[r]; br.pos[r] = row_pos[r]; }

	std::vector<half> hx( static_cast<size_t>( rows ) * H );
	std::vector<float> fx( hx.size() );
	for ( size_t i = 0; i < hx.size(); ++i ) { hx[i] = float_to_half( rnd( seed ) * 8.0f ); fx[i] = half_to_float( hx[i] ); }

	SteerTables t;
	t.hidden = H; t.layers = layers; t.slots = slots;
	CUDA_CHECK( cudaMalloc( &t.vec, vec.size() * 4 ) );  CUDA_CHECK( cudaMemcpy( t.vec, vec.data(), vec.size() * 4, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMalloc( &t.par, par.size() * 4 ) );  CUDA_CHECK( cudaMemcpy( t.par, par.data(), par.size() * 4, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMalloc( &t.kind, kind.size() * 4 ) ); CUDA_CHECK( cudaMemcpy( t.kind, kind.data(), kind.size() * 4, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMalloc( &t.ctl, ctl.size() * sizeof( SteerSlotCtl ) ) ); CUDA_CHECK( cudaMemcpy( t.ctl, ctl.data(), ctl.size() * sizeof( SteerSlotCtl ), cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMalloc( &t.penalty_ids, pen_ids.size() * 4 ) ); CUDA_CHECK( cudaMemcpy( t.penalty_ids, pen_ids.data(), pen_ids.size() * 4, cudaMemcpyHostToDevice ) );
	pod::BatchRows* d_rows;
	CUDA_CHECK( cudaMalloc( &d_rows, sizeof( pod::BatchRows ) ) ); CUDA_CHECK( cudaMemcpy( d_rows, &br, sizeof( br ), cudaMemcpyHostToDevice ) );
	half* d_x;
	CUDA_CHECK( cudaMalloc( &d_x, hx.size() * sizeof( half ) ) );

	std::printf( "[*] residual edits: add, clamp, scopes, untouched slots...\n" );
	for ( int layer = 0; layer < layers; ++layer ) {
		CUDA_CHECK( cudaMemcpy( d_x, hx.data(), hx.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
		ResidualSteer::apply( t, d_x, d_rows, rows, layer, stream );
		CUDA_CHECK( cudaStreamSynchronize( stream ) );
		std::vector<half> got( hx.size() );
		CUDA_CHECK( cudaMemcpy( got.data(), d_x, got.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
		int edited = 0;
		for ( int r = 0; r < rows; ++r ) {
			const int s = row_slot[r], p = row_pos[r];
			const int cls = p < ctl[s].prompt_len ? STEER_PROMPT : ( p < ctl[s].think_end ? STEER_THINK : STEER_ANSWER );
			bool any = false;
			for ( int k = 0; k < K; ++k ) any |= kind[( static_cast<size_t>( s ) * layers + layer ) * K + k] != STEER_OFF;
			const bool active = ( ctl[s].scope & cls ) != 0 && any;
			std::vector<double> x( fx.begin() + static_cast<size_t>( r ) * H, fx.begin() + static_cast<size_t>( r + 1 ) * H );
			if ( active ) {
				++edited;
				for ( int k = 0; k < K; ++k ) {
					const size_t e = ( static_cast<size_t>( s ) * layers + layer ) * K + k;
					if ( kind[e] == STEER_OFF ) continue;
					const float* v = &vec[e * H];
					if ( kind[e] == STEER_ADD ) { for ( int i = 0; i < H; ++i ) x[i] += par[e] * v[i]; }
					else {
						double dot = 0, n0 = 0; for ( int i = 0; i < H; ++i ) { dot += x[i] * v[i]; n0 += x[i] * x[i]; }
						if ( dot - par[e] > 0 ) {
							double n1 = 0;
							for ( int i = 0; i < H; ++i ) { x[i] -= ( dot - par[e] ) * v[i]; n1 += x[i] * x[i]; }
							if ( kind[e] == STEER_CLAMP_KEEP ) for ( int i = 0; i < H; ++i ) x[i] *= std::sqrt( n0 / n1 );
						}
					}
				}
			}
			for ( int i = 0; i < H; ++i ) {
				const half g = got[static_cast<size_t>( r ) * H + i];
				if ( !active ) ENGINE_ASSERT( std::memcmp( &g, &hx[static_cast<size_t>( r ) * H + i], sizeof( half ) ) == 0, "a row outside the scope changed" );
				else ENGINE_ASSERT( std::fabs( half_to_float( g ) - x[i] ) <= 2e-3 * std::fabs( x[i] ) + 2e-3, "steered row differs from the fp32 reference" );
			}
		}
		std::printf( "    layer %d: %d of %d rows edited, the rest bit-identical\n", layer, edited, rows );
	}

	std::printf( "[*] streams (hc = 4): the entries act on the mean, every stream takes the change...\n" );
	{
		const int hc = 4;
		std::vector<float> hs( static_cast<size_t>( rows ) * hc * H );
		for ( float& v : hs ) v = rnd( seed ) * 6.0f;
		float* d_s;
		CUDA_CHECK( cudaMalloc( &d_s, hs.size() * sizeof( float ) ) );
		for ( int layer = 0; layer < layers; ++layer ) {
			CUDA_CHECK( cudaMemcpy( d_s, hs.data(), hs.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
			ResidualSteer::apply_streams( t, d_s, hc, d_rows, rows, layer, stream );
			CUDA_CHECK( cudaStreamSynchronize( stream ) );
			std::vector<float> got( hs.size() );
			CUDA_CHECK( cudaMemcpy( got.data(), d_s, got.size() * sizeof( float ), cudaMemcpyDeviceToHost ) );
			int edited = 0;
			for ( int r = 0; r < rows; ++r ) {
				const int sl = row_slot[r], p = row_pos[r];
				const int cls = p < ctl[sl].prompt_len ? STEER_PROMPT : ( p < ctl[sl].think_end ? STEER_THINK : STEER_ANSWER );
				bool any = false;
				for ( int k = 0; k < K; ++k ) any |= kind[( static_cast<size_t>( sl ) * layers + layer ) * K + k] != STEER_OFF;
				const bool active = ( ctl[sl].scope & cls ) != 0 && any;
				const float* x = &hs[static_cast<size_t>( r ) * hc * H];
				std::vector<double> mean( H, 0.0 ), edit;
				for ( int i = 0; i < H; ++i ) { for ( int q = 0; q < hc; ++q ) mean[i] += x[q * H + i]; mean[i] /= hc; }
				edit = mean;
				if ( active ) {
					++edited;
					for ( int k = 0; k < K; ++k ) {
						const size_t e = ( static_cast<size_t>( sl ) * layers + layer ) * K + k;
						if ( kind[e] == STEER_OFF ) continue;
						const float* v = &vec[e * H];
						if ( kind[e] == STEER_ADD ) { for ( int i = 0; i < H; ++i ) edit[i] += par[e] * v[i]; }
						else {
							double dot = 0, n0 = 0; for ( int i = 0; i < H; ++i ) { dot += edit[i] * v[i]; n0 += edit[i] * edit[i]; }
							if ( dot - par[e] > 0 ) {
								double n1 = 0;
								for ( int i = 0; i < H; ++i ) { edit[i] -= ( dot - par[e] ) * v[i]; n1 += edit[i] * edit[i]; }
								if ( kind[e] == STEER_CLAMP_KEEP ) for ( int i = 0; i < H; ++i ) edit[i] *= std::sqrt( n0 / n1 );
							}
						}
					}
				}
				for ( int q = 0; q < hc; ++q ) {
					for ( int i = 0; i < H; ++i ) {
						const float g = got[( static_cast<size_t>( r ) * hc + q ) * H + i];
						if ( !active ) ENGINE_ASSERT( std::memcmp( &g, &x[q * H + i], sizeof( float ) ) == 0, "a stream row outside the scope changed" );
						else ENGINE_ASSERT( std::fabs( g - ( x[q * H + i] + ( edit[i] - mean[i] ) ) ) <= 1e-4 * std::fabs( g ) + 1e-4, "steered streams differ from the fp32 reference" );
					}
				}
			}
			std::printf( "    layer %d: %d of %d rows edited, the rest bit-identical\n", layer, edited, rows );
		}
	}

	std::printf( "[*] logit penalty: shard window, think scope...\n" );
	{
		std::vector<half> hl( static_cast<size_t>( rows ) * logit_cols );
		for ( size_t i = 0; i < hl.size(); ++i ) hl[i] = float_to_half( rnd( seed ) * 4.0f );
		half* d_l;
		CUDA_CHECK( cudaMalloc( &d_l, hl.size() * sizeof( half ) ) );
		CUDA_CHECK( cudaMemcpy( d_l, hl.data(), hl.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
		ResidualSteer::penalize( t, d_l, logit_cols, d_rows, 0, rows, shard_lo, shard_cols, stream );
		CUDA_CHECK( cudaStreamSynchronize( stream ) );
		std::vector<half> got( hl.size() );
		CUDA_CHECK( cudaMemcpy( got.data(), d_l, got.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
		int hit = 0;
		for ( int r = 0; r < rows; ++r ) {
			const int s = row_slot[r], q = row_pos[r] + 1;
			const bool in_think = ctl[s].penalty_n > 0 && q >= ctl[s].prompt_len && q < ctl[s].think_end;
			for ( int c = 0; c < logit_cols; ++c ) {
				const size_t i = static_cast<size_t>( r ) * logit_cols + c;
				const bool listed = c == 10 || c == 299;   // ids 410 and 699 in the slice
				if ( in_think && listed ) { ++hit; ENGINE_ASSERT( std::fabs( half_to_float( got[i] ) - ( half_to_float( hl[i] ) - 4.0f ) ) <= 4e-3, "penalised logit wrong" ); }
				else ENGINE_ASSERT( std::memcmp( &got[i], &hl[i], sizeof( half ) ) == 0, "an unpenalised logit changed" );
			}
		}
		ENGINE_ASSERT( hit == 3 * 2, "expected the three in-think rows of slot 2 (predicted positions 4, 5, 6) to be penalised at both in-shard ids" );
		std::printf( "    %d logits penalised, all others bit-identical\n", hit );
	}
	std::printf( "[+] ResidualSteer PASS\n" );
	return 0;
}
