// Sequence-parallel DiT (H3DiTGroup over N GPUs) against the golden.   test_h3_dit_sp <checkpoint> <golden> <ngpus> [lora]
#include "engine/config.h"
#include "engine/loader/safetensors.h"
#include "engine/video/h3_dit_group.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static pod::SafeTensorsFile g_st;
static const pod::SafeTensorMeta* need( const std::string& n ) { const pod::SafeTensorMeta* m = engine::loader::safetensors_find( g_st, n.c_str() ); ENGINE_ASSERT( m, n.c_str() ); return m; }
template <typename T> static std::vector<T> vec( const std::string& n ) { const pod::SafeTensorMeta* m = need( n ); const T* p = static_cast<const T*>( engine::loader::safetensors_get_data_cpu( g_st, m ) ); return std::vector<T>( p, p + m->size_bytes / sizeof( T ) ); }

int main( int argc, char** argv ) {
	if ( argc < 4 ) { std::printf( "usage: test_h3_dit_sp <checkpoint> <golden> <ngpus> [lora]\n" ); return 1; }
	ENGINE_ASSERT( engine::loader::safetensors_open( g_st, argv[2] ), "golden" );
	const int ngpu = std::atoi( argv[3] );
	std::vector<int> devs;
	for ( int i = 0; i < ngpu; ++i ) { devs.push_back( i ); }
	const int used = need( "s0.embed" )->shape[0];
	int layers = 0;
	for ( int i = 0; i < 64; ++i ) { if ( engine::loader::safetensors_find( g_st, ( "s0.block" + std::to_string( i ) ).c_str() ) ) { layers = i + 1; } }
	engine::video::H3DiTGroup grp;
	ENGINE_ASSERT( grp.load( argv[1], devs, ( used + 63 ) / 64 * 64 ), "load" );
	if ( argc > 4 ) { ENGINE_ASSERT( grp.add_lora( argv[4], 1.0f ) >= 0, "lora" ); }
	( void )layers;
	const std::vector<float> x = vec<float>( "x" ), ax = vec<float>( "audio_x" );
	const std::vector<int> ip = vec<int>( "img_pos" ), ap = vec<int>( "audio_pos" ), tp = vec<int>( "text_pos" ), tags = vec<int>( "token_tags" );
	engine::video::H3HostRequest r;
	r.used = used; r.text_len = need( "prompt_embeds" )->shape[0]; r.n_img = static_cast<int>( ip.size() ); r.n_audio = static_cast<int>( ap.size() );
	for ( int i : ip ) { r.video_rows.insert( r.video_rows.end(), x.begin() + static_cast<size_t>( i ) * 96, x.begin() + static_cast<size_t>( i + 1 ) * 96 ); }
	for ( int i : ap ) { r.audio_rows.insert( r.audio_rows.end(), ax.begin() + static_cast<size_t>( i ) * 32, ax.begin() + static_cast<size_t>( i + 1 ) * 32 ); }
	r.prompt = vec<float>( "prompt_embeds" );
	r.position_ids = vec<float>( "img_position_ids" );
	r.img_pos = ip; r.audio_pos = ap; r.text_pos = tp;
	r.tags.assign( tags.begin(), tags.begin() + used );
	const std::vector<uint8_t> mk = vec<uint8_t>( "update_mask" );
	r.update_mask = mk;
	int bad = 0;
	for ( int step = 0; step < 2; ++step ) {
		const std::string pre = "s" + std::to_string( step ) + ".";
		r.unique_t = vec<float>( pre + "unique_timesteps" );
		const std::vector<int> inv = vec<int>( pre + "inverse_indices" );
		r.inverse.assign( inv.begin(), inv.begin() + used );
		std::vector<float> vo, ao;
		auto t0 = std::chrono::steady_clock::now();
		ENGINE_ASSERT( grp.forward( r, &vo, &ao ), "forward" );
		const double secs = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
		auto cmp = [&]( const std::vector<float>& a, const std::vector<float>& b, double* rel ) {
			double dot = 0, na = 0, nb = 0, nd = 0;
			for ( size_t i = 0; i < a.size(); ++i ) { dot += static_cast<double>( a[i] ) * b[i]; na += static_cast<double>( a[i] ) * a[i]; nb += static_cast<double>( b[i] ) * b[i]; nd += ( static_cast<double>( a[i] ) - b[i] ) * ( static_cast<double>( a[i] ) - b[i] ); }
			*rel = std::sqrt( nd / nb );
			return dot / std::sqrt( na * nb );
		};
		double rv, ra;
		const double cv = cmp( vo, vec<float>( pre + "video_out" ), &rv ), ca = cmp( ao, vec<float>( pre + "audio_out" ), &ra );
		{ // per-owner diagnosis: rows below / above the first rank's boundary
			const std::vector<float> ref = vec<float>( pre + "video_out" );
			const int chunk = ( used + ngpu - 1 ) / ngpu;
			for ( int rk = 0; rk < ngpu; ++rk ) {
				double dot = 0, na = 0, nb = 0;
				for ( size_t i = 0; i < ip.size(); ++i ) {
					if ( ip[i] / chunk != rk ) { continue; }
					for ( int c = 0; c < 96; ++c ) { const double a = vo[i * 96 + c], b = ref[i * 96 + c]; dot += a * b; na += a * a; nb += b * b; }
				}
				std::printf( "   rank %d rows: video cosine %.6f (|a| %.3g |b| %.3g)\n", rk, dot / std::sqrt( na * nb + 1e-30 ), std::sqrt( na ), std::sqrt( nb ) );
			}
		}
		std::printf( "step %d on %d GPU(s) (%.2f s): video cosine %.6f rel %.4g | audio cosine %.6f rel %.4g\n", step, ngpu, secs, cv, rv, ca, ra );
		if ( cv < 0.999 || ca < 0.999 ) { ++bad; }
	}
	std::printf( bad ? "FAIL\n" : "ok\n" );
	return bad ? 1 : 0;
}
