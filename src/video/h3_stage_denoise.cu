// H3 stage: layout, initial noise and sigma schedules; then the DiT denoise loop (one or several GPUs, sequence parallel, Euler eta 0).
#include "engine/core/log.h"
#include "engine/video/h3_dit_group.h"
#include "h3_run.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <random>

namespace engine::video {

namespace {

void gaussian( std::mt19937_64& rng, std::vector<float>& out ) {
	std::normal_distribution<float> nd( 0.0f, 1.0f );
	for ( float& v : out ) { v = nd( rng ); }
}

} // namespace

void H3Run::plan_denoise() {
	lay = ref2va ? h3_build_layout_ref( static_cast<int>( ids.size() ), latent_t, lh, lw, audio_t, blocks ) : h3_build_layout( static_cast<int>( ids.size() ), latent_t, lh, lw, audio_t, keyframes, frames );
	for ( size_t i = 0; i < ids.size(); ++i ) { lay.tags[i] = text_tags[i]; }
	const int target_rows = latent_t * ( lh / 2 ) * ( lw / 2 );
	n_cond = lay.n_img - target_rows;       // leading video rows that are conditions (keyframes / reference images)
	n_aref = lay.n_audio - 2 * audio_t;     // leading audio rows that are reference audio
	std::mt19937_64 rng( req.seed );
	vrows.assign( static_cast<size_t>( lay.n_img ) * 96, 0.0f );
	arows.assign( static_cast<size_t>( lay.n_audio ) * 32, 0.0f );
	gaussian( rng, vrows );
	gaussian( rng, arows );
	// condition rows are the fixed (noised) anchors; only the target rows are denoised
	std::copy( cond_rows.begin(), cond_rows.end(), vrows.begin() );
	std::copy( ref_audio_rows.begin(), ref_audio_rows.end(), arows.begin() ); // the audio reference is pinned at timestep 1: its clean latents
	video_shift = req.video_shift;
	if ( video_shift <= 0.0f ) {
		video_shift = 12.0f;
		for ( const H3Lora& l : req.loras ) {
			if ( l.path.find( "_768p" ) != std::string::npos ) { video_shift = 6.0f; }
		}
	}
	const int n_sigma = req.steps + 1;
	sv.assign( n_sigma, 0.0 );
	sa.assign( n_sigma, 0.0 );
	for ( int i = 0; i < n_sigma; ++i ) {
		const double u = n_sigma > 1 ? 1.0 - static_cast<double>( i ) / ( n_sigma - 1 ) : 1.0;
		sv[i] = video_shift * u / ( 1.0 + ( video_shift - 1.0 ) * u );
		sa[i] = req.audio_shift * u / ( 1.0 + ( req.audio_shift - 1.0 ) * u );
	}

}

bool H3Run::denoise() {
	report( "load", 0, 1 );
	auto t0 = std::chrono::steady_clock::now();
	std::vector<int> devices = paths.devices;
	if ( devices.empty() ) { devices.push_back( paths.device ); }
	{
		H3DiTGroup dit;
		const std::string& dit_path = ref2va && !paths.dit_ref2va.empty() ? paths.dit_ref2va : paths.dit;
		if ( ref2va ) {
			std::string low = dit_path;
			std::transform( low.begin(), low.end(), low.begin(), []( unsigned char c ) { return static_cast<char>( std::tolower( c ) ); } );
			if ( low.find( "ref2va" ) == std::string::npos ) { LOG_WARN( "H3: Ref2VA needs the Ref2VA transformer, but %s does not look like one (set dit_ref2va / --dit-ref2va)", dit_path.c_str() ); }
		}
		if ( !dit.load( dit_path.c_str(), devices, lay.used ) ) { LOG_ERROR( "H3: cannot load %s", dit_path.c_str() ); return false; }
		for ( const H3Lora& l : req.loras ) {
			if ( dit.add_lora( l.path.c_str(), l.scale ) < 0 ) { LOG_ERROR( "H3: cannot use LoRA %s", l.path.c_str() ); return false; }
		}
		res.t_load = seconds_since( t0 );
		LOG_INFO( "H3: DiT and %zu adapter(s) ready on %zu GPU(s) in %.1f s; %d rows (%d video, %d audio, %d text), %d steps, shifts %g/%g", req.loras.size(), devices.size(), res.t_load, lay.used,
			lay.n_img, lay.n_audio, lay.text_len, req.steps, video_shift, req.audio_shift );
		H3HostRequest r;
		r.used = lay.used; r.text_len = lay.text_len; r.n_img = lay.n_img; r.n_audio = lay.n_audio;
		r.prompt = std::move( prompt );
		r.position_ids = lay.position_ids;
		r.img_pos = lay.img_pos; r.audio_pos = lay.audio_pos; r.text_pos = lay.text_pos;
		r.update_mask = lay.update_mask; r.tags = lay.tags;
		r.inverse.resize( lay.used );
		t0 = std::chrono::steady_clock::now();
		std::vector<float> vout, aout;
		report( "denoise", 0, req.steps );
		for ( int step = 0; step < req.steps; ++step ) {
			if ( cancelled() ) { return false; }
			const double tv = 1.0 - sv[step], ta = 1.0 - sa[step];
			// per-row timesteps: text rows inherit the video timestep, condition rows sit at the condition timestep, audio rows at the audio one
			const double tc = std::max( tv, 0.999 );
			std::vector<double> row_t( lay.used, tv );
			for ( int i = 0; i < lay.n_audio; ++i ) { row_t[lay.audio_pos[i]] = ta; }
			for ( int i = 0; i < n_cond; ++i ) { row_t[lay.img_pos[i]] = tc; }
			for ( int i = 0; i < n_aref; ++i ) { row_t[lay.audio_pos[i]] = std::max( ta, 1.0 ); }
			std::vector<float> uniq;
			for ( double t : row_t ) {
				const float f = static_cast<float>( t );
				if ( std::find( uniq.begin(), uniq.end(), f ) == uniq.end() ) { uniq.push_back( f ); }
			}
			std::sort( uniq.begin(), uniq.end() );
			for ( int i = 0; i < lay.used; ++i ) { r.inverse[i] = static_cast<int>( std::lower_bound( uniq.begin(), uniq.end(), static_cast<float>( row_t[i] ) ) - uniq.begin() ); }
			r.unique_t = uniq;
			r.video_rows = vrows;
			r.audio_rows = arows;
			if ( !dit.forward( r, &vout, &aout ) ) { return false; }
			{ double sum = 0, sa2 = 0; size_t bad = 0; for ( float v : vout ) { if ( !std::isfinite( v ) ) { ++bad; } else { sum += std::fabs( v ); } } for ( float v : aout ) { if ( std::isfinite( v ) ) { sa2 += std::fabs( v ); } }
				LOG_INFO( "H3: step %d velocity |v|_1 %.6f |a|_1 %.6f, %zu non-finite", step, sum, sa2, bad ); }
			// Euler (eta 0): x += (sigma_curr - sigma_next) * v; condition rows stay at their anchors
			const float cv = static_cast<float>( sv[step] - sv[step + 1] ), ca = static_cast<float>( sa[step] - sa[step + 1] );
			for ( size_t i = static_cast<size_t>( n_cond ) * 96; i < vrows.size(); ++i ) { vrows[i] += cv * vout[i]; }
			for ( size_t i = 0; i < arows.size(); ++i ) { arows[i] += ca * aout[i]; }
			LOG_INFO( "H3: step %d/%d done (%.1f s elapsed)", step + 1, req.steps, seconds_since( t0 ) );
			report( "denoise", step + 1, req.steps );
		}
		res.t_denoise = seconds_since( t0 );
	}
	return true;

}

} // namespace engine::video
