// tts_generate: the vall-e v3 TTS model from the command line (guidance: the export's own, or --cfg / --cfg-prompt).
//   tts_generate --model <export.sft>                        load and report
//   tts_generate --model <export.sft> --golden <case.safetensors>
//       the length head (its frame count, and the raw log patches where stored);
//       parity against a vall_e/golden.py reference: the backbone at demask step 0 (the input of every block, the final
//       hidden states) for the three guidance rows (cond / null / promptless); then the demasking loop teacher-forced (every
//       step from the reference's previous digits: how many revealed digits agree, and the step-0 guided velocities),
//       and a free-running generation from the reference's reveal order and noise, written with --out as .npy digits
//       [frames, 32]; with --codec, the codec's decode of the reference's digits against the golden's wav and its encode
//       of the prompt's audio against the prompt's digits, and --wav
//       writes the free-running generation as audio
#include "engine/backend/runtime.h"
#include "engine/kernels/cpu/cpu_util.h"
#include "engine/loader/safetensors.h"
#include "engine/tts/audio.h"
#include "engine/tts/codec.h"
#include "engine/tts/tts_model.h"
#ifdef ENGINE_TTS
#include "engine/tts/tts.h"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using engine::loader::safetensors_find;
using engine::loader::safetensors_get_data_cpu;

// golden.py writes torch tensors: ids / digits as int64, hidden states as FP16
static std::vector<int64_t> i64( const pod::SafeTensorsFile& g, const char* name ) {
	const pod::SafeTensorMeta* m = safetensors_find( g, name );
	if ( !m ) { std::fprintf( stderr, "golden: no %s\n", name ); std::exit( 1 ); }
	const int64_t* p = static_cast<const int64_t*>( safetensors_get_data_cpu( g, m ) );
	return std::vector<int64_t>( p, p + m->size_bytes / 8 );
}

static std::vector<float> f32( const pod::SafeTensorsFile& g, const std::string& name ) {
	const pod::SafeTensorMeta* m = safetensors_find( g, name.c_str() );
	if ( !m ) { return {}; }
	const float* p = static_cast<const float*>( safetensors_get_data_cpu( g, m ) );
	return std::vector<float>( p, p + m->size_bytes / 4 );
}

static void write_npy_u8( const std::string& path, const std::vector<uint8_t>& v, int rows, int cols ) {
	std::string hdr = "{'descr': '|u1', 'fortran_order': False, 'shape': (" + std::to_string( rows ) + ", " + std::to_string( cols ) + "), }";
	while ( ( 10 + hdr.size() + 1 ) % 64 ) { hdr += ' '; }
	hdr += '\n';
	FILE* f = std::fopen( path.c_str(), "wb" );
	const unsigned char magic[8] = { 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0 };
	std::fwrite( magic, 1, 8, f );
	const uint16_t n = static_cast<uint16_t>( hdr.size() );
	std::fwrite( &n, 2, 1, f );
	std::fwrite( hdr.data(), 1, hdr.size(), f );
	std::fwrite( v.data(), 1, v.size(), f );
	std::fclose( f );
}

// 16-bit PCM mono WAV
static void write_wav( const std::string& path, const std::vector<float>& x, int rate ) {
	if ( !engine::tts::write_wav( path, x, rate ) ) { std::fprintf( stderr, "can't write %s\n", path.c_str() ); }
}

// The codec against the golden: its reference digits decoded here and by vall_e (the golden's wav, FP32): SNR and the worst sample
static int check_codec( engine::tts::NemoCodec& codec, const char* path ) {
	pod::SafeTensorsFile g;
	if ( !engine::loader::safetensors_open( g, path ) ) { std::fprintf( stderr, "can't open %s\n", path ); return 1; }
	const auto d64 = i64( g, "digits" );
	const auto ref = f32( g, "wav" );
	std::vector<uint8_t> digits( d64.begin(), d64.end() );
	const int n_frames = static_cast<int>( digits.size() / 32 );
	codec.decode( digits, n_frames );   // warm-up (buffers, kernel loads), then timed
	engine::backend::device_sync();
	const auto t0 = std::chrono::steady_clock::now();
	const auto wav = codec.decode( digits, n_frames );
	const double dt = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
	if ( ref.size() != wav.size() ) {
		std::printf( "codec: %zu samples, the reference %zu -> FAIL\n", wav.size(), ref.size() );
		return 1;
	}
	double sig = 0, err = 0, worst = 0;
	for ( size_t i = 0; i < wav.size(); ++i ) {
		const double e = wav[i] - ref[i];
		sig += static_cast<double>( ref[i] ) * ref[i]; err += e * e; worst = std::max( worst, std::fabs( e ) );
	}
	const double snr = 10.0 * std::log10( sig / std::max( err, 1e-30 ) );
	std::printf( "codec: %d frames -> %zu samples in %.0f ms (%.0fx real time), SNR %.1f dB against vall_e's decode, worst sample %.2e -> %s\n", n_frames,
		wav.size(), dt * 1e3, wav.size() / static_cast<double>( codec.config().sample_rate ) / dt, snr, worst, snr > 60.0 ? "PASS" : "FAIL" );
	engine::loader::safetensors_close( g );
	return snr > 60.0 ? 0 : 1;
}

// The codec's encoder against the golden: the prompt's audio encoded here, against the prompt digits vall_e encoded
static int check_encode( engine::tts::NemoCodec& codec, const char* path ) {
	pod::SafeTensorsFile g;
	if ( !engine::loader::safetensors_open( g, path ) ) { std::fprintf( stderr, "can't open %s\n", path ); return 1; }
	const auto audio = f32( g, "prompt_wav" );
	if ( audio.empty() ) { std::printf( "codec encoder: no prompt_wav in the golden (python -m vall_e.golden stores it)\n" ); return 0; }
	const auto ref = i64( g, "prompt" );
	codec.encode( audio );   // warm-up, then timed
	engine::backend::device_sync();
	const auto t0 = std::chrono::steady_clock::now();
	const auto digits = codec.encode( audio );
	const double dt = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
	// the golden's prompt is whole patches: compare the frames both have
	const size_t n = std::min( digits.size(), ref.size() );
	long same = 0;
	for ( size_t i = 0; i < n; ++i ) { same += digits[i] == ref[i]; }
	const double agree = same / static_cast<double>( n );
	std::printf( "codec encoder: %.2f s of audio -> %zu frames in %.0f ms, %.5f of the reference's digits (%zu frames) -> %s\n",
		audio.size() / static_cast<double>( codec.config().sample_rate ), digits.size() / 32, dt * 1e3, agree, ref.size() / 32, agree > 0.999 ? "PASS" : "FAIL" );
	engine::loader::safetensors_close( g );
	return agree > 0.999 ? 0 : 1;
}

static int check_demask( engine::tts::TtsModel& m, const char* path, const std::string& out, const engine::tts::TtsModel::Sampling& s,
	engine::tts::NemoCodec* codec, const std::string& wav_out ) {
	pod::SafeTensorsFile g;
	engine::loader::safetensors_open( g, path );
	const auto& c = m.config();
	const int V = c.values();
	const auto text64 = i64( g, "text" ), prompt64 = i64( g, "prompt" ), nf = i64( g, "n_frames" ), order64 = i64( g, "order" );
	std::vector<int> text( text64.begin(), text64.end() ), order( order64.begin(), order64.end() );
	std::vector<uint8_t> prompt( prompt64.begin(), prompt64.end() );
	const int n_frames = static_cast<int>( nf[0] ), La = static_cast<int>( order.size() );
	std::vector<std::vector<float>> noise( s.steps );
	for ( int st = 0; st < s.steps; ++st ) { noise[st] = f32( g, "noise_" + std::to_string( st ) ); }
	// teacher-forced: each step from the reference's digits after the previous step
	std::vector<uint8_t> ref_prev( static_cast<size_t>( La ) * V, 0 );
	long agree = 0, total = 0;
	int done = 0;
	std::printf( "teacher-forced steps (revealed digits that agree):" );
	for ( int st = 0; st < s.steps; ++st ) {
		const int target = engine::tts::TtsModel::revealed_after( st, s.steps, La );
		if ( target == done ) { continue; }
		std::vector<int> reveal( order.begin() + done, order.begin() + target );
		std::vector<uint8_t> digits = ref_prev, masked( La, 1 );
		for ( int i = 0; i < done; ++i ) { masked[order[i]] = 0; }
		std::vector<std::vector<float>> vt;
		m.demask_step( text, prompt, digits, masked, reveal, noise[st].data(), s, st == 0 ? &vt : nullptr );
		const auto ref = i64( g, ( "digits_" + std::to_string( st ) ).c_str() );
		long a = 0;
		for ( int p : reveal ) { for ( int e = 0; e < V; ++e ) { a += digits[static_cast<size_t>( p ) * V + e] == ref[static_cast<size_t>( p ) * V + e]; } }
		agree += a; total += static_cast<long>( reveal.size() ) * V;
		std::printf( " %d:%.4f", st, a / double( reveal.size() * V ) );
		if ( st == 0 ) {
			std::printf( " [velocity cosine per flow step:" );
			for ( size_t j = 0; j < vt.size(); ++j ) {
				const auto gv = f32( g, "step0_v_" + std::to_string( j ) );
				double dot = 0, na = 0, nb = 0;
				for ( size_t i = 0; i < gv.size(); ++i ) { dot += vt[j][i] * gv[i]; na += vt[j][i] * vt[j][i]; nb += gv[i] * gv[i]; }
				std::printf( " %.5f", dot / std::sqrt( na * nb + 1e-30 ) );
			}
			std::printf( "]" );
		}
		for ( size_t i = 0; i < ref_prev.size(); ++i ) { ref_prev[i] = static_cast<uint8_t>( ref[i] ); }
		done = target;
	}
	const double tf = agree / double( total );
	std::printf( "\nteacher-forced agreement %.5f -> %s\n", tf, tf > 0.995 ? "PASS" : "FAIL" );
	// free-running
	engine::backend::device_sync();
	const auto t0 = std::chrono::steady_clock::now();
	const auto digits = m.generate( text, prompt, n_frames, order, noise, s );
	const double dt = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
	const auto ref = i64( g, "digits" );
	long same = 0;
	for ( size_t i = 0; i < digits.size(); ++i ) { same += digits[i] == ref[i]; }
	std::printf( "free-running: %d frames in %.2f s (%d steps), %.3f of digits equal to the reference's (bf16 vs fp32 Python: 0.58-0.72)\n", n_frames, dt,
		s.steps, same / double( digits.size() ) );
	if ( !out.empty() ) {
		write_npy_u8( out, digits, n_frames, 32 );
		std::printf( "wrote %s\n", out.c_str() );
	}
	if ( codec && !wav_out.empty() ) {
		write_wav( wav_out, codec->decode( digits, n_frames ), codec->config().sample_rate );
		std::printf( "wrote %s\n", wav_out.c_str() );
	}
	engine::loader::safetensors_close( g );
	return tf > 0.995 ? 0 : 1;
}

// The length head against the golden's: its raw output (log patches, where golden.py stored it) within 0.01, 1% of the length, and
// the frame count within one (FP16 activations against an FP32 reference move the output by up to ~0.004, which tips a length
// sitting near a rounding boundary; the head's own error against real lengths is ~15%)
static int check_length( engine::tts::TtsModel& m, const char* path ) {
	if ( !m.config().length ) { std::printf( "length head: none in this export\n" ); return 0; }
	pod::SafeTensorsFile g;
	if ( !engine::loader::safetensors_open( g, path ) ) { std::fprintf( stderr, "can't open %s\n", path ); return 1; }
	const auto text64 = i64( g, "text" ), prompt64 = i64( g, "prompt" ), nf = i64( g, "n_frames" );
	std::vector<int> text( text64.begin(), text64.end() );
	std::vector<uint8_t> prompt( prompt64.begin(), prompt64.end() );
	float lp = 0.0f;
	const int n = m.predict_length( text, prompt, &lp );
	const auto ref_lp = f32( g, "log_patches" );
	bool ok = std::abs( n - static_cast<int>( nf[0] ) ) <= 1;
	if ( ref_lp.empty() ) {
		std::printf( "length head: %d frames (reference %d; log patches %.5f) -> %s\n", n, static_cast<int>( nf[0] ), lp, ok ? "PASS" : "FAIL" );
	} else {
		ok &= std::fabs( lp - ref_lp[0] ) < 0.01f;
		std::printf( "length head: %d frames (reference %d), log patches %.5f (reference %.5f, diff %+.5f) -> %s\n", n, static_cast<int>( nf[0] ), lp, ref_lp[0],
			lp - ref_lp[0], ok ? "PASS" : "FAIL" );
	}
	engine::loader::safetensors_close( g );
	return ok ? 0 : 1;
}

static int check_backbone( engine::tts::TtsModel& m, const char* path ) {
	pod::SafeTensorsFile g;
	if ( !engine::loader::safetensors_open( g, path ) ) { std::fprintf( stderr, "can't open %s\n", path ); return 1; }
	const auto& c = m.config();
	const int V = c.values(), d = c.d_model;
	const auto text = i64( g, "text" ), prompt = i64( g, "prompt" ), nf = i64( g, "n_frames" );
	const int La = static_cast<int>( ( nf[0] + c.patch - 1 ) / c.patch );
	engine::tts::TtsRow rows[3];
	for ( int r = 0; r < 3; ++r ) {
		if ( r != 1 ) { rows[r].text.assign( text.begin(), text.end() ); }
		if ( r == 0 ) { for ( int64_t v : prompt ) { rows[r].prompt.push_back( static_cast<uint8_t>( v ) ); } }
		rows[r].target.assign( static_cast<size_t>( La ) * V, 0 );
		rows[r].masked.assign( La, 1 );
	}
	// the rows golden.py's guidance ran: cond, then null (with guidance), then promptless (with speaker guidance)
	const pod::SafeTensorMeta* h0 = safetensors_find( g, "step0_hidden_0" );
	if ( !h0 ) { std::fprintf( stderr, "golden: no step0_hidden_0\n" ); return 1; }
	const int R = std::min( 3, static_cast<int>( h0->shape[0] ) );
	std::printf( "%s: text %zu, prompt %zu patches, target %d patches, %d guidance row%s\n", path, text.size(), prompt.size() / V, La, R, R == 1 ? "" : "s" );
	float* out = static_cast<float*>( engine::backend::device_malloc( static_cast<size_t>( engine::tts::TtsModel::MAX_LEN ) * d * 4 ) );
	double worst = 1.0;
	for ( int r = 0; r < R; ++r ) {
		std::vector<std::vector<float>> taps;
		m.hidden( rows[r], out, &taps );
		const int L = rows[r].length();
		std::printf( "  row %d (%s), %d positions: cosine per block input", r, r == 0 ? "cond" : r == 1 ? "null" : "promptless", L );
		for ( size_t l = 0; l < taps.size(); ++l ) {
			const std::string name = "step0_hidden_" + std::to_string( l );
			const pod::SafeTensorMeta* gm = safetensors_find( g, name.c_str() );
			const uint16_t* gh = static_cast<const uint16_t*>( safetensors_get_data_cpu( g, gm ) );
			const int Lpad = gm->shape[1];
			double dot = 0, na = 0, nb = 0;
			for ( int p = 0; p < L; ++p ) {
				for ( int e = 0; e < d; ++e ) {
					const double a = taps[l][static_cast<size_t>( p ) * d + e], b = engine::kernels::cpu::half_bits_to_float( gh[( static_cast<size_t>( r ) * Lpad + p ) * d + e] );
					dot += a * b; na += a * a; nb += b * b;
				}
			}
			const double cs = dot / std::sqrt( na * nb + 1e-30 );
			worst = std::min( worst, cs );
			if ( l == 0 || l == 1 || l == taps.size() / 2 || l + 1 == taps.size() ) { std::printf( " [%zu] %.5f", l, cs ); }
		}
		std::printf( "\n" );
	}
	std::printf( "worst cosine %.5f (1 - cos %.3e) -> %s\n", worst, 1.0 - worst, worst > 0.999 ? "PASS" : "FAIL" );
	engine::backend::device_free( out );
	engine::loader::safetensors_close( g );
	return worst > 0.999 ? 0 : 1;
}

int main( int argc, char** argv ) {
	std::string model, golden, out, codec_path, wav_out, text, prompt_path, language = "en";
	unsigned long long seed = 0;
	float cfg = -1.0f, cfg_prompt = -1.0f;
	for ( int i = 1; i < argc; ++i ) {
		if ( !std::strcmp( argv[i], "--model" ) && i + 1 < argc ) { model = argv[++i]; }
		else if ( !std::strcmp( argv[i], "--golden" ) && i + 1 < argc ) { golden = argv[++i]; }
		else if ( !std::strcmp( argv[i], "--out" ) && i + 1 < argc ) { out = argv[++i]; }
		else if ( !std::strcmp( argv[i], "--codec" ) && i + 1 < argc ) { codec_path = argv[++i]; }
		else if ( !std::strcmp( argv[i], "--wav" ) && i + 1 < argc ) { wav_out = argv[++i]; }
		else if ( !std::strcmp( argv[i], "--text" ) && i + 1 < argc ) { text = argv[++i]; }
		else if ( !std::strcmp( argv[i], "--prompt" ) && i + 1 < argc ) { prompt_path = argv[++i]; }
		else if ( !std::strcmp( argv[i], "--language" ) && i + 1 < argc ) { language = argv[++i]; }
		else if ( !std::strcmp( argv[i], "--seed" ) && i + 1 < argc ) { seed = std::strtoull( argv[++i], nullptr, 10 ); }
		else if ( !std::strcmp( argv[i], "--cfg" ) && i + 1 < argc ) { cfg = std::strtof( argv[++i], nullptr ); }
		else if ( !std::strcmp( argv[i], "--cfg-prompt" ) && i + 1 < argc ) { cfg_prompt = std::strtof( argv[++i], nullptr ); }
	}
	if ( model.empty() ) {
		std::fprintf( stderr, "usage: tts_generate --model <vall-e v3 export .sft> [--golden <golden.py case .safetensors>] [--cfg x] [--cfg-prompt x]\n"
			"  [--codec <codec .sft> [--wav <out.wav>]]\n"
			"  tts_generate --model <.sft> --codec <codec .sft> --text \"...\" [--prompt voice.wav] [--language en] [--seed n] --wav out.wav\n" );
		return 2;
	}
	if ( !text.empty() ) {
#ifdef ENGINE_TTS
		// text to speech end to end (engine::tts::Tts, as vall_e/tts.py)
		if ( codec_path.empty() || wav_out.empty() ) { std::fprintf( stderr, "--text needs --codec and --wav\n" ); return 2; }
		engine::tts::Tts tts;
		if ( !tts.load( model, codec_path ) ) { return 1; }
		std::vector<float> prompt, audio;
		int rate = 0;
		std::string err;
		if ( !prompt_path.empty() && !engine::tts::read_wav( prompt_path, prompt, rate, &err ) ) { std::fprintf( stderr, "%s\n", err.c_str() ); return 1; }
		engine::tts::SynthOptions o;
		o.language = language;
		o.seed = seed;
		o.cfg = cfg;
		o.cfg_prompt = cfg_prompt;
		const auto t0 = std::chrono::steady_clock::now();
		if ( !tts.synthesize( text, prompt, rate, o, audio, &err ) ) { std::fprintf( stderr, "%s\n", err.c_str() ); return 1; }
		const double dt = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
		write_wav( wav_out, audio, tts.sample_rate() );
		std::printf( "wrote %s: %.2f s of speech in %.2f s (%.1fx real time)\n", wav_out.c_str(), audio.size() / static_cast<double>( tts.sample_rate() ), dt,
			audio.size() / static_cast<double>( tts.sample_rate() ) / dt );
		return 0;
#else
		std::fprintf( stderr, "--text needs a build with propan2ol (the text front-end)\n" );
		return 2;
#endif
	}
	engine::tts::TtsModel m;
	if ( !m.load( model ) ) { return 1; }
	const auto& c = m.config();
	std::printf( "loaded %s: %d layers, d_model %d, %d heads (head_dim %d), ffn %d, patch %d, flow %d x %d, %d modes, length head %s, %zu phonemes; %.0f MB of weights\n",
		model.c_str(), c.layers, c.d_model, c.heads, c.head_dim(), c.ffn(), c.patch, c.flow_depth, c.flow_width, c.n_modes, c.length ? "yes" : "no",
		c.vocab.size(), m.weight_bytes() / 1048576.0 );
	// the export's own guidance unless given (a golden made at other settings: golden.py's meta.json names them)
	engine::tts::TtsModel::Sampling s = m.default_sampling();
	if ( cfg >= 0.0f ) { s.cfg = cfg; }
	if ( cfg_prompt >= 0.0f ) { s.cfg_prompt = cfg_prompt; }
	std::printf( "guidance %.2f, speaker guidance %.2f\n", s.cfg, s.cfg_prompt );
	engine::tts::NemoCodec codec;
	const bool have_codec = !codec_path.empty();
	if ( have_codec && !codec.load( codec_path ) ) { return 1; }
	if ( !golden.empty() ) {
		const int rl = check_length( m, golden.c_str() );
		const int rb = check_backbone( m, golden.c_str() );
		const int rc = have_codec ? check_codec( codec, golden.c_str() ) | check_encode( codec, golden.c_str() ) : 0;
		const int rd = check_demask( m, golden.c_str(), out, s, have_codec ? &codec : nullptr, wav_out );
		return rl | rb | rc | rd;
	}
	return 0;
}
