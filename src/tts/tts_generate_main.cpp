// tts_generate: the vall-e v3 TTS model from the command line.
//   tts_generate --model <export.sft>                        load and report
//   tts_generate --model <export.sft> --golden <case.safetensors>
//       parity against a vall_e/golden.py reference: the backbone at demask step 0 (the input of every block, the final
//       hidden states) for the three guidance rows (cond / null / promptless)
#include "engine/loader/safetensors.h"
#include "engine/tts/tts_model.h"
#include <cmath>
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

static float h2f( uint16_t h ) {
	const uint32_t s = ( h >> 15 ) & 1, e = ( h >> 10 ) & 31, f = h & 1023;
	float v = e == 0 ? std::ldexp( static_cast<float>( f ), -24 ) : e == 31 ? INFINITY : std::ldexp( static_cast<float>( f | 1024 ), static_cast<int>( e ) - 25 );
	return s ? -v : v;
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
	std::printf( "%s: text %zu, prompt %zu patches, target %d patches\n", path, text.size(), prompt.size() / V, La );
	float* out = nullptr;
	cudaMalloc( &out, static_cast<size_t>( engine::tts::TtsModel::MAX_LEN ) * d * 4 );
	double worst = 1.0;
	for ( int r = 0; r < 3; ++r ) {
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
					const double a = taps[l][static_cast<size_t>( p ) * d + e], b = h2f( gh[( static_cast<size_t>( r ) * Lpad + p ) * d + e] );
					dot += a * b; na += a * a; nb += b * b;
				}
			}
			const double cs = dot / std::sqrt( na * nb + 1e-30 );
			worst = std::min( worst, cs );
			if ( l == 0 || l == 1 || l == taps.size() / 2 || l + 1 == taps.size() ) { std::printf( " [%zu] %.5f", l, cs ); }
		}
		std::printf( "\n" );
	}
	std::printf( "worst cosine %.5f -> %s\n", worst, worst > 0.999 ? "PASS" : "FAIL" );
	cudaFree( out );
	engine::loader::safetensors_close( g );
	return worst > 0.999 ? 0 : 1;
}

int main( int argc, char** argv ) {
	std::string model, golden;
	for ( int i = 1; i < argc; ++i ) {
		if ( !std::strcmp( argv[i], "--model" ) && i + 1 < argc ) { model = argv[++i]; }
		else if ( !std::strcmp( argv[i], "--golden" ) && i + 1 < argc ) { golden = argv[++i]; }
	}
	if ( model.empty() ) {
		std::fprintf( stderr, "usage: tts_generate --model <vall-e v3 export .sft> [--golden <golden.py case .safetensors>]\n" );
		return 2;
	}
	engine::tts::TtsModel m;
	if ( !m.load( model ) ) { return 1; }
	const auto& c = m.config();
	std::printf( "loaded %s: %d layers, d_model %d, %d heads (head_dim %d), ffn %d, patch %d, flow %d x %d, %d modes, length head %s, %zu phonemes; %.0f MB of weights\n",
		model.c_str(), c.layers, c.d_model, c.heads, c.head_dim(), c.ffn(), c.patch, c.flow_depth, c.flow_width, c.n_modes, c.length ? "yes" : "no",
		c.vocab.size(), m.weight_bytes() / 1048576.0 );
	if ( !golden.empty() ) { return check_backbone( m, golden.c_str() ); }
	return 0;
}
