// tts_generate: the vall-e v3 TTS model from the command line. For now: loads an export and reports it.
//   tts_generate --model <export.sft>
#include "engine/tts/tts_model.h"
#include <cstdio>
#include <cstring>
#include <string>

int main( int argc, char** argv ) {
	std::string model;
	for ( int i = 1; i < argc; ++i ) {
		if ( !std::strcmp( argv[i], "--model" ) && i + 1 < argc ) { model = argv[++i]; }
	}
	if ( model.empty() ) {
		std::fprintf( stderr, "usage: tts_generate --model <vall-e v3 export .sft>\n" );
		return 2;
	}
	engine::tts::TtsModel m;
	if ( !m.load( model ) ) { return 1; }
	const auto& c = m.config();
	std::printf( "loaded %s: %d layers, d_model %d, %d heads (head_dim %d), ffn %d, patch %d, flow %d x %d, %d modes, length head %s, %zu phonemes; %.0f MB of weights\n",
		model.c_str(), c.layers, c.d_model, c.heads, c.head_dim(), c.ffn(), c.patch, c.flow_depth, c.flow_width, c.n_modes, c.length ? "yes" : "no",
		c.vocab.size(), m.weight_bytes() / 1048576.0 );
	return 0;
}
