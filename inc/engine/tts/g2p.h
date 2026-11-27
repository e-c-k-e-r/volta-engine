// The vall-e text front-end (vall_e/emb/g2p.py encode, then TTS.phonemes): text to the model's phoneme ids, through
// propan2ol (espeak-ng's phonemizing with phonemizer 3.3's espeak-backend processing, character for character): English
// abbreviations spelled out first (normalize_en), then punctuation preserved, IPA with stress marks, the language-switch flags
// removed, words separated by spaces; Mandarin tone digits remapped; characters outside the model's vocabulary dropped.
// Japanese must come as kana (pykakasi's kanji readings aren't ported): katakana becomes hiragana.
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace propan2ol { class Phonemizer; }

namespace engine::tts {

class G2P {
public:
	G2P();
	~G2P();
	// vocab: the model's phonemes (TtsConfig::vocab, id i + 1 is vocab[i])
	bool init( const std::vector<std::string>& vocab );
	// The phoneme string g2p.encode gives for `text` in `language` (vall_e's codes: en, de, fr, ja, ko, zh, ...); empty
	// with an error logged if it can't be phonemized (an unknown language, kanji)
	std::string phonemize( const std::string& text, const std::string& language );
	// Its phoneme ids (1-based; characters outside the vocabulary dropped)
	std::vector<int> encode( const std::string& text, const std::string& language );

	// The steps on their own (exposed for the parity test)
	static std::string normalize_en( const std::string& text );
	static std::string coerce_language( const std::string& language );

private:
	const propan2ol::Phonemizer* phonemizer( const std::string& code );   // one per language, made on first use (null: unknown)

	std::unordered_map<char32_t, int> ids_;
	std::unordered_map<std::string, std::unique_ptr<propan2ol::Phonemizer>> phonemizers_;
};

} // namespace engine::tts
