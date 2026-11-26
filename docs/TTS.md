# vall-e text to speech

The vall-e v3 model (flow-head masked NAR over NeMo 44.1 kHz FSQ codes, [github.com/e-c-k-e-r/vall-e](https://github.com/e-c-k-e-r/vall-e)) end to end:
text and a voice prompt in, 44.1 kHz speech out. `engine::tts::Tts` (inc/engine/tts/tts.h) is the pipeline; `tts_generate` runs it from the command line and
volta_server serves it as `POST /v1/audio/speech` (see CONFIG.md, the `tts` section).

## Pieces

| Piece | Source | What it reproduces |
|---|---|---|
| Text front-end | src/tts/g2p.cpp | vall_e/emb/g2p.py `encode` and phonemizer 3.3's espeak backend, through libespeak-ng |
| Model | src/tts/tts_model.cu, tts_backbone.cu, tts_flow.cu | vall_e/model.py: the backbone, the length head, the flow head and the demasking loop with guidance |
| Codec | src/tts/codec.cu | the NeMo codec's HiFi-GAN encoder, FSQ and HiFi-GAN decoder, in FP32 |
| Pipeline | src/tts/tts.cpp | vall_e/tts.py `TTS.generate`: the prompt (mono, resampled as torchaudio does, its first 10 s), sentence splitting past 160 phonemes, trailing-silence trimming, pauses |

The model and codec build in every GPU configuration; the front-end, the pipeline and the endpoint need libespeak-ng at build time (`ENGINE_TTS`; GPL-3.0,
which the AGPL-3.0 allows combining with).

## Files

```
python -m vall_e.export <checkpoint> model.sft --cfg 3.0 --cfg-prompt 2.0   # a model, with the guidance it was tuned for
python -m vall_e.export codec codec.sft                                     # the codec (both halves, FP32, weight norm folded)
tts_generate --model model.sft --codec codec.sft --text "Hello there." --prompt voice.wav --wav out.wav
```

The export's guidance is the default (v2.0 3.0 / 2.0, the distilled v2.0-fast 0 / 0, older exports 6.0 / 2.0); `--cfg` / `--cfg-prompt` override it.

## Parity (tts_generate --golden, test_tts_g2p, test_tts_resample; references from vall_e/golden.py in FP32)

| Check | Result |
|---|---|
| Backbone hidden states, step 0 | cosine 1.00000 at every block (v2.0, v2.0-fast, v1.2) |
| Demasking, teacher-forced | 99.82-99.90% of revealed digits equal (v2.0 / v2.0-fast), 99.65-99.72% (v1.2) |
| Length head | log length within 0.004; frames equal or one off at a rounding boundary |
| Codec decoder | 121-125 dB SNR against vall_e's decode |
| Codec encoder + FSQ | every digit of a 6 s prompt equal |
| Text front-end | 11608 of 11609 store texts equal (en, de, fr, ko, zh: all 2000; ja: 1608 of 1609 kana lines) |
| Prompt resampling | 141-145 dB SNR against torchaudio (16, 22.05, 24, 48 kHz) |

## Limits

- Japanese must be given as kana: vall_e reads kanji through pykakasi's dictionary, which isn't ported (kanji are refused with a message).
- No language detection (vall_e's `auto`): requests name the language.
- Voices are single WAV files; vall_e's multi-clip folder prompts (speaker-embedding and MOS selection) aren't ported.
- Speed is about vall_e's (v2.0 2.5x real time, v2.0-fast 4.8x on an RX 7900 XTX): the backbone's attention is plain batched GEMMs and the guidance
  rows run one after another.
