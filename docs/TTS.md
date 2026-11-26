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
| Pipeline | src/tts/tts.cpp, audio.cpp | vall_e/tts.py `TTS.generate`: the prompt (mono, resampled as torchaudio does, its first 10 s), sentence splitting past 160 phonemes, trailing-silence trimming, pauses; WAV files in and out |

The model and codec build in every GPU configuration; the front-end, the pipeline and the endpoint need libespeak-ng at build time (`ENGINE_TTS`; GPL-3.0,
which the AGPL-3.0 allows combining with).

## Files

```
python -m vall_e.export <checkpoint> model.sft --cfg 3.0 --cfg-prompt 2.0   # a model, with the guidance it was tuned for
python -m vall_e.export codec codec.sft                                     # the codec (both halves, FP32, weight norm folded)
tts_generate --model model.sft --codec codec.sft --text "Hello there." --prompt voice.wav --wav out.wav
```

The server (CONFIG.md, the `tts` section) also has a page at `/tts` (at `/` when it serves speech alone): text, a named or uploaded voice,
the sampling settings, and the session's results with players and downloads. A named voice's prompt is encoded once, at its first use.

Each demasking step runs the backbone over its guidance rows (cond, null, promptless) in one pass of up to 2560 positions: the linears see
them as one block, attention stays within each row (on CUDA the fused WMMA attention kernel of the video service, `h3_attention`; on ROCm,
where that kernel isn't correct, batched GEMMs around a softmax).

The export's guidance is the default (v2.0 3.0 / 2.0, the distilled v2.0-fast 0 / 0, older exports 6.0 / 2.0); `--cfg` / `--cfg-prompt` override it.

## Memory

v2.0 takes 2.9 GiB of its GPU at load: the weights (1.4 GiB), the model's workspaces for 2560 positions (on ROCm also the attention
scores, in batches of heads within 128 MiB), and the codec with its activations for 10 s of audio. A longer sentence grows the codec's activations
by 42 MB a second; when the GPU has no room for them the request fails with an error.

Beside a language model the server loads the TTS first, so the model's arena is sized from what is left on that GPU (GLM slices its
experts over every rank, so its expert cache shrinks by that much on each). The TTS kernels run on the default stream: on a GPU the
language model also uses, they queue with its work instead of overlapping it.

## Parity (tts_generate --golden, test_tts_g2p, test_tts_resample; references from vall_e/golden.py in FP32; on ROCm and CUDA)

| Check | Result |
|---|---|
| Backbone hidden states, step 0 | cosine 1.00000 at every block (v2.0, v2.0-fast, v1.2) |
| Demasking, teacher-forced | 99.82-99.90% of revealed digits equal (v2.0 / v2.0-fast), 99.65-99.72% (v1.2) |
| Length head | log length within 0.004; frames equal or one off at a rounding boundary |
| Codec decoder | 118-125 dB SNR against vall_e's decode |
| Codec encoder + FSQ | every digit of a 6 s prompt equal |
| Text front-end | 11608 of 11609 store texts equal (en, de, fr, ko, zh: all 2000; ja: 1608 of 1609 kana lines) |
| Prompt resampling | 141-145 dB SNR against torchaudio (16, 22.05, 24, 48 kHz) |

## Limits

- Japanese must be given as kana: vall_e reads kanji through pykakasi's dictionary, which isn't ported (kanji are refused with a message).
- No language detection (vall_e's `auto`): requests name the language.
- Voices are single WAV files; vall_e's multi-clip folder prompts (speaker-embedding and MOS selection) aren't ported.
- Speed (a 7 s line with a voice prompt, one run of tts_generate): V100 v2.0 4.7x real time, v2.0-fast 6.9x, the 225M 8.8x; RX 7900 XTX
  (measured before the guidance rows shared a pass) v2.0 about 4.3x, v2.0-fast 8x, the 225M 9.5x. The backbone is compute-bound (a pass
  over ~670 positions is ~1 TFLOP); on a V100 a v2.0 generation's GPU time is ~57% the backbone's GEMMs, ~21% the FP32 codec, ~11%
  attention.
