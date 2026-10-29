# Runtime steering

Changes how a model behaves at inference time, per request, with the weights untouched: control vectors added to the residual stream, a direction clamp built by a probe of the prompt itself, and a logit penalty on reflection markers inside the reasoning block. The edits act on the fp16 residual after a range of layers, so they are independent of the weight format (EXL3, NVFP4). Qwen3.5-family dense models (Qwen3.8-27B) and Qwen3.8-Flash-Next; the CPU and Vulkan builds carry stubs.

Everything is off unless `steer` is set (`on`, or `file` to give every request the `steer_file` vector). With `steer=off` no table is allocated and no kernel is launched.

## What runs

* **Per-slot state on the device** (`inc/engine/modules/steer.h`): for each slot, two edit entries per steered layer (`vec` fp32 [hidden], a strength or reference level, a kind), a control block (scope bits, prompt length, position where the reasoning block ends, penalty), and a list of up to 256 penalised token ids. About 0.9 MB per slot for the 23 layers of the default range on the 27B. Kernels read the slot of each row from the row descriptor, so the decode CUDA graphs are unchanged by the feature and a request's settings change between steps without a recapture. A row of an unsteered slot returns before touching the residual.
* **`ResidualSteer::apply`** (after each steered layer): per row, the slot's entries in order, accumulated in fp32 in shared memory and written back saturated to fp16.
  * `add`: `x += strength * v`
  * `clamp`: `x -= max(x . v - ref, 0) * v` (the part of the projection above `ref` is removed; `v` is a unit vector)
* **`ResidualSteer::penalize`** (after the LM head): subtracts the slot's penalty from its token ids, on the rank that owns them, for rows whose predicted token lies inside the reasoning block. Plain decode, verify blocks and prefill rows all go through it.
* **Scope** per request: a row is a *prompt* row below the prompt length, a *think* row from there to the end of the reasoning block, an *answer* row after it. The edit applies to the rows named by the scope (`prompt`, `think`, `answer`, `generated` = think + answer, `all`). The reasoning block opens when the rendered prompt ends in the template's `<think>`, and ends when the model emits `</think>`; inside a speculative verify block a row past the boundary is classified with the old boundary until the next step.
* **Prefix cache**: a request whose scope covers prompt rows and that has edits neither restores from nor stores into the prefix cache (its prompt KV is not the cache's). Scopes without `prompt` share the prompt cache normally.
* Under tensor parallelism the residual is replicated, so every rank applies the same edit and nothing is communicated.

## Request fields

On `/v1/chat/completions` and `/v1/messages`, as top-level keys of the body:

| Field | Meaning |
|---|---|
| `steer_vector` | `""` = the server's choice (`steer_file` when `steer=file`), `"file"` = the `steer_file` option, `"none"`, or a name: `<steer_dir>/<name>.steer` |
| `steer_scale` | multiplies the strengths of the `add` entries (default `steer_scale` option) |
| `steer_scope` | comma list of `prompt`, `think`, `answer`, `generated`, `all` (default: the file's own) |
| `think_penalty` | logit taken off the reflection markers inside the reasoning block (default `think_penalty` option, 0 = off) |
| `steer_probe` | run the probe below before the real prefill |
| `steer_gate_threshold`, `steer_edit_strength`, `steer_hedge_strength`, `steer_push` | per-request values of the probe options |

A bad name, an unreadable or mismatching file, or `steer_probe` without `steer` answers 400.

## Control-vector files

`.steer` files hold, per entry, a layer, a kind (`add` or `clamp`), a strength or reference level and a unit direction (fp32 [hidden]); at most two entries per layer, all inside `steer_layers`. The header carries the hidden size, the layer count, the file's default scope and its recommended scale.

`build/steer_build --pos pos.txt --neg neg.txt --out v.steer [--layers 21-42] [--model dir] [--scope generated] [--scale 1.5]` builds one from two sets of prompts (one per line): it prefills each (as a user turn with an empty reasoning block), reads the last token's residual after every steered layer, and stores per layer the unit mean difference between the sets with a strength of the difference's length divided by the layer count. At `steer_scale` 1 the layers together move the state by the contrast the prompts show; the recommended scale in the file header (`--scale`, see below) is what a request multiplies.

## Recommended scales and the contrast sets

A `.steer` file (version 2) also carries a recommended scale; a request's `steer_scale` (or the `steer_scale` option) multiplies it, so 1 means as recommended. `tests/tools/steer_file.py info FILE...` prints a file's header and `steer_file.py scale FILE X` sets the scale; `steer_build --scale X` writes it at build time.

`tests/tools/steer_sets.py DIR` writes matched contrast sets (the same question with and without a style instruction) and held-out prompts; `steer_build --batch DIR` builds `terse`, `direct` and `english` from them in one model load, and `tests/tools/steer_vec_sweep.py` sweeps a vector's scale against a running server (answer length, the share of answers that finished, caveat phrases, CJK share, repetition, doubled words, similarity to the unsteered text). On Qwen3.8-27B EXL3 4.0 bpw, 8 held-out questions at 700 tokens (10 Chinese questions at 120 for `english`), unsteered answers average 434 words:

| Vector | Recommended scale | Effect |
|---|---|---|
| `terse` (one short sentence) | 1.0 | 38 words per answer, all finished (2: 13 words; 4: 4 words) |
| `direct` (no disclaimers or caveats) | 1.0 | 150 words, caveat share 0.25 to 0.12; at 2 it gives 37 words and no caveats; repetition rises at 4 |
| `english` (English only, against a Chinese request) | 2.0 | share of CJK characters in the answers 0.60 to 0.00; at 1 a Chinese term stays in the English text |

Below about 0.5 the vectors change the wording without changing the length or language. The strengths of a file add up over its layers, so a file built before the per-layer division (strength = the full contrast on every layer) needs scales around 1/23 of these.

## The probe and the gate (`steer_probe`)

For a request with `steer_probe`, before its prefill (on the scheduler thread; other slots wait):

1. The prompt is prefilled unedited and its recurrent state checkpointed. A few short openings are forced after it (5 accepting, 8 declining, 4 plain, 4 hedged; English, in `src/runtime/generator_probe.cpp`), each run as a block whose state is rolled back, and the residual after every layer of the band (`steer_band`, default 1/3 to 2/3 of the layers) at the last token is kept. These blocks run without the LM head.
2. Per band layer, the axis is the unit difference of the mean declining and accepting states, with the spread of the accepting samples projected out (`src/runtime/steer_probe.cpp`). `pa`, `pb` are the mean projections of the two sets and `sd` the standard deviation of the accepting one.
3. Look-ahead: the model decodes 24 greedy tokens unedited; for each token the band-averaged fraction `(h . axis - pa) / max(pb - pa, 1e-3)` is taken, and the largest over the tokens is the score.
4. Score above `steer_gate_threshold`: the gate fires. The slot gets a clamp on each edit layer (`steer_edit_layers`, default 17/32 to 21/32 of the layers, 34 to 42 of 64) with reference `pa - steer_edit_strength * sd`, optionally a second clamp (plain against hedged openings, reference `plain mean + steer_hedge_strength * gap`, off when negative), scope `all`, and the prompt is prefilled again with the edit active. Below it the request runs as if the probe had not been asked for, from the prefix cache like any other.

Evaluation on Qwen3.8-27B EXL3 4.0 bpw (`tests/tools/steer_eval_sets.py` writes the prompt sets, `tests/tools/steer_sweep.py` runs them): 56 target prompts (requests the model tends to answer with a stock reply, plain, with a purpose given, and as a second turn after a harmless first turn) and 56 control prompts (ordinary tasks and benign questions on the same topics, plain and as conversations). The gate score separates the two sets with an AUC of 0.949. At the default threshold 0.65 the gate fires on 47 of 56 targets and 3 of 56 controls; at 0.70 on 42 and 0. Of the 40 targets whose unprobed answer opens with a stock reply, all fire the gate and 8 still open with one afterwards. Controls that do not fire give the same text as without the probe (53 of 53). With a long neutral passage in front (6 target and 6 control prompts) the gate fires on all 6 targets (stock openings 1.00 to 0.17) and on no control. With the gate forced on every control prompt, the text stays coherent but moves: at edit strengths -12, -8 and -4 the share of repeated word trigrams is 0.02, 0.03 and 0.05 (0.00 unprobed), doubled words per answer 0.2, 1.0 and 2.2, and the answers share 13%, 10% and 7% of their leading words with the unprobed text.

The score is logged (`[steer] req N gate fired|clean score=... (threshold ..., ms)`) and kept in the request's metrics. The probe replaces `steer_vector` when it fires.

## Options

`steer`, `steer_layers`, `steer_file`, `steer_dir`, `steer_scale`, `steer_scope`, `think_penalty`, `think_penalty_words`, `steer_gate_threshold`, `steer_edit_strength`, `steer_hedge_strength`, `steer_push`, `steer_band`, `steer_edit_layers`; see [CONFIG.md](CONFIG.md). `think_penalty_words` is a comma list or a file with one word per line (default `Wait, Hmm, Alternatively, But`); each word's first token in the spellings as is, lower case, capitalised, each plain, after a space and after a newline goes on the list, which the server logs at start.

`steer_gate_threshold`, `steer_edit_strength` and `steer_edit_layers` say "auto" (or are empty) until set, and then take the model's value from `inc/engine/models/steer_defaults.h`; a request's own fields override both. The table, with the checkpoint each row was tuned on:

| model_type | gate threshold | edit strength | edit layers | tuned on |
|---|---|---|---|---|
| `qwen3_5` (also the fallback) | 0.65 | -8 | 17/32 to 21/32 of the layers (34 to 42 of 64) | Qwen3.8-27B EXL3 4.0 bpw |
| `qwen4_exp` | 0.60 | 8 | 1/3 to 2/3 (16 to 32 of 48) | Qwen3.8-Flash-Next, 2 bpw experts |


## Running the sweeps

The tools in `tests/tools/` are plain Python clients (standard library only) of a running server with `steer` on, plus `steer_build` (a binary that loads the model itself). Nothing in them depends on the backend, so the same steps work on a CUDA or ROCm build.

1. **Build** the engine for the backend. A build that reads the Volta tile order (`-DENGINE_EXL3_VOLTA_LAYOUT=ON`) loads only `tile_layout: volta` EXL3 checkpoints, and a stock build only the others; the loader refuses a mismatch.
2. **Prompt sets.** `steer_eval_sets.py DIR` writes `target.txt`, `control.txt` and their long-prompt variants for the clamp; `steer_sets.py DIR` writes the contrast and held-out sets for the control vectors. Both are one prompt per line (a line starting with `{` is a conversation, `{"messages": [...]}`), so the sets are easy to replace with your own.
3. **Vectors.** `build/steer_build --batch DIR --model CHECKPOINT` builds every `NAME.pos.txt` / `NAME.neg.txt` pair in DIR into `NAME.steer` in one model load (the model type comes from the checkpoint's `config.json`; `ENGINE_*` variables such as `ENGINE_TP` and `ENGINE_EXPERT_CACHE` apply as for the server). `steer_file.py info FILE...` prints the headers and `steer_file.py scale FILE X` sets a recommended scale.
4. **Server.** A launch config with the steering options, for example `{"model_dir": "...", "tp_size": 1, "port": 8011, "options": {"steer": "on", "steer_dir": "DIR"}}`, or `--set steer=on --set steer_dir=DIR` on the command line. `steer_probe` requests need nothing more; the gate score comes back in `usage.steer_gate`.
5. **Sweeps** (`--url` names the server; every tool writes a `.jsonl` with one row per prompt and prints a summary line per setting):
   * `steer_sweep.py --target T --control C --strength=-8,-4,0 --layers=22-32,16-32 --scope=all --hedge=-1`: the clamp's gate fire rate, stock-opening rate, repetition, doubled words and similarity to the unprobed text; `--no-force-control` leaves the gate alone on the control prompts (false positives), the default forces it (damage when it fires).
   * `steer_vec_sweep.py --name NAME --prompts P --scales 0.5,1,2,4 --max-tokens 500`: answer length, finished share, caveat share, CJK share, repetition, doubled words and similarity per scale; the length is only meaningful when `--max-tokens` is above the longest answer.
   * `steer_think_sweep.py --penalties 2,4,8`: reasoning length, marker rate and correctness on generated counting questions.
   * `steer_spec_bench.py --prompts P --vector NAME --penalty 8`: speculation acceptance and decode speed with and without steering.
6. **Read the files** with `steer_report.py FILE --list` (the settings a file holds) and, with filters, as a per-prompt text report (`--set`, `--strength`, `--layers`, `--fired`, `--stock`, `--scale`, `--limit`).
7. **Keep the winners** per model in `inc/engine/models/steer_defaults.h` (gate threshold, edit strength and edit window as fractions of the layer count), or set the options in the launch config.

A sweep is serial (one request at a time), and the probe is several times the cost of a plain request. For speed on a machine with several GPUs, run one single-GPU server per card on separate ports (`CUDA_VISIBLE_DEVICES`, `tp_size` 1) and give each a slice of the grid. The probe stalls the other slots of its server while it runs, so extra slots do not help a sweep. Every tuned number is per model, bit width and quantization format; repeat a short sweep on a new checkpoint before trusting the defaults.

## The reflection-marker penalty

`think_penalty` (option, or per request) subtracts a fixed logit from the first token of each marker spelling while the predicted token lies inside the reasoning block. Only first tokens that are word pieces of at least three letters go on the list (a bare newline or a single letter would be penalised everywhere); the server logs the list at start (15 tokens for the default words). `tests/tools/steer_think_sweep.py` measures it: seeded counting questions with exact answers, thinking on, per penalty the reasoning words, marker words per 100 reasoning words, the share of answers that finished and the share that are correct.

On Qwen3.8-27B EXL3 4.0 bpw, 12 such questions at 3500 tokens, the model hesitates rarely (0.19 marker words per 100 reasoning words, 307 reasoning words per answer on average); penalties of 2, 4 and 8 take the marker rate to 0.05, 0.03 and 0.00 and change neither the length (307 to 315 words) nor the share finished or correct (0.83 each; the two unfinished questions are long enumerations with 1000-word reasoning that the penalty does not shorten). The default stays 0.

## Push

`steer_push` (option and request field, default 0) goes the other way: with `steer_probe` on, it adds the probe's declining direction to the states of the edit layers, `steer_push` times the gap between the declining and the accepting openings (their mean projections on the axis), on every prompt whether or not the gate fires; negative values subtract it. It shares the layers' two edit slots with the clamp, so a push replaces the hedge clamp.

Measured on Qwen3.8-27B EXL3 4.0 bpw (24 target and 24 control prompts, the gate left unforced on the controls, 80 tokens, clamp strength -8 on the targets; the stock rate is the share of replies that open with a refusal):
* Control prompts' refusal rate by push: 0 at 0.03, 0.08 at 0.06, 0.79 at 0.10 ("I cannot fulfill this request."), then the text breaks down: repeated fragments at 0.15, dots from 0.2, garbage at 0.5 and above. The useful band is narrow, about 0.08 to 0.12.
* Targets under the clamp alone refuse 0.21 of the time (0.71 unprobed); with the push 0.25 / 0.29 / 0.33 at 0.03 / 0.06 / 0.10 and 0.50 at 0.3, and at 0.5 to 1 the replies are refusals again (0.79 and 0.92) while the controls are already broken.
* Negative push (-1) breaks every reply into one repeated token.
* The sum over the edit layers is what counts: a push of 1 puts about ten gaps into the residual stream. Fine steps are needed, and the tuned value moves with the edit window (`steer_edit_layers`).

## Flash-Next

Flash-Next (`qwen4_exp`) keeps four fp32 residual streams per row. The steering entries act on the mean of the four and the change of the mean is added to every stream (`ResidualSteer::apply_streams`), after the finished layer; a layer that is steered or captured skips the fused all-reduce hand-off of pre-normed streams, since it edits the streams after it. The probe captures the stream mean, and the penalty runs after the LM head as for Qwen. `steer_build` reads `model_type` from the checkpoint's `config.json`. The default steered layers are 1/3 to 2/3 of the 48 (16 to 32).

On the 24 GB card this model loads with `expert_cache` 0.4 (0.5 does not fit the arena) and decodes at about 16 to 20 tokens per second; a probe takes about 9.5 s here against 2.5 s on the 27B.

Measured on Qwen3.8-Flash-Next-exl3 (2 bpw experts), numbers tuned for the 27B left as they are:
* Clamp, 56 target and 56 control prompts (the same sets as for the 27B): 39 of the 56 targets are answered with a stock reply unprobed (stock-opening rate 0.70). They score 0.62 to 1.01, the 17 targets the model simply answers 0.26 to 0.67, the controls 0.26 to 0.49. A threshold of 0.60 fires on all 39 declined prompts and on no control (the 27B's 0.65 would miss 3); AUC of target against control 0.873. Controls that do not fire give the same text as without the probe (56 of 56).
* Edit strength and window (56 targets with the gate on, 16 controls with it forced; target stock-opening rate, baseline 0.70): at strength -8 the rate is 0.54 to 0.62 over the windows 20-28, 22-32 and 28-32 (the 27B's setting is too gentle here); -4: 0.43 to 0.52; 0: 0.36 (22-32) and 0.30 (16-32); 4: 0.30 and 0.29; **8 on 16-32: 0.21 with clean text**; 16: 0.21 with slight damage (repetition 0.02, 0.3 doubled words per target answer); 32: 0.09 with clear damage (repetition 0.11, 2.1 doubled words). At strength 4 on 16-32, 23 of the 36 declined prompts that fire are answered normally afterwards; hotwiring, insulting nicknames and the cruel text message still decline. Forced on controls, the text stays coherent in every row and keeps 0.38 to 0.58 of its leading words.
* Vectors (6 held-out prompts each, `steer_vec_sweep.py`): `terse` at scale 1 takes the answers from 336 words (none finished at 500 tokens) to 73 (all finished), 14 at 2; `direct` at 2 gives 56 words and no caveat phrases (caveat share 0.50 unsteered, 0.17 at scale 1); `english` takes the CJK share of the answers to Chinese questions from 0.61 to 0.29 at scale 1 and 0.00 at 2. Recommended scales stored in the files: terse 1.0, direct 2.0, english 2.0. Repetition and doubled words stay at the unsteered level at these scales.
* The penalty path is live: a thinking request with `think_penalty` 8 differs from the plain one (53 against 61 reasoning words on a short question).

## Tests

* `test_steer`: the kernels against an fp32 reference on a mixed batch (add and clamp, every scope, slots with and without edits, the penalty over a vocabulary slice); rows the scope does not cover and unlisted logits come back bit-identical.
* `test_steer_probe`: the probe's axis and fraction on synthetic states (host only).
* `test_steer_model`: the 27B with three slots: a neighbouring slot keeps its tokens while another is steered in the same batches, a random-direction vector changes the steered slot's text, and a penalty on the tokens a reasoning block starts with removes them. `--tokens` prints the plain greedy tokens for a comparison with `steer=off`.

## Limits

* The Qwen3.5 family and Flash-Next implement the hooks; GLM and DeepSeek (also several residual streams) do not.
* Speculation (DFlash2 drafter, MTP head) stays correct under steering, because the verify block runs the same steered forward as plain decode; the draft proposals are not steered by the edits of their own layers and do not apply the logit penalty. `tests/tools/steer_spec_bench.py` measures the cost on Qwen3.8-27B (8 questions, 200 tokens): the accepted share of drafted tokens is 0.358 (DFlash2) and 0.486 (MTP) without steering, 0.323 and 0.520 with the terse vector, 0.332 and 0.479 with the clamp forced, and with a penalty of 8 on common sentence starters inside the reasoning block 0.325 against 0.330 and 0.479 against 0.486. The steered text equals the unspeculated steered text on 7 of 8 prompts with either speculator (the eighth differs late in the answer, at the same place for both).
* The probe prefills a probed prompt twice when the gate fires and once more than a plain request when it does not (the probe's own prefill is not taken from the prefix cache).
* The probe's defaults are per model (`inc/engine/models/steer_defaults.h`) and were tuned on one checkpoint and bit width each: Qwen3.8-27B EXL3 4.0 bpw (the Evaluation paragraph above) and Qwen3.8-Flash-Next with 2 bpw experts (the Flash-Next section). A wider edit window or a stronger edit than the tuned one degrades unrelated text when the gate fires; other models, depths and quantization formats need their own sweep (see "Running the sweeps"). The openings are English and short; the second clamp degrades the text quickly.
* Tensor parallelism above 1 is untested with steering: the residual is replicated and every rank applies the same edit, but the 2- and 4-rank runs, the probe's capture on rank 0 and Flash-Next's skipped fused all-reduce hand-off on steered layers have not been run.
