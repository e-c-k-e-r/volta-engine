# Runtime steering

Changes how a model behaves at inference time, per request, with the weights untouched: control vectors added to the residual stream, a direction clamp built by a probe of the prompt itself, and a logit penalty on reflection markers inside the reasoning block. The edits act on the fp16 residual after a range of layers, so they are independent of the weight format (EXL3, NVFP4). Qwen3.5-family dense models (Qwen3.8-27B) only; the CPU and Vulkan builds carry stubs.

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
| `steer_gate_threshold`, `steer_edit_strength`, `steer_hedge_strength` | per-request values of the probe options |

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

The score is logged (`[steer] req N gate fired|clean score=... (threshold ..., ms)`) and kept in the request's metrics. The probe replaces `steer_vector` when it fires.

## Options

`steer`, `steer_layers`, `steer_file`, `steer_dir`, `steer_scale`, `steer_scope`, `think_penalty`, `think_penalty_words`, `steer_gate_threshold`, `steer_edit_strength`, `steer_hedge_strength`, `steer_band`, `steer_edit_layers`; see [CONFIG.md](CONFIG.md). `think_penalty_words` is a comma list or a file with one word per line (default `Wait, Hmm, Alternatively, But`); each word's first token in the spellings as is, lower case, capitalised, each plain, after a space and after a newline goes on the list, which the server logs at start.

## Sweeps

`tests/tools/steer_sweep.py` sends two prompt files (target and control, one prompt per line) to a running server with `steer=on` over a grid of `steer_edit_strength`, `steer_edit_layers`, `steer_scope` and `steer_hedge_strength`, and reports per point the gate fire rate, the share of outputs that start with a stock opening, repetition and doubled words, and how far the control text moves from the unprobed text when the gate is forced. The response carries the gate score in `usage.steer_gate`.

## The reflection-marker penalty

`think_penalty` (option, or per request) subtracts a fixed logit from the first token of each marker spelling while the predicted token lies inside the reasoning block. Only first tokens that are word pieces of at least three letters go on the list (a bare newline or a single letter would be penalised everywhere); the server logs the list at start (15 tokens for the default words). `tests/tools/steer_think_sweep.py` measures it: seeded counting questions with exact answers, thinking on, per penalty the reasoning words, marker words per 100 reasoning words, the share of answers that finished and the share that are correct.

On Qwen3.8-27B EXL3 4.0 bpw, 12 such questions at 3500 tokens, the model hesitates rarely (0.19 marker words per 100 reasoning words, 307 reasoning words per answer on average); penalties of 2, 4 and 8 take the marker rate to 0.05, 0.03 and 0.00 and change neither the length (307 to 315 words) nor the share finished or correct (0.83 each; the two unfinished questions are long enumerations with 1000-word reasoning that the penalty does not shorten). The default stays 0.

## Tests

* `test_steer`: the kernels against an fp32 reference on a mixed batch (add and clamp, every scope, slots with and without edits, the penalty over a vocabulary slice); rows the scope does not cover and unlisted logits come back bit-identical.
* `test_steer_probe`: the probe's axis and fraction on synthetic states (host only).
* `test_steer_model`: the 27B with three slots: a neighbouring slot keeps its tokens while another is steered in the same batches, a random-direction vector changes the steered slot's text, and a penalty on the tokens a reasoning block starts with removes them. `--tokens` prints the plain greedy tokens for a comparison with `steer=off`.

## Limits

* Qwen3.5-family models with a single residual stream; the other models do not implement the hooks.
* The DFlash2 and MTP heads do not apply the logit penalty, and see the steered hidden states; acceptance on steered slots is not measured.
* The probe prefills a probed prompt twice when the gate fires and once more than a plain request when it does not (the probe's own prefill is not taken from the prefix cache).
* The defaults (gate threshold 0.65, edit strength -8, edit layers 34-42 of 64, second clamp off) were tuned on Qwen3.8-27B EXL3 4.0 bpw with `tests/tools/steer_sweep.py` over 8 target and 10 control prompts: the gate fired on 7 of 8 target prompts and on none of the control prompts (scores 0.66 to 0.83 against 0.45 to 0.61), and the control text was unchanged. A wider edit window or a higher strength degrades unrelated text when the gate fires; other models, depths and quant formats need their own sweep. The openings are English and short; the second clamp degrades the text quickly.
