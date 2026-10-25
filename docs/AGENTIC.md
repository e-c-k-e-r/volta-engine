# Agentic benchmark

`tests/agentic/bench.py` runs the Claude Code CLI against any server that speaks the Anthropic messages API (`volta_server` serves `/v1/messages`) and scores
it on small software tasks. It needs only a base URL and a model name; nothing in it is specific to this engine.

```
tests/agentic/bench.py selftest
tests/agentic/bench.py run --base-url http://localhost:8000 --model <name> --out results/agentic/<run> [--tier quick] [--only ID,ID] \
                           [--repeats N] [--thinking off|low|medium|high] [--max-turns N] [--timeout S]
tests/agentic/bench.py summary results/agentic/<run>
```

## Tasks

A task is a directory `tests/agentic/tasks/<id>/`:

| file | content |
|---|---|
| `task.json` | `id`, `tier` (quick), `prompt`, `tags` |
| `repo/` | the starting files; each run gets a fresh copy |
| `verify.sh` | runs in the work directory after the client finished (files of `hidden/` are copied in first); exit 0 is a pass |
| `solution.sh` | a reference solution; `selftest` checks that the verifier fails on `repo/` and passes after it |

The quick tier covers a bug fix, a feature with tests, a rename across files, a shell fix, a read-only question and a config edit.

## Sandbox

Every run, and every verifier, executes under bubblewrap: the system is read-only, `/home` and `/mnt` are empty, the environment is cleared, and the task's
work directory is the only writable path. The client run has network access (it must reach the server); verifiers do not. The client's own
configuration lives in a throwaway directory, so the user's Claude Code settings are never read or written.

## Results

A run directory holds `meta.json` (client version, server URL, model, thinking level, limits, repository revision), `results.jsonl` (per run: pass, wall
time, turns, tool calls and errors, input and output tokens, thinking and answer characters) and, per task and repetition, the client's event stream
(`events.jsonl`) and the final work directory. `summary` prints pass rate, median wall time, mean turns, tool calls and output tokens, and the share of
thinking characters among thinking plus answer characters. Runs are not deterministic (batching and speculation change the sampled text), so use
`--repeats` and compare pass rates, not single outcomes.

`--thinking` sets `MAX_THINKING_TOKENS` for the client (off 0, low 4000, medium 10000, high 31999); `default` leaves it unset.

## Calibration corpora

Started with `--set dump_requests=<dir>`, the server writes every request body and the rendered prompt; those are the token sequences the model saw, and
they serve as calibration text for Hessian capture (`docs/QUANTIZATION.md`).
