# Server (`src/server/`)

This section of the project handles `volta_server`: the HTTP server, the chat protocols spoken over it, and the monitor page. Everything else (generation, scheduling, models) is behind `runtime/`, and the video service (MiniMax-H3) is its own sub-folder.

Start it with a launch config or flags, see [GETTING_STARTED.md](GETTING_STARTED.md) and [CONFIG.md](CONFIG.md). The model family is read from the model directory:
* `qwen3_5`, `glm5_next`, `qwen4_exp`, `deepseek_v41`, `gemma4`: the language-model path (below; per-model setup in [MODELS.md](MODELS.md))
* a `model_index.json` of class `MiniMaxH3Pipeline`: the video service (see [VIDEO.md](VIDEO.md))
* anything else is a fatal error

## Settings for several requests

| Requests at once | `max_slots` | Other settings |
|---|---|---|
| 1 | 1 or 2 | `spec_suffix=8` when the output copies its input |
| 2 | 2 | the same, `prefill_budget_ms=300` if token gaps matter more than prompt speed |
| 3 to 4 | 4 | the same; on GLM the slots share the KV pool (`kv_pool`), so their contexts add up to `kv_pool` tokens |

* `spec_suffix` offers matches of the request's own text to the DFlash2 and DSpark drafters' rounds (and is what `speculator=suffix` drafts from); the MTP head does not take them.
* while other requests decode, a new prompt is prefilled in pieces of `prefill_chunk` rows (512) with decode steps between them; `prefill_budget_ms` sizes each piece from its measured time instead, and `prefill_cooldown` runs more decode steps between pieces.
* `max_batch_size` (`--max-batch-size`, 8) caps how many of the slots the scheduler fills.

## `http.cpp`

A small HTTP/1.1 server on one epoll thread.
* a `Connection` is one client socket
  * writes never block the caller: what the socket does not take is queued and flushed on `EPOLLOUT`, and a peer that lets the queue grow past a limit is dropped
  * this is what lets the scheduler thread stream tokens without a slow client stalling generation
* `Router` is a method + path table: exact paths win over prefixes, and a method of `*` matches any (the handler answers 405 itself)
  * handlers run on the epoll thread, unless the route is added with `blocking = true`, which hands the request to a short-lived thread (decoding a large upload, serving a file)
* helpers for JSON and SSE responses, error bodies shaped like OpenAI's and Anthropic's, and files with single-range `Range` support (used for the generated videos)
* a connection that sends nothing for `request_timeout_s` before its request is complete is dropped

## `server.cpp`, `main.cpp`

Owns the socket, the router, and the model-family choice.

Routes:
* `POST /v1/chat/completions` (OpenAI), `GET /v1/models`
* `POST /v1/messages` and `/v1/messages/count_tokens` (Anthropic), so Claude Code can use it with `ANTHROPIC_BASE_URL=http://host:8000`. Claude Code assumes a 200K window for a model it does not know, so give it the server's (`context_length` in `GET /v1/models`) with `CLAUDE_CODE_MAX_CONTEXT_TOKENS=<max_seq>`; it then compacts in time. A messages request that does not fit is refused with the API's own errors (`prompt is too long: N tokens > W maximum`, ``input length and `max_tokens` exceed context limit: N + M > W``), which Claude Code acts on (compacting, or retrying with a smaller `max_tokens`); W is `max_seq` minus one verify block. The OpenAI route instead ends the reply at the window (finish reason `length`)
* `GET /health` (also `/`), `/metrics` (JSON), `/metrics/prometheus`
* `GET /monitor`, `/api/stats`, `/api/experts`. The recent requests in `/api/stats` carry the split of the time before the first token: `queue_ms` (waiting for a slot; `ttft_ms` counts from the slot being taken), `restore_ms` (prefix cache resume), `prefill_ms` (prefill kernels), `store_ms` (checkpoint stores); the rest of `ttft_ms` went to other slots' decode steps. The server prints two lines per request, one when it takes a slot and one when it finishes
* the video routes when the model is MiniMax-H3

## `config.cpp`

The launch file (`--config`), `--set key=value`, `ENGINE_*` environment variables, and the CLI flags all end up in one `ServerConfig`.
* the option table that backs `docs/CONFIG.md` lives with the runtime options (`inc/engine/runtime/options.def`): `volta_server --print-config-docs` regenerates the doc, and `gate.sh static` fails when it's stale
* the `video` object holds the MiniMax-H3 settings

## `chat.cpp`, `openai.cpp`, `anthropic.cpp`

* `chat.cpp`: the protocol-independent request. It renders the prompt through the model's Jinja chat template (DeepSeek-V4.1: the built-in port of the checkpoint's `encoding/encoding.py`, `src/loader/deepseek_encoding.cpp`), applies server defaults (sampling, max tokens), and starts the generation.
  * message content is tokenized as plain text: the template's role markers are real special tokens, but the same strings inside message content are escaped first (`escape_control_text`), so quoted source cannot end or open a turn
  * an API reasoning effort is mapped onto what the model's template accepts (DeepSeek-V4.1: low 50, medium 60, high 75, max 100)
  * for Gemma 4 the tool results of consecutive tool messages go into one message's `tool_responses`, named after the calls they answer, as its template expects
  * without a loaded vision tower an image part becomes the text note `[image omitted: this model cannot view images]`, so a request with images still runs
* `openai.cpp`, `anthropic.cpp`: the two protocol front ends. Each parses its payload into a chat request, and writes one response (streamed or not) from parsed pieces as they arrive.
  * tools are accepted in either shape (`{"type":"function","function":{...}}` or Anthropic's `{name, description, input_schema}`)
  * images arrive as base64 `data:` URLs (OpenAI `image_url`) or `image` blocks (Anthropic), also inside `tool_result` blocks. Remote URLs are not fetched.
  * `tool_choice` `required` / a named function tells the model to call a tool (best effort, no constrained decoding)
* usage reports the prefix cache hits: OpenAI `prompt_tokens_details.cached_tokens`, Anthropic `cache_read_input_tokens` (with `input_tokens` excluding them)

## `output_parser.cpp`

An incremental parser for what the model writes: reasoning (`<think>`; Gemma 4: `<|channel>thought ... <channel|>`), plain content, and tool calls in the family's syntax (inside `<tool_call>` ... `</tool_call>`, Qwen: `<function=name><parameter=key>value</parameter></function>`, GLM: `name<arg_key>k</arg_key><arg_value>v</arg_value>`; Gemma 4: `<|tool_call>call:name{key:value}<tool_call|>` with `<|"|>`-quoted strings; DeepSeek-V4.1 DSML: `<｜DSML｜ invoke name="n">` with `<｜DSML｜ parameter name="k" string="true|false">v</｜DSML｜ parameter>` per argument, inside `<｜DSML｜ calls>`, where `string="false"` values are JSON).
* the result does not depend on how the text is split across `feed()` calls
* argument values written as text are decoded using the parameter types from the request's tool schemas
* `starts_in_reasoning` handles a rendered prompt that ended inside an open `<think>` block

## `monitor.cpp`, `monitor_page.cpp`, `system_stats.cpp`

The `/monitor` page: one self-contained HTML string (Monokai or Tomorrow Night theme) reading `/api/stats` and `/api/experts`.
* throughput, slots with context fill, token-gap and TTFT percentiles, speculation, the prefix cache, and recent requests
* per-GPU power, clocks, temperature, and throttle reasons through NVML, which is loaded at run time (a machine without it just reports no GPUs)
* for MoE models, a layer-by-expert routing heat map with the load per rank (`expert_stats=false` turns the counters off)
  * the expert tier and the monitor both read and clear the same counters, so the monitor undercounts with the tier on

## `src/server/tui/`

The terminal dashboard, one view in two hosts: `volta_top [host[:port]]` polls a running server's `/api/stats` (so it attaches to any server, at any time), and `volta_server --tui` (or `"tui": true`) runs the same view on a thread inside the server.
* `view` renders a frame from the `/api/stats` document: Overview (throughput sparklines from the 15-minute history, token-gap histogram), Slots (pick a slot, `Enter` for its detail, recent requests), GPUs, Experts (the `/api/experts` routing heat map, fetched only while that tab is open; `m` switches between the recent minute and since boot) and Log; scrolling; `p` pauses; a missing key never fails a frame
* an alert row appears under the tabs and the title bar turns red (yellow for warnings) for: no connection, a GPU at 85 C or more or throttled by hardware or thermal limits (the software power cap is not one), more than 2 GiB in use on a GPU outside the engine arena (the arena fills the card by design), host memory over 95%, waiting requests, a p99 token gap over 500 ms
* `term` is the raw-mode terminal on the alternate screen; Ctrl-C still reaches the server's handler
* `app` has the viewer loop (one reading per second), `StdioCapture` (descriptors 1 and 2 go to a pipe while the dashboard is up, so `printf` and library output land in the Log tab instead of tearing the screen) and `ServerTui`; `q`, or the server stopping, restores the terminal and the console output
* `http_get` is the small blocking client `volta_top` uses; the Log tab is empty there, since console output is only available inside the server process

## `src/server/video/`

The MiniMax-H3 service: `adapters` (LoRA scan and labels), `jobs` (the queue and the on-disk job history), `service` (the worker thread that owns the GPUs), `api` (the routes), `metrics`, and `page` (the web page). Picture caption requests (`POST /api/caption`, with `video.captioner`) are queued beside the jobs and run by the same worker between jobs. See [VIDEO.md](VIDEO.md).
