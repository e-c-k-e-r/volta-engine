# `src/server/`

This section of the project handles `volta_server`: the HTTP server, the chat protocols spoken over it, and the monitor page. Everything else (generation, scheduling, models) is behind `runtime/`, and the video service (MiniMax-H3) is its own sub-folder.

Start it with a launch config or flags, see [GETTING_STARTED.md](GETTING_STARTED.md) and [CONFIG.md](CONFIG.md). The model family is read from the model directory:
* `qwen3_5`, `glm5_next`, `qwen4_exp`: the language-model path (below)
* a `model_index.json` of class `MiniMaxH3Pipeline`: the video service (see [video.md](video.md))
* anything else is a fatal error

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
* `POST /v1/messages` and `/v1/messages/count_tokens` (Anthropic), so Claude Code can use it with `ANTHROPIC_BASE_URL=http://host:8000`
* `GET /health` (also `/`), `/metrics` (JSON), `/metrics/prometheus`
* `GET /monitor`, `/api/stats`, `/api/experts`. The recent requests in `/api/stats` carry the split of the time before the first token: `queue_ms` (waiting for a slot; `ttft_ms` counts from the slot being taken), `restore_ms` (prefix cache resume), `prefill_ms` (prefill kernels), `store_ms` (checkpoint stores); the rest of `ttft_ms` went to other slots' decode steps. The server prints two lines per request, one when it takes a slot and one when it finishes
* the video routes when the model is MiniMax-H3

## `config.cpp`

The launch file (`--config`), `--set key=value`, `ENGINE_*` environment variables, and the CLI flags all end up in one `ServerConfig`.
* the option table that backs `docs/CONFIG.md` lives with the runtime options (`inc/engine/runtime/options.def`): `volta_server --print-config-docs` regenerates the doc, and `gate.sh static` fails when it's stale
* the `video` object holds the MiniMax-H3 settings

## `chat.cpp`, `openai.cpp`, `anthropic.cpp`

* `chat.cpp`: the protocol-independent request. It renders the prompt through the model's Jinja chat template, applies server defaults (sampling, max tokens), and starts the generation.
  * message content is tokenized as plain text: the template's role markers are real special tokens, but the same strings inside message content are escaped first (`escape_control_text`), so quoted source cannot end or open a turn
* `openai.cpp`, `anthropic.cpp`: the two protocol front ends. Each parses its payload into a chat request, and writes one response (streamed or not) from parsed pieces as they arrive.
  * tools are accepted in either shape (`{"type":"function","function":{...}}` or Anthropic's `{name, description, input_schema}`)
  * images arrive as base64 `data:` URLs (OpenAI `image_url`) or `image` blocks (Anthropic), also inside `tool_result` blocks. Remote URLs are not fetched.
  * `tool_choice` `required` / a named function tells the model to call a tool (best effort, no constrained decoding)
* usage reports the prefix cache hits: OpenAI `prompt_tokens_details.cached_tokens`, Anthropic `cache_read_input_tokens` (with `input_tokens` excluding them)

## `output_parser.cpp`

An incremental parser for what the model writes: `<think>` reasoning, plain content, and tool calls in the family's syntax (Qwen: `<function=name><parameter=key>value</parameter></function>`; GLM: `name<arg_key>k</arg_key><arg_value>v</arg_value>`).
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

The MiniMax-H3 service: `adapters` (LoRA scan and labels), `jobs` (the queue and the on-disk job history), `service` (the worker thread that owns the GPUs), `api` (the routes), `metrics`, and `page` (the web page). Picture caption requests (`POST /api/caption`, with `video.captioner`) are queued beside the jobs and run by the same worker between jobs. See [video.md](video.md).
