# libvolta

The engine as a shared library for a host application (a game engine): a text model and / or a vall-e speech model loaded from a launch
config, driven from the host's threads through one C++ header, `inc/volta/volta.h`. It reuses volta_server's pieces without the HTTP front end:
the launch config, the generator, the OpenAI chat-completions parser and chat layer (reasoning and tool-call parsing included), and the TTS service.

## Building

The Vulkan configuration with `ENGINE_LIBRARY=ON`:

```
cmake -S . -B build_vklib -G Ninja -DENGINE_BACKEND=VULKAN -DENGINE_LIBRARY=ON
ninja -C build_vklib volta test_library
```

- `build_vklib/libvolta.so` exports namespace `volta` only (`src/library/volta.map`); the objects it is built from are position-independent with
  hidden symbols, so the engine's copies of nlohmann::json and the rest don't clash with the host's.
- Every kernel's SPIR-V is linked in (a generated table of `.incbin` blobs, looked up by name), so the library needs no kernel files at run time.
  `ENGINE_SPV_DIR` still makes the loader read a directory instead.
- With propan2ol (the deps/propan2ol submodule, or an installed one) the library has speech, the same condition as volta_server; propan2ol's
  data is built in, so speech needs no files beyond the models. Without it `speak` returns an error.
- The host builds with the same C++ standard library (std::string, std::vector and std::function cross the boundary). Linux only.

## The API

```cpp
#include "volta/volta.h"

volta::set_log_callback( []( volta::LogLevel level, const std::string& line ) { ... } );   // optional: the engine's log lines
std::string error;
auto engine = volta::Engine::create( config_json, &error );   // or ( config_json, &error, vulkan_device )
```

- `Engine::create( config_json, error, vulkan )`: `config_json` is a volta_server launch config as text ([CONFIG.md](CONFIG.md)): `model_dir`
  (the text model), `options`, and a `tts` section (`model`, `codec`, `voices_dir`, `device`) for speech; either part may be absent. Inside the
  library `tp_size` defaults to 1 and the `pin_threads` option to false (the scheduler thread is left to the OS); the config overrides both.
  One Engine per process: a second `create` while one exists fails.
- Text: `chat( request_json, on_event, error )` takes an OpenAI chat-completions body (messages, tools, tool_choice, max_tokens, sampling,
  stop, reasoning_effort, ...) and returns the request's id. `on_event` receives `ChatEvent`s in order on the engine's scheduler thread:
  Content and Reasoning deltas, ToolCall (name and a JSON object of arguments), and a final Finish with `finish_reason` (`stop`, `length`,
  `tool_calls`, `cancelled`) and the usage (prompt / cached / completion tokens, time to first token, decode rate). Returning false from
  `on_event` cancels the reply; so does `cancel( id )`. `count_tokens( request_json, error )` is the prompt length a body renders to.
- Media: `chat( request_json, media, on_event, error )` and `count_tokens( request_json, media, error )` take the host's videos and audio
  (`volta::Media`): `Video`s as frames (`VideoFrame`s of 8-bit RGB, one size, each with its time in seconds) and `Audio` clips as mono float
  samples with their rate. Each `{"type": "video"}` content part without a source of its own takes the next video, each `{"type": "input_audio"}`
  part without `input_audio` data the next clip, in message order. The engine prepares them as the server prepares a video's frames and an
  audio file's samples ([FEATURES.md](FEATURES.md#multimodal)). Parts with sources work as in the server (video files and audio other than WAV
  need a library built with FFmpeg's libraries, which it then links).
- Speech: `speak( request_json, &speech, error )` takes a `/v1/audio/speech` body (input, voice, speed, and vall-e's language, seed, cfg,
  cfg_prompt, steps, duration, temperature, ...; [TTS.md](TTS.md)), blocks the calling thread and returns mono float PCM with its sample rate
  and the seed it was sampled with. One synthesis runs at a time. `speak_async( request_json, done )` queues it on the engine's speech
  thread and calls `done( ok, speech, error )` there. `add_voice( name, pcm, rate, error )` encodes a voice prompt from PCM once and makes it
  usable by name; `voices()` lists those and `tts.voices_dir`'s WAV files.
- Errors come back as strings: an invalid config, a model that fails to load (the last error line logged), a bad request body. An internal
  invariant failing (`ENGINE_ASSERT`, including an arena that runs out of memory) still ends the process, as it does in volta_server.

## Contexts

Every chat resumes from the longest cached beginning of its prompt: the prefix cache keeps KV snapshots of prompts of at least 512 tokens in
pinned host memory (`prefix_cache_gib`) and, with `prefix_cache_dir`, evicted ones on disk across restarts ([SERVER.md](SERVER.md)). A host can
also warm that cache and keep its content itself, e.g. a character's long backstory in a save file:

```cpp
engine->prefill( body, &error );                     // the prompt into the cache, no reply (one token sampled and dropped)
volta::Context ctx;
engine->export_context( body, &ctx, &error );        // the cached beginning of the prompt the body renders: ctx.tokens, ctx.data
...
engine->import_context( ctx, /*pinned*/ true, &error );   // back into the cache (another run, another process)
engine->chat( body, on_event, &error );               // resumes after ctx.tokens
engine->release_context( ctx.tokens, /*drop*/ false, &error );   // evictable again (drop: freed now)
```

- A context is the cache's chain of snapshots for that prefix (a shared root, such as a world's lore, is a node of its own, and importing a
  context whose root the cache holds already reuses it), with a fingerprint of the model files (names, sizes and a hash of each file's first
  MiB), the options the snapshot layout depends on (`kv_format`, ...) and the device layout. It loads only into an engine with the same;
  `import_context` refuses others.
- Its size is the prompt's KV: for Gemma 4 E4B 16 KiB a token in FP16 (8 in FP8, about 4 in NVFP4, `kv_format`) on its four full-attention
  layers, plus 20 MiB for the sliding-window layers' last 512 positions (FP16): a 1561-token prompt is 45 MiB, imported in 32 ms (`test_library`).
- `export_context` fails when nothing of the prompt is cached, `prefill` for prompts under 512 tokens. A pinned context counts against
  `prefix_cache_gib` until it is released.
- Only a prompt's exact beginning is reused: whatever comes after the first token that differs (a memory list in another order, a changed
  line) is prefilled again, so a prompt that keeps its unchanging parts first gets the most from the cache.
- A system block of at least 1024 tokens is pinned when it is first prefilled, for every later conversation that shares it
  (`prefix_pin_system`); a host with many distinct system prompts (one per character) can set it false so they age out like any prefix.
- These calls are not for an `on_event` callback: `prefill` waits for the scheduler thread that runs the callbacks, and the others run on it.

## The Vulkan device

`volta::VulkanDevice` chooses the device of a Vulkan build:

- the engine's own (the default): physical device `index` (-1: the `ENGINE_VK_DEVICE` environment variable, else 0) or the first whose
  `VkPhysicalDeviceProperties::deviceID` is `pci_device_id`; the engine creates the instance and device.
- the host's: `instance`, `physical_device`, `device` and a `queue` of a family with compute (`queue_family`), which the engine never
  destroys; `queue_mutex`, held around the engine's `vkQueueSubmit` / `vkQueueWaitIdle` when the host submits to the same queue; and what the
  device was created with: `enabled_features` (the `VkPhysicalDeviceFeatures2` chain given to `vkCreateDevice`, read for the Vulkan 1.1 / 1.2 /
  1.3 structs and the single-feature ones) and `enabled_extensions`. The device must have every feature the engine enables on a device of
  its own where the GPU supports it: `bufferDeviceAddress`, `shaderFloat16`, `timelineSemaphore`, `shaderInt64`, `shaderInt16`,
  `shaderFloat64`, `shaderInt8`, 16- and 8-bit storage, and `subgroupSizeControl` on GPUs whose subgroups are not 32 wide. The engine uses
  `VK_KHR_push_descriptor`, `VK_EXT_external_memory_host`, `VK_EXT_memory_budget`, `VK_KHR_shader_integer_dot_product`,
  `VK_EXT_subgroup_size_control` and `VK_NV_cuda_kernel_launch` (with its `cudaKernelLaunchFeatures`: the tensor-core GEMM, NVFP4 and attention kernels on Volta) when they
  are enabled. A device that lacks a required feature is refused with an error naming it. The
  host's device is the only one: `tp_size` 1 and `tts.device` 0.

The arena is sized from the memory the device reports free (`VK_EXT_memory_budget`), so what the host has already allocated is left alone.

## The test

`test_library <launch config> [--adopt | --adopt-bad] [--out <dir>]` links `libvolta.so` and includes `volta/volta.h` alone: log lines reach
the callback, a second Engine is refused, a streamed chat, a tool call, cancelling from the callback and with `cancel`, `count_tokens` (also
with a video of the host's frames and with a clip of its samples, and a video part without one refused), a bad body, contexts (a long character prompt prefilled, a chat resuming from it, `export_context`, and a second process that refuses a tampered
copy, imports the context and gets the same reply from it); with speech, a line spoken to `library_speech.wav`, that speech added as a voice and a second line spoken in it, `speak_async`, and a
bad speech body. `--adopt` creates the test's own Vulkan instance and device first, as a game would, and runs the same checks on it;
`--adopt-bad` creates it without `shaderInt64` and expects `create` to refuse it by name.
