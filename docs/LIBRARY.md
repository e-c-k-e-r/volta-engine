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
- Speech: `speak( request_json, &speech, error )` takes a `/v1/audio/speech` body (input, voice, speed, and vall-e's language, seed, cfg,
  cfg_prompt, steps, duration, temperature, ...; [TTS.md](TTS.md)), blocks the calling thread and returns mono float PCM with its sample rate
  and the seed it was sampled with. One synthesis runs at a time. `speak_async( request_json, done )` queues it on the engine's speech
  thread and calls `done( ok, speech, error )` there. `add_voice( name, pcm, rate, error )` encodes a voice prompt from PCM once and makes it
  usable by name; `voices()` lists those and `tts.voices_dir`'s WAV files.
- Errors come back as strings: an invalid config, a model that fails to load (the last error line logged), a bad request body. An internal
  invariant failing (`ENGINE_ASSERT`, including an arena that runs out of memory) still ends the process, as it does in volta_server.

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
  `VK_KHR_push_descriptor`, `VK_EXT_external_memory_host`, `VK_EXT_memory_budget`, `VK_KHR_shader_integer_dot_product` and
  `VK_EXT_subgroup_size_control` when they are enabled. A device that lacks a required feature is refused with an error naming it. The
  host's device is the only one: `tp_size` 1 and `tts.device` 0.

The arena is sized from the memory the device reports free (`VK_EXT_memory_budget`), so what the host has already allocated is left alone.

## The test

`test_library <launch config> [--adopt | --adopt-bad] [--out <dir>]` links `libvolta.so` and includes `volta/volta.h` alone: log lines reach
the callback, a second Engine is refused, a streamed chat, a tool call, cancelling from the callback and with `cancel`, `count_tokens`, a bad
body; with speech, a line spoken to `library_speech.wav`, that speech added as a voice and a second line spoken in it, `speak_async`, and a
bad speech body. `--adopt` creates the test's own Vulkan instance and device first, as a game would, and runs the same checks on it;
`--adopt-bad` creates it without `shaderInt64` and expects `create` to refuse it by name.
