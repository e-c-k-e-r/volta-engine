#pragma once

// volta-engine as a library (libvolta, CMake ENGINE_LIBRARY=ON): a text model and / or a vall-e speech model loaded from a launch config,
// driven from the host's threads. This header stands alone (no engine headers); strings, vectors and callbacks cross the boundary as std types,
// so the host builds with the same C++ standard library. One Engine per process (the engine's options and devices are process-wide).

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if defined( _WIN32 )
	#if defined( VOLTA_BUILD )
		#define VOLTA_API __declspec( dllexport )
	#else
		#define VOLTA_API __declspec( dllimport )
	#endif
#else
	#define VOLTA_API __attribute__( ( visibility( "default" ) ) )
#endif

namespace volta {

enum class LogLevel { Error = 0, Warn = 1, Info = 2, Debug = 3 };

// Where the engine's log lines go: one call per line, without its trailing newline, from whichever thread logs. Unset (or an empty function):
// stdout (info, debug) and stderr (errors, warnings). The threshold is the log_level option. Process-wide; set it before create
VOLTA_API void set_log_callback( std::function<void( LogLevel level, const std::string& line )> callback );

// The Vulkan device of a Vulkan build. By default the engine makes its own instance and device (the physical device by index, or by PCI device
// ID); with `device` set it runs on the host's instead, which it never destroys
struct VulkanDevice {
	// own device: physical device `index` (-1: the ENGINE_VK_DEVICE environment variable, else 0), or the first whose
	// VkPhysicalDeviceProperties::deviceID is `pci_device_id` (when not 0)
	int index = -1;
	uint32_t pci_device_id = 0;
	// adopted device (all four set): VkInstance, VkPhysicalDevice, VkDevice and the VkQueue the engine submits to (from a family with compute)
	void* instance = nullptr;
	void* physical_device = nullptr;
	void* device = nullptr;
	void* queue = nullptr;
	uint32_t queue_family = 0;
	// held around the engine's vkQueueSubmit / vkQueueWaitIdle on that queue; nullptr: no other thread submits to it
	std::mutex* queue_mutex = nullptr;
	// what the device was created with: the VkPhysicalDeviceFeatures2 chain given to vkCreateDevice (the engine needs bufferDeviceAddress,
	// shaderFloat16, timelineSemaphore and shaderInt64 enabled, and uses 16- / 8-bit storage, shaderIntegerDotProduct and subgroupSizeControl
	// when they are) and its enabled extension names (it uses VK_KHR_push_descriptor, VK_EXT_external_memory_host, VK_EXT_memory_budget,
	// VK_KHR_shader_integer_dot_product and VK_EXT_subgroup_size_control when present)
	const void* enabled_features = nullptr;
	std::vector<std::string> enabled_extensions;
};

// One event of a chat reply, in order; the last is always Finish
struct ChatEvent {
	enum class Type { Content, Reasoning, ToolCall, Finish };
	Type type = Type::Content;
	std::string text;             // Content, Reasoning: the next piece of text
	std::string tool_name;        // ToolCall
	std::string tool_arguments;   // ToolCall: a JSON object
	// Finish: "stop", "length", "tool_calls" or "cancelled", and the request's usage
	std::string finish_reason;
	int prompt_tokens = 0, cached_tokens = 0, completion_tokens = 0;
	double ttft_ms = 0.0, decode_tok_s = 0.0;
};

// One frame of a video: 8-bit RGB, rows of width x 3 bytes from the top, and its time in seconds from the video's start
struct VideoFrame {
	int width = 0;
	int height = 0;
	std::vector<uint8_t> rgb;
	double time = 0.0;
};

// A video as its frames, all of one size, in order (the engine keeps them as given, resized to the model's canvas and paired into the tower's
// temporal patches; past the max_video_tokens option evenly spaced frames drop)
struct Video {
	std::vector<VideoFrame> frames;
};

// Mono float PCM in [-1, 1]
struct Speech {
	std::vector<float> pcm;
	int sample_rate = 0;
	uint64_t seed = 0;   // the seed it was sampled with (the request's, or a random one)
};

class VOLTA_API Engine {
public:
	// config_json: a volta_server launch config (docs/CONFIG.md) as text: model_dir (the text model), tp_size, options, and tts {model, codec,
	// voices_dir, device} for speech; either part may be absent. Null with *error set on failure
	static std::unique_ptr<Engine> create( const std::string& config_json, std::string* error, const VulkanDevice& vulkan = VulkanDevice() );
	~Engine();
	Engine( const Engine& ) = delete;
	Engine& operator=( const Engine& ) = delete;

	bool has_text() const;
	bool has_speech() const;

	// Text: an OpenAI chat-completions body (messages, tools, tool_choice, max_tokens, temperature, top_p, top_k, seed, stop, reasoning_effort,
	// ...; "stream" is ignored). The events come on the engine's scheduler thread; returning false from on_event cancels the reply. Returns the
	// request's id, or 0 with *error set (nothing is called then)
	uint64_t chat( const std::string& request_json, std::function<bool( const ChatEvent& )> on_event, std::string* error );
	// With videos from the host: each {"type": "video"} content part without a source of its own takes the next of `videos`, in message
	// order (as many parts as videos). Parts with sources are the server's: image_url and video_url as base64 data: URLs, video with a
	// video file's data: URL or its frames' (and "timestamps" in seconds, or "fps"); video files need a build with FFmpeg
	uint64_t chat( const std::string& request_json, const std::vector<Video>& videos, std::function<bool( const ChatEvent& )> on_event, std::string* error );
	// Ends a reply early (its Finish event reports "cancelled"); unknown or finished ids are ignored
	void cancel( uint64_t id );
	// The prompt length a chat body renders to, or -1 with *error set
	int count_tokens( const std::string& request_json, std::string* error );
	int count_tokens( const std::string& request_json, const std::vector<Video>& videos, std::string* error );

	// Speech: an OpenAI /v1/audio/speech body (input, voice, speed) with vall-e's fields (language, seed, cfg, cfg_prompt, steps, duration,
	// temperature, ...; docs/TTS.md). Blocks the calling thread; one synthesis runs at a time
	bool speak( const std::string& request_json, Speech* out, std::string* error );
	// The same on the engine's speech thread (requests run in order); done is called there
	void speak_async( const std::string& request_json, std::function<void( bool ok, Speech& speech, const std::string& error )> done );
	// A voice from PCM (any rate; channels averaged beforehand), encoded once and used by name as "voice"; replaces one of the same name
	bool add_voice( const std::string& name, const std::vector<float>& pcm, int sample_rate, std::string* error );
	// The voices by name: tts.voices_dir's WAV files and the added ones
	std::vector<std::string> voices() const;

private:
	Engine();
	struct Impl;
	std::unique_ptr<Impl> p_;
};

} // namespace volta
