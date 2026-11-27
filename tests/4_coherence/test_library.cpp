// libvolta from a host's side: links libvolta.so and includes volta/volta.h alone. A streamed chat, a tool call, count_tokens (also with a
// video from the host's frames), a cancelled reply; with speech, a line spoken to a WAV, that speech added as a voice and a second line in it, and speak_async. --adopt: the test makes
// its own Vulkan instance and device, as a game would, and hands them to the engine.
// --adopt-bad: the host's device lacks shaderInt64, which create must refuse with an error naming it.
//   test_library <launch config> [--adopt | --adopt-bad] [--out <dir>]
#include "volta/volta.h"
#include <vulkan/vulkan.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>

namespace {

int g_fail = 0;
void check( bool ok, const char* what ) {
	std::printf( "  %-52s %s\n", what, ok ? "PASS" : "FAIL" );
	if ( !ok ) ++g_fail;
}

void write_wav( const std::string& path, const volta::Speech& s ) {
	std::ofstream f( path, std::ios::binary );
	auto u32 = [&]( uint32_t v ) { f.write( reinterpret_cast<const char*>( &v ), 4 ); };
	auto u16 = [&]( uint16_t v ) { f.write( reinterpret_cast<const char*>( &v ), 2 ); };
	const uint32_t bytes = static_cast<uint32_t>( s.pcm.size() * 2 );
	f.write( "RIFF", 4 );
	u32( 36 + bytes );
	f.write( "WAVEfmt ", 8 );
	u32( 16 );
	u16( 1 );
	u16( 1 );
	u32( static_cast<uint32_t>( s.sample_rate ) );
	u32( static_cast<uint32_t>( s.sample_rate ) * 2 );
	u16( 2 );
	u16( 16 );
	f.write( "data", 4 );
	u32( bytes );
	for ( float x : s.pcm ) {
		const float c = x < -1.0f ? -1.0f : ( x > 1.0f ? 1.0f : x );
		u16( static_cast<uint16_t>( static_cast<int16_t>( c * 32767.0f ) ) );
	}
}

// A host application's device: every feature of the engine's list the GPU supports enabled, and the extensions it uses
struct HostVulkan {
	VkInstance instance = VK_NULL_HANDLE;
	VkPhysicalDevice phys = VK_NULL_HANDLE;
	VkDevice device = VK_NULL_HANDLE;
	VkQueue queue = VK_NULL_HANDLE;
	uint32_t family = 0;
	std::mutex queue_mutex;
	VkPhysicalDeviceVulkan11Features f11{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
	VkPhysicalDeviceVulkan12Features f12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
	VkPhysicalDeviceShaderIntegerDotProductFeatures fdot{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES };
	VkPhysicalDeviceSubgroupSizeControlFeatures fsg{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES };
	VkPhysicalDeviceFeatures2 f2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
	std::vector<std::string> exts;

	bool create( int index, bool without_int64 = false ) {
		VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
		app.pApplicationName = "test_library host";
		app.apiVersion = VK_API_VERSION_1_2;
		VkInstanceCreateInfo ici{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
		ici.pApplicationInfo = &app;
		if ( vkCreateInstance( &ici, nullptr, &instance ) != VK_SUCCESS ) return false;
		uint32_t n = 0;
		vkEnumeratePhysicalDevices( instance, &n, nullptr );
		std::vector<VkPhysicalDevice> all( n );
		vkEnumeratePhysicalDevices( instance, &n, all.data() );
		if ( index < 0 || static_cast<uint32_t>( index ) >= n ) return false;
		phys = all[index];
		uint32_t en = 0;
		vkEnumerateDeviceExtensionProperties( phys, nullptr, &en, nullptr );
		std::vector<VkExtensionProperties> avail( en );
		vkEnumerateDeviceExtensionProperties( phys, nullptr, &en, avail.data() );
		auto has = [&]( const char* e ) {
			for ( const auto& a : avail ) {
				if ( std::strcmp( a.extensionName, e ) == 0 ) return true;
			}
			return false;
		};
		for ( const char* e : { VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME,
				  VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME } ) {
			if ( has( e ) ) exts.push_back( e );
		}
		void* chain = nullptr;
		if ( has( VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME ) ) { fsg.pNext = chain; chain = &fsg; }
		if ( has( VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME ) ) { fdot.pNext = chain; chain = &fdot; }
		f11.pNext = chain;
		f12.pNext = &f11;
		f2.pNext = &f12;
		vkGetPhysicalDeviceFeatures2( phys, &f2 );   // supported: enable them all (a game would enable what it and the engine need)
		f2.features.robustBufferAccess = VK_FALSE;
		if ( without_int64 ) f2.features.shaderInt64 = VK_FALSE;
		fsg.computeFullSubgroups = VK_FALSE;
		uint32_t qn = 0;
		vkGetPhysicalDeviceQueueFamilyProperties( phys, &qn, nullptr );
		std::vector<VkQueueFamilyProperties> qf( qn );
		vkGetPhysicalDeviceQueueFamilyProperties( phys, &qn, qf.data() );
		bool found = false;
		for ( uint32_t i = 0; i < qn && !found; ++i ) {
			if ( qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT ) { family = i; found = true; }
		}
		if ( !found ) return false;
		const float prio = 1.0f;
		VkDeviceQueueCreateInfo qci{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
		qci.queueFamilyIndex = family;
		qci.queueCount = 1;
		qci.pQueuePriorities = &prio;
		std::vector<const char*> names;
		for ( const auto& e : exts ) names.push_back( e.c_str() );
		VkDeviceCreateInfo dci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2 };
		dci.queueCreateInfoCount = 1;
		dci.pQueueCreateInfos = &qci;
		dci.enabledExtensionCount = static_cast<uint32_t>( names.size() );
		dci.ppEnabledExtensionNames = names.data();
		if ( vkCreateDevice( phys, &dci, nullptr, &device ) != VK_SUCCESS ) return false;
		vkGetDeviceQueue( device, family, 0, &queue );
		return true;
	}
	~HostVulkan() {
		if ( device ) vkDestroyDevice( device, nullptr );
		if ( instance ) vkDestroyInstance( instance, nullptr );
	}
};

struct Reply {
	std::mutex mu;
	std::condition_variable cv;
	bool done = false;
	std::string content, reasoning, tool_name, tool_args, finish;
	int deltas = 0, completion_tokens = 0, prompt_tokens = 0;
	double ttft_ms = 0.0, tok_s = 0.0;
	std::function<bool( const volta::ChatEvent& )> handler( int stop_after = -1 ) {
		return [this, stop_after]( const volta::ChatEvent& e ) {
			std::lock_guard<std::mutex> lock( mu );
			switch ( e.type ) {
				case volta::ChatEvent::Type::Content: {
					content += e.text;
					++deltas;
				} break;
				case volta::ChatEvent::Type::Reasoning: {
					reasoning += e.text;
				} break;
				case volta::ChatEvent::Type::ToolCall: {
					tool_name = e.tool_name;
					tool_args = e.tool_arguments;
				} break;
				case volta::ChatEvent::Type::Finish: {
					finish = e.finish_reason;
					completion_tokens = e.completion_tokens;
					prompt_tokens = e.prompt_tokens;
					ttft_ms = e.ttft_ms;
					tok_s = e.decode_tok_s;
					done = true;
					cv.notify_all();
				} break;
			}
			return stop_after < 0 || deltas < stop_after;
		};
	}
	void wait() {
		std::unique_lock<std::mutex> lock( mu );
		cv.wait( lock, [&] { return done; } );
	}
};

} // namespace

int main( int argc, char** argv ) {
	if ( argc < 2 ) {
		std::printf( "usage: test_library <launch config> [--adopt | --adopt-bad] [--out <dir>]\n" );
		return 2;
	}
	bool adopt = false, adopt_bad = false;
	std::string out_dir = ".";
	for ( int i = 2; i < argc; ++i ) {
		if ( !std::strcmp( argv[i], "--adopt" ) ) adopt = true;
		if ( !std::strcmp( argv[i], "--adopt-bad" ) ) adopt = adopt_bad = true;
		if ( !std::strcmp( argv[i], "--out" ) && i + 1 < argc ) out_dir = argv[++i];
	}
	std::ifstream cf( argv[1] );
	std::stringstream ss;
	ss << cf.rdbuf();

	std::atomic<int> log_lines{ 0 };
	volta::set_log_callback( [&]( volta::LogLevel level, const std::string& line ) {
		++log_lines;
		if ( level <= volta::LogLevel::Warn ) std::printf( "  [engine %s] %s\n", level == volta::LogLevel::Error ? "error" : "warn", line.c_str() );
	} );

	HostVulkan host;
	volta::VulkanDevice vk;
	if ( adopt ) {
		if ( !host.create( 0, adopt_bad ) ) {
			std::printf( "FAIL: the host's Vulkan device\n" );
			return 1;
		}
		vk.instance = host.instance;
		vk.physical_device = host.phys;
		vk.device = host.device;
		vk.queue = host.queue;
		vk.queue_family = host.family;
		vk.queue_mutex = &host.queue_mutex;
		vk.enabled_features = &host.f2;
		vk.enabled_extensions = host.exts;
	}

	std::printf( "libvolta (%s device)\n", adopt ? "adopted" : "own" );
	std::string error;
	const auto t0 = std::chrono::steady_clock::now();
	auto engine = volta::Engine::create( ss.str(), &error, vk );
	if ( adopt_bad ) {
		std::printf( "  create: %s\n", engine ? "succeeded" : error.c_str() );
		check( !engine && error.find( "shaderInt64" ) != std::string::npos, "a device without shaderInt64 is refused, by name" );
		std::printf( "%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail );
		return g_fail ? 1 : 0;
	}
	if ( !engine ) {
		std::printf( "FAIL: create: %s\n", error.c_str() );
		return 1;
	}
	std::printf( "  loaded in %.1f s (%d log lines through the callback), text %d, speech %d\n",
		std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count(), log_lines.load(), engine->has_text(), engine->has_speech() );
	check( log_lines.load() > 0, "log lines reach the host's callback" );
	std::string err2;
	check( !volta::Engine::create( ss.str(), &err2 ) && !err2.empty(), "a second Engine is refused" );

	if ( engine->has_text() ) {
		{
			Reply r;
			const uint64_t id = engine->chat( R"({"messages":[{"role":"user","content":"A player asks the blacksmith NPC: where can I find iron ore? Answer in one sentence, in character."}],"max_tokens":64,"temperature":0})",
				r.handler(), &error );
			check( id != 0, "chat starts" );
			if ( id ) {
				r.wait();
				std::printf( "    \"%s\"\n    %d prompt + %d tokens, ttft %.0f ms, %.1f tok/s, finish %s\n", r.content.c_str(), r.prompt_tokens, r.completion_tokens, r.ttft_ms, r.tok_s, r.finish.c_str() );
				check( !r.content.empty() && r.deltas > 1 && ( r.finish == "stop" || r.finish == "length" ), "streamed content and a finish" );
			}
		}
		{
			Reply r;
			const char* body = R"({"messages":[{"role":"user","content":"Give me three iron ingots, blacksmith."}],"max_tokens":64,"temperature":0,
				"tools":[{"type":"function","function":{"name":"give_item","description":"Give the player an item","parameters":{"type":"object","properties":{"item":{"type":"string"},"count":{"type":"integer"}},"required":["item","count"]}}}]})";
			if ( engine->chat( body, r.handler(), &error ) ) {
				r.wait();
				std::printf( "    tool %s %s, finish %s\n", r.tool_name.c_str(), r.tool_args.c_str(), r.finish.c_str() );
				check( r.tool_name == "give_item" && r.finish == "tool_calls", "a tool call" );
			} else {
				check( false, ( "tool chat: " + error ).c_str() );
			}
		}
		{
			Reply r;
			if ( engine->chat( R"({"messages":[{"role":"user","content":"Count from one to one hundred in words."}],"max_tokens":400,"temperature":0})", r.handler( 4 ), &error ) ) {
				r.wait();
				check( r.finish == "cancelled" && r.completion_tokens < 100, "the callback's false cancels" );
			}
			Reply r2;
			const uint64_t id = engine->chat( R"({"messages":[{"role":"user","content":"Count from one to one hundred in words."}],"max_tokens":400,"temperature":0})", r2.handler(), &error );
			std::this_thread::sleep_for( std::chrono::milliseconds( 300 ) );
			engine->cancel( id );
			r2.wait();
			check( r2.finish == "cancelled" && r2.completion_tokens < 300, "Engine::cancel" );
		}
		const int n = engine->count_tokens( R"({"messages":[{"role":"user","content":"Hello there"}]})", &error );
		check( n > 2 && n < 64, "count_tokens" );
		{
			// a video from the host's frames: one sourceless part takes it (a model without video renders its note instead); a part with no
			// video given is an error
			volta::Video clip;
			for ( int i = 0; i < 4; ++i ) {
				volta::VideoFrame f;
				f.width = 64;
				f.height = 48;
				f.rgb.assign( static_cast<size_t>( 64 ) * 48 * 3, static_cast<uint8_t>( 60 * i ) );
				f.time = 0.5 * i;
				clip.frames.push_back( std::move( f ) );
			}
			const char* body = R"({"messages":[{"role":"user","content":[{"type":"video"},{"type":"text","text":"What happens?"}]}]})";
			const int nv = engine->count_tokens( body, { clip }, &error );
			check( nv > n, "count_tokens with the host's video" );
			std::string verr;
			check( engine->count_tokens( body, {}, &verr ) < 0 && !verr.empty(), "a video part without a video is an error" );
		}
		check( engine->chat( "{ not json", Reply().handler(), &error ) == 0 && !error.empty(), "a bad body is an error" );
	}

	if ( engine->has_speech() ) {
		volta::Speech a;
		const auto s0 = std::chrono::steady_clock::now();
		const bool ok = engine->speak( R"({"input":"Hmph. You'll find iron ore in the mountains beyond the village.","seed":7})", &a, &error );
		const double dt = std::chrono::duration<double>( std::chrono::steady_clock::now() - s0 ).count();
		check( ok && !a.pcm.empty() && a.sample_rate > 0, "speak" );
		if ( ok ) {
			std::printf( "    %.2f s of speech at %d Hz in %.2f s, seed %llu\n", a.pcm.size() / double( a.sample_rate ), a.sample_rate, dt, static_cast<unsigned long long>( a.seed ) );
			write_wav( out_dir + "/library_speech.wav", a );
			check( engine->add_voice( "blacksmith", a.pcm, a.sample_rate, &error ), "add_voice from PCM" );
			bool listed = false;
			for ( const auto& v : engine->voices() ) listed = listed || v == "blacksmith";
			check( listed, "the voice is listed" );
			volta::Speech b;
			check( engine->speak( R"({"input":"Bring me coal as well.","voice":"blacksmith","seed":7})", &b, &error ) && !b.pcm.empty(), "speak in the added voice" );
			if ( !b.pcm.empty() ) write_wav( out_dir + "/library_speech_voice.wav", b );
		} else {
			std::printf( "    %s\n", error.c_str() );
		}
		std::mutex mu;
		std::condition_variable cv;
		int done = 0, good = 0;
		for ( int i = 0; i < 2; ++i ) {
			engine->speak_async( R"({"input":"Next!","seed":3})", [&]( bool ok, volta::Speech& s, const std::string& ) {
				std::lock_guard<std::mutex> lock( mu );
				good += ok && !s.pcm.empty();
				++done;
				cv.notify_all();
			} );
		}
		{
			std::unique_lock<std::mutex> lock( mu );
			cv.wait( lock, [&] { return done == 2; } );
		}
		check( good == 2, "speak_async" );
		volta::Speech c;
		check( !engine->speak( R"({"input":""})", &c, &error ) && !error.empty(), "a bad speech body is an error" );
	}

	engine.reset();
	std::printf( "%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail );
	return g_fail ? 1 : 0;
}
