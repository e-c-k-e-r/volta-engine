// libvolta: the C++ API of inc/volta/volta.h over volta_server's pieces without its HTTP front end: the launch config, the Generator (text),
// the OpenAI chat-completions parser and the chat layer (start_chat with a Responder that forwards to the host's callback), and the TTS service
// (speech, its voices). Built by CMake with ENGINE_LIBRARY=ON; only namespace volta is exported (src/library/volta.map).
#include "volta/volta.h"
#include "engine/backend/runtime.h"
#include "engine/core/log.h"
#include "engine/runtime/generator.h"
#include "engine/runtime/options.h"
#include "engine/server/chat.h"
#include "engine/server/config.h"
#if ENGINE_TTS
	#include "engine/server/tts.h"
#endif
#if ENGINE_BACKEND_VULKAN
	#include "engine/backend/vulkan/device.h"
#endif
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <map>
#include <thread>

namespace volta {

namespace {

// The engine's log lines: to the host's callback (or the console), the last error kept for the error strings of create()
struct LogState {
	std::mutex mu;
	std::function<void( LogLevel, const std::string& )> callback;
	std::string last_error;
	bool installed = false;
};
LogState& log_state() {
	static LogState s;
	return s;
}

void install_log_sink() {
	LogState& s = log_state();
	std::lock_guard<std::mutex> lock( s.mu );
	if ( s.installed ) return;
	s.installed = true;
	engine::core::set_log_sink( []( engine::core::LogLevel level, const char* line ) {
		LogState& st = log_state();
		std::function<void( LogLevel, const std::string& )> cb;
		{
			std::lock_guard<std::mutex> lk( st.mu );
			if ( level == engine::core::LogLevel::Error ) st.last_error = line;
			cb = st.callback;
		}
		if ( cb ) {
			cb( static_cast<LogLevel>( level ), line );
			return;
		}
		static const char* const prefix[] = { "[-] ", "[!] ", "[*] ", "    " };
		std::fprintf( level <= engine::core::LogLevel::Warn ? stderr : stdout, "%s%s\n", prefix[static_cast<int>( level )], line );
	} );
}

std::string take_last_error( const char* fallback ) {
	LogState& s = log_state();
	std::lock_guard<std::mutex> lock( s.mu );
	std::string e = s.last_error.empty() ? std::string( fallback ) : s.last_error;
	s.last_error.clear();
	return e;
}

std::atomic<bool> g_engine_live{ false };   // one Engine per process

// A chat reply's events, forwarded to the host's callback; cancelled by the callback's false or Engine::cancel
class LibResponder : public engine::server::Responder {
public:
	LibResponder( std::function<bool( const ChatEvent& )> cb, std::shared_ptr<std::atomic<bool>> cancelled, std::function<void()> done )
		: cb_( std::move( cb ) ), cancelled_( std::move( cancelled ) ), done_( std::move( done ) ) {}

	bool on_event( const engine::server::OutputEvent& ev ) override {
		if ( cancelled_->load() ) return false;
		ChatEvent e;
		switch ( ev.type ) {
			case engine::server::OutputEvent::Type::Content: {
				e.type = ChatEvent::Type::Content;
				e.text = ev.text;
			} break;
			case engine::server::OutputEvent::Type::Reasoning: {
				e.type = ChatEvent::Type::Reasoning;
				e.text = ev.text;
			} break;
			case engine::server::OutputEvent::Type::ToolCall: {
				e.type = ChatEvent::Type::ToolCall;
				e.tool_name = ev.call.name;
				e.tool_arguments = ev.call.arguments;
			} break;
		}
		if ( cb_( e ) ) return true;
		cancelled_->store( true );
		return false;
	}

	bool on_wait() override { return !cancelled_->load(); }

	void on_finish( engine::runtime::FinishReason reason, const std::string&, bool had_tool_calls, const engine::server::Usage& usage ) override {
		ChatEvent e;
		e.type = ChatEvent::Type::Finish;
		if ( reason == engine::runtime::FinishReason::Cancelled || cancelled_->load() ) e.finish_reason = "cancelled";
		else if ( had_tool_calls ) e.finish_reason = "tool_calls";
		else if ( reason == engine::runtime::FinishReason::Length ) e.finish_reason = "length";
		else e.finish_reason = "stop";
		e.prompt_tokens = usage.prompt_tokens;
		e.cached_tokens = usage.cached_tokens;
		e.completion_tokens = usage.completion_tokens;
		e.ttft_ms = usage.ttft_ms;
		e.decode_tok_s = usage.decode_tok_s;
		cb_( e );
		done_();
	}

private:
	std::function<bool( const ChatEvent& )> cb_;
	std::shared_ptr<std::atomic<bool>> cancelled_;
	std::function<void()> done_;
};

// The host's videos into the request's video parts that have no source of their own, in order
bool attach_videos( engine::server::ChatRequest* req, const std::vector<Video>& videos, std::string* error ) {
	size_t next = 0;
	for ( auto& v : req->videos ) {
		if ( !v.file.empty() || !v.frames.empty() ) continue;
		if ( next == videos.size() ) {
			*error = "the request has more video parts without a source than videos were given";
			return false;
		}
		auto frames = std::make_shared<engine::loader::VideoFrames>();
		for ( const VideoFrame& f : videos[next].frames ) {
			engine::loader::DecodedImage img;
			img.width = f.width;
			img.height = f.height;
			img.rgb = f.rgb;
			frames->frames.push_back( std::move( img ) );
			frames->times.push_back( f.time );
		}
		v.decoded = std::move( frames );
		++next;
	}
	if ( next != videos.size() ) {
		*error = "the request has fewer video parts without a source than videos were given";
		return false;
	}
	return true;
}

} // namespace

void set_log_callback( std::function<void( LogLevel level, const std::string& line )> callback ) {
	install_log_sink();
	LogState& s = log_state();
	std::lock_guard<std::mutex> lock( s.mu );
	s.callback = std::move( callback );
}

struct Engine::Impl {
	engine::server::ServerConfig cfg;
	std::unique_ptr<engine::server::ChatProtocol> openai;
	std::mutex live_mu;
	std::map<uint64_t, std::shared_ptr<std::atomic<bool>>> live;   // replies in flight: their cancel flags
	std::atomic<uint64_t> next_id{ 0 };
#if ENGINE_TTS
	std::unique_ptr<engine::server::TtsService> tts;
#endif
	// speak_async's thread and queue (started on first use)
	struct SpeechJob {
		std::string body;
		std::function<void( bool, Speech&, const std::string& )> done;
	};
	std::mutex speech_mu;
	std::condition_variable speech_cv;
	std::deque<SpeechJob> speech_jobs;
	bool speech_stop = false;
	std::thread speech_thread;
	// last: destroyed first, so the scheduler thread (which calls the responders) is gone before the rest
	std::unique_ptr<engine::runtime::Generator> gen;

	bool speak( const std::string& request_json, Speech* out, std::string* error ) {
#if ENGINE_TTS
		if ( !tts ) {
			*error = "no speech model (the config has no tts section)";
			return false;
		}
		engine::server::SpeechBody body;
		if ( !engine::server::parse_speech_body( request_json, &body, error ) ) return false;
		out->pcm.clear();
		if ( !tts->speak( body, &out->pcm, error ) ) return false;
		out->sample_rate = tts->sample_rate();
		out->seed = body.options.seed;
		return true;
#else
		( void )request_json;
		( void )out;
		*error = "this libvolta was built without propan2ol: no speech";
		return false;
#endif
	}

	void speech_loop() {
		for ( ;; ) {
			SpeechJob job;
			{
				std::unique_lock<std::mutex> lk( speech_mu );
				speech_cv.wait( lk, [&] { return speech_stop || !speech_jobs.empty(); } );
				if ( speech_jobs.empty() ) return;   // stopping, nothing left
				job = std::move( speech_jobs.front() );
				speech_jobs.pop_front();
			}
			Speech speech;
			std::string error;
			bool stopping;
			{
				std::lock_guard<std::mutex> lk( speech_mu );
				stopping = speech_stop;
			}
			const bool ok = !stopping && speak( job.body, &speech, &error );
			if ( stopping ) error = "the engine is shutting down";
			if ( job.done ) job.done( ok, speech, error );
		}
	}

	~Impl() {
		{
			std::lock_guard<std::mutex> lock( speech_mu );
			speech_stop = true;
		}
		speech_cv.notify_all();
		if ( speech_thread.joinable() ) speech_thread.join();
		if ( gen ) gen->shutdown();
		gen.reset();
	}
};

Engine::Engine() : p_( std::make_unique<Impl>() ) {}

Engine::~Engine() {
	p_.reset();
	g_engine_live.store( false );
}

std::unique_ptr<Engine> Engine::create( const std::string& config_json, std::string* error, const VulkanDevice& vulkan ) {
	install_log_sink();
	take_last_error( "" );
	if ( g_engine_live.exchange( true ) ) {
		*error = "an Engine already exists in this process";
		return nullptr;
	}
	std::unique_ptr<Engine> e( new Engine() );
	Impl& P = *e->p_;
	// Inside a host application: the scheduler thread is left to the OS, and one GPU unless the config says otherwise (its values win)
	engine::runtime::set_option( "pin_threads", "false" );
	P.cfg.tp_size = 1;
	if ( !P.cfg.load_from_json_text( config_json ) || !P.cfg.resolve() ) {
		*error = take_last_error( "invalid launch config" );
		return nullptr;
	}
	if ( P.cfg.model_type == "minimax_h3" ) {
		*error = "MiniMax-H3 video generation is not part of the library";
		return nullptr;
	}
#if ENGINE_BACKEND_VULKAN
	if ( vulkan.device != nullptr ) {
		if ( P.cfg.tp_size != 1 || P.cfg.tts.device != 0 ) {
			*error = "an adopted Vulkan device is the only device: tp_size 1 and tts.device 0";
			return nullptr;
		}
		engine::backend::vk::HostDevice host;
		host.instance = vulkan.instance;
		host.physical_device = vulkan.physical_device;
		host.device = vulkan.device;
		host.queue = vulkan.queue;
		host.queue_family = vulkan.queue_family;
		host.queue_mutex = vulkan.queue_mutex;
		host.enabled_features = vulkan.enabled_features;
		host.enabled_extensions = vulkan.enabled_extensions;
		if ( !engine::backend::vk::adopt_device( host, error ) ) return nullptr;
	} else {
		engine::backend::vk::select_device( vulkan.index, vulkan.pci_device_id );
	}
#else
	if ( vulkan.device != nullptr ) {
		*error = "this libvolta is not a Vulkan build";
		return nullptr;
	}
#endif
	// speech first: the language model's arena is sized from the memory left (as the server does)
	if ( !P.cfg.tts.model.empty() ) {
#if ENGINE_TTS
		P.tts = std::make_unique<engine::server::TtsService>();
		if ( !P.tts->init( P.cfg ) ) {
			*error = take_last_error( "the speech model failed to load" );
			return nullptr;
		}
#else
		*error = "this libvolta was built without propan2ol: no speech";
		return nullptr;
#endif
	}
	if ( P.cfg.model_type != "vall_e" ) {
		P.gen = std::make_unique<engine::runtime::Generator>();
		if ( !P.gen->init( P.cfg ) ) {
			*error = take_last_error( "the text model failed to load" );
			return nullptr;
		}
		P.openai = engine::server::make_openai_protocol();
	}
	return e;
}

bool Engine::has_text() const { return p_->gen != nullptr; }

bool Engine::has_speech() const {
#if ENGINE_TTS
	return p_->tts != nullptr;
#else
	return false;
#endif
}

uint64_t Engine::chat( const std::string& request_json, std::function<bool( const ChatEvent& )> on_event, std::string* error ) {
	return chat( request_json, {}, std::move( on_event ), error );
}

uint64_t Engine::chat( const std::string& request_json, const std::vector<Video>& videos, std::function<bool( const ChatEvent& )> on_event, std::string* error ) {
	Impl& P = *p_;
	if ( !P.gen ) {
		*error = "no text model (the config has no model_dir)";
		return 0;
	}
	engine::server::ChatRequest req;
	engine::server::ApiError err;
	if ( !P.openai->parse( request_json, &req, &err ) ) {
		*error = err.message;
		return 0;
	}
	if ( !attach_videos( &req, videos, error ) ) return 0;
	const uint64_t id = ++P.next_id;
	auto flag = std::make_shared<std::atomic<bool>>( false );
	{
		std::lock_guard<std::mutex> lock( P.live_mu );
		P.live[id] = flag;
	}
	Impl* impl = p_.get();
	auto responder = std::make_shared<LibResponder>( std::move( on_event ), flag, [impl, id] {
		std::lock_guard<std::mutex> lock( impl->live_mu );
		impl->live.erase( id );
	} );
	if ( !engine::server::start_chat( *P.gen, P.cfg, std::move( req ), id, responder, nullptr, &err ) ) {
		std::lock_guard<std::mutex> lock( P.live_mu );
		P.live.erase( id );
		*error = err.message;
		return 0;
	}
	return id;
}

void Engine::cancel( uint64_t id ) {
	std::lock_guard<std::mutex> lock( p_->live_mu );
	auto it = p_->live.find( id );
	if ( it != p_->live.end() ) it->second->store( true );
}

int Engine::count_tokens( const std::string& request_json, std::string* error ) {
	return count_tokens( request_json, {}, error );
}

int Engine::count_tokens( const std::string& request_json, const std::vector<Video>& videos, std::string* error ) {
	Impl& P = *p_;
	if ( !P.gen ) {
		*error = "no text model (the config has no model_dir)";
		return -1;
	}
	engine::server::ChatRequest req;
	engine::server::ApiError err;
	int tokens = 0;
	if ( !P.openai->parse( request_json, &req, &err ) ) {
		*error = err.message;
		return -1;
	}
	if ( !attach_videos( &req, videos, error ) ) return -1;
	if ( !engine::server::count_chat_tokens( *P.gen, P.cfg, req, ++P.next_id, &tokens, &err ) ) {
		*error = err.message;
		return -1;
	}
	return tokens;
}

bool Engine::speak( const std::string& request_json, Speech* out, std::string* error ) { return p_->speak( request_json, out, error ); }

void Engine::speak_async( const std::string& request_json, std::function<void( bool ok, Speech& speech, const std::string& error )> done ) {
	Impl& P = *p_;
	std::lock_guard<std::mutex> lock( P.speech_mu );
	P.speech_jobs.push_back( { request_json, std::move( done ) } );
	if ( !P.speech_thread.joinable() ) P.speech_thread = std::thread( &Impl::speech_loop, &P );
	P.speech_cv.notify_one();
}

bool Engine::add_voice( const std::string& name, const std::vector<float>& pcm, int sample_rate, std::string* error ) {
#if ENGINE_TTS
	if ( !p_->tts ) {
		*error = "no speech model (the config has no tts section)";
		return false;
	}
	return p_->tts->add_voice( name, pcm, sample_rate, error );
#else
	( void )name;
	( void )pcm;
	( void )sample_rate;
	*error = "this libvolta was built without propan2ol: no speech";
	return false;
#endif
}

std::vector<std::string> Engine::voices() const {
#if ENGINE_TTS
	if ( p_->tts ) return p_->tts->voice_names();
#endif
	return {};
}

} // namespace volta
