#pragma once

#include "engine/server/tui/term.h"
#include "engine/server/tui/view.h"

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace engine::server::tui {

// Fills `stats` with a fresh /api/stats document; false (with `error`) when there is none
using FetchFn = std::function<bool( nlohmann::json* stats, std::string* error )>;
// Replaces `lines` with the current log; an empty function means the source has no log
using LogFn = std::function<void( std::vector<std::string>* lines )>;

struct ViewerOptions {
	std::string source;          // shown in the title bar
	int interval_ms = 1000;      // time between readings
};

// The viewer loop on an open terminal: reads, draws and handles keys until the quit key or *stop
void run_viewer( Terminal& term, const ViewerOptions& opt, const FetchFn& fetch, const LogFn& log, const std::atomic<bool>& stop );

// Copies everything written to descriptors 1 and 2 (printf, stderr, library output) into a line buffer instead of the terminal,
// until stop() puts the descriptors back.
class StdioCapture {
public:
	~StdioCapture() { stop(); }
	bool start();
	void stop();
	void snapshot( std::vector<std::string>* lines ) const;

private:
	void reader();

	int read_fd_ = -1;
	int saved_out_ = -1, saved_err_ = -1;
	std::thread thread_;
	mutable std::mutex mu_;
	std::vector<std::string> lines_;
	std::string partial_;
};

// The dashboard inside the server process: the viewer runs on its own thread and the console output moves into the log tab.
// Pressing q (or stop()) restores the terminal and the console output; the server keeps running.
class ServerTui {
public:
	~ServerTui() { stop(); }
	// False when there is no terminal
	bool start( FetchFn fetch, const std::string& source );
	void stop();

private:
	Terminal term_;
	StdioCapture capture_;
	std::thread thread_;
	std::atomic<bool> stop_{ false };
	bool started_ = false;
};

} // namespace engine::server::tui
