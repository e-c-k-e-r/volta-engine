#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace engine::server::tui {

enum Tab { TAB_OVERVIEW, TAB_SLOTS, TAB_GPUS, TAB_EXPERTS, TAB_LOG, TAB_COUNT };

// What the viewer remembers between frames. scroll[] is the first visible body line, except on the log tab, where it counts
// the lines above the newest one (0 follows the tail).
struct UiState {
	int tab = TAB_OVERVIEW;
	int scroll[TAB_COUNT] = {};
	bool paused = false;          // p: stop taking readings, keep the last one on screen
	bool experts_total = false;   // m: routing counters since boot instead of the decayed recent ones
	int slot_sel = 0;             // Slots tab: the highlighted slot
	bool slot_detail = false;     // Enter: the detail block for it
};

// One frame's inputs. `stats` is the /api/stats document (null until the first reading arrives); `log` is null when the
// source has no log to show (a remote viewer).
struct ViewData {
	const nlohmann::json* stats = nullptr;
	const nlohmann::json* experts = nullptr;   // the /api/experts document; null until the Experts tab has fetched it
	std::string experts_error;
	bool connected = false;
	std::string error;                       // why the last reading failed
	std::string source;                      // where the readings come from, for the title bar
	const std::vector<std::string>* log = nullptr;
};

// The whole screen as one string of ANSI text (cursor home first, every line cleared to its end). Clamps ui.scroll.
std::string render_frame( const ViewData& data, UiState& ui, int rows, int cols );

// Applies a key to the view state; false when the key asks to quit. `page` is the number of body lines a page key moves.
bool apply_key( UiState& ui, int key, int page );

} // namespace engine::server::tui
