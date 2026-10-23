#pragma once

#include <string>
#include <termios.h>

namespace engine::server::tui {

// Key codes: printable characters are their own value; the rest start at 1000
enum : int {
	KEY_NONE = -1,
	KEY_UP = 1000, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_PGUP, KEY_PGDN, KEY_HOME, KEY_END, KEY_TAB, KEY_BTAB, KEY_ESC
};

// The controlling terminal in raw mode on the alternate screen. Output goes to a private copy of the standard output
// descriptor, so redirecting descriptors 1 and 2 afterwards (StdioCapture) does not touch the screen. Ctrl-C still raises SIGINT.
class Terminal {
public:
	Terminal() = default;
	Terminal( const Terminal& ) = delete;
	Terminal& operator=( const Terminal& ) = delete;
	~Terminal() { close(); }

	bool open();                       // false when standard input or output is not a terminal
	void close();                      // restores the screen and the terminal mode; safe to call twice
	bool is_open() const { return out_fd_ >= 0; }
	void size( int* rows, int* cols ) const;
	int read_key( int timeout_ms );    // KEY_NONE after the timeout
	void write( const std::string& s );

private:
	int out_fd_ = -1;
	termios saved_ = {};
};

} // namespace engine::server::tui
