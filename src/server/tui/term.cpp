#include "engine/server/tui/term.h"

#include <cerrno>
#include <poll.h>
#include <sys/ioctl.h>
#include <unistd.h>

namespace engine::server::tui {

bool Terminal::open() {
	if ( out_fd_ >= 0 ) return true;
	if ( !isatty( STDIN_FILENO ) || !isatty( STDOUT_FILENO ) ) return false;
	if ( tcgetattr( STDIN_FILENO, &saved_ ) != 0 ) return false;
	termios raw = saved_;
	raw.c_lflag &= ~static_cast<tcflag_t>( ICANON | ECHO );   // ISIG stays on: Ctrl-C reaches the server's own handler
	raw.c_iflag &= ~static_cast<tcflag_t>( IXON | ICRNL );
	raw.c_cc[VMIN] = 0;
	raw.c_cc[VTIME] = 0;
	if ( tcsetattr( STDIN_FILENO, TCSANOW, &raw ) != 0 ) return false;
	out_fd_ = dup( STDOUT_FILENO );
	if ( out_fd_ < 0 ) {
		tcsetattr( STDIN_FILENO, TCSANOW, &saved_ );
		return false;
	}
	write( "\x1b[?1049h\x1b[?25l\x1b[2J" );   // alternate screen, hidden cursor
	return true;
}

void Terminal::close() {
	if ( out_fd_ < 0 ) return;
	write( "\x1b[0m\x1b[?25h\x1b[?1049l" );
	tcsetattr( STDIN_FILENO, TCSANOW, &saved_ );
	::close( out_fd_ );
	out_fd_ = -1;
}

void Terminal::size( int* rows, int* cols ) const {
	winsize ws = {};
	if ( ioctl( out_fd_ >= 0 ? out_fd_ : STDOUT_FILENO, TIOCGWINSZ, &ws ) == 0 && ws.ws_row > 0 && ws.ws_col > 0 ) {
		*rows = ws.ws_row;
		*cols = ws.ws_col;
	}
	else {
		*rows = 24;
		*cols = 80;
	}
}

namespace {

bool next_byte( int timeout_ms, unsigned char* c ) {
	pollfd p = { STDIN_FILENO, POLLIN, 0 };
	if ( poll( &p, 1, timeout_ms ) <= 0 ) return false;
	return read( STDIN_FILENO, c, 1 ) == 1;
}

} // namespace

int Terminal::read_key( int timeout_ms ) {
	unsigned char c = 0;
	if ( !next_byte( timeout_ms, &c ) ) return KEY_NONE;
	if ( c == '\t' ) return KEY_TAB;
	if ( c != 0x1b ) return c;
	unsigned char b = 0;
	if ( !next_byte( 25, &b ) ) return KEY_ESC;
	if ( b != '[' && b != 'O' ) return KEY_ESC;
	unsigned char d = 0;
	if ( !next_byte( 25, &d ) ) return KEY_ESC;
	switch ( d ) {
		case 'A': return KEY_UP;
		case 'B': return KEY_DOWN;
		case 'C': return KEY_RIGHT;
		case 'D': return KEY_LEFT;
		case 'H': return KEY_HOME;
		case 'F': return KEY_END;
		case 'Z': return KEY_BTAB;
		default: break;
	}
	if ( d >= '0' && d <= '9' ) {   // CSI n ~
		unsigned char t = 0;
		while ( next_byte( 25, &t ) && t != '~' && ( t < 0x40 || t > 0x7e ) ) {}
		switch ( d ) {
			case '1': case '7': return KEY_HOME;
			case '4': case '8': return KEY_END;
			case '5': return KEY_PGUP;
			case '6': return KEY_PGDN;
			default: break;
		}
	}
	return KEY_NONE;
}

void Terminal::write( const std::string& s ) {
	if ( out_fd_ < 0 ) return;
	size_t done = 0;
	while ( done < s.size() ) {
		const ssize_t n = ::write( out_fd_, s.data() + done, s.size() - done );
		if ( n < 0 ) {
			if ( errno == EINTR ) continue;
			if ( errno == EAGAIN ) {
				pollfd p = { out_fd_, POLLOUT, 0 };
				poll( &p, 1, 50 );
				continue;
			}
			return;
		}
		done += static_cast<size_t>( n );
	}
}

} // namespace engine::server::tui
