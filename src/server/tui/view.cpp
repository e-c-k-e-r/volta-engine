#include "engine/server/tui/view.h"
#include "engine/server/tui/term.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace engine::server::tui {

namespace {

using json = nlohmann::json;

constexpr const char* RESET = "\x1b[0m";
constexpr const char* BOLD = "\x1b[1m";
constexpr const char* DIM = "\x1b[2m";
constexpr const char* REVERSE = "\x1b[7m";
constexpr const char* RED = "\x1b[31m";
constexpr const char* GREEN = "\x1b[32m";
constexpr const char* YELLOW = "\x1b[33m";
constexpr const char* CYAN = "\x1b[36m";

const char* const TAB_NAMES[TAB_COUNT] = { "Overview", "Slots", "GPUs", "Log" };

// ---- reading the stats document; every accessor tolerates a missing key, so an older or smaller server still renders

const json* get( const json& j, const char* a ) {
	if ( !j.is_object() ) return nullptr;
	const auto it = j.find( a );
	return it == j.end() ? nullptr : &*it;
}

const json* get( const json& j, const char* a, const char* b ) {
	const json* x = get( j, a );
	return x ? get( *x, b ) : nullptr;
}

double num( const json* x, double def = 0.0 ) { return x && x->is_number() ? x->get<double>() : def; }
double num( const json& j, const char* a, double def = 0.0 ) { return num( get( j, a ), def ); }
double num( const json& j, const char* a, const char* b, double def = 0.0 ) { return num( get( j, a, b ), def ); }
std::string str( const json* x ) { return x && x->is_string() ? x->get<std::string>() : std::string(); }
std::string str( const json& j, const char* a ) { return str( get( j, a ) ); }
std::string str( const json& j, const char* a, const char* b ) { return str( get( j, a, b ) ); }
bool flag( const json& j, const char* a, const char* b ) { const json* x = get( j, a, b ); return x && x->is_boolean() && x->get<bool>(); }

// ---- formatting

std::string format( const char* fmt, double v ) {
	char buf[64];
	std::snprintf( buf, sizeof( buf ), fmt, v );
	return buf;
}

std::string count( double v ) {   // 1234567 -> 1,234,567
	std::string digits = format( "%.0f", std::fabs( v ) );
	std::string out;
	for ( size_t i = 0; i < digits.size(); ++i ) {
		if ( i && ( digits.size() - i ) % 3 == 0 ) out += ',';
		out += digits[i];
	}
	return v < 0 ? "-" + out : out;
}

std::string bytes( double b ) {
	const char* unit[] = { "B", "KiB", "MiB", "GiB", "TiB" };
	int u = 0;
	while ( b >= 1024.0 && u < 4 ) { b /= 1024.0; ++u; }
	return format( u >= 3 ? "%.1f" : "%.0f", b ) + " " + unit[u];
}

std::string ms( double v ) {
	if ( v >= 10000.0 ) return format( "%.1f", v / 1000.0 ) + " s";
	if ( v >= 100.0 ) return format( "%.0f", v ) + " ms";
	return format( "%.1f", v ) + " ms";
}

std::string duration( double s ) {
	const int64_t t = static_cast<int64_t>( s );
	char buf[48];
	if ( t >= 86400 ) std::snprintf( buf, sizeof( buf ), "%lldd %02lldh", static_cast<long long>( t / 86400 ), static_cast<long long>( t % 86400 / 3600 ) );
	else if ( t >= 3600 ) std::snprintf( buf, sizeof( buf ), "%lldh %02lldm", static_cast<long long>( t / 3600 ), static_cast<long long>( t % 3600 / 60 ) );
	else std::snprintf( buf, sizeof( buf ), "%lldm %02llds", static_cast<long long>( t / 60 ), static_cast<long long>( t % 60 ) );
	return buf;
}

std::string pad( const std::string& s, size_t width ) { return s.size() >= width ? s : s + std::string( width - s.size(), ' ' ); }
std::string lpad( const std::string& s, size_t width ) { return s.size() >= width ? s : std::string( width - s.size(), ' ' ) + s; }

const char* level_color( double frac ) { return frac >= 0.9 ? RED : frac >= 0.7 ? YELLOW : GREEN; }

std::string bar( double frac, int width, bool colored = true ) {
	frac = std::min( 1.0, std::max( 0.0, frac ) );
	const int full = static_cast<int>( std::lround( frac * width ) );
	std::string s = "[";
	if ( colored ) s += level_color( frac );
	s += std::string( static_cast<size_t>( full ), '#' );
	if ( colored ) s += DIM;
	s += std::string( static_cast<size_t>( width - full ), '-' );
	s += RESET;
	return s + "]";
}

std::string heading( const std::string& s ) { return std::string( BOLD ) + CYAN + s + RESET; }

// Width of the text as shown: escape sequences take no room, a UTF-8 sequence takes one column
int visible_width( const std::string& s ) {
	int w = 0;
	for ( size_t i = 0; i < s.size(); ++i ) {
		const unsigned char c = static_cast<unsigned char>( s[i] );
		if ( c == 0x1b && i + 1 < s.size() && s[i + 1] == '[' ) {
			i += 2;
			while ( i < s.size() && !( s[i] >= 0x40 && s[i] <= 0x7e ) ) ++i;
		}
		else if ( ( c & 0xc0 ) != 0x80 ) ++w;
	}
	return w;
}

// The line cut to `cols` visible columns (never inside an escape sequence or a UTF-8 sequence)
std::string fit( const std::string& s, int cols ) {
	std::string out;
	int w = 0;
	for ( size_t i = 0; i < s.size(); ) {
		const unsigned char c = static_cast<unsigned char>( s[i] );
		if ( c == 0x1b ) {
			size_t j = i + 1;
			if ( j < s.size() && s[j] == '[' ) {
				++j;
				while ( j < s.size() && !( s[j] >= 0x40 && s[j] <= 0x7e ) ) ++j;
				if ( j < s.size() ) ++j;
			}
			out.append( s, i, j - i );
			i = j;
			continue;
		}
		const size_t len = c < 0xc0 ? 1 : c >= 0xf0 ? 4 : c >= 0xe0 ? 3 : 2;
		if ( w >= cols ) break;
		out.append( s, i, std::min( len, s.size() - i ) );
		i += len;
		++w;
	}
	return out + RESET;
}

// ---- tab bodies

using Lines = std::vector<std::string>;

void overview( const json& s, int cols, Lines* out ) {
	const int barw = std::max( 10, std::min( 30, cols / 4 ) );
	out->push_back( heading( "Server" ) );
	std::string l = "  " + str( s, "server", "model" ) + "  (" + str( s, "server", "model_type" ) + ")";
	l += "   tp " + format( "%.0f", num( s, "server", "tp" ) );
	l += "   context " + count( num( s, "server", "max_context" ) ) + " per slot";
	l += "   kv " + str( s, "server", "kv_format" );
	out->push_back( l );
	int active = 0;
	if ( const json* sl = get( s, "slots" ); sl && sl->is_array() ) {
		for ( const auto& x : *sl ) {
			const json* a = get( x, "active" );
			if ( a && a->is_boolean() && a->get<bool>() ) ++active;
		}
	}
	const double slots = num( s, "server", "slots" );
	out->push_back( "  slots " + bar( slots > 0 ? active / slots : 0.0, barw ) + " " + std::to_string( active ) + " / " + format( "%.0f", slots ) +
		" busy   queued " + format( "%.0f", num( s, "server", "queued" ) ) );
	const bool spec = flag( s, "server", "speculation" );
	if ( spec ) {
		out->push_back( "  speculation " + str( s, "server", "speculator" ) + "   accept " + format( "%.1f", num( s, "speculation", "accept_rate" ) * 100.0 ) + "%   " +
			format( "%.2f", num( s, "speculation", "tokens_per_round" ) ) + " tokens per round   " + count( num( s, "speculation", "rounds" ) ) + " rounds" );
	}
	else {
		out->push_back( std::string( DIM ) + "  speculation off" + RESET );
	}
	out->push_back( "" );

	out->push_back( heading( "Throughput" ) );
	out->push_back( "  decode " + format( "%.1f", num( s, "averages", "decode_tok_s" ) ) + " tok/s    prefill " + format( "%.0f", num( s, "averages", "prefill_tok_s" ) ) +
		" tok/s    ttft " + ms( num( s, "averages", "ttft_ms" ) ) + "   (averages over finished requests)" );
	const json* gap = get( s, "latency", "gap_ms" );
	if ( gap && gap->is_array() && gap->size() >= 5 ) {
		out->push_back( "  token gap  p50 " + ms( num( &( *gap )[0] ) ) + "   p90 " + ms( num( &( *gap )[1] ) ) + "   p99 " + ms( num( &( *gap )[2] ) ) +
			"   p99.9 " + ms( num( &( *gap )[3] ) ) + "   max " + ms( num( &( *gap )[4] ) ) );
	}
	const json* tt = get( s, "latency", "ttft_ms" );
	if ( tt && tt->is_array() && tt->size() >= 3 ) {
		out->push_back( "  ttft       p50 " + ms( num( &( *tt )[0] ) ) + "   p90 " + ms( num( &( *tt )[1] ) ) + "   p99 " + ms( num( &( *tt )[2] ) ) );
	}
	out->push_back( "" );

	out->push_back( heading( "Requests" ) );
	const double prompt = num( s, "totals", "prompt_tokens" ), cached = num( s, "totals", "cached_tokens" );
	out->push_back( "  " + count( num( s, "totals", "requests" ) ) + " received   " + count( num( s, "totals", "completed" ) ) + " completed   " +
		count( num( s, "totals", "tokens_out" ) ) + " tokens out   " + count( prompt ) + " prompt tokens (" + format( "%.1f", prompt > 0 ? cached * 100.0 / prompt : 0.0 ) + "% from cache)" );
	if ( const json* f = get( s, "totals", "finished" ) ) {
		out->push_back( "  finished  stop " + count( num( *f, "stop" ) ) + "   length " + count( num( *f, "length" ) ) + "   stop string " + count( num( *f, "stop_string" ) ) +
			"   cancelled " + count( num( *f, "cancelled" ) ) );
	}
	if ( const json* pc = get( s, "prefix_cache" ) ) {
		const double look = num( *pc, "lookups" );
		out->push_back( "  prefix cache  " + count( num( *pc, "entries" ) ) + " entries   " + bytes( num( *pc, "bytes" ) ) + " of " + bytes( num( *pc, "budget_bytes" ) ) +
			"   hit rate " + format( "%.1f", look > 0 ? num( *pc, "hits" ) * 100.0 / look : 0.0 ) + "%   " + count( num( *pc, "evictions" ) ) + " evictions" );
	}
	out->push_back( "" );

	out->push_back( heading( "GPUs" ) + std::string( DIM ) + "  (details on tab 3)" + RESET );
	if ( const json* g = get( s, "gpus" ); g && g->is_array() && !g->empty() ) {
		for ( const auto& x : *g ) {
			const double limit = num( x, "power_limit_w" ), used = num( x, "mem_used" ), total = num( x, "mem_total" );
			out->push_back( "  " + std::to_string( static_cast<int>( num( x, "index" ) ) ) + " util " + bar( num( x, "util_gpu" ) / 100.0, barw / 2 ) + lpad( format( "%.0f", num( x, "util_gpu" ) ), 4 ) + "%   " +
				lpad( format( "%.0f", num( x, "power_w" ) ), 4 ) + " / " + format( "%.0f", limit ) + " W   " + format( "%.0f", num( x, "temp_c" ) ) + " C   vram " +
				bytes( used ) + " / " + bytes( total ) );
		}
	}
	else {
		out->push_back( std::string( DIM ) + "  no NVML readings" + RESET );
	}
	if ( const json* h = get( s, "host" ) ) {
		const double mt = num( *h, "mem_total" );
		out->push_back( "  host cpu " + bar( num( *h, "cpu_pct" ) / 100.0, barw / 2 ) + lpad( format( "%.0f", num( *h, "cpu_pct" ) ), 4 ) + "%   ram " + bytes( num( *h, "mem_used" ) ) + " / " + bytes( mt ) +
			"   server rss " + bytes( num( *h, "rss" ) ) + "   load " + format( "%.2f", num( *h, "load1" ) ) );
	}
}

void slots( const json& s, int cols, Lines* out ) {
	const int barw = std::max( 8, std::min( 24, cols / 6 ) );
	const double ctx = num( s, "server", "max_context" );
	out->push_back( heading( "Slots" ) );
	out->push_back( std::string( DIM ) + "  " + lpad( "slot", 4 ) + " " + pad( "phase", 9 ) + " " + pad( "req", 8 ) + " " + pad( "context", static_cast<size_t>( barw ) + 2 ) + " " + lpad( "pos", 7 ) + "  " + lpad( "prompt", 6 ) + "  " +
		lpad( "cached", 6 ) + "  " + lpad( "generated", 9 ) + "/" + pad( "max", 6 ) + lpad( "age", 8 ) + "  " + lpad( "tok/s", 6 ) + RESET );
	if ( const json* sl = get( s, "slots" ); sl && sl->is_array() ) {
		for ( const auto& x : *sl ) {
			const std::string phase = str( x, "phase" );
			const bool idle = phase == "idle";
			const double pos = num( x, "pos" );
			std::string l = "  " + lpad( format( "%.0f", num( x, "slot" ) ), 4 ) + " " + pad( phase, 9 ) + " ";
			if ( idle ) {
				l = std::string( DIM ) + l + "-" + RESET;
			}
			else {
				const char* pc = phase == "prefill" ? YELLOW : phase == "thinking" ? CYAN : GREEN;
				l = "  " + lpad( format( "%.0f", num( x, "slot" ) ), 4 ) + " " + pc + pad( phase, 9 ) + RESET + " " + pad( "#" + format( "%.0f", num( x, "req_id" ) ), 8 ) + " " +
					bar( ctx > 0 ? pos / ctx : 0.0, barw ) + " " + lpad( count( pos ), 7 ) + "  " + lpad( count( num( x, "prompt_tokens" ) ), 6 ) + "  " +
					lpad( count( num( x, "cached_tokens" ) ), 6 ) + "  " + lpad( count( num( x, "generated" ) ), 9 ) + "/" + pad( count( num( x, "max_tokens" ) ), 6 ) +
					lpad( duration( num( x, "age_s" ) ), 8 ) + "  " + lpad( format( "%.1f", num( x, "decode_tok_s" ) ), 6 );
			}
			out->push_back( l );
		}
	}
	out->push_back( "" );
	out->push_back( heading( "Recent requests" ) + std::string( DIM ) + "  (newest first)" + RESET );
	out->push_back( std::string( DIM ) + "     req    ago  prompt  cached  output   queue    ttft  prefill   total  tok/s  reason" + RESET );
	const json* rc = get( s, "recent" );
	if ( !rc || !rc->is_array() || rc->empty() ) {
		out->push_back( std::string( DIM ) + "  no finished requests yet" + RESET );
		return;
	}
	for ( const auto& r : *rc ) {
		const std::string reason = str( r, "reason" );
		out->push_back( "  " + lpad( "#" + format( "%.0f", num( r, "req_id" ) ), 6 ) + " " + lpad( duration( num( r, "age_s" ) ), 6 ) + " " + lpad( count( num( r, "prompt_tokens" ) ), 7 ) + " " +
			lpad( count( num( r, "cached_tokens" ) ), 7 ) + " " + lpad( count( num( r, "generated" ) ), 7 ) + " " + lpad( ms( num( r, "queue_ms" ) ), 7 ) + " " +
			lpad( ms( num( r, "ttft_ms" ) ), 7 ) + " " + lpad( ms( num( r, "prefill_ms" ) ), 8 ) + " " + lpad( ms( num( r, "total_ms" ) ), 8 ) + " " +
			lpad( format( "%.1f", num( r, "decode_tok_s" ) ), 6 ) + "  " + ( reason == "stop" ? reason : std::string( YELLOW ) + reason + RESET ) );
	}
}

void gpus( const json& s, int cols, Lines* out ) {
	const int barw = std::max( 10, std::min( 30, cols / 4 ) );
	const json* g = get( s, "gpus" );
	if ( !g || !g->is_array() || g->empty() ) {
		out->push_back( std::string( DIM ) + "no NVML readings (libnvidia-ml.so.1 not found, or no devices)" + RESET );
	}
	else {
		for ( const auto& x : *g ) {
			const double power = num( x, "power_w" ), limit = num( x, "power_limit_w" ), used = num( x, "mem_used" ), total = num( x, "mem_total" );
			out->push_back( heading( "GPU " + std::to_string( static_cast<int>( num( x, "index" ) ) ) ) + "  " + str( x, "name" ) );
			out->push_back( "  util    " + bar( num( x, "util_gpu" ) / 100.0, barw ) + lpad( format( "%.0f", num( x, "util_gpu" ) ), 5 ) + "%     memory bandwidth " + format( "%.0f", num( x, "util_mem" ) ) + "%" );
			out->push_back( "  power   " + bar( limit > 0 ? power / limit : 0.0, barw ) + lpad( format( "%.0f", power ), 5 ) + " / " + format( "%.0f", limit ) + " W     energy " + format( "%.1f", num( x, "energy_kj" ) ) + " kJ" );
			const double temp = num( x, "temp_c" );
			out->push_back( "  temp    " + std::string( temp >= 85 ? RED : temp >= 75 ? YELLOW : GREEN ) + format( "%.0f", temp ) + " C" + RESET + "     sm " + format( "%.0f", num( x, "sm_mhz" ) ) + " MHz     mem " + format( "%.0f", num( x, "mem_mhz" ) ) + " MHz" );
			out->push_back( "  vram    " + bar( total > 0 ? used / total : 0.0, barw ) + " " + bytes( used ) + " / " + bytes( total ) + "     engine arena " + bytes( num( x, "arena_bytes" ) ) );
			std::string th;
			if ( const json* t = get( x, "throttle" ); t && t->is_array() ) {
				for ( const auto& n : *t ) th += ( th.empty() ? "" : ", " ) + str( &n );
			}
			out->push_back( "  throttle " + ( th.empty() ? std::string( GREEN ) + "none" + RESET : std::string( YELLOW ) + th + RESET ) );
			out->push_back( "" );
		}
	}
	if ( const json* h = get( s, "host" ) ) {
		const double mt = num( *h, "mem_total" );
		out->push_back( heading( "Host" ) );
		out->push_back( "  cpu     " + bar( num( *h, "cpu_pct" ) / 100.0, barw ) + lpad( format( "%.0f", num( *h, "cpu_pct" ) ), 5 ) + "%     load " + format( "%.2f", num( *h, "load1" ) ) );
		out->push_back( "  ram     " + bar( mt > 0 ? num( *h, "mem_used" ) / mt : 0.0, barw ) + " " + bytes( num( *h, "mem_used" ) ) + " / " + bytes( mt ) + "     server rss " + bytes( num( *h, "rss" ) ) );
	}
}

// ---- frame

int body_rows( int rows ) { return std::max( 1, rows - 3 ); }

} // namespace

bool apply_key( UiState& ui, int key, int page ) {
	switch ( key ) {
		case 'q': case 'Q': return false;
		case KEY_TAB: case KEY_RIGHT: ui.tab = ( ui.tab + 1 ) % TAB_COUNT; break;
		case KEY_BTAB: case KEY_LEFT: ui.tab = ( ui.tab + TAB_COUNT - 1 ) % TAB_COUNT; break;
		case '1': case '2': case '3': case '4': ui.tab = key - '1'; break;
		default: {
			const int sign = ui.tab == TAB_LOG ? 1 : -1;   // the log counts lines above the tail
			int& sc = ui.scroll[ui.tab];
			if ( key == KEY_UP || key == 'k' ) sc += sign;
			else if ( key == KEY_DOWN || key == 'j' ) sc -= sign;
			else if ( key == KEY_PGUP ) sc += sign * page;
			else if ( key == KEY_PGDN ) sc -= sign * page;
			else if ( key == KEY_HOME ) sc = ui.tab == TAB_LOG ? 1 << 30 : 0;
			else if ( key == KEY_END ) sc = ui.tab == TAB_LOG ? 0 : 1 << 30;
			break;
		}
	}
	return true;
}

std::string render_frame( const ViewData& data, UiState& ui, int rows, int cols ) {
	rows = std::max( rows, 6 );
	cols = std::max( cols, 20 );
	std::vector<std::string> screen;

	// title bar
	std::string title = " volta-top";
	if ( data.stats ) title += "  " + str( *data.stats, "server", "model" ) + "  up " + duration( num( *data.stats, "server", "uptime_s" ) );
	std::string right = data.connected ? "connected " : "NO CONNECTION ";
	if ( !data.source.empty() ) right = data.source + "  " + right;
	const int gap = cols - visible_width( title ) - visible_width( right );
	title = fit( title, cols );
	std::string bar_line = std::string( REVERSE ) + title + ( gap > 0 ? std::string( static_cast<size_t>( gap ), ' ' ) + right : std::string() );
	screen.push_back( bar_line );

	// tab bar
	std::string tabs;
	for ( int t = 0; t < TAB_COUNT; ++t ) {
		const std::string label = " " + std::to_string( t + 1 ) + " " + TAB_NAMES[t] + " ";
		tabs += ( t == ui.tab ? std::string( REVERSE ) + BOLD + label + RESET : label ) + " ";
	}
	screen.push_back( tabs );

	// body
	Lines body;
	if ( ui.tab == TAB_LOG ) {
		if ( !data.log ) body.push_back( std::string( DIM ) + "The log is shown only when the server runs with --tui; this viewer has no access to its console output." + RESET );
		else body = *data.log;
	}
	else if ( !data.stats ) {
		body.push_back( data.connected ? "waiting for the first reading..." : std::string( RED ) + "no reading yet: " + data.error + RESET );
	}
	else {
		if ( !data.connected ) body.push_back( std::string( RED ) + "connection lost (" + data.error + "); the numbers below are the last reading" + RESET );
		if ( ui.tab == TAB_OVERVIEW ) overview( *data.stats, cols, &body );
		else if ( ui.tab == TAB_SLOTS ) slots( *data.stats, cols, &body );
		else gpus( *data.stats, cols, &body );
	}
	const int avail = body_rows( rows );
	const int total = static_cast<int>( body.size() );
	const int max_scroll = std::max( 0, total - avail );
	int& sc = ui.scroll[ui.tab];
	sc = std::min( std::max( sc, 0 ), max_scroll );
	const int first = ui.tab == TAB_LOG ? max_scroll - sc : sc;
	for ( int i = 0; i < avail; ++i ) {
		const int at = first + i;
		screen.push_back( at < total ? body[static_cast<size_t>( at )] : std::string() );
	}

	// footer
	std::string foot = std::string( DIM ) + " Tab/1-4 switch   Up/Down PgUp/PgDn Home/End scroll   q quit" + RESET;
	if ( max_scroll > 0 ) {
		const int shown = ui.tab == TAB_LOG ? max_scroll - sc : sc;
		foot += std::string( DIM ) + "   lines " + std::to_string( shown + 1 ) + "-" + std::to_string( std::min( total, shown + avail ) ) + " of " + std::to_string( total ) + RESET;
	}
	screen.push_back( foot );

	std::string out = "\x1b[H";
	for ( size_t i = 0; i < screen.size(); ++i ) {
		out += fit( screen[i], cols );
		out += "\x1b[K";
		if ( i + 1 < screen.size() ) out += "\r\n";
	}
	out += "\x1b[J";
	return out;
}

} // namespace engine::server::tui
