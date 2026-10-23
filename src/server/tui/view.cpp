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

const char* const TAB_NAMES[TAB_COUNT] = { "Overview", "Slots", "GPUs", "Experts", "Log" };

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

// The newest `width` columns of a series as block characters, one column per bucket of samples (the bucket's maximum, so a short spike
// survives the thinning), scaled to `top`; zero is blank so idle stretches stay empty. Shorter series sit at the right edge.
std::string spark( const std::vector<double>& v, int width, double top ) {
	static const char* const BLOCKS[] = { "\xe2\x96\x81", "\xe2\x96\x82", "\xe2\x96\x83", "\xe2\x96\x84", "\xe2\x96\x85", "\xe2\x96\x86", "\xe2\x96\x87", "\xe2\x96\x88" };
	const size_t n = v.size();
	const size_t cols = static_cast<size_t>( width );
	std::string out;
	const size_t shown = std::min( n, cols * ( ( n + cols - 1 ) / cols ) );   // whole buckets only
	const size_t per = n == 0 ? 1 : ( n + cols - 1 ) / cols;
	const size_t used = ( shown + per - 1 ) / per;
	out.append( cols - std::min( cols, used ), ' ' );
	const size_t start = n - shown;
	for ( size_t b = 0; b < used; ++b ) {
		double m = 0.0;
		for ( size_t i = start + b * per; i < std::min( n, start + ( b + 1 ) * per ); ++i ) m = std::max( m, v[i] );
		if ( m <= 0.0 || top <= 0.0 ) out += ' ';
		else out += BLOCKS[std::min( 7, std::max( 0, static_cast<int>( std::ceil( m / top * 8.0 ) ) - 1 ) )];
	}
	return out;
}

std::vector<double> series( const json& s, const char* key ) {
	std::vector<double> v;
	if ( const json* a = get( s, "history", key ); a && a->is_array() ) {
		for ( const auto& x : *a ) v.push_back( num( &x ) );
	}
	return v;
}


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
	if ( const json* h = get( s, "latency", "gap_hist" ); h && h->is_array() && h->size() == 12 ) {   // log2 buckets from 1 ms
		static const char* const LABELS[12] = { "<2", "4", "8", "16", "32", "64", "128", "256", "512", "1k", "2k", ">2k" };
		std::vector<double> v;
		double top = 0.0;
		for ( const auto& x : *h ) { v.push_back( num( &x ) ); top = std::max( top, v.back() ); }
		if ( top > 0.0 ) {
			std::string blocks = "  gap ms  ", names = "          ";
			for ( int i = 0; i < 12; ++i ) {
				const std::string cell = spark( { v[static_cast<size_t>( i )] }, 1, top );
				blocks += std::string( CYAN ) + cell + RESET + "    ";
				names += pad( LABELS[i], 5 );
			}
			out->push_back( DIM + std::string( "  token gaps by size (time between tokens, buckets doubling from 2 ms; height = share of samples)" ) + RESET );
			out->push_back( blocks );
			out->push_back( std::string( DIM ) + names + RESET );
		}
	}
	const json* tt = get( s, "latency", "ttft_ms" );
	if ( tt && tt->is_array() && tt->size() >= 3 ) {
		out->push_back( "  ttft       p50 " + ms( num( &( *tt )[0] ) ) + "   p90 " + ms( num( &( *tt )[1] ) ) + "   p99 " + ms( num( &( *tt )[2] ) ) );
	}
	const int sw = std::max( 10, cols - 46 );
	struct Row { const char* label; const char* key; const char* unit; double top; };
	const double slot_count = num( s, "server", "slots" );
	for ( const Row& r : { Row{ "decode ", "decode", " tok/s", 0.0 }, Row{ "prefill", "prefill", " tok/s", 0.0 }, Row{ "busy   ", "active", " slots", slot_count } } ) {
		const std::vector<double> v = series( s, r.key );
		if ( v.empty() ) continue;
		const double top = r.top > 0.0 ? r.top : std::max( 1.0, *std::max_element( v.begin(), v.end() ) );
		out->push_back( "  " + std::string( DIM ) + r.label + RESET + " " + CYAN + spark( v, sw, top ) + RESET + "  now " + format( "%.0f", v.back() ) + " / max " + format( "%.0f", top ) + r.unit );
	}
	out->push_back( std::string( DIM ) + "  last " + duration( static_cast<double>( series( s, "decode" ).size() ) ) + ", one sample per second" + RESET );
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

void slots( const json& s, const UiState& ui, int cols, Lines* out ) {
	const int barw = std::max( 8, std::min( 24, cols / 6 ) );
	const double ctx = num( s, "server", "max_context" );
	out->push_back( heading( "Slots" ) );
	out->push_back( std::string( DIM ) + "  " + lpad( "slot", 4 ) + " " + pad( "phase", 9 ) + " " + pad( "req", 8 ) + " " + pad( "context", static_cast<size_t>( barw ) + 2 ) + " " + lpad( "pos", 7 ) + "  " + lpad( "prompt", 6 ) + "  " +
		lpad( "cached", 6 ) + "  " + lpad( "generated", 9 ) + "/" + pad( "max", 6 ) + lpad( "age", 8 ) + "  " + lpad( "tok/s", 6 ) + RESET );
	const json* sl = get( s, "slots" );
	if ( sl && sl->is_array() ) {
		int row = 0;
		for ( const auto& x : *sl ) {
			const bool selected = row++ == ui.slot_sel;
			const std::string phase = str( x, "phase" );
			const bool idle = phase == "idle";
			const double pos = num( x, "pos" );
			const std::string lead = selected ? std::string( BOLD ) + "> " + RESET : "  ";
			std::string l = lead + lpad( format( "%.0f", num( x, "slot" ) ), 4 ) + " " + pad( phase, 9 ) + " ";
			if ( idle ) {
				l = lead + DIM + lpad( format( "%.0f", num( x, "slot" ) ), 4 ) + " " + pad( phase, 9 ) + " -" + RESET;
			}
			else {
				const char* pc = phase == "prefill" ? YELLOW : phase == "thinking" ? CYAN : GREEN;
				l = lead + lpad( format( "%.0f", num( x, "slot" ) ), 4 ) + " " + pc + pad( phase, 9 ) + RESET + " " + pad( "#" + format( "%.0f", num( x, "req_id" ) ), 8 ) + " " +
					bar( ctx > 0 ? pos / ctx : 0.0, barw ) + " " + lpad( count( pos ), 7 ) + "  " + lpad( count( num( x, "prompt_tokens" ) ), 6 ) + "  " +
					lpad( count( num( x, "cached_tokens" ) ), 6 ) + "  " + lpad( count( num( x, "generated" ) ), 9 ) + "/" + pad( count( num( x, "max_tokens" ) ), 6 ) +
					lpad( duration( num( x, "age_s" ) ), 8 ) + "  " + lpad( format( "%.1f", num( x, "decode_tok_s" ) ), 6 );
			}
			out->push_back( l );
		}
	}
	if ( ui.slot_detail && sl && sl->is_array() && ui.slot_sel < static_cast<int>( sl->size() ) ) {
		const json& x = ( *sl )[static_cast<size_t>( ui.slot_sel )];
		out->push_back( "" );
		out->push_back( heading( "Slot " + std::to_string( ui.slot_sel ) ) + std::string( DIM ) + "  (Enter closes)" + RESET );
		if ( str( x, "phase" ) == "idle" ) {
			out->push_back( "  idle" );
		}
		else {
			const double pos = num( x, "pos" ), prompt = num( x, "prompt_tokens" ), cached = num( x, "cached_tokens" ), gen = num( x, "generated" ), mx = num( x, "max_tokens" ), rate = num( x, "decode_tok_s" );
			out->push_back( "  request #" + format( "%.0f", num( x, "req_id" ) ) + "   " + str( x, "phase" ) + " for " + duration( num( x, "age_s" ) ) );
			out->push_back( "  context   " + count( pos ) + " of " + count( ctx ) + " (" + format( "%.1f", ctx > 0 ? pos * 100.0 / ctx : 0.0 ) + "% full)" );
			out->push_back( "  prompt    " + count( prompt ) + " tokens, " + count( cached ) + " from the prefix cache (" + format( "%.1f", prompt > 0 ? cached * 100.0 / prompt : 0.0 ) + "%)" );
			std::string g = "  output    " + count( gen ) + " of at most " + count( mx ) + " tokens at " + format( "%.1f", rate ) + " tok/s";
			if ( rate > 0.0 && mx > gen ) g += "; the limit is " + duration( ( mx - gen ) / rate ) + " away";
			out->push_back( g );
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


// Cell colours for the routing heat map: dark (cold) through blue and green to orange and red (hot)
const int HEAT_RAMP[] = { 236, 24, 31, 36, 71, 142, 208, 196 };

std::string heat_color( bool background, double f ) {
	f = std::min( 0.999, std::max( 0.0, f ) );
	return "\x1b[" + std::string( background ? "48" : "38" ) + ";5;" + std::to_string( HEAT_RAMP[static_cast<int>( f * 8.0 )] ) + "m";
}

void experts_tab( const json* e, const std::string& error, const UiState& ui, int cols, int rows, Lines* out ) {
	if ( !e ) {
		out->push_back( error.empty() ? std::string( "loading the routing counters..." ) : std::string( RED ) + "no routing data: " + error + RESET );
		return;
	}
	const int R = static_cast<int>( num( *e, "ranks" ) ), L = static_cast<int>( num( *e, "layers" ) ), E = static_cast<int>( num( *e, "experts" ) );
	const json* arr = get( *e, ui.experts_total ? "total" : "recent" );
	if ( R <= 0 || L <= 0 || E <= 0 || !arr || !arr->is_array() || arr->size() != static_cast<size_t>( R ) * L * E ) {
		out->push_back( std::string( DIM ) + "No routing counters: this model has no mixture-of-experts layers, or expert_stats is off." + RESET );
		return;
	}
	const int C = R * E;
	std::vector<double> v;
	v.reserve( arr->size() );
	for ( const auto& x : *arr ) v.push_back( num( &x ) );
	auto at = [&]( int r, int l, int x ) { return v[( static_cast<size_t>( r ) * L + l ) * E + x]; };
	double sum = 0.0;
	std::vector<double> per_rank( static_cast<size_t>( R ), 0.0 ), per_expert( static_cast<size_t>( C ), 0.0 ), imb;
	for ( int l = 0; l < L; ++l ) {
		double lsum = 0.0, lmax = 0.0;
		for ( int r = 0; r < R; ++r ) {
			for ( int x = 0; x < E; ++x ) {
				const double c = at( r, l, x );
				sum += c; lsum += c; per_rank[static_cast<size_t>( r )] += c; per_expert[static_cast<size_t>( r ) * E + x] += c;
				lmax = std::max( lmax, c );
			}
		}
		if ( lsum > 0.0 ) imb.push_back( lmax / ( lsum / C ) );
	}
	int live = 0;
	for ( double c : per_expert ) live += c > 0.0;
	double imbalance = 0.0;
	for ( double x : imb ) imbalance += x;
	imbalance = imb.empty() ? 0.0 : imbalance / static_cast<double>( imb.size() );
	std::vector<int> order( static_cast<size_t>( C ) );
	for ( int i = 0; i < C; ++i ) order[static_cast<size_t>( i )] = i;
	std::sort( order.begin(), order.end(), [&]( int a, int b ) { return per_expert[static_cast<size_t>( a )] > per_expert[static_cast<size_t>( b )]; } );
	double top10 = 0.0;
	for ( int i = 0; i < std::min( 10, C ); ++i ) top10 += per_expert[static_cast<size_t>( order[static_cast<size_t>( i )] )];

	out->push_back( heading( "Expert routing" ) + std::string( DIM ) + ( ui.experts_total ? "  since boot (m: recent minute)" : "  recent minute (m: since boot)" ) + RESET );
	out->push_back( "  " + count( sum ) + " routes, " + std::to_string( L ) + " layers x " + std::to_string( C ) + " experts" + ( R > 1 ? " on " + std::to_string( R ) + " ranks" : std::string() ) );
	out->push_back( "  experts used " + std::string( live < C ? YELLOW : GREEN ) + std::to_string( live ) + " / " + std::to_string( C ) + RESET + "   layer imbalance " + format( "%.2f", imbalance ) + "x   top 10 share " +
		format( "%.1f", sum > 0 ? top10 * 100.0 / sum : 0.0 ) + "% (even: " + format( "%.1f", 1000.0 / C ) + "%)" );
	if ( R > 1 ) {
		const int bw = std::max( 10, std::min( 30, cols / 4 ) );
		for ( int r = 0; r < R; ++r ) {
			const double share = sum > 0 ? per_rank[static_cast<size_t>( r )] / sum : 0.0;
			out->push_back( "  rank " + std::to_string( r ) + " " + bar( share * R / 2.0, bw, false ) + lpad( format( "%.1f", share * 100.0 ), 6 ) + "%" +
				( std::fabs( share * R - 1.0 ) > 0.15 ? std::string( YELLOW ) + "  uneven" + RESET : std::string() ) );
		}
	}
	out->push_back( "" );
	out->push_back( DIM + std::string( "  hottest experts" ) + RESET );
	for ( int i = 0; i < std::min( 5, C ); ++i ) {
		const int id = order[static_cast<size_t>( i )];
		out->push_back( "    expert " + lpad( std::to_string( id ), 5 ) + ( R > 1 ? "  (rank " + std::to_string( id / E ) + ")" : std::string() ) + "  " + lpad( count( per_expert[static_cast<size_t>( id )] ), 12 ) + "  " +
			lpad( format( "%.2f", sum > 0 ? per_expert[static_cast<size_t>( id )] * 100.0 / sum : 0.0 ), 6 ) + "%" );
	}
	out->push_back( "" );

	// heat map: layers down (two per text row, as the halves of a block character), experts across; cells that do not fit are averaged
	const int label = 5;
	const int width = std::max( 8, cols - label - 1 );
	const int per_col = ( C + width - 1 ) / width;
	const int ncols = ( C + per_col - 1 ) / per_col;
	const int max_rows = std::max( 4, rows - static_cast<int>( out->size() ) - 6 );
	const int per_row = std::max( 1, ( L + 2 * max_rows - 1 ) / ( 2 * max_rows ) );   // layers per half row
	const int groups = ( L + per_row - 1 ) / per_row;
	std::vector<double> cell( static_cast<size_t>( groups ) * ncols, 0.0 );
	double top = 0.0;
	for ( int g = 0; g < groups; ++g ) {
		for ( int c = 0; c < ncols; ++c ) {
			double acc = 0.0;
			int n = 0;
			for ( int l = g * per_row; l < std::min( L, ( g + 1 ) * per_row ); ++l ) {
				for ( int x = c * per_col; x < std::min( C, ( c + 1 ) * per_col ); ++x ) { acc += at( x / E, l, x % E ); ++n; }
			}
			cell[static_cast<size_t>( g ) * ncols + c] = n ? acc / n : 0.0;
			top = std::max( top, cell[static_cast<size_t>( g ) * ncols + c] );
		}
	}
	const double lg = std::log1p( top );
	auto level = [&]( int g, int c ) { return g < groups && lg > 0.0 ? std::log1p( cell[static_cast<size_t>( g ) * ncols + c] ) / lg : 0.0; };
	std::string ruler = std::string( static_cast<size_t>( label ), ' ' );
	for ( int c = 0; c < ncols; ++c ) {
		const bool edge = R > 1 && c > 0 && ( c * per_col ) % E < per_col;   // the first column of a rank
		ruler += edge ? '|' : ' ';
	}
	out->push_back( DIM + std::string( "  layers down, experts across" ) + ( R > 1 ? "; | marks a rank" : "" ) + ( per_col > 1 ? "; a column is " + std::to_string( per_col ) + " experts" : std::string() ) +
		( per_row > 1 ? "; a half row is " + std::to_string( per_row ) + " layers" : std::string() ) + RESET );
	out->push_back( DIM + ruler + RESET );
	for ( int g = 0; g < groups; g += 2 ) {
		std::string line = std::string( DIM ) + lpad( std::to_string( g * per_row ), 4 ) + RESET + " ";
		for ( int c = 0; c < ncols; ++c ) line += heat_color( false, level( g, c ) ) + heat_color( true, level( g + 1, c ) ) + "\xe2\x96\x80";
		out->push_back( line + RESET );
	}
	out->push_back( DIM + std::string( "  cold" ) + RESET + " " + [&] { std::string k; for ( int i = 0; i < 8; ++i ) k += heat_color( true, ( i + 0.5 ) / 8.0 ) + "  "; return k + RESET; }() + " " + DIM + "hot (log scale)" + RESET );
}

struct Alert { bool critical; std::string text; };

// Conditions worth a second look; the always-on power cap and idle clocks are not among them
std::vector<Alert> alerts( const ViewData& d ) {
	std::vector<Alert> a;
	if ( !d.connected ) a.push_back( { true, "no connection" + ( d.error.empty() ? std::string() : ": " + d.error ) } );
	if ( !d.stats ) return a;
	const json& s = *d.stats;
	if ( const json* g = get( s, "gpus" ); g && g->is_array() ) {
		for ( const auto& x : *g ) {
			const std::string id = "GPU " + format( "%.0f", num( x, "index" ) );
			if ( num( x, "temp_c" ) >= 85.0 ) a.push_back( { true, id + " at " + format( "%.0f", num( x, "temp_c" ) ) + " C" } );
			if ( const json* t = get( x, "throttle" ); t && t->is_array() ) {
				for ( const auto& n : *t ) {
					const std::string name = str( &n );
					if ( name == "hw slowdown" || name == "hw thermal" || name == "sw thermal" || name == "hw power brake" ) a.push_back( { true, id + " throttled: " + name } );
				}
			}
			// The engine takes its arena up to the last megabyte, so a full card is normal; what matters is memory outside the arena
			// (another process, a leak), which is what the arena leaves free for the CUDA context, NCCL and graphs (about 1.3 GiB measured)
			const double arena = num( x, "arena_bytes" ), outside = num( x, "mem_used" ) - arena;
			if ( arena > 0.0 && outside > 2.0 * 1073741824.0 ) a.push_back( { false, id + " has " + bytes( outside ) + " in use outside the engine arena" } );
		}
	}
	if ( num( s, "server", "queued" ) > 0.0 ) a.push_back( { false, format( "%.0f", num( s, "server", "queued" ) ) + " request(s) waiting for a slot" } );
	if ( const json* gap = get( s, "latency", "gap_ms" ); gap && gap->is_array() && gap->size() >= 3 && num( &( *gap )[2] ) > 500.0 ) {
		a.push_back( { false, "token gap p99 " + ms( num( &( *gap )[2] ) ) } );
	}
	const double mt = num( s, "host", "mem_total" );
	if ( mt > 0.0 && num( s, "host", "mem_used" ) / mt > 0.95 ) a.push_back( { false, "host memory " + format( "%.0f", num( s, "host", "mem_used" ) * 100.0 / mt ) + "% full" } );
	return a;
}

// ---- frame

int body_rows( int rows, bool has_alerts ) { return std::max( 1, rows - 3 - ( has_alerts ? 1 : 0 ) ); }

} // namespace

bool apply_key( UiState& ui, int key, int page ) {
	switch ( key ) {
		case 'q': case 'Q': return false;
		case KEY_TAB: case KEY_RIGHT: ui.tab = ( ui.tab + 1 ) % TAB_COUNT; break;
		case KEY_BTAB: case KEY_LEFT: ui.tab = ( ui.tab + TAB_COUNT - 1 ) % TAB_COUNT; break;
		case 'p': case 'P': ui.paused = !ui.paused; break;
		case 'm': case 'M': ui.experts_total = !ui.experts_total; break;
		case '\r': case '\n': if ( ui.tab == TAB_SLOTS ) ui.slot_detail = !ui.slot_detail; break;
		default: {
			if ( key >= '1' && key < '1' + TAB_COUNT ) { ui.tab = key - '1'; break; }
			const int sign = ui.tab == TAB_LOG ? 1 : -1;   // the log counts lines above the tail
			int& sc = ui.scroll[ui.tab];
			if ( ui.tab == TAB_SLOTS && ( key == KEY_UP || key == 'k' ) ) { --ui.slot_sel; break; }   // the arrows pick a slot there; the page keys scroll
			if ( ui.tab == TAB_SLOTS && ( key == KEY_DOWN || key == 'j' ) ) { ++ui.slot_sel; break; }
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
	const std::vector<Alert> alert = alerts( data );
	bool critical = false;
	for ( const Alert& a : alert ) critical = critical || a.critical;

	// title bar: red when something needs attention, yellow for a warning
	std::string title = " volta-top";
	if ( data.stats ) title += "  " + str( *data.stats, "server", "model" ) + "  up " + duration( num( *data.stats, "server", "uptime_s" ) );
	std::string right = ( ui.paused ? "PAUSED  " : std::string() ) + ( data.connected ? "connected " : "NO CONNECTION " );
	if ( !data.source.empty() ) right = data.source + "  " + right;
	const int gap = cols - visible_width( title ) - visible_width( right );
	title = fit( title, cols );
	const char* title_style = critical ? "\x1b[41;97m" : !alert.empty() ? "\x1b[43;30m" : REVERSE;
	screen.push_back( std::string( title_style ) + title + ( gap > 0 ? std::string( title_style ) + std::string( static_cast<size_t>( gap ), ' ' ) + right : std::string() ) );

	// tab bar
	std::string tabs;
	for ( int t = 0; t < TAB_COUNT; ++t ) {
		const std::string label = " " + std::to_string( t + 1 ) + " " + TAB_NAMES[t] + " ";
		tabs += ( t == ui.tab ? std::string( REVERSE ) + BOLD + label + RESET : label ) + " ";
	}
	screen.push_back( tabs );
	if ( !alert.empty() ) {
		std::string line;
		for ( const Alert& a : alert ) line += std::string( a.critical ? RED : YELLOW ) + "! " + a.text + RESET + "   ";
		screen.push_back( line );
	}

	// body
	Lines body;
	const int avail = body_rows( rows, !alert.empty() );
	if ( ui.tab == TAB_LOG ) {
		if ( !data.log ) body.push_back( std::string( DIM ) + "The log is shown only when the server runs with --tui; this viewer has no access to its console output." + RESET );
		else body = *data.log;
	}
	else if ( !data.stats ) {
		body.push_back( data.connected ? "waiting for the first reading..." : std::string( RED ) + "no reading yet: " + data.error + RESET );
	}
	else {
		if ( !data.connected ) body.push_back( std::string( RED ) + "connection lost (" + data.error + "); the numbers below are the last reading" + RESET );
		if ( ui.tab == TAB_SLOTS ) {
			const json* sl = get( *data.stats, "slots" );
			const int n = sl && sl->is_array() ? static_cast<int>( sl->size() ) : 0;
			ui.slot_sel = std::min( std::max( ui.slot_sel, 0 ), std::max( 0, n - 1 ) );
		}
		if ( ui.tab == TAB_OVERVIEW ) overview( *data.stats, cols, &body );
		else if ( ui.tab == TAB_SLOTS ) slots( *data.stats, ui, cols, &body );
		else if ( ui.tab == TAB_GPUS ) gpus( *data.stats, cols, &body );
		else experts_tab( data.experts, data.experts_error, ui, cols, avail, &body );
	}
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
	std::string keys = " Tab/1-" + std::to_string( TAB_COUNT ) + " switch   PgUp/PgDn Home/End scroll   p pause   q quit";
	if ( ui.tab == TAB_SLOTS ) keys += "   Up/Down pick a slot   Enter detail";
	else if ( ui.tab == TAB_EXPERTS ) keys += "   m since boot / recent";
	else keys += "   Up/Down scroll";
	std::string foot = std::string( DIM ) + keys + RESET;
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
