#include "engine/core/log.h"
#include "engine/runtime/options.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace engine::core {

LogLevel log_level() {
	static const LogLevel level = [] {
		const std::string& s = runtime::options().log_level;
		if ( s == "error" ) return LogLevel::Error;
		if ( s == "warn" ) return LogLevel::Warn;
		if ( s == "debug" ) return LogLevel::Debug;
		return LogLevel::Info;
	}();
	return level;
}

void log_write( LogLevel level, const char* fmt, ... ) {
	if ( level > log_level() ) return;
	static const char* const prefix[] = { "[-] ", "[!] ", "[*] ", "    " };
	char buf[2048];
	int n = std::snprintf( buf, sizeof( buf ), "%s", prefix[static_cast<int>( level )] );
	va_list ap;
	va_start( ap, fmt );
	n += std::vsnprintf( buf + n, sizeof( buf ) - static_cast<size_t>( n ) - 2, fmt, ap );
	va_end( ap );
	if ( n >= static_cast<int>( sizeof( buf ) ) - 2 ) n = static_cast<int>( sizeof( buf ) ) - 3;
	if ( n == 0 || buf[n - 1] != '\n' ) buf[n++] = '\n';
	buf[n] = '\0';
	std::fputs( buf, level <= LogLevel::Warn ? stderr : stdout );
}

} // namespace engine::core
