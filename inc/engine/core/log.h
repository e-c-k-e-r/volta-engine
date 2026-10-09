#pragma once

namespace engine::core {

enum class LogLevel { Error = 0, Warn = 1, Info = 2, Debug = 3 };

// Threshold from the log_level option (ENGINE_LOG_LEVEL: error, warn, info, debug; default info)
LogLevel log_level();

// printf-style line with a level prefix ("[-] ", "[!] ", "[*] ", "    "): errors and warnings go to stderr, the rest to stdout.
// A trailing newline is added when the format lacks one.
void log_write( LogLevel level, const char* fmt, ... ) __attribute__( ( format( printf, 2, 3 ) ) );

} // namespace engine::core

#define LOG_ERROR( ... ) ::engine::core::log_write( ::engine::core::LogLevel::Error, __VA_ARGS__ )
#define LOG_WARN( ... ) ::engine::core::log_write( ::engine::core::LogLevel::Warn, __VA_ARGS__ )
#define LOG_INFO( ... ) ::engine::core::log_write( ::engine::core::LogLevel::Info, __VA_ARGS__ )
#define LOG_DEBUG( ... ) ::engine::core::log_write( ::engine::core::LogLevel::Debug, __VA_ARGS__ )
