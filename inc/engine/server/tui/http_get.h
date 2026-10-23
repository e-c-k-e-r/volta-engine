#pragma once

#include <string>

namespace engine::server::tui {

// A blocking HTTP/1.0 GET with one overall timeout per step (connect, send, receive). On success `body` holds the response body of a
// 200 answer; otherwise `error` says what went wrong.
bool http_get( const std::string& host, int port, const std::string& path, int timeout_ms, std::string* body, std::string* error );

} // namespace engine::server::tui
