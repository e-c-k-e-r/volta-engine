#pragma once

#include <nlohmann/json.hpp>
#include <string>

namespace engine::server {

// JSON text that cannot abort the process: the library is built without exceptions, so dump() on a string that is not valid UTF-8 (a streamed
// model answer can end or split inside a multi-byte character) would call abort(); here the bad bytes become U+FFFD
template <typename J>
inline std::string dump_safe( const J& j, int indent = -1 ) {
	return j.dump( indent, ' ', false, J::error_handler_t::replace );
}

} // namespace engine::server
