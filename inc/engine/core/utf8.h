#pragma once

#include <cstddef>
#include <string>

namespace engine::core {

// Number of bytes at the end of `s` that start a UTF-8 sequence which is not complete yet (0 when the text ends on a character boundary).
// A model's tokens can split a multi-byte character; text is streamed only up to such a boundary.
inline size_t utf8_incomplete_tail( const std::string& s ) {
	const size_t n = s.size();
	for ( size_t back = 1; back <= 3 && back <= n; ++back ) {
		const unsigned char c = static_cast<unsigned char>( s[n - back] );
		if ( ( c & 0xC0 ) == 0x80 ) continue;                 // continuation byte: keep looking for the lead
		const size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
		return need > back ? back : 0;
	}
	return 0;
}

} // namespace engine::core
