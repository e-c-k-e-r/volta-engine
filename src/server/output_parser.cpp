#include "engine/server/output_parser.h"

#define JSON_NOEXCEPTION 1
#include <nlohmann/json.hpp>

#include <algorithm>

namespace engine::server {

namespace {

const char* const kThinkOpen = "<think>";
const char* const kThinkClose = "</think>";
const char* const kCallOpen = "<tool_call>";
const char* const kCallClose = "</tool_call>";

bool is_ws( char c ) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

// Length of the longest suffix of s that is a proper prefix of marker (text that may still become the marker)
size_t partial_suffix( const std::string& s, const std::string& marker ) {
	const size_t max = std::min( s.size(), marker.size() - 1 );
	for ( size_t n = max; n > 0; --n ) {
		if ( s.compare( s.size() - n, n, marker, 0, n ) == 0 ) return n;
	}
	return 0;
}

} // namespace

ToolSchemas tool_schemas_from_json( const std::string& tools_json ) {
	ToolSchemas out;
	if ( tools_json.empty() ) return out;
	auto j = nlohmann::json::parse( tools_json, nullptr, false );
	if ( j.is_discarded() || !j.is_array() ) return out;
	for ( const auto& t : j ) {
		const nlohmann::json& f = ( t.is_object() && t.contains( "function" ) ) ? t["function"] : t;
		if ( !f.is_object() || !f.contains( "name" ) || !f["name"].is_string() ) continue;
		auto& params = out[f["name"].get<std::string>()];
		const nlohmann::json* schema = nullptr;
		if ( f.contains( "parameters" ) ) schema = &f["parameters"];
		else if ( f.contains( "input_schema" ) ) schema = &f["input_schema"];
		if ( !schema || !schema->is_object() || !schema->contains( "properties" ) || !( *schema )["properties"].is_object() ) continue;
		for ( auto it = ( *schema )["properties"].begin(); it != ( *schema )["properties"].end(); ++it ) {
			std::string type;
			if ( it.value().is_object() && it.value().contains( "type" ) && it.value()["type"].is_string() ) type = it.value()["type"].get<std::string>();
			params[it.key()] = type;
		}
	}
	return out;
}

OutputParser::OutputParser( engine::loader::ModelFamily family, bool starts_in_reasoning, ToolSchemas schemas )
	: family_( family ), schemas_( std::move( schemas ) ), state_( starts_in_reasoning ? State::Reasoning : State::Content ) {
	if ( family_ == engine::loader::ModelFamily::DEEPSEEK ) {
		call_open_ = "<｜DSML｜ invoke";
		call_close_ = "</｜DSML｜ invoke>";
		wrappers_ = { "<｜DSML｜ calls>", "</｜DSML｜ calls>" };
	}
	else {
		call_open_ = kCallOpen;
		call_close_ = kCallClose;
	}
}

void OutputParser::feed( const std::string& text, std::vector<OutputEvent>& events ) {
	buf_ += text;
	process( events, false );
}

void OutputParser::finish( std::vector<OutputEvent>& events ) {
	process( events, true );
}

void OutputParser::end_section() {
	lead_trim_ = true;
	pending_ws_.clear();
}

void OutputParser::emit_text( OutputEvent::Type type, const std::string& s, std::vector<OutputEvent>& events ) {
	std::string t = s;
	if ( lead_trim_ ) {
		size_t k = 0;
		while ( k < t.size() && is_ws( t[k] ) ) ++k;
		t.erase( 0, k );
		if ( t.empty() ) return;
		lead_trim_ = false;
	}
	t = pending_ws_ + t;
	size_t e = t.size();
	while ( e > 0 && is_ws( t[e - 1] ) ) --e;
	pending_ws_ = t.substr( e );
	t.resize( e );
	if ( t.empty() ) return;
	OutputEvent ev;
	ev.type = type;
	ev.text = std::move( t );
	events.push_back( std::move( ev ) );
	if ( type == OutputEvent::Type::Content ) emitted_any_ = true;
}

void OutputParser::process( std::vector<OutputEvent>& events, bool final ) {
	for ( ;; ) {
		if ( state_ == State::Reasoning ) {
			const size_t p = buf_.find( kThinkClose );
			if ( p != std::string::npos ) {
				emit_text( OutputEvent::Type::Reasoning, buf_.substr( 0, p ), events );
				buf_.erase( 0, p + std::char_traits<char>::length( kThinkClose ) );
				end_section();
				state_ = State::Content;
				continue;
			}
			const size_t safe = final ? buf_.size() : buf_.size() - partial_suffix( buf_, kThinkClose );
			emit_text( OutputEvent::Type::Reasoning, buf_.substr( 0, safe ), events );
			buf_.erase( 0, safe );
			return;
		}
		if ( state_ == State::Content ) {
			const size_t pc = buf_.find( call_open_ );
			const size_t pt = emitted_any_ || saw_tool_call_ ? std::string::npos : buf_.find( kThinkOpen );
			// a calls-block wrapper: the text before it is content, the marker itself is dropped
			size_t pw = std::string::npos, wlen = 0;
			for ( const std::string& w : wrappers_ ) {
				const size_t p = buf_.find( w );
				if ( p < pw ) { pw = p; wlen = w.size(); }
			}
			if ( pw != std::string::npos && pw < pc && pw < pt ) {
				emit_text( OutputEvent::Type::Content, buf_.substr( 0, pw ), events );
				buf_.erase( 0, pw + wlen );
				continue;
			}
			if ( pt != std::string::npos && ( pc == std::string::npos || pt < pc ) ) {
				emit_text( OutputEvent::Type::Content, buf_.substr( 0, pt ), events );
				buf_.erase( 0, pt + std::char_traits<char>::length( kThinkOpen ) );
				end_section();
				state_ = State::Reasoning;
				continue;
			}
			if ( pc != std::string::npos ) {
				emit_text( OutputEvent::Type::Content, buf_.substr( 0, pc ), events );
				buf_.erase( 0, pc + call_open_.size() );
				end_section();
				state_ = State::ToolCall;
				continue;
			}
			size_t hold = partial_suffix( buf_, call_open_ );
			for ( const std::string& w : wrappers_ ) hold = std::max( hold, partial_suffix( buf_, w ) );
			if ( !emitted_any_ && !saw_tool_call_ ) hold = std::max( hold, partial_suffix( buf_, kThinkOpen ) );
			const size_t safe = final ? buf_.size() : buf_.size() - hold;
			emit_text( OutputEvent::Type::Content, buf_.substr( 0, safe ), events );
			buf_.erase( 0, safe );
			return;
		}
		// State::ToolCall
		const size_t p = buf_.find( call_close_ );
		if ( p == std::string::npos ) {
			if ( final ) {
				// unterminated call: hand the raw text back as content rather than dropping it
				emit_text( OutputEvent::Type::Content, call_open_ + buf_, events );
				buf_.clear();
			}
			return;
		}
		ToolCallOut call;
		const std::string body = buf_.substr( 0, p );
		buf_.erase( 0, p + call_close_.size() );
		if ( parse_tool_call( body, &call ) ) {
			OutputEvent ev;
			ev.type = OutputEvent::Type::ToolCall;
			ev.call = std::move( call );
			events.push_back( std::move( ev ) );
			saw_tool_call_ = true;
		}
		else {
			emit_text( OutputEvent::Type::Content, call_open_ + body + call_close_, events );
		}
		end_section();
		state_ = State::Content;
	}
}

std::string OutputParser::decode_value( const std::string& tool, const std::string& key, const std::string& raw ) const {
	// returns the JSON text of the argument value
	std::string type;
	auto t = schemas_.find( tool );
	if ( t != schemas_.end() ) {
		auto k = t->second.find( key );
		if ( k != t->second.end() ) type = k->second;
	}
	if ( type != "string" ) {
		std::string trimmed = raw;
		size_t b = 0, e = trimmed.size();
		while ( b < e && is_ws( trimmed[b] ) ) ++b;
		while ( e > b && is_ws( trimmed[e - 1] ) ) --e;
		trimmed = trimmed.substr( b, e - b );
		auto j = nlohmann::json::parse( trimmed, nullptr, false );
		if ( !trimmed.empty() && !j.is_discarded() ) return j.dump();
	}
	return nlohmann::json( raw ).dump();
}

bool OutputParser::parse_tool_call( const std::string& body, ToolCallOut* out ) const {
	std::string name;
	std::vector<std::pair<std::string, std::string>> raw_args;
	if ( family_ == engine::loader::ModelFamily::DEEPSEEK ) {
		// body: ` name="n">\n<｜DSML｜ parameter name="k" string="true|false">v</｜DSML｜ parameter>\n...`; string="false" values are JSON
		const std::string popen = "<｜DSML｜ parameter name=\"", pclose = "</｜DSML｜ parameter>";
		const size_t ns = body.find( "name=\"" );
		const size_t ne = ns == std::string::npos ? ns : body.find( '"', ns + 6 );
		if ( ns == std::string::npos || ne == std::string::npos ) return false;
		name = body.substr( ns + 6, ne - ns - 6 );
		std::string args = "{";
		size_t pos = ne;
		for ( int k = 0;; ++k ) {
			const size_t ps = body.find( popen, pos );
			if ( ps == std::string::npos ) break;
			const size_t ke = body.find( '"', ps + popen.size() );
			if ( ke == std::string::npos ) return false;
			const std::string key = body.substr( ps + popen.size(), ke - ps - popen.size() );
			const size_t ss = body.find( "string=\"", ke );
			const size_t gt = body.find( '>', ke );
			if ( gt == std::string::npos ) return false;
			const bool is_str = ss != std::string::npos && ss < gt && body.compare( ss + 8, 4, "true" ) == 0;
			size_t ve = body.find( pclose, gt );
			const bool last_open = ve == std::string::npos;   // a missing close on the last parameter: the value runs to the end
			if ( last_open ) ve = body.size();
			const std::string raw = body.substr( gt + 1, ve - gt - 1 );
			std::string value = nlohmann::json( raw ).dump();
			if ( !is_str ) {
				const auto j = nlohmann::json::parse( raw, nullptr, false );
				if ( !j.is_discarded() ) value = j.dump();
			}
			args += ( k ? "," : "" ) + nlohmann::json( key ).dump() + ":" + value;
			if ( last_open ) break;
			pos = ve + pclose.size();
		}
		out->name = name;
		out->arguments = args + "}";
		return !name.empty();
	}
	if ( family_ == engine::loader::ModelFamily::GLM ) {
		const size_t k = body.find( "<arg_key>" );
		name = body.substr( 0, k );
		size_t pos = k;
		while ( pos != std::string::npos && pos < body.size() ) {
			const size_t ks = body.find( "<arg_key>", pos );
			if ( ks == std::string::npos ) break;
			const size_t ke = body.find( "</arg_key>", ks );
			const size_t vs = body.find( "<arg_value>", ke == std::string::npos ? ks : ke );
			size_t ve = vs == std::string::npos ? std::string::npos : body.find( "</arg_value>", vs );
			if ( ke == std::string::npos || vs == std::string::npos ) return false;
			// a missing </arg_value> on the last argument is tolerated: the value runs to the end of the call
			const bool last_open = ve == std::string::npos;
			if ( last_open ) ve = body.size();
			raw_args.emplace_back( body.substr( ks + 9, ke - ks - 9 ), body.substr( vs + 11, ve - vs - 11 ) );
			pos = last_open ? body.size() : ve + 12;
		}
	}
	else {
		const size_t fs = body.find( "<function=" );
		if ( fs == std::string::npos ) return false;
		const size_t fe = body.find( '>', fs );
		if ( fe == std::string::npos ) return false;
		name = body.substr( fs + 10, fe - fs - 10 );
		size_t pos = fe + 1;
		for ( ;; ) {
			const size_t ps = body.find( "<parameter=", pos );
			if ( ps == std::string::npos ) break;
			const size_t pe = body.find( '>', ps );
			if ( pe == std::string::npos ) return false;
			size_t ve = body.find( "</parameter>", pe );
			// a missing </parameter> on the last parameter is tolerated: the value runs to </function> or the end
			bool last_open = false;
			if ( ve == std::string::npos ) {
				last_open = true;
				ve = body.find( "</function>", pe );
				if ( ve == std::string::npos ) ve = body.size();
			}
			std::string v = body.substr( pe + 1, ve - pe - 1 );
			// the template wraps each value in one newline on either side
			if ( !v.empty() && v.front() == '\n' ) v.erase( 0, 1 );
			if ( !v.empty() && v.back() == '\n' ) v.pop_back();
			raw_args.emplace_back( body.substr( ps + 11, pe - ps - 11 ), std::move( v ) );
			if ( last_open ) break;
			pos = ve + 12;
		}
	}
	while ( !name.empty() && is_ws( name.back() ) ) name.pop_back();
	while ( !name.empty() && is_ws( name.front() ) ) name.erase( 0, 1 );
	if ( name.empty() ) return false;
	std::string args = "{";
	for ( size_t k = 0; k < raw_args.size(); ++k ) {
		if ( k ) args += ",";
		args += nlohmann::json( raw_args[k].first ).dump() + ":" + decode_value( name, raw_args[k].first, raw_args[k].second );
	}
	args += "}";
	out->name = name;
	out->arguments = args;
	return true;
}

} // namespace engine::server
