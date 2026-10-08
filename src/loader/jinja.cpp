#include "engine/loader/jinja.h"

#define JSON_NOEXCEPTION 1
#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

namespace engine::loader::jinja {

// ---------------------------------------------------------------------------------------------------------------------
// Value
// ---------------------------------------------------------------------------------------------------------------------

Value Value::none() { Value v; v.kind = Kind::None; return v; }
Value Value::boolean( bool x ) { Value v; v.kind = Kind::Bool; v.b = x; return v; }
Value Value::integer( int64_t x ) { Value v; v.kind = Kind::Int; v.i = x; return v; }
Value Value::number( double x ) { Value v; v.kind = Kind::Float; v.f = x; return v; }
Value Value::string( std::string x ) { Value v; v.kind = Kind::Str; v.s = std::move( x ); return v; }
Value Value::make_list() { Value v; v.kind = Kind::List; v.list = std::make_shared<std::vector<Value>>(); return v; }
Value Value::make_map() { Value v; v.kind = Kind::Map; v.map = std::make_shared<MapData>(); return v; }

bool Value::truthy() const {
	switch ( kind ) {
		case Kind::Undefined: return false;
		case Kind::None: return false;
		case Kind::Bool: return b;
		case Kind::Int: return i != 0;
		case Kind::Float: return f != 0.0;
		case Kind::Str: return !s.empty();
		case Kind::List: return !list->empty();
		case Kind::Map: return !map->items.empty();
	}
	return false;
}

void Value::set( const std::string& key, Value v ) {
	for ( auto& kv : map->items ) {
		if ( kv.first == key ) {
			kv.second = std::move( v );
			return;
		}
	}
	map->items.emplace_back( key, std::move( v ) );
}

void Value::push( Value v ) { list->push_back( std::move( v ) ); }

const Value* Value::find( const std::string& key ) const {
	if ( kind != Kind::Map ) return nullptr;
	for ( const auto& kv : map->items ) {
		if ( kv.first == key ) return &kv.second;
	}
	return nullptr;
}

namespace {

Value convert_json( const nlohmann::ordered_json& j ) {
	if ( j.is_null() ) return Value::none();
	if ( j.is_boolean() ) return Value::boolean( j.get<bool>() );
	if ( j.is_number_integer() ) return Value::integer( j.get<int64_t>() );
	if ( j.is_number() ) return Value::number( j.get<double>() );
	if ( j.is_string() ) return Value::string( j.get<std::string>() );
	if ( j.is_array() ) {
		Value l = Value::make_list();
		for ( const auto& e : j ) l.push( convert_json( e ) );
		return l;
	}
	Value m = Value::make_map();
	for ( auto it = j.begin(); it != j.end(); ++it ) m.set( it.key(), convert_json( it.value() ) );
	return m;
}

std::string float_repr( double x ) {
	if ( std::isnan( x ) ) return "nan";
	if ( std::isinf( x ) ) return x < 0 ? "-inf" : "inf";
	char buf[64];
	if ( x == std::floor( x ) && std::fabs( x ) < 1e16 ) {
		std::snprintf( buf, sizeof( buf ), "%.1f", x );
		return buf;
	}
	auto r = std::to_chars( buf, buf + sizeof( buf ), x );
	return std::string( buf, r.ptr );
}

void json_string( const std::string& s, std::string& out ) {
	out.push_back( '"' );
	for ( unsigned char c : s ) {
		switch ( c ) {
			case '"': out += "\\\""; break;
			case '\\': out += "\\\\"; break;
			case '\n': out += "\\n"; break;
			case '\r': out += "\\r"; break;
			case '\t': out += "\\t"; break;
			case '\b': out += "\\b"; break;
			case '\f': out += "\\f"; break;
			default: {
				if ( c < 0x20 ) {
					char buf[8];
					std::snprintf( buf, sizeof( buf ), "\\u%04x", c );
					out += buf;
				}
				else {
					out.push_back( static_cast<char>( c ) );
				}
			} break;
		}
	}
	out.push_back( '"' );
}

void dump_json( const Value& v, std::string& out ) {
	switch ( v.kind ) {
		case Value::Kind::Undefined:
		case Value::Kind::None: out += "null"; break;
		case Value::Kind::Bool: out += v.b ? "true" : "false"; break;
		case Value::Kind::Int: out += std::to_string( v.i ); break;
		case Value::Kind::Float: out += float_repr( v.f ); break;
		case Value::Kind::Str: json_string( v.s, out ); break;
		case Value::Kind::List: {
			out.push_back( '[' );
			for ( size_t k = 0; k < v.list->size(); ++k ) {
				if ( k ) out += ", ";
				dump_json( ( *v.list )[k], out );
			}
			out.push_back( ']' );
		} break;
		case Value::Kind::Map: {
			out.push_back( '{' );
			bool first = true;
			for ( const auto& kv : v.map->items ) {
				if ( !first ) out += ", ";
				first = false;
				json_string( kv.first, out );
				out += ": ";
				dump_json( kv.second, out );
			}
			out.push_back( '}' );
		} break;
	}
}

// Python repr/str of a value
void py_str( const Value& v, std::string& out, bool quote_strings ) {
	switch ( v.kind ) {
		case Value::Kind::Undefined: break;
		case Value::Kind::None: out += "None"; break;
		case Value::Kind::Bool: out += v.b ? "True" : "False"; break;
		case Value::Kind::Int: out += std::to_string( v.i ); break;
		case Value::Kind::Float: out += float_repr( v.f ); break;
		case Value::Kind::Str: {
			if ( quote_strings ) {
				out.push_back( '\'' );
				out += v.s;
				out.push_back( '\'' );
			}
			else {
				out += v.s;
			}
		} break;
		case Value::Kind::List: {
			out.push_back( '[' );
			for ( size_t k = 0; k < v.list->size(); ++k ) {
				if ( k ) out += ", ";
				py_str( ( *v.list )[k], out, true );
			}
			out.push_back( ']' );
		} break;
		case Value::Kind::Map: {
			out.push_back( '{' );
			bool first = true;
			for ( const auto& kv : v.map->items ) {
				if ( !first ) out += ", ";
				first = false;
				out += "'" + kv.first + "': ";
				py_str( kv.second, out, true );
			}
			out.push_back( '}' );
		} break;
	}
}

bool is_number( const Value& v ) { return v.kind == Value::Kind::Int || v.kind == Value::Kind::Float || v.kind == Value::Kind::Bool; }
double as_double( const Value& v ) { return v.kind == Value::Kind::Float ? v.f : v.kind == Value::Kind::Bool ? ( v.b ? 1.0 : 0.0 ) : static_cast<double>( v.i ); }
int64_t as_int( const Value& v ) { return v.kind == Value::Kind::Float ? static_cast<int64_t>( v.f ) : v.kind == Value::Kind::Bool ? ( v.b ? 1 : 0 ) : v.i; }

bool values_equal( const Value& a, const Value& b ) {
	if ( is_number( a ) && is_number( b ) ) {
		if ( a.kind == Value::Kind::Float || b.kind == Value::Kind::Float ) return as_double( a ) == as_double( b );
		return as_int( a ) == as_int( b );
	}
	if ( a.kind != b.kind ) return false;
	switch ( a.kind ) {
		case Value::Kind::Undefined:
		case Value::Kind::None: return true;
		case Value::Kind::Str: return a.s == b.s;
		case Value::Kind::List: {
			if ( a.list->size() != b.list->size() ) return false;
			for ( size_t k = 0; k < a.list->size(); ++k ) {
				if ( !values_equal( ( *a.list )[k], ( *b.list )[k] ) ) return false;
			}
			return true;
		}
		case Value::Kind::Map: {
			if ( a.map->items.size() != b.map->items.size() ) return false;
			for ( const auto& kv : a.map->items ) {
				const Value* o = b.find( kv.first );
				if ( !o || !values_equal( kv.second, *o ) ) return false;
			}
			return true;
		}
		default: return false;
	}
}

std::string trim_ws( const std::string& s, bool left, bool right, const std::string& chars = " \t\n\r\f\v" ) {
	size_t b = 0, e = s.size();
	if ( left ) while ( b < e && chars.find( s[b] ) != std::string::npos ) ++b;
	if ( right ) while ( e > b && chars.find( s[e - 1] ) != std::string::npos ) --e;
	return s.substr( b, e - b );
}

} // namespace

Value from_json( const std::string& text ) {
	auto j = nlohmann::ordered_json::parse( text, nullptr, false );
	if ( j.is_discarded() ) return Value();
	return convert_json( j );
}

std::string to_json( const Value& v ) {
	std::string out;
	dump_json( v, out );
	return out;
}

// ---------------------------------------------------------------------------------------------------------------------
// Lexing: template text -> segments, then tag content -> tokens
// ---------------------------------------------------------------------------------------------------------------------

namespace {

enum class SegType { Text, Var, Block };

struct Segment {
	SegType type = SegType::Text;
	std::string text;
};

// Returns the index of the closing delimiter, honoring quoted strings; npos when unterminated.
size_t find_tag_end( const std::string& src, size_t from, const char* close ) {
	const size_t clen = std::strlen( close );
	char quote = 0;
	for ( size_t k = from; k + clen <= src.size(); ++k ) {
		const char c = src[k];
		if ( quote ) {
			if ( c == '\\' ) ++k;
			else if ( c == quote ) quote = 0;
			continue;
		}
		if ( c == '\'' || c == '"' ) {
			quote = c;
			continue;
		}
		if ( src.compare( k, clen, close ) == 0 ) return k;
	}
	return std::string::npos;
}

bool split_segments( const std::string& src, std::vector<Segment>& segs, std::string* err ) {
	// trim_blocks and lstrip_blocks are on (transformers' chat template environment)
	size_t pos = 0;
	std::string text;
	bool strip_next_ws = false;
	bool trim_next_newline = false;
	auto flush_text = [&]() {
		if ( strip_next_ws ) {
			text = trim_ws( text, true, false );
			strip_next_ws = false;
		}
		if ( trim_next_newline ) {
			if ( !text.empty() && text[0] == '\n' ) text.erase( 0, 1 );
			else if ( text.size() > 1 && text[0] == '\r' && text[1] == '\n' ) text.erase( 0, 2 );
			trim_next_newline = false;
		}
		if ( !text.empty() ) segs.push_back( { SegType::Text, text } );
		text.clear();
	};
	while ( pos < src.size() ) {
		size_t open = std::string::npos;
		for ( size_t k = pos; k + 1 < src.size(); ++k ) {
			if ( src[k] == '{' && ( src[k + 1] == '{' || src[k + 1] == '%' || src[k + 1] == '#' ) ) {
				open = k;
				break;
			}
		}
		if ( open == std::string::npos ) {
			text.append( src, pos, std::string::npos );
			pos = src.size();
			break;
		}
		text.append( src, pos, open - pos );
		const char kind = src[open + 1];
		const char* close = kind == '{' ? "}}" : kind == '%' ? "%}" : "#}";
		const size_t end = find_tag_end( src, open + 2, close );
		if ( end == std::string::npos ) {
			if ( err ) *err = "unterminated template tag";
			return false;
		}
		std::string inner = src.substr( open + 2, end - ( open + 2 ) );
		bool left_strip = false, right_strip = false, left_keep = false;
		if ( !inner.empty() && inner[0] == '-' ) {
			left_strip = true;
			inner.erase( 0, 1 );
		}
		else if ( !inner.empty() && inner[0] == '+' ) {
			left_keep = true;
			inner.erase( 0, 1 );
		}
		if ( !inner.empty() && inner.back() == '-' ) {
			right_strip = true;
			inner.pop_back();
		}
		if ( left_strip ) {
			text = trim_ws( text, false, true );
		}
		else if ( kind != '{' && !left_keep ) {
			// lstrip_blocks: drop blanks between the start of the source line and the tag
			size_t b = open;
			while ( b > 0 && ( src[b - 1] == ' ' || src[b - 1] == '\t' ) ) --b;
			if ( ( b == 0 || src[b - 1] == '\n' ) && open - b <= text.size() ) text.erase( text.size() - ( open - b ) );
		}
		flush_text();
		if ( kind == '{' ) segs.push_back( { SegType::Var, inner } );
		else if ( kind == '%' ) segs.push_back( { SegType::Block, inner } );
		// comments produce no segment
		if ( right_strip ) strip_next_ws = true;
		else if ( kind != '{' ) trim_next_newline = true;
		pos = end + 2;
	}
	flush_text();
	return true;
}

enum class TokType { Name, Int, Float, Str, Op, End };

struct Token {
	TokType type = TokType::End;
	std::string text;
	int64_t i = 0;
	double f = 0.0;
};

bool tokenize( const std::string& s, std::vector<Token>& toks, std::string* err ) {
	size_t k = 0;
	while ( k < s.size() ) {
		const char c = s[k];
		if ( std::isspace( static_cast<unsigned char>( c ) ) ) {
			++k;
			continue;
		}
		Token t;
		if ( std::isalpha( static_cast<unsigned char>( c ) ) || c == '_' ) {
			size_t e = k;
			while ( e < s.size() && ( std::isalnum( static_cast<unsigned char>( s[e] ) ) || s[e] == '_' ) ) ++e;
			t.type = TokType::Name;
			t.text = s.substr( k, e - k );
			k = e;
		}
		else if ( std::isdigit( static_cast<unsigned char>( c ) ) ) {
			size_t e = k;
			bool fl = false;
			while ( e < s.size() && std::isdigit( static_cast<unsigned char>( s[e] ) ) ) ++e;
			if ( e + 1 < s.size() && s[e] == '.' && std::isdigit( static_cast<unsigned char>( s[e + 1] ) ) ) {
				fl = true;
				++e;
				while ( e < s.size() && std::isdigit( static_cast<unsigned char>( s[e] ) ) ) ++e;
			}
			t.text = s.substr( k, e - k );
			if ( fl ) {
				t.type = TokType::Float;
				t.f = std::strtod( t.text.c_str(), nullptr );
			}
			else {
				t.type = TokType::Int;
				t.i = std::strtoll( t.text.c_str(), nullptr, 10 );
			}
			k = e;
		}
		else if ( c == '\'' || c == '"' ) {
			std::string v;
			size_t e = k + 1;
			while ( e < s.size() && s[e] != c ) {
				if ( s[e] == '\\' && e + 1 < s.size() ) {
					++e;
					switch ( s[e] ) {
						case 'n': v.push_back( '\n' ); break;
						case 't': v.push_back( '\t' ); break;
						case 'r': v.push_back( '\r' ); break;
						case '\\': v.push_back( '\\' ); break;
						case '\'': v.push_back( '\'' ); break;
						case '"': v.push_back( '"' ); break;
						default: v.push_back( '\\' ); v.push_back( s[e] ); break;
					}
				}
				else {
					v.push_back( s[e] );
				}
				++e;
			}
			if ( e >= s.size() ) {
				if ( err ) *err = "unterminated string literal";
				return false;
			}
			t.type = TokType::Str;
			t.text = v;
			k = e + 1;
		}
		else {
			static const char* ops2[] = { "==", "!=", "<=", ">=", "//", "**" };
			t.type = TokType::Op;
			bool two = false;
			if ( k + 1 < s.size() ) {
				for ( const char* o : ops2 ) {
					if ( s[k] == o[0] && s[k + 1] == o[1] ) {
						t.text = o;
						k += 2;
						two = true;
						break;
					}
				}
			}
			if ( !two ) {
				if ( !std::strchr( "+-*/%<>=|.,:()[]{}~", c ) ) {
					if ( err ) *err = std::string( "unexpected character '" ) + c + "'";
					return false;
				}
				t.text = std::string( 1, c );
				++k;
			}
		}
		toks.push_back( std::move( t ) );
	}
	toks.push_back( Token() );
	return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// AST
// ---------------------------------------------------------------------------------------------------------------------

struct Node;
using NodeP = std::shared_ptr<Node>;
using Body = std::vector<NodeP>;

enum class NT {
	// expressions
	Lit, Name, Attr, Index, Slice, Call, Filter, Test, Unary, Binary, Cond, ListLit, DictLit,
	// statements
	Text, Output, If, For, Set, Macro, Break, Continue
};

struct Node {
	NT type = NT::Lit;
	std::string name;                              // Name, Attr, Filter, Test, Macro, operator text
	Value lit;                                     // Lit
	std::vector<NodeP> kids;                       // operands / args
	std::vector<std::pair<std::string, NodeP>> kw; // keyword args, macro defaults
	bool negate = false;                           // Test: "is not"
	// statements
	std::vector<std::pair<NodeP, Body>> branches;  // If: condition + body (null condition = else)
	Body body, else_body;                          // For
	std::vector<std::string> targets;              // For / Set names
	NodeP expr;                                    // For iterable, Set value
	NodeP cond;                                    // For filter
	std::vector<std::string> params;               // Macro
};

NodeP make( NT t ) {
	auto n = std::make_shared<Node>();
	n->type = t;
	return n;
}

class ExprParser {
public:
	ExprParser( const std::vector<Token>& toks, std::string* err ) : toks_( toks ), err_( err ) {}

	size_t pos = 0;
	bool failed = false;

	const Token& peek( size_t ahead = 0 ) const { return toks_[std::min( pos + ahead, toks_.size() - 1 )]; }
	bool at_end() const { return peek().type == TokType::End; }
	bool is_op( const char* o, size_t ahead = 0 ) const { return peek( ahead ).type == TokType::Op && peek( ahead ).text == o; }
	bool is_name( const char* n, size_t ahead = 0 ) const { return peek( ahead ).type == TokType::Name && peek( ahead ).text == n; }

	void fail( const std::string& msg ) {
		if ( !failed && err_ ) *err_ = msg;
		failed = true;
	}

	bool accept_op( const char* o ) {
		if ( is_op( o ) ) {
			++pos;
			return true;
		}
		return false;
	}

	bool accept_name( const char* n ) {
		if ( is_name( n ) ) {
			++pos;
			return true;
		}
		return false;
	}

	NodeP parse_expression() { return parse_cond(); }

	NodeP parse_cond() {
		NodeP e = parse_or();
		if ( failed ) return e;
		while ( is_name( "if" ) ) {
			++pos;
			NodeP c = parse_or();
			NodeP alt;
			if ( accept_name( "else" ) ) alt = parse_cond();
			NodeP n = make( NT::Cond );
			n->kids = { c, e, alt };
			e = n;
		}
		return e;
	}

	NodeP parse_or() {
		NodeP l = parse_and();
		while ( !failed && is_name( "or" ) ) {
			++pos;
			NodeP n = make( NT::Binary );
			n->name = "or";
			n->kids = { l, parse_and() };
			l = n;
		}
		return l;
	}

	NodeP parse_and() {
		NodeP l = parse_not();
		while ( !failed && is_name( "and" ) ) {
			++pos;
			NodeP n = make( NT::Binary );
			n->name = "and";
			n->kids = { l, parse_not() };
			l = n;
		}
		return l;
	}

	NodeP parse_not() {
		if ( is_name( "not" ) ) {
			++pos;
			NodeP n = make( NT::Unary );
			n->name = "not";
			n->kids = { parse_not() };
			return n;
		}
		return parse_compare();
	}

	NodeP parse_compare() {
		NodeP l = parse_concat();
		while ( !failed ) {
			std::string op;
			if ( peek().type == TokType::Op && ( peek().text == "==" || peek().text == "!=" || peek().text == "<" || peek().text == ">" || peek().text == "<=" || peek().text == ">=" ) ) {
				op = peek().text;
				++pos;
			}
			else if ( is_name( "in" ) ) {
				op = "in";
				++pos;
			}
			else if ( is_name( "not" ) && is_name( "in", 1 ) ) {
				op = "not in";
				pos += 2;
			}
			else {
				break;
			}
			NodeP n = make( NT::Binary );
			n->name = op;
			n->kids = { l, parse_concat() };
			l = n;
		}
		return l;
	}

	NodeP parse_concat() {
		NodeP l = parse_add();
		while ( !failed && is_op( "~" ) ) {
			++pos;
			NodeP n = make( NT::Binary );
			n->name = "~";
			n->kids = { l, parse_add() };
			l = n;
		}
		return l;
	}

	NodeP parse_add() {
		NodeP l = parse_mul();
		while ( !failed && ( is_op( "+" ) || is_op( "-" ) ) ) {
			NodeP n = make( NT::Binary );
			n->name = peek().text;
			++pos;
			n->kids = { l, parse_mul() };
			l = n;
		}
		return l;
	}

	NodeP parse_mul() {
		NodeP l = parse_unary();
		while ( !failed && ( is_op( "*" ) || is_op( "/" ) || is_op( "//" ) || is_op( "%" ) ) ) {
			NodeP n = make( NT::Binary );
			n->name = peek().text;
			++pos;
			n->kids = { l, parse_unary() };
			l = n;
		}
		return l;
	}

	NodeP parse_unary() {
		if ( is_op( "-" ) || is_op( "+" ) ) {
			NodeP n = make( NT::Unary );
			n->name = peek().text;
			++pos;
			n->kids = { parse_unary() };
			return n;
		}
		return parse_filters( parse_postfix( parse_primary() ) );
	}

	NodeP parse_primary() {
		const Token t = peek();
		if ( t.type == TokType::Int ) {
			++pos;
			NodeP n = make( NT::Lit );
			n->lit = Value::integer( t.i );
			return n;
		}
		if ( t.type == TokType::Float ) {
			++pos;
			NodeP n = make( NT::Lit );
			n->lit = Value::number( t.f );
			return n;
		}
		if ( t.type == TokType::Str ) {
			++pos;
			std::string s = t.text;
			// adjacent string literals concatenate
			while ( peek().type == TokType::Str ) {
				s += peek().text;
				++pos;
			}
			NodeP n = make( NT::Lit );
			n->lit = Value::string( s );
			return n;
		}
		if ( t.type == TokType::Name ) {
			++pos;
			NodeP n = make( NT::Lit );
			if ( t.text == "true" || t.text == "True" ) n->lit = Value::boolean( true );
			else if ( t.text == "false" || t.text == "False" ) n->lit = Value::boolean( false );
			else if ( t.text == "none" || t.text == "None" ) n->lit = Value::none();
			else {
				n->type = NT::Name;
				n->name = t.text;
			}
			return n;
		}
		if ( t.type == TokType::Op ) {
			if ( t.text == "(" ) {
				++pos;
				NodeP first = parse_expression();
				if ( is_op( "," ) ) {
					NodeP l = make( NT::ListLit );
					l->kids.push_back( first );
					while ( accept_op( "," ) ) {
						if ( is_op( ")" ) ) break;
						l->kids.push_back( parse_expression() );
					}
					expect_op( ")" );
					return l;
				}
				expect_op( ")" );
				return first;
			}
			if ( t.text == "[" ) {
				++pos;
				NodeP l = make( NT::ListLit );
				while ( !is_op( "]" ) && !at_end() && !failed ) {
					l->kids.push_back( parse_expression() );
					if ( !accept_op( "," ) ) break;
				}
				expect_op( "]" );
				return l;
			}
			if ( t.text == "{" ) {
				++pos;
				NodeP d = make( NT::DictLit );
				while ( !is_op( "}" ) && !at_end() && !failed ) {
					NodeP k = parse_expression();
					expect_op( ":" );
					NodeP v = parse_expression();
					d->kids.push_back( k );
					d->kids.push_back( v );
					if ( !accept_op( "," ) ) break;
				}
				expect_op( "}" );
				return d;
			}
		}
		fail( "unexpected token '" + t.text + "' in expression" );
		return make( NT::Lit );
	}

	void expect_op( const char* o ) {
		if ( !accept_op( o ) ) fail( std::string( "expected '" ) + o + "'" );
	}

	void parse_call_args( NodeP& call ) {
		// after '('
		while ( !is_op( ")" ) && !at_end() && !failed ) {
			if ( peek().type == TokType::Name && is_op( "=", 1 ) ) {
				const std::string key = peek().text;
				pos += 2;
				call->kw.emplace_back( key, parse_expression() );
			}
			else {
				call->kids.push_back( parse_expression() );
			}
			if ( !accept_op( "," ) ) break;
		}
		expect_op( ")" );
	}

	NodeP parse_postfix( NodeP e ) {
		while ( !failed ) {
			if ( is_op( "." ) ) {
				++pos;
				const Token t = peek();
				if ( t.type == TokType::Name ) {
					++pos;
					NodeP n = make( NT::Attr );
					n->name = t.text;
					n->kids = { e };
					e = n;
				}
				else if ( t.type == TokType::Int ) {
					++pos;
					NodeP idx = make( NT::Lit );
					idx->lit = Value::integer( t.i );
					NodeP n = make( NT::Index );
					n->kids = { e, idx };
					e = n;
				}
				else {
					fail( "expected attribute name" );
				}
			}
			else if ( is_op( "[" ) ) {
				++pos;
				NodeP start, stop, step;
				bool is_slice = false;
				if ( !is_op( ":" ) ) start = parse_expression();
				if ( accept_op( ":" ) ) {
					is_slice = true;
					if ( !is_op( ":" ) && !is_op( "]" ) ) stop = parse_expression();
					if ( accept_op( ":" ) ) {
						if ( !is_op( "]" ) ) step = parse_expression();
					}
				}
				expect_op( "]" );
				if ( is_slice ) {
					NodeP n = make( NT::Slice );
					n->kids = { e, start, stop, step };
					e = n;
				}
				else {
					NodeP n = make( NT::Index );
					n->kids = { e, start };
					e = n;
				}
			}
			else if ( is_op( "(" ) ) {
				++pos;
				NodeP n = make( NT::Call );
				n->kids.push_back( e );
				parse_call_args( n );
				e = n;
			}
			else {
				break;
			}
		}
		return e;
	}

	NodeP parse_filters( NodeP e ) {
		while ( !failed ) {
			if ( is_op( "|" ) ) {
				++pos;
				if ( peek().type != TokType::Name ) {
					fail( "expected filter name" );
					break;
				}
				NodeP n = make( NT::Filter );
				n->name = peek().text;
				++pos;
				// dotted filter names are not used by chat templates
				n->kids.push_back( e );
				if ( is_op( "(" ) ) {
					++pos;
					parse_call_args( n );
				}
				e = n;
			}
			else if ( is_name( "is" ) ) {
				++pos;
				NodeP n = make( NT::Test );
				if ( accept_name( "not" ) ) n->negate = true;
				if ( peek().type != TokType::Name ) {
					fail( "expected test name" );
					break;
				}
				n->name = peek().text;
				++pos;
				n->kids.push_back( e );
				if ( is_op( "(" ) ) {
					++pos;
					parse_call_args( n );
				}
				e = n;
			}
			else {
				break;
			}
		}
		return e;
	}

private:
	const std::vector<Token>& toks_;
	std::string* err_;
};

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// Template: parsing of segments into statements
// ---------------------------------------------------------------------------------------------------------------------

struct Template::Impl {
	Body root;
	bool ok = false;
};

namespace {

class StmtParser {
public:
	StmtParser( const std::vector<Segment>& segs, std::string* err ) : segs_( segs ), err_( err ) {}

	bool parse( Body& root ) {
		std::string stop;
		parse_body( root, {}, stop );
		return !failed_ && stop.empty();
	}

private:
	const std::vector<Segment>& segs_;
	std::string* err_;
	size_t pos_ = 0;
	bool failed_ = false;
	std::vector<Token> last_tail_; // tokens of the terminating tag (for 'elif')

	void fail( const std::string& msg ) {
		if ( !failed_ && err_ ) *err_ = msg;
		failed_ = true;
	}

	static std::string first_word( const std::string& s ) {
		size_t b = 0;
		while ( b < s.size() && std::isspace( static_cast<unsigned char>( s[b] ) ) ) ++b;
		size_t e = b;
		while ( e < s.size() && ( std::isalnum( static_cast<unsigned char>( s[e] ) ) || s[e] == '_' ) ) ++e;
		return s.substr( b, e - b );
	}

	// Parses statements until a block tag whose keyword is in stops; stop receives that keyword (empty at end of input).
	void parse_body( Body& body, const std::vector<std::string>& stops, std::string& stop ) {
		stop.clear();
		while ( pos_ < segs_.size() && !failed_ ) {
			const Segment& seg = segs_[pos_];
			if ( seg.type == SegType::Text ) {
				NodeP n = make( NT::Text );
				n->name = seg.text;
				body.push_back( n );
				++pos_;
				continue;
			}
			if ( seg.type == SegType::Var ) {
				std::vector<Token> toks;
				if ( !tokenize( seg.text, toks, err_ ) ) {
					failed_ = true;
					return;
				}
				ExprParser ep( toks, err_ );
				NodeP e = ep.parse_expression();
				if ( !ep.failed && !ep.at_end() ) ep.fail( "unexpected trailing tokens in '{{ " + seg.text + " }}'" );
				if ( ep.failed ) {
					failed_ = true;
					return;
				}
				NodeP n = make( NT::Output );
				n->expr = e;
				body.push_back( n );
				++pos_;
				continue;
			}
			const std::string kw = first_word( seg.text );
			if ( std::find( stops.begin(), stops.end(), kw ) != stops.end() ) {
				stop = kw;
				return;
			}
			parse_statement( body, kw, seg.text );
		}
	}

	std::vector<Token> tokens_after_keyword( const std::string& text, const std::string& kw ) {
		std::vector<Token> toks;
		const size_t at = text.find( kw );
		if ( !tokenize( text.substr( at + kw.size() ), toks, err_ ) ) failed_ = true;
		return toks;
	}

	void parse_statement( Body& body, const std::string& kw, const std::string& text ) {
		++pos_;
		if ( kw == "if" ) {
			NodeP n = make( NT::If );
			std::vector<Token> toks = tokens_after_keyword( text, "if" );
			ExprParser ep( toks, err_ );
			NodeP cond = ep.parse_expression();
			if ( ep.failed ) { failed_ = true; return; }
			Body b;
			std::string stop;
			parse_body( b, { "elif", "else", "endif" }, stop );
			n->branches.emplace_back( cond, std::move( b ) );
			while ( !failed_ && ( stop == "elif" || stop == "else" ) ) {
				const std::string tag = segs_[pos_].text;
				++pos_;
				NodeP c;
				if ( stop == "elif" ) {
					std::vector<Token> t2 = tokens_after_keyword( tag, "elif" );
					ExprParser ep2( t2, err_ );
					c = ep2.parse_expression();
					if ( ep2.failed ) { failed_ = true; return; }
				}
				Body b2;
				parse_body( b2, { "elif", "else", "endif" }, stop );
				n->branches.emplace_back( c, std::move( b2 ) );
			}
			if ( stop != "endif" ) { fail( "unterminated if" ); return; }
			++pos_;
			body.push_back( n );
			return;
		}
		if ( kw == "for" ) {
			NodeP n = make( NT::For );
			std::vector<Token> toks = tokens_after_keyword( text, "for" );
			size_t k = 0;
			while ( k < toks.size() && toks[k].type == TokType::Name && !( toks[k].text == "in" ) ) {
				n->targets.push_back( toks[k].text );
				++k;
				if ( k < toks.size() && toks[k].type == TokType::Op && toks[k].text == "," ) ++k;
			}
			if ( n->targets.empty() || k >= toks.size() || toks[k].text != "in" ) { fail( "malformed for statement" ); return; }
			++k;
			std::vector<Token> rest( toks.begin() + static_cast<long>( k ), toks.end() );
			ExprParser ep( rest, err_ );
			n->expr = ep.parse_expression();
			if ( !ep.failed && ep.accept_name( "if" ) ) n->cond = ep.parse_or();
			if ( ep.failed ) { failed_ = true; return; }
			std::string stop;
			parse_body( n->body, { "else", "endfor" }, stop );
			if ( stop == "else" ) {
				++pos_;
				parse_body( n->else_body, { "endfor" }, stop );
			}
			if ( stop != "endfor" ) { fail( "unterminated for" ); return; }
			++pos_;
			body.push_back( n );
			return;
		}
		if ( kw == "set" ) {
			NodeP n = make( NT::Set );
			std::vector<Token> toks = tokens_after_keyword( text, "set" );
			// target: name, name.attr, or a comma separated list of names
			size_t eq = 0;
			while ( eq < toks.size() && !( toks[eq].type == TokType::Op && toks[eq].text == "=" ) ) ++eq;
			if ( eq == toks.size() ) { fail( "block set is not supported" ); return; }
			std::vector<Token> target( toks.begin(), toks.begin() + static_cast<long>( eq ) );
			target.push_back( Token() );
			ExprParser tp( target, err_ );
			std::vector<NodeP> names;
			while ( !tp.at_end() && !tp.failed ) {
				names.push_back( tp.parse_postfix( tp.parse_primary() ) );
				if ( !tp.accept_op( "," ) ) break;
			}
			if ( tp.failed || !tp.at_end() || names.empty() ) { fail( "malformed set target" ); return; }
			for ( const NodeP& t : names ) n->kids.push_back( t );
			std::vector<Token> rest( toks.begin() + static_cast<long>( eq ) + 1, toks.end() );
			ExprParser ep( rest, err_ );
			n->expr = ep.parse_expression();
			if ( ep.failed ) { failed_ = true; return; }
			body.push_back( n );
			return;
		}
		if ( kw == "macro" ) {
			NodeP n = make( NT::Macro );
			std::vector<Token> toks = tokens_after_keyword( text, "macro" );
			if ( toks.empty() || toks[0].type != TokType::Name ) { fail( "malformed macro" ); return; }
			n->name = toks[0].text;
			ExprParser ep( toks, err_ );
			ep.pos = 1;
			if ( ep.accept_op( "(" ) ) {
				while ( !ep.is_op( ")" ) && !ep.at_end() && !ep.failed ) {
					if ( ep.peek().type != TokType::Name ) { ep.fail( "malformed macro parameter" ); break; }
					const std::string pn = ep.peek().text;
					++ep.pos;
					n->params.push_back( pn );
					if ( ep.accept_op( "=" ) ) n->kw.emplace_back( pn, ep.parse_expression() );
					if ( !ep.accept_op( "," ) ) break;
				}
				ep.expect_op( ")" );
			}
			if ( ep.failed ) { failed_ = true; return; }
			std::string stop;
			parse_body( n->body, { "endmacro" }, stop );
			if ( stop != "endmacro" ) { fail( "unterminated macro" ); return; }
			++pos_;
			body.push_back( n );
			return;
		}
		if ( kw == "break" ) { body.push_back( make( NT::Break ) ); return; }
		if ( kw == "continue" ) { body.push_back( make( NT::Continue ) ); return; }
		fail( "unsupported statement '" + kw + "'" );
	}
};

// ---------------------------------------------------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------------------------------------------------

struct Frame {
	std::vector<std::pair<std::string, Value>> vars;
	Frame* parent = nullptr;
};

enum class Flow { Normal, Break, Continue };

class Evaluator {
public:
	Evaluator( const Value& vars, std::string* error ) : error_( error ) {
		global_.parent = nullptr;
		if ( vars.kind == Value::Kind::Map ) {
			for ( const auto& kv : vars.map->items ) global_.vars.emplace_back( kv.first, kv.second );
		}
		cur_ = &global_;
	}

	bool failed = false;

	void fail( const std::string& msg ) {
		if ( !failed && error_ ) *error_ = msg;
		failed = true;
	}

	Flow exec_body( const Body& body, std::string& out ) {
		for ( const NodeP& n : body ) {
			if ( failed ) return Flow::Normal;
			const Flow f = exec( *n, out );
			if ( f != Flow::Normal ) return f;
		}
		return Flow::Normal;
	}

private:
	std::string* error_;
	Frame global_;
	Frame* cur_ = nullptr;
	std::map<std::string, NodeP> macros_;
	int depth_ = 0;

	Value lookup( const std::string& name ) {
		for ( Frame* f = cur_; f; f = f->parent ) {
			for ( auto it = f->vars.rbegin(); it != f->vars.rend(); ++it ) {
				if ( it->first == name ) return it->second;
			}
		}
		return Value();
	}

	void assign( Frame& f, const std::string& name, Value v ) {
		for ( auto& kv : f.vars ) {
			if ( kv.first == name ) {
				kv.second = std::move( v );
				return;
			}
		}
		f.vars.emplace_back( name, std::move( v ) );
	}

	static std::string to_str( const Value& v ) {
		std::string s;
		py_str( v, s, false );
		return s;
	}

	Flow exec( const Node& n, std::string& out ) {
		switch ( n.type ) {
			case NT::Text: out += n.name; return Flow::Normal;
			case NT::Output: {
				const Value v = eval( *n.expr );
				if ( !failed ) out += to_str( v );
				return Flow::Normal;
			}
			case NT::If: {
				for ( const auto& br : n.branches ) {
					if ( !br.first || eval( *br.first ).truthy() ) {
						if ( failed ) return Flow::Normal;
						return exec_body( br.second, out );
					}
					if ( failed ) return Flow::Normal;
				}
				return Flow::Normal;
			}
			case NT::For: return exec_for( n, out );
			case NT::Set: {
				const Value v = eval( *n.expr );
				if ( failed ) return Flow::Normal;
				if ( n.kids.size() == 1 ) {
					assign_target( *n.kids[0], v );
				}
				else {
					if ( v.kind != Value::Kind::List || v.list->size() != n.kids.size() ) {
						fail( "cannot unpack value in set" );
						return Flow::Normal;
					}
					for ( size_t k = 0; k < n.kids.size(); ++k ) assign_target( *n.kids[k], ( *v.list )[k] );
				}
				return Flow::Normal;
			}
			case NT::Macro: {
				macros_[n.name] = std::make_shared<Node>( n );
				return Flow::Normal;
			}
			case NT::Break: return Flow::Break;
			case NT::Continue: return Flow::Continue;
			default: return Flow::Normal;
		}
	}

	void assign_target( const Node& t, const Value& v ) {
		if ( t.type == NT::Name ) {
			assign( *cur_, t.name, v );
		}
		else if ( t.type == NT::Attr ) {
			Value obj = eval( *t.kids[0] );
			if ( obj.kind != Value::Kind::Map || !obj.map->is_namespace ) {
				fail( "can only assign attributes of a namespace" );
				return;
			}
			obj.set( t.name, v );
		}
		else {
			fail( "invalid assignment target" );
		}
	}

	Flow exec_for( const Node& n, std::string& out ) {
		const Value iter = eval( *n.expr );
		if ( failed ) return Flow::Normal;
		std::vector<Value> items;
		if ( iter.kind == Value::Kind::List ) {
			items = *iter.list;
		}
		else if ( iter.kind == Value::Kind::Map ) {
			for ( const auto& kv : iter.map->items ) items.push_back( Value::string( kv.first ) );
		}
		else if ( iter.kind == Value::Kind::Str ) {
			for ( char c : iter.s ) items.push_back( Value::string( std::string( 1, c ) ) );
		}
		else if ( !iter.is_undefined() && iter.kind != Value::Kind::None ) {
			fail( "value is not iterable" );
			return Flow::Normal;
		}
		Frame frame;
		frame.parent = cur_;
		Frame* saved = cur_;
		cur_ = &frame;
		// the filter condition selects the iterated items (loop.length counts the selected ones)
		std::vector<Value> selected;
		if ( n.cond ) {
			for ( const Value& item : items ) {
				bind_targets( n, frame, item );
				if ( eval( *n.cond ).truthy() ) selected.push_back( item );
				if ( failed ) break;
			}
		}
		else {
			selected = std::move( items );
		}
		Flow result = Flow::Normal;
		if ( selected.empty() && !failed ) {
			cur_ = saved;
			Flow f = exec_body( n.else_body, out );
			return f;
		}
		for ( size_t k = 0; k < selected.size() && !failed; ++k ) {
			frame.vars.clear();
			bind_targets( n, frame, selected[k] );
			Value loop = Value::make_map();
			loop.set( "index0", Value::integer( static_cast<int64_t>( k ) ) );
			loop.set( "index", Value::integer( static_cast<int64_t>( k ) + 1 ) );
			loop.set( "revindex0", Value::integer( static_cast<int64_t>( selected.size() - k - 1 ) ) );
			loop.set( "revindex", Value::integer( static_cast<int64_t>( selected.size() - k ) ) );
			loop.set( "first", Value::boolean( k == 0 ) );
			loop.set( "last", Value::boolean( k + 1 == selected.size() ) );
			loop.set( "length", Value::integer( static_cast<int64_t>( selected.size() ) ) );
			if ( k > 0 ) loop.set( "previtem", selected[k - 1] );
			if ( k + 1 < selected.size() ) loop.set( "nextitem", selected[k + 1] );
			frame.vars.emplace_back( "loop", loop );
			const Flow f = exec_body( n.body, out );
			if ( f == Flow::Break ) break;
		}
		cur_ = saved;
		return result;
	}

	void bind_targets( const Node& n, Frame& frame, const Value& item ) {
		if ( n.targets.size() == 1 ) {
			assign( frame, n.targets[0], item );
			return;
		}
		if ( item.kind != Value::Kind::List || item.list->size() != n.targets.size() ) {
			fail( "cannot unpack loop item" );
			return;
		}
		for ( size_t k = 0; k < n.targets.size(); ++k ) assign( frame, n.targets[k], ( *item.list )[k] );
	}

	// -- expressions --------------------------------------------------------------------------------------------------

	Value eval( const Node& n ) {
		if ( failed ) return Value();
		switch ( n.type ) {
			case NT::Lit: return n.lit;
			case NT::Name: return lookup( n.name );
			case NT::ListLit: {
				Value l = Value::make_list();
				for ( const NodeP& k : n.kids ) l.push( eval( *k ) );
				return l;
			}
			case NT::DictLit: {
				Value m = Value::make_map();
				for ( size_t k = 0; k + 1 < n.kids.size(); k += 2 ) m.set( to_str( eval( *n.kids[k] ) ), eval( *n.kids[k + 1] ) );
				return m;
			}
			case NT::Attr: {
				const Value obj = eval( *n.kids[0] );
				if ( obj.kind == Value::Kind::Map ) {
					const Value* v = obj.find( n.name );
					return v ? *v : Value();
				}
				return Value();
			}
			case NT::Index: return eval_index( eval( *n.kids[0] ), eval( *n.kids[1] ) );
			case NT::Slice: return eval_slice( n );
			case NT::Unary: {
				const Value v = eval( *n.kids[0] );
				if ( n.name == "not" ) return Value::boolean( !v.truthy() );
				if ( !is_number( v ) ) { fail( "unary operator on a non-number" ); return Value(); }
				if ( n.name == "-" ) return v.kind == Value::Kind::Float ? Value::number( -v.f ) : Value::integer( -as_int( v ) );
				return v;
			}
			case NT::Binary: return eval_binary( n );
			case NT::Cond: {
				const bool c = eval( *n.kids[0] ).truthy();
				if ( c ) return eval( *n.kids[1] );
				return n.kids[2] ? eval( *n.kids[2] ) : Value();
			}
			case NT::Test: return eval_test( n );
			case NT::Filter: return eval_filter( n );
			case NT::Call: return eval_call( n );
			default: return Value();
		}
	}

	Value eval_index( const Value& obj, const Value& idx ) {
		if ( obj.kind == Value::Kind::List && is_number( idx ) ) {
			int64_t k = as_int( idx );
			const int64_t n = static_cast<int64_t>( obj.list->size() );
			if ( k < 0 ) k += n;
			return ( k >= 0 && k < n ) ? ( *obj.list )[static_cast<size_t>( k )] : Value();
		}
		if ( obj.kind == Value::Kind::Str && is_number( idx ) ) {
			int64_t k = as_int( idx );
			const int64_t n = static_cast<int64_t>( obj.s.size() );
			if ( k < 0 ) k += n;
			return ( k >= 0 && k < n ) ? Value::string( std::string( 1, obj.s[static_cast<size_t>( k )] ) ) : Value();
		}
		if ( obj.kind == Value::Kind::Map && idx.kind == Value::Kind::Str ) {
			const Value* v = obj.find( idx.s );
			return v ? *v : Value();
		}
		return Value();
	}

	Value eval_slice( const Node& n ) {
		const Value obj = eval( *n.kids[0] );
		const Value a = n.kids[1] ? eval( *n.kids[1] ) : Value();
		const Value b = n.kids[2] ? eval( *n.kids[2] ) : Value();
		const Value c = n.kids[3] ? eval( *n.kids[3] ) : Value();
		const int64_t step = is_number( c ) ? as_int( c ) : 1;
		if ( step == 0 ) { fail( "slice step cannot be zero" ); return Value(); }
		int64_t len;
		if ( obj.kind == Value::Kind::List ) len = static_cast<int64_t>( obj.list->size() );
		else if ( obj.kind == Value::Kind::Str ) len = static_cast<int64_t>( obj.s.size() );
		else return Value();
		auto norm = [&]( const Value& v, int64_t dflt ) {
			if ( !is_number( v ) ) return dflt;
			int64_t k = as_int( v );
			if ( k < 0 ) k += len;
			if ( step > 0 ) return std::max<int64_t>( 0, std::min<int64_t>( k, len ) );
			return std::max<int64_t>( -1, std::min<int64_t>( k, len - 1 ) );
		};
		const int64_t start = norm( a, step > 0 ? 0 : len - 1 );
		const int64_t stop = norm( b, step > 0 ? len : -1 );
		Value out = obj.kind == Value::Kind::List ? Value::make_list() : Value::string( "" );
		for ( int64_t k = start; step > 0 ? k < stop : k > stop; k += step ) {
			if ( obj.kind == Value::Kind::List ) out.push( ( *obj.list )[static_cast<size_t>( k )] );
			else out.s.push_back( obj.s[static_cast<size_t>( k )] );
		}
		return out;
	}

	bool contains( const Value& hay, const Value& needle ) {
		if ( hay.kind == Value::Kind::List ) {
			for ( const Value& v : *hay.list ) {
				if ( values_equal( v, needle ) ) return true;
			}
			return false;
		}
		if ( hay.kind == Value::Kind::Str && needle.kind == Value::Kind::Str ) return hay.s.find( needle.s ) != std::string::npos;
		if ( hay.kind == Value::Kind::Map && needle.kind == Value::Kind::Str ) return hay.find( needle.s ) != nullptr;
		return false;
	}

	Value eval_binary( const Node& n ) {
		const std::string& op = n.name;
		if ( op == "and" ) {
			const Value l = eval( *n.kids[0] );
			return l.truthy() ? eval( *n.kids[1] ) : l;
		}
		if ( op == "or" ) {
			const Value l = eval( *n.kids[0] );
			return l.truthy() ? l : eval( *n.kids[1] );
		}
		const Value l = eval( *n.kids[0] );
		const Value r = eval( *n.kids[1] );
		if ( failed ) return Value();
		if ( op == "==" ) return Value::boolean( values_equal( l, r ) );
		if ( op == "!=" ) return Value::boolean( !values_equal( l, r ) );
		if ( op == "in" ) return Value::boolean( contains( r, l ) );
		if ( op == "not in" ) return Value::boolean( !contains( r, l ) );
		if ( op == "~" ) return Value::string( to_str( l ) + to_str( r ) );
		if ( op == "<" || op == ">" || op == "<=" || op == ">=" ) {
			int cmp = 0;
			if ( is_number( l ) && is_number( r ) ) {
				const double a = as_double( l ), b = as_double( r );
				cmp = a < b ? -1 : a > b ? 1 : 0;
			}
			else if ( l.kind == Value::Kind::Str && r.kind == Value::Kind::Str ) {
				cmp = l.s.compare( r.s );
			}
			else {
				fail( "cannot order these values" );
				return Value();
			}
			return Value::boolean( op == "<" ? cmp < 0 : op == ">" ? cmp > 0 : op == "<=" ? cmp <= 0 : cmp >= 0 );
		}
		if ( op == "+" ) {
			if ( l.kind == Value::Kind::Str && r.kind == Value::Kind::Str ) return Value::string( l.s + r.s );
			if ( l.kind == Value::Kind::List && r.kind == Value::Kind::List ) {
				Value out = Value::make_list();
				for ( const Value& v : *l.list ) out.push( v );
				for ( const Value& v : *r.list ) out.push( v );
				return out;
			}
		}
		if ( is_number( l ) && is_number( r ) ) {
			const bool fl = l.kind == Value::Kind::Float || r.kind == Value::Kind::Float;
			if ( op == "+" ) return fl ? Value::number( as_double( l ) + as_double( r ) ) : Value::integer( as_int( l ) + as_int( r ) );
			if ( op == "-" ) return fl ? Value::number( as_double( l ) - as_double( r ) ) : Value::integer( as_int( l ) - as_int( r ) );
			if ( op == "*" ) return fl ? Value::number( as_double( l ) * as_double( r ) ) : Value::integer( as_int( l ) * as_int( r ) );
			if ( op == "/" ) return Value::number( as_double( l ) / as_double( r ) );
			if ( op == "//" || op == "%" ) {
				if ( !fl ) {
					const int64_t a = as_int( l ), b = as_int( r );
					if ( b == 0 ) { fail( "division by zero" ); return Value(); }
					int64_t q = a / b;
					if ( ( a % b != 0 ) && ( ( a < 0 ) != ( b < 0 ) ) ) --q;
					return op == "//" ? Value::integer( q ) : Value::integer( a - q * b );
				}
				const double a = as_double( l ), b = as_double( r );
				return op == "//" ? Value::number( std::floor( a / b ) ) : Value::number( a - std::floor( a / b ) * b );
			}
		}
		if ( op == "*" && l.kind == Value::Kind::Str && is_number( r ) ) {
			std::string s;
			for ( int64_t k = 0; k < as_int( r ); ++k ) s += l.s;
			return Value::string( s );
		}
		fail( "unsupported operands for '" + op + "'" );
		return Value();
	}

	Value eval_test( const Node& n ) {
		const Value v = eval( *n.kids[0] );
		std::vector<Value> args;
		for ( size_t k = 1; k < n.kids.size(); ++k ) args.push_back( eval( *n.kids[k] ) );
		bool r = false;
		const std::string& t = n.name;
		if ( t == "defined" ) r = !v.is_undefined();
		else if ( t == "undefined" ) r = v.is_undefined();
		else if ( t == "none" ) r = v.kind == Value::Kind::None;
		else if ( t == "string" ) r = v.kind == Value::Kind::Str;
		else if ( t == "mapping" ) r = v.kind == Value::Kind::Map;
		else if ( t == "number" ) r = v.kind == Value::Kind::Int || v.kind == Value::Kind::Float;
		else if ( t == "integer" ) r = v.kind == Value::Kind::Int;
		else if ( t == "float" ) r = v.kind == Value::Kind::Float;
		else if ( t == "boolean" ) r = v.kind == Value::Kind::Bool;
		else if ( t == "true" ) r = v.kind == Value::Kind::Bool && v.b;
		else if ( t == "false" ) r = v.kind == Value::Kind::Bool && !v.b;
		else if ( t == "iterable" ) r = v.kind == Value::Kind::List || v.kind == Value::Kind::Str || v.kind == Value::Kind::Map;
		else if ( t == "sequence" ) r = v.kind == Value::Kind::List || v.kind == Value::Kind::Str;
		else if ( t == "eq" || t == "equalto" ) r = !args.empty() && values_equal( v, args[0] );
		else if ( t == "ne" ) r = !args.empty() && !values_equal( v, args[0] );
		else if ( t == "in" ) r = !args.empty() && contains( args[0], v );
		else { fail( "unknown test '" + t + "'" ); return Value(); }
		return Value::boolean( n.negate ? !r : r );
	}

	Value eval_filter( const Node& n ) {
		const Value v = eval( *n.kids[0] );
		std::vector<Value> args;
		for ( size_t k = 1; k < n.kids.size(); ++k ) args.push_back( eval( *n.kids[k] ) );
		const std::string& f = n.name;
		if ( failed ) return Value();
		if ( f == "tojson" ) return Value::string( to_json( v ) );
		if ( f == "trim" ) return Value::string( trim_ws( to_str( v ), true, true ) );
		if ( f == "safe" || f == "forceescape" ) return v;
		if ( f == "string" ) return Value::string( to_str( v ) );
		if ( f == "default" || f == "d" ) {
			bool by_truth = false;
			if ( args.size() > 1 ) by_truth = args[1].truthy();
			for ( const auto& kv : n.kw ) {
				if ( kv.first == "boolean" ) by_truth = eval( *kv.second ).truthy();
			}
			const Value dflt = args.empty() ? Value::string( "" ) : args[0];
			if ( v.is_undefined() || ( by_truth && !v.truthy() ) ) return dflt;
			return v;
		}
		if ( f == "length" || f == "count" ) {
			if ( v.kind == Value::Kind::List ) return Value::integer( static_cast<int64_t>( v.list->size() ) );
			if ( v.kind == Value::Kind::Str ) return Value::integer( static_cast<int64_t>( v.s.size() ) );
			if ( v.kind == Value::Kind::Map ) return Value::integer( static_cast<int64_t>( v.map->items.size() ) );
			return Value::integer( 0 );
		}
		if ( f == "items" ) {
			Value out = Value::make_list();
			if ( v.kind == Value::Kind::Map ) {
				for ( const auto& kv : v.map->items ) {
					Value pair = Value::make_list();
					pair.push( Value::string( kv.first ) );
					pair.push( kv.second );
					out.push( pair );
				}
			}
			return out;
		}
		if ( f == "capitalize" ) {
			std::string s = to_str( v );
			for ( size_t k = 0; k < s.size(); ++k ) s[k] = static_cast<char>( k == 0 ? std::toupper( static_cast<unsigned char>( s[k] ) ) : std::tolower( static_cast<unsigned char>( s[k] ) ) );
			return Value::string( s );
		}
		if ( f == "lower" ) {
			std::string s = to_str( v );
			for ( char& c : s ) c = static_cast<char>( std::tolower( static_cast<unsigned char>( c ) ) );
			return Value::string( s );
		}
		if ( f == "upper" ) {
			std::string s = to_str( v );
			for ( char& c : s ) c = static_cast<char>( std::toupper( static_cast<unsigned char>( c ) ) );
			return Value::string( s );
		}
		if ( f == "replace" && args.size() >= 2 ) return Value::string( replace_all( to_str( v ), to_str( args[0] ), to_str( args[1] ) ) );
		if ( f == "list" ) {
			if ( v.kind == Value::Kind::List ) return v;
			Value out = Value::make_list();
			if ( v.kind == Value::Kind::Map ) for ( const auto& kv : v.map->items ) out.push( Value::string( kv.first ) );
			if ( v.kind == Value::Kind::Str ) for ( char c : v.s ) out.push( Value::string( std::string( 1, c ) ) );
			return out;
		}
		if ( f == "first" || f == "last" ) {
			if ( v.kind == Value::Kind::List && !v.list->empty() ) return f == "first" ? v.list->front() : v.list->back();
			return Value();
		}
		if ( f == "join" ) {
			std::string sep = args.empty() ? "" : to_str( args[0] ), out;
			if ( v.kind == Value::Kind::List ) {
				for ( size_t k = 0; k < v.list->size(); ++k ) {
					if ( k ) out += sep;
					out += to_str( ( *v.list )[k] );
				}
			}
			return Value::string( out );
		}
		if ( f == "int" ) return Value::integer( v.kind == Value::Kind::Str ? std::strtoll( v.s.c_str(), nullptr, 10 ) : is_number( v ) ? as_int( v ) : 0 );
		if ( f == "float" ) return Value::number( v.kind == Value::Kind::Str ? std::strtod( v.s.c_str(), nullptr ) : is_number( v ) ? as_double( v ) : 0.0 );
		if ( f == "abs" && is_number( v ) ) return v.kind == Value::Kind::Float ? Value::number( std::fabs( v.f ) ) : Value::integer( std::llabs( as_int( v ) ) );
		fail( "unknown filter '" + f + "'" );
		return Value();
	}

	static std::string replace_all( std::string s, const std::string& from, const std::string& to ) {
		if ( from.empty() ) return s;
		size_t at = 0;
		while ( ( at = s.find( from, at ) ) != std::string::npos ) {
			s.replace( at, from.size(), to );
			at += to.size();
		}
		return s;
	}

	Value eval_call( const Node& n ) {
		const Node& callee = *n.kids[0];
		std::vector<Value> args;
		for ( size_t k = 1; k < n.kids.size(); ++k ) args.push_back( eval( *n.kids[k] ) );
		std::vector<std::pair<std::string, Value>> kwargs;
		for ( const auto& kv : n.kw ) kwargs.emplace_back( kv.first, eval( *kv.second ) );
		if ( failed ) return Value();
		if ( callee.type == NT::Name ) return call_function( callee.name, args, kwargs );
		if ( callee.type == NT::Attr ) return call_method( eval( *callee.kids[0] ), callee.name, args );
		fail( "value is not callable" );
		return Value();
	}

	Value call_function( const std::string& name, const std::vector<Value>& args, const std::vector<std::pair<std::string, Value>>& kwargs ) {
		auto m = macros_.find( name );
		if ( m != macros_.end() ) return call_macro( *m->second, args, kwargs );
		if ( name == "range" ) {
			int64_t a = 0, b = 0, step = 1;
			if ( args.size() == 1 ) b = as_int( args[0] );
			else if ( args.size() >= 2 ) { a = as_int( args[0] ); b = as_int( args[1] ); }
			if ( args.size() >= 3 ) step = as_int( args[2] );
			if ( step == 0 ) { fail( "range step cannot be zero" ); return Value(); }
			Value out = Value::make_list();
			for ( int64_t k = a; step > 0 ? k < b : k > b; k += step ) out.push( Value::integer( k ) );
			return out;
		}
		if ( name == "namespace" || name == "dict" ) {
			Value ns = Value::make_map();
			ns.map->is_namespace = ( name == "namespace" );
			for ( const auto& kv : kwargs ) ns.set( kv.first, kv.second );
			return ns;
		}
		if ( name == "raise_exception" ) {
			fail( "template error: " + ( args.empty() ? std::string() : to_str( args[0] ) ) );
			return Value();
		}
		fail( "unknown function '" + name + "'" );
		return Value();
	}

	Value call_macro( const Node& macro, const std::vector<Value>& args, const std::vector<std::pair<std::string, Value>>& kwargs ) {
		if ( ++depth_ > 64 ) { fail( "macro recursion too deep" ); --depth_; return Value(); }
		Frame frame;
		frame.parent = &global_;
		for ( size_t k = 0; k < macro.params.size(); ++k ) {
			const std::string& p = macro.params[k];
			Value v;
			bool have = false;
			if ( k < args.size() ) { v = args[k]; have = true; }
			for ( const auto& kv : kwargs ) {
				if ( kv.first == p ) { v = kv.second; have = true; }
			}
			if ( !have ) {
				for ( const auto& d : macro.kw ) {
					if ( d.first == p ) {
						Frame* saved = cur_;
						cur_ = &frame;
						v = eval( *d.second );
						cur_ = saved;
					}
				}
			}
			frame.vars.emplace_back( p, v );
		}
		Frame* saved = cur_;
		cur_ = &frame;
		std::string out;
		exec_body( macro.body, out );
		cur_ = saved;
		--depth_;
		return Value::string( out );
	}

	Value call_method( const Value& obj, const std::string& name, const std::vector<Value>& args ) {
		if ( obj.kind == Value::Kind::Map ) {
			if ( name == "items" ) {
				Value out = Value::make_list();
				for ( const auto& kv : obj.map->items ) {
					Value pair = Value::make_list();
					pair.push( Value::string( kv.first ) );
					pair.push( kv.second );
					out.push( pair );
				}
				return out;
			}
			if ( name == "keys" || name == "values" ) {
				Value out = Value::make_list();
				for ( const auto& kv : obj.map->items ) out.push( name == "keys" ? Value::string( kv.first ) : kv.second );
				return out;
			}
			if ( name == "get" && !args.empty() ) {
				const Value* v = obj.find( to_str( args[0] ) );
				return v ? *v : ( args.size() > 1 ? args[1] : Value::none() );
			}
		}
		if ( obj.kind == Value::Kind::Str ) {
			const std::string& s = obj.s;
			if ( name == "strip" || name == "lstrip" || name == "rstrip" ) {
				const std::string chars = args.empty() ? std::string( " \t\n\r\f\v" ) : to_str( args[0] );
				return Value::string( trim_ws( s, name != "rstrip", name != "lstrip", chars ) );
			}
			if ( name == "startswith" && !args.empty() ) return Value::boolean( s.compare( 0, to_str( args[0] ).size(), to_str( args[0] ) ) == 0 );
			if ( name == "endswith" && !args.empty() ) {
				const std::string e = to_str( args[0] );
				return Value::boolean( s.size() >= e.size() && s.compare( s.size() - e.size(), e.size(), e ) == 0 );
			}
			if ( name == "split" ) {
				Value out = Value::make_list();
				if ( args.empty() || args[0].kind == Value::Kind::None ) {
					size_t k = 0;
					while ( k < s.size() ) {
						while ( k < s.size() && std::isspace( static_cast<unsigned char>( s[k] ) ) ) ++k;
						size_t e = k;
						while ( e < s.size() && !std::isspace( static_cast<unsigned char>( s[e] ) ) ) ++e;
						if ( e > k ) out.push( Value::string( s.substr( k, e - k ) ) );
						k = e;
					}
					return out;
				}
				const std::string sep = to_str( args[0] );
				if ( sep.empty() ) { fail( "empty separator" ); return Value(); }
				size_t from = 0, at;
				while ( ( at = s.find( sep, from ) ) != std::string::npos ) {
					out.push( Value::string( s.substr( from, at - from ) ) );
					from = at + sep.size();
				}
				out.push( Value::string( s.substr( from ) ) );
				return out;
			}
			if ( name == "replace" && args.size() >= 2 ) return Value::string( replace_all( s, to_str( args[0] ), to_str( args[1] ) ) );
			if ( name == "lower" || name == "upper" ) {
				std::string r = s;
				for ( char& c : r ) c = static_cast<char>( name == "lower" ? std::tolower( static_cast<unsigned char>( c ) ) : std::toupper( static_cast<unsigned char>( c ) ) );
				return Value::string( r );
			}
		}
		fail( "unsupported method '" + name + "'" );
		return Value();
	}
};

} // namespace

Template::Template() : impl_( std::make_unique<Impl>() ) {}
Template::~Template() = default;
Template::Template( Template&& ) noexcept = default;
Template& Template::operator=( Template&& ) noexcept = default;

bool Template::valid() const { return impl_ && impl_->ok; }

bool Template::compile( const std::string& source, std::string* error ) {
	impl_ = std::make_unique<Impl>();
	std::vector<Segment> segs;
	if ( !split_segments( source, segs, error ) ) return false;
	StmtParser parser( segs, error );
	if ( !parser.parse( impl_->root ) ) {
		if ( error && error->empty() ) *error = "template syntax error";
		return false;
	}
	impl_->ok = true;
	return true;
}

bool Template::render( const Value& vars, std::string* out, std::string* error ) const {
	if ( !valid() ) {
		if ( error ) *error = "template not compiled";
		return false;
	}
	Evaluator ev( vars, error );
	std::string result;
	ev.exec_body( impl_->root, result );
	if ( ev.failed ) return false;
	if ( out ) *out = std::move( result );
	return true;
}

} // namespace engine::loader::jinja
