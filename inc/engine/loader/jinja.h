#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace engine::loader::jinja {

struct MapData;

// Dynamic value of the template language (Python semantics for truthiness, equality and str()).
struct Value {
	enum class Kind { Undefined, None, Bool, Int, Float, Str, List, Map };

	Kind kind = Kind::Undefined;
	bool b = false;
	int64_t i = 0;
	double f = 0.0;
	std::string s;
	std::shared_ptr<std::vector<Value>> list;
	std::shared_ptr<MapData> map;

	static Value none();
	static Value boolean( bool v );
	static Value integer( int64_t v );
	static Value number( double v );
	static Value string( std::string v );
	static Value make_list();
	static Value make_map();

	bool is_undefined() const { return kind == Kind::Undefined; }
	bool truthy() const;

	// Maps keep insertion order. set() replaces an existing key.
	void set( const std::string& key, Value v );
	void push( Value v );
	const Value* find( const std::string& key ) const;
};

struct MapData {
	std::vector<std::pair<std::string, Value>> items;
	bool is_namespace = false;
};

// Parses JSON text into a Value, preserving object key order. Undefined on a parse error.
Value from_json( const std::string& text );

// Python-style json.dumps (separators ", " and ": ", non-ASCII left as UTF-8).
std::string to_json( const Value& v );

// Subset of Jinja2 sufficient for model chat templates (the environment transformers uses: trim_blocks, lstrip_blocks,
// loop controls). Errors (syntax, raise_exception, type errors) are reported through the message, never thrown.
class Template {
public:
	Template();
	~Template();
	Template( Template&& ) noexcept;
	Template& operator=( Template&& ) noexcept;

	bool compile( const std::string& source, std::string* error );
	// vars must be a Map
	bool render( const Value& vars, std::string* out, std::string* error ) const;
	bool valid() const;

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

} // namespace engine::loader::jinja
