#include "engine/models/config_check.h"
#include "engine/core/log.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>

namespace engine::models {

struct ConfigCheck::Impl {
	nlohmann::json root;
	const nlohmann::json* text = nullptr;
	// The field at a dotted path under text_config (or the top level), or null when it is absent or null
	const nlohmann::json* at( const char* key, bool top = false ) const {
		if ( text == nullptr ) return nullptr;
		const nlohmann::json* j = top ? &root : text;
		std::string path( key );
		for ( size_t b = 0; b <= path.size(); ) {
			const size_t e = std::min( path.find( '.', b ), path.size() );
			const std::string part = path.substr( b, e - b );
			if ( !j->is_object() || !j->contains( part ) ) return nullptr;
			j = &( *j )[part];
			b = e + 1;
		}
		return j->is_null() ? nullptr : j;
	}
};

ConfigCheck::ConfigCheck( const char* model_dir, const char* model, bool log ) : impl_( std::make_unique<Impl>() ), model_( model ), log_( log ) {
	const std::string path = std::string( model_dir ) + "/config.json";
	std::ifstream f( path );
	if ( f ) impl_->root = nlohmann::json::parse( f, nullptr, false );
	if ( !f || impl_->root.is_discarded() || !impl_->root.is_object() ) {
		if ( log_ ) LOG_ERROR( "[%s] %s is unreadable or not a JSON object\n", model_.c_str(), path.c_str() );
		ok_ = false;
		return;
	}
	impl_->text = impl_->root.contains( "text_config" ) && impl_->root["text_config"].is_object() ? &impl_->root["text_config"] : &impl_->root;
}

ConfigCheck::~ConfigCheck() = default;

void ConfigCheck::fail( const char* key, const std::string& got, const std::string& want ) {
	if ( log_ ) LOG_ERROR( "[%s] config.json: %s is %s; this pipeline implements %s (another variant of the architecture, not supported)\n", model_.c_str(), key,
		got.c_str(), want.c_str() );
	ok_ = false;
}

void ConfigCheck::expect_int( const char* key, int64_t want ) {
	const nlohmann::json* j = impl_->at( key );
	if ( j == nullptr ) return;
	if ( !j->is_number_integer() || j->get<int64_t>() != want ) fail( key, j->dump(), std::to_string( want ) );
}

void ConfigCheck::expect_num( const char* key, double want ) {
	const nlohmann::json* j = impl_->at( key );
	if ( j == nullptr ) return;
	if ( !j->is_number() || std::fabs( j->get<double>() - want ) > 1e-6 * std::fabs( want ) ) {
		char w[64];
		std::snprintf( w, sizeof( w ), "%g", want );
		fail( key, j->dump(), w );
	}
}

void ConfigCheck::expect_bool( const char* key, bool want ) {
	const nlohmann::json* j = impl_->at( key );
	if ( j == nullptr ) return;
	if ( !j->is_boolean() || j->get<bool>() != want ) fail( key, j->dump(), want ? "true" : "false" );
}

void ConfigCheck::expect_str( const char* key, const char* want ) {
	const nlohmann::json* j = impl_->at( key );
	if ( j == nullptr ) return;
	if ( !j->is_string() || j->get<std::string>() != want ) fail( key, j->dump(), std::string( "\"" ) + want + "\"" );
}

void ConfigCheck::expect_ints( const char* key, const std::vector<int64_t>& want, bool prefix ) {
	const nlohmann::json* j = impl_->at( key );
	if ( j == nullptr ) return;
	bool same = j->is_array() && ( prefix ? j->size() >= want.size() : j->size() == want.size() );
	for ( size_t i = 0; same && i < want.size(); ++i ) same = ( *j )[i].is_number_integer() && ( *j )[i].get<int64_t>() == want[i];
	if ( !same ) fail( key, j->dump(), nlohmann::json( want ).dump() );
}

void ConfigCheck::expect_strs( const char* key, const std::vector<std::string>& want ) {
	const nlohmann::json* j = impl_->at( key );
	if ( j == nullptr ) return;
	bool same = j->is_array() && j->size() == want.size();
	for ( size_t i = 0; same && i < want.size(); ++i ) same = ( *j )[i].is_string() && ( *j )[i].get<std::string>() == want[i];
	if ( same ) return;
	// the first entry that differs, not the whole list
	size_t i = 0;
	while ( j->is_array() && i < j->size() && i < want.size() && ( *j )[i].is_string() && ( *j )[i].get<std::string>() == want[i] ) ++i;
	const std::string got = !j->is_array() ? j->dump() : "[" + std::to_string( j->size() ) + " entries, entry " + std::to_string( i ) + ": " +
		( i < j->size() ? ( *j )[i].dump() : std::string( "none" ) ) + "]";
	fail( key, got, "[" + std::to_string( want.size() ) + " entries, entry " + std::to_string( i ) + ": " + ( i < want.size() ? "\"" + want[i] + "\"" : std::string( "none" ) ) + "]" );
}

bool ConfigCheck::read_num( const char* key, float* out ) const {
	const nlohmann::json* j = impl_->at( key );
	if ( j == nullptr || !j->is_number() ) return false;
	*out = j->get<float>();
	return true;
}

bool ConfigCheck::read_int( const char* key, int* out ) const {
	const nlohmann::json* j = impl_->at( key );
	if ( j == nullptr ) j = impl_->at( key, true );
	if ( j == nullptr || !j->is_number_integer() ) return false;
	*out = j->get<int>();
	return true;
}

} // namespace engine::models
