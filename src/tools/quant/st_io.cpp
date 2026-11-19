#include "st_io.h"
#include "engine/core/log.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstdio>
#include <set>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <sstream>

namespace engine::tools::quant {

int64_t StEntry::numel() const {
	int64_t n = 1;
	for ( int64_t s : shape ) n *= s;
	return n;
}

StFile::~StFile() {
	if ( map_ ) munmap( map_, map_bytes_ );
	if ( fd_ >= 0 ) close( fd_ );
}

bool StFile::open( const std::string& path ) {
	fd_ = ::open( path.c_str(), O_RDONLY );
	if ( fd_ < 0 ) {
		LOG_ERROR( "[st_io] cannot open %s", path.c_str() );
		return false;
	}
	struct stat st {};
	if ( fstat( fd_, &st ) != 0 || st.st_size < 8 ) {
		LOG_ERROR( "[st_io] %s is not a safetensors file", path.c_str() );
		return false;
	}
	map_bytes_ = static_cast<size_t>( st.st_size );
	map_ = mmap( nullptr, map_bytes_, PROT_READ, MAP_SHARED, fd_, 0 );
	if ( map_ == MAP_FAILED ) {
		map_ = nullptr;
		LOG_ERROR( "[st_io] cannot map %s", path.c_str() );
		return false;
	}
	const auto* base = static_cast<const uint8_t*>( map_ );
	uint64_t header_len = 0;
	std::memcpy( &header_len, base, 8 );
	if ( 8 + header_len > map_bytes_ ) {
		LOG_ERROR( "[st_io] %s: header runs past the end of the file", path.c_str() );
		return false;
	}
	const auto header = nlohmann::json::parse( base + 8, base + 8 + header_len, nullptr, false );
	if ( header.is_discarded() || !header.is_object() ) {
		LOG_ERROR( "[st_io] %s: header is not JSON", path.c_str() );
		return false;
	}
	const uint8_t* data = base + 8 + header_len;
	const size_t data_bytes = map_bytes_ - 8 - header_len;
	for ( auto it = header.begin(); it != header.end(); ++it ) {
		if ( it.key() == "__metadata__" ) {
			for ( auto m = it.value().begin(); m != it.value().end(); ++m ) {
				if ( m.value().is_string() ) metadata_[m.key()] = m.value().get<std::string>();
			}
			continue;
		}
		const auto& v = it.value();
		StEntry e;
		e.dtype = v.value( "dtype", "" );
		for ( const auto& s : v["shape"] ) e.shape.push_back( s.get<int64_t>() );
		const auto& off = v["data_offsets"];
		const uint64_t a = off[0].get<uint64_t>(), b = off[1].get<uint64_t>();
		if ( b < a || b > data_bytes ) {
			LOG_ERROR( "[st_io] %s: %s lies outside the file", path.c_str(), it.key().c_str() );
			return false;
		}
		e.data = data + a;
		e.bytes = static_cast<size_t>( b - a );
		entries_[it.key()] = std::move( e );
	}
	// Header order: by data offset, the order the writer laid the tensors out in
	for ( const auto& kv : entries_ ) names_.push_back( kv.first );
	std::sort( names_.begin(), names_.end(), [&]( const std::string& x, const std::string& y ) { return entries_[x].data < entries_[y].data; } );
	return true;
}

const StEntry* StFile::find( const std::string& name ) const {
	const auto it = entries_.find( name );
	return it == entries_.end() ? nullptr : &it->second;
}

bool StRepo::open( const std::string& dir ) {
	dir_ = dir;
	where_.clear();
	files_.clear();
	std::ifstream idx( dir + "/model.safetensors.index.json" );
	if ( idx ) {
		std::stringstream ss;
		ss << idx.rdbuf();
		const auto j = nlohmann::json::parse( ss.str(), nullptr, false );
		if ( !j.is_discarded() && j.contains( "weight_map" ) ) {
			for ( auto it = j["weight_map"].begin(); it != j["weight_map"].end(); ++it ) where_[it.key()] = it.value().get<std::string>();
		}
	}
	// Shards the index does not list (n-gram tables, vision files next to the checkpoint): open them to list their keys
	DIR* d = opendir( dir.c_str() );
	if ( !d ) {
		LOG_ERROR( "[st_io] cannot read the directory %s", dir.c_str() );
		return false;
	}
	std::vector<std::string> shards;
	while ( dirent* de = readdir( d ) ) {
		const std::string fn = de->d_name;
		if ( fn.size() > 12 && fn.compare( fn.size() - 12, 12, ".safetensors" ) == 0 ) shards.push_back( fn );
	}
	closedir( d );
	std::sort( shards.begin(), shards.end() );
	std::set<std::string> indexed;
	for ( const auto& kv : where_ ) indexed.insert( kv.second );
	for ( const auto& fn : shards ) {
		if ( indexed.count( fn ) ) continue;
		StFile* f = shard( fn );
		if ( !f ) return false;
		for ( const auto& kv : f->entries() ) where_[kv.first] = fn;
	}
	return true;
}

StFile* StRepo::shard( const std::string& file ) {
	auto& slot = files_[file];
	if ( !slot ) {
		auto f = std::make_unique<StFile>();
		if ( !f->open( dir_ + "/" + file ) ) {
			files_.erase( file );
			return nullptr;
		}
		slot = std::move( f );
	}
	return slot.get();
}

const StEntry* StRepo::find( const std::string& name ) {
	const auto it = where_.find( name );
	if ( it == where_.end() ) return nullptr;
	StFile* f = shard( it->second );
	return f ? f->find( name ) : nullptr;
}

bool st_write( const std::string& path, const std::vector<StOut>& tensors, const std::map<std::string, std::string>& metadata ) {
	nlohmann::json header = nlohmann::json::object();
	if ( !metadata.empty() ) {
		nlohmann::json m = nlohmann::json::object();
		for ( const auto& kv : metadata ) m[kv.first] = kv.second;
		header["__metadata__"] = m;
	}
	uint64_t off = 0;
	for ( const auto& t : tensors ) {
		header[t.name] = { { "dtype", t.dtype }, { "shape", t.shape }, { "data_offsets", { off, off + t.bytes() } } };
		off += t.bytes();
	}
	std::string text = header.dump();
	while ( text.size() % 8 ) text.push_back( ' ' );
	const std::string tmp = path + ".tmp";
	FILE* f = std::fopen( tmp.c_str(), "wb" );
	if ( !f ) {
		LOG_ERROR( "[st_io] cannot write %s", tmp.c_str() );
		return false;
	}
	const uint64_t len = text.size();
	bool ok = std::fwrite( &len, 8, 1, f ) == 1 && std::fwrite( text.data(), 1, text.size(), f ) == text.size();
	for ( const auto& t : tensors ) {
		if ( !ok ) break;
		if ( t.bytes() ) ok = std::fwrite( t.data(), 1, t.bytes(), f ) == t.bytes();
	}
	ok = std::fclose( f ) == 0 && ok;
	if ( !ok || std::rename( tmp.c_str(), path.c_str() ) != 0 ) {
		LOG_ERROR( "[st_io] writing %s failed", path.c_str() );
		std::remove( tmp.c_str() );
		return false;
	}
	return true;
}

} // namespace engine::tools::quant
