#include "adapters.h"
#include <sys/stat.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <dirent.h>
#include <regex>

namespace engine::server {

namespace {

// The JSON header of a safetensors file (empty when unreadable)
std::string safetensors_header( const std::string& path ) {
	FILE* f = std::fopen( path.c_str(), "rb" );
	uint64_t len = 0;
	std::string hdr;
	if ( f && std::fread( &len, 8, 1, f ) == 1 && len < ( 64u << 20 ) ) {
		hdr.resize( len );
		if ( std::fread( hdr.data(), 1, len, f ) != len ) hdr.clear();
	}
	if ( f ) std::fclose( f );
	return hdr;
}

} // namespace

void AdapterCatalog::scan( const std::vector<std::string>& dirs, bool ref2va ) {
	list_.clear();
	static const std::regex turbo_re( R"(^minimax_h3_(fl2v|ref2v)_turbo_(\d)step_v([\d.]+)(_768p)?(_bf16)?\.safetensors$)" );
	const std::string ext = ".safetensors";
	for ( const std::string& dir : dirs ) {
		DIR* d = ::opendir( dir.c_str() );
		if ( !d ) continue;
		while ( dirent* e = ::readdir( d ) ) {
			const std::string name = e->d_name;
			if ( name.size() < ext.size() || name.compare( name.size() - ext.size(), ext.size(), ext ) != 0 ) continue;
			AdapterInfo a;
			a.file = name;
			a.path = dir + "/" + name;
			struct stat st;
			if ( ::stat( a.path.c_str(), &st ) != 0 ) continue;
			a.mb = static_cast<long>( st.st_size >> 20 );
			std::smatch m;
			if ( std::regex_match( name, m, turbo_re ) ) {
				if ( m[1] == "ref2v" && !ref2va ) continue; // Ref2V adapters need the Ref2VA transformer
				a.turbo = true;
				a.task = m[1].str();
				a.label = m[1].str() + " " + m[2].str() + "-step turbo v" + m[3].str() + ( m[4].matched ? " (768p)" : "" ) + " - fast";
				a.format = "diffusers";
			} else {
				a.label = name.substr( 0, name.size() - ext.size() );
				const std::string hdr = safetensors_header( a.path );
				if ( hdr.find( "lora_unet_blocks_" ) != std::string::npos ) a.format = "kohya";
				else if ( hdr.find( "diffusion_model.blocks." ) != std::string::npos ) a.format = "ai-toolkit";
				else if ( hdr.find( "transformer_blocks." ) != std::string::npos ) a.format = "diffusers";
				else continue; // not an adapter for this model
				if ( name.find( "ref2va" ) != std::string::npos || hdr.find( "\"ss_h3_training_mode\":\"ref2va\"" ) != std::string::npos ) a.task = "ref2va";
			}
			list_.push_back( a );
		}
		::closedir( d );
	}
	std::sort( list_.begin(), list_.end(), []( const AdapterInfo& a, const AdapterInfo& b ) { return a.turbo != b.turbo ? a.turbo > b.turbo : a.label < b.label; } );
}

const AdapterInfo* AdapterCatalog::find( const std::string& file ) const {
	for ( const AdapterInfo& a : list_ ) {
		if ( a.file == file ) return &a;
	}
	return nullptr;
}

} // namespace engine::server
