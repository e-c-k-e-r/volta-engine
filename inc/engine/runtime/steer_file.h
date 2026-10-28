#pragma once

#include "engine/modules/steer.h"
#include <string>
#include <vector>

namespace engine::runtime {

// Control-vector file (.steer), written by tests/tools/steer_build and read at start or per request:
//   char[4] "VSTR", int32 version (1), int32 hidden, int32 model layers, int32 default scope (SteerScope bits), int32 entries,
//   then per entry: int32 layer, int32 kind (SteerKind), float par, float vec[hidden]   (native little-endian)
struct SteerFile {
	int hidden = 0;
	int layers = 0;
	int scope = engine::modules::STEER_ALL;
	std::vector<engine::modules::SteerEntry> entries;
};

bool load_steer_file( const std::string& path, SteerFile* out, std::string* err );
bool save_steer_file( const std::string& path, const SteerFile& file, std::string* err );

// A scope list ("prompt,think", "generated", "all") to SteerScope bits; false when a word is unknown
bool parse_steer_scope( const std::string& spec, int* scope );

} // namespace engine::runtime
