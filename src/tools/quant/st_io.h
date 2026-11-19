#pragma once

// Safetensors files for the quantizer: a reader that keeps every header entry as written (dtype string, shape, the mmapped bytes; the engine's
// loader folds the FP8 and unknown dtypes into its own enum), a repository over a directory's shards, and a writer.
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace engine::tools::quant {

struct StEntry {
	std::string dtype;            // "BF16", "F8_E4M3", "I16", ...
	std::vector<int64_t> shape;
	const uint8_t* data = nullptr;
	size_t bytes = 0;
	int64_t numel() const;
};

class StFile {
public:
	~StFile();
	bool open( const std::string& path );
	const StEntry* find( const std::string& name ) const;
	const std::map<std::string, StEntry>& entries() const { return entries_; }
	const std::map<std::string, std::string>& metadata() const { return metadata_; }
	// The order the header lists the tensors in
	const std::vector<std::string>& names() const { return names_; }

private:
	int fd_ = -1;
	void* map_ = nullptr;
	size_t map_bytes_ = 0;
	std::map<std::string, StEntry> entries_;
	std::map<std::string, std::string> metadata_;
	std::vector<std::string> names_;
};

// Every *.safetensors file of a directory, opened on first use (a network share is slow to open whole)
class StRepo {
public:
	bool open( const std::string& dir );
	const StEntry* find( const std::string& name );
	bool has( const std::string& name ) const { return where_.count( name ) != 0; }
	const std::string& dir() const { return dir_; }
	// name -> shard file name, from the index when there is one
	const std::map<std::string, std::string>& weight_map() const { return where_; }
	StFile* shard( const std::string& file );

private:
	std::string dir_;
	std::map<std::string, std::string> where_;
	std::map<std::string, std::unique_ptr<StFile>> files_;
};

// A tensor to write: either owned bytes or a view of someone else's (a template tensor kept as it is)
struct StOut {
	std::string name;
	std::string dtype;
	std::vector<int64_t> shape;
	std::vector<uint8_t> owned;
	const uint8_t* view = nullptr;
	size_t view_bytes = 0;
	const uint8_t* data() const { return view ? view : owned.data(); }
	size_t bytes() const { return view ? view_bytes : owned.size(); }
};

// Writes path + ".tmp" and renames it into place
bool st_write( const std::string& path, const std::vector<StOut>& tensors, const std::map<std::string, std::string>& metadata );

} // namespace engine::tools::quant
