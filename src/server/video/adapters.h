#pragma once

#include <string>
#include <vector>

namespace engine::server {

// A LoRA adapter file found in one of the configured directories
struct AdapterInfo {
	std::string file, path, label, format, task; // format: diffusers | kohya | ai-toolkit; task: fl2v | ref2v (turbo) or ref2va (community)
	bool turbo = false;                          // a LightX2V few-step adapter (fixes the step count and the flow shift)
	long mb = 0;
};

class AdapterCatalog {
public:
	// Scans `dirs` for *.safetensors adapters; Ref2V turbo adapters are listed only when the Ref2VA transformer is configured.
	void scan( const std::vector<std::string>& dirs, bool ref2va );
	const AdapterInfo* find( const std::string& file ) const;
	const std::vector<AdapterInfo>& all() const { return list_; }

private:
	std::vector<AdapterInfo> list_;
};

} // namespace engine::server
