#pragma once

// A checkpoint's config.json (its text_config) against the one layout a pipeline implements, for the model families that have one (their
// fine-tunes share it; another size of the architecture fails the load loudly instead of loading wrong). Every expect names a field (a dotted
// path for a nested one: "linear_attn_config.num_heads") and the value the pipeline is built for; a field that is there with another value, or
// of another kind, fails the check and is logged with both values. A field that is absent or null passes: the reference's default is the
// implemented one. Gemma 4 reads its layout from the config instead (read_gemma4_config).
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models {

class ConfigCheck {
public:
	// model: the pipeline's name in the messages; log false: the same checks, silently (the ranks after the first)
	ConfigCheck( const char* model_dir, const char* model, bool log = true );
	~ConfigCheck();

	void expect_int( const char* key, int64_t want );
	void expect_num( const char* key, double want );   // within a relative 1e-6
	void expect_bool( const char* key, bool want );
	void expect_str( const char* key, const char* want );
	// prefix: the list starts with want (it may go on: entries for an MTP head the pipeline reads elsewhere)
	void expect_ints( const char* key, const std::vector<int64_t>& want, bool prefix = false );
	void expect_strs( const char* key, const std::vector<std::string>& want );
	// A value the pipeline takes from the config (not a check): *out is set when the field is a number (read_int: also a top-level field, e.g.
	// eos_token_id, when text_config has none)
	bool read_num( const char* key, float* out ) const;
	bool read_int( const char* key, int* out ) const;

	// False when the file is unreadable or any expect failed
	bool ok() const { return ok_; }

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
	std::string model_;
	bool log_ = true, ok_ = true;
	void fail( const char* key, const std::string& got, const std::string& want );
};

} // namespace engine::models
