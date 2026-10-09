#pragma once

#include <string>

namespace engine::runtime {

// Runtime and model options (see options.def for the list). The only place that reads the environment is options.cpp:
// defaults, then ENGINE_<NAME> / ENGINE_DEBUG_<NAME> environment variables, then whatever the launcher (config file,
// CLI, tests) assigns before the first model loads.
struct Options {
#define OPTION( type, name, def, help ) type name = def;
#define DEBUG_OPTION( type, name, def, help ) type name = def;
#include "engine/runtime/options.def"
#undef OPTION
#undef DEBUG_OPTION
};

// Populated from the environment on first use. An invalid environment value is fatal.
Options& options();

// Assigns one option by name (launch config "options" object, --set key=value). False for an unknown key
// or an unparsable / out-of-range value; the caller treats that as a fatal configuration error.
bool set_option( const std::string& key, const std::string& value );

// docs/CONFIG.md: the option table (name, type, default, environment variable, description)
std::string options_markdown();

} // namespace engine::runtime
