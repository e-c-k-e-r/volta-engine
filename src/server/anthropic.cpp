#include "engine/server/chat.h"

namespace engine::server {

// Implemented in the next step; /v1/messages stays unrouted until then.
std::unique_ptr<ChatProtocol> make_anthropic_protocol() { return nullptr; }

} // namespace engine::server
