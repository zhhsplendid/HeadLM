#pragma once

namespace comm_backend {
namespace head_ccl {

#define HEADLM_TIMEOUT_SEC 3600

#define DEFAULT_PORT_START 32768

#define SOCKET_EXCHANGE_SENDER_PORT DEFAULT_PORT_START + 1
#define SOCKET_EXCHANGE_RECEIVER_PORT DEFAULT_PORT_START + 2

} // namespace head_ccl
} // namespace comm_backend