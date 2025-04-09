#pragma once

#include <cstdlib>
#include <sys/socket.h>
#include <netdb.h>
#include <ifaddrs.h>
#include <string>

#include "utils/logging.h"


namespace comm_backend {
namespace utils {

std::string get_master_addr();

int get_master_port();

std::string get_local_ip();

int get_local_port(int local_rank = 0);

} // namespace utils
} // namespace comm_backend