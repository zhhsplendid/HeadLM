#include "utils.h"

#include <arpa/inet.h>
#include <boost/stacktrace.hpp>

#include <iomanip>
#include <iostream>
#include <ostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "info_struct/config.h"
#include "logging.h"

void signal_handler(int signum) {
  SLIME_LOG_INFO("Interrupt signal (" + std::to_string(signum) + ") received.");
  boost::stacktrace::stacktrace st;
  std::ostringstream oss;
  oss << st;
  SLIME_ERROR("Stacktrace:\n" + oss.str());
  exit(1);
}

std::string human_readable_gid(rdma_conn_info_t *info) {
  std::string gid_str;
  bool is_ipv4_mapped = true;

  // Check if the GID is an IPv4-mapped IPv6 address
  for (int i = 0; i < 10; ++i) {
    if (info->gid.raw[i] != 0) {
      is_ipv4_mapped = false;
      break;
    }
  }
  if (info->gid.raw[10] != 0xff || info->gid.raw[11] != 0xff) {
    is_ipv4_mapped = false;
  }

  if (is_ipv4_mapped) {
    // Convert the last 4 bytes to an IPv4 address
    char ipv4_str[INET_ADDRSTRLEN];
    uint8_t ipv4_addr[4] = {info->gid.raw[12], info->gid.raw[13],
                            info->gid.raw[14], info->gid.raw[15]};
    inet_ntop(AF_INET, ipv4_addr, ipv4_str, INET_ADDRSTRLEN);
    gid_str = ipv4_str;
  } else {
    // Convert the GID to a standard IPv6 address string
    for (int i = 0; i < 16; ++i) {
      std::ostringstream oss;
      oss << std::hex << std::setw(2) << std::setfill('0')
          << static_cast<uint8_t>(info->gid.raw[i]);
      gid_str += oss.str();
      if (i % 2 == 1 && i != 15) {
        gid_str += ":";
      }
    }
  }
  return gid_str;
}

void print_rdma_conn_info(rdma_conn_info_t *info, bool is_remote) {
  std::string gid_str = human_readable_gid(info);
  if (is_remote) {
    std::string debug_str =
        "remote rdma_conn_info: psn: " + std::to_string((uint32_t)info->psn) +
        ", qpn: " + std::to_string((uint32_t)info->qpn);
    +", gid: " + gid_str + ", enum mtu: " + std::to_string(info->mtu);
    SLIME_LOG_DEBUG(debug_str);
  } else {
    std::string debug_str =
        "local rdma_conn_info: " + std::to_string((uint32_t)info->psn) +
        ", qpn: " + std::to_string((uint32_t)info->qpn);
    +", gid: " + gid_str + ", enum mtu: " + std::to_string(info->mtu);
    SLIME_LOG_DEBUG(debug_str);
  }
}

