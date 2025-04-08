#pragma once

#include <cstdlib>
#include <sys/socket.h>
#include <netdb.h>
#include <ifaddrs.h>
#include <string>

#include "utils/logging.h"


std::string get_master_addr() {
    const char* master_addr = std::getenv("MASTER_ADDR");
    HEADLM_ASSERT_NE(master_addr, nullptr, "Please set MASTER_ADDR to use HeadLM");
    return std::string(master_addr);
}

int get_master_port() {
    const char* master_port = std::getenv("MASTER_PORT");
    HEADLM_ASSERT_NE(master_port, nullptr, "Please set MASTER_PORT to use HeadLM");
    return std::atoi(master_port);
}

std::string get_local_ip() {
    struct ifaddrs *ifaddr, *ifa;
    std::string local_ip = "127.0.0.1"; // Default to localhost

    if (getifaddrs(&ifaddr) == -1) {
        return local_ip; // Handle error if needed
    }

    // Loop through interfaces
    for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr) continue;
        int family = ifa->ifa_addr->sa_family;
        if (family == AF_INET) { // IPv4
            char host[NI_MAXHOST];
            if (getnameinfo(ifa->ifa_addr, sizeof(sockaddr_in),
                            host, NI_MAXHOST, nullptr, 0, NI_NUMERICHOST) == 0) {
                // Check for non-loopback interface
                if (std::string(ifa->ifa_name) != "lo") {
                    local_ip = host;
                    break;
                }
            }
        }
    }

    freeifaddrs(ifaddr);
    return local_ip;
}
