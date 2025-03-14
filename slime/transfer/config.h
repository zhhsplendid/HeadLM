#ifndef LIBCONFIG_H
#define LIBCONFIG_H

#include <infiniband/verbs.h>
#include <string>

typedef struct ServerConfig {
    int service_port;
    std::string log_level;
    std::string dev_name;
    size_t prealloc_size;  // unit: GB
    int ib_port;
    std::string link_type;
    int minimal_allocate_size;  // unit: KB
    int num_stream;             // can only be 1,2,4, number of stream for each client
    bool auto_increase;
} server_config_t;

typedef struct ClientConfig {
    int service_port;
    std::string log_level;
    std::string dev_name;
    std::string host_addr;
    int ib_port;
    std::string link_type;
} client_config_t;

typedef struct __attribute__((packed)) rdma_conn_info_t {
    uint32_t qpn;
    uint32_t psn;
    union ibv_gid gid;  // RoCE v2
    uint16_t lid;       // IB
    uint32_t mtu;       // peers should have the same mtu
} rdma_conn_info_t;

#endif