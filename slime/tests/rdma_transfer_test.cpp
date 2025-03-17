#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#include "transfer/rdma_transport.h"
#include "info_struct/config.h"

using slime::transfer::RDMAContext;

std::vector<std::string> getActiveRdmaDeviceNames() {
    // TODO: change hardcode to get from command line output
    std::vector<std::string> active_rdma_device_names;
    for (int i = 0; i < 8; ++i) {
        active_rdma_device_names.push_back("mlx5_bond_" + std::to_string(i));
    }
    return active_rdma_device_names;
}

int main() {
    std::vector<std::string> rdma_devs = getActiveRdmaDeviceNames();
    ClientConfig client_config;
    client_config.host_addr = "127.0.0.1";
    client_config.service_port = 92345;
    client_config.link_type = "Ethernet";
    client_config.dev_name = rdma_devs[7];
    client_config.ib_port = 1;

    RDMAContext rdma_context;
    int ret_code = rdma_context.connect_client(client_config);
    std::cout << "Connected to client returns code: " << ret_code << std::endl;
    ret_code = rdma_context.setup_rdma(client_config);
    std::cout << "Setup RDMA returns code: " << ret_code << std::endl;

    rdma_context.close_conn();
    std::cout << "Closed connection." << std::endl;
    return 0;
}