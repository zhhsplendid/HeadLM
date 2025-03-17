#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#include "transfer/rdma_transport.h"
#include "info_struct/config.h"

std::vector<std::string> getActiveRdmaDeviceNames() {
    // TODO: change hardcode to get from command line output
    std::vector<std::string> active_rdma_device_names;
    for (int i = 0; i < 8; ++i) {
        active_rdma_device_names.push_back("mlx5_bond_" + std::to_string(i));
    }
    return active_rdma_device_names;
}

int main() {
    std::cout << getActiveRdmaDeviceNames()[0] << std::endl;
    return 0;
}