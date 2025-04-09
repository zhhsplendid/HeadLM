#pragma once

#include <string>

#include "utils/device.hpp"
#include "utils/json.hpp"

namespace comm_backend {
namespace head_ccl {

using utils::DeviceCompany;
using nlohmann::json;

class TopoNode {
public:
    TopoNode(DeviceCompany dev_brand,
             const std::string& card_ver,
             const std::string& addr,
             int port,
             int first_rank,
             int nproc_in_node) :
             device_brand(dev_brand),
             card_version(card_ver),
             address(addr),
             port(port),
             first_rank(first_rank),
             nproc_in_node(nproc_in_node) {}

    TopoNode(const json& j);

    static TopoNode fromQueryMachine();

    json to_json() const;

    DeviceCompany device_brand;
    std::string card_version;

    std::string address;
    int port;
    int first_rank;
    int nproc_in_node;
};

}  // namespace head_ccl
}  // namespace comm_backend

