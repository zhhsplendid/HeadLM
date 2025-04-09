#include "head_ccl/topo_node.hpp"

#include <string>

#include "utils/addr_util.hpp"
#include "utils/device.hpp"
#include "utils/json.hpp"
#include "topo_node.hpp"

namespace comm_backend {
namespace head_ccl {

using utils::DeviceCompany;
using nlohmann::json;


TopoNode::TopoNode(const json &j)
{
    std::string s = j.at("device_brand").get<std::string>();
    device_brand = utils::device_company_from_string(s);
    card_version = j.at("card_version").get<std::string>();
    address =    j.at("address").get<std::string>();
    port = j.at("port").get<int>();
    first_rank = j.at("first_rank").get<int>();
    nproc_in_node = j.at("nproc_in_node").get<int>();
}

json TopoNode::to_json() const {
    std::string device_brand_str = utils::device_company_to_string(device_brand);
    return {
        {"device_brand", device_brand_str},
        {"card_version", card_version},
        {"address", address},
        {"port", port},
        {"first_rank", first_rank},
        {"nproc_in_node", nproc_in_node}
    };
}


}  // namespace head_ccl
}  // namespace comm_backend