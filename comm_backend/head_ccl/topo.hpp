#pragma once

#include <signal.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <zmq.hpp>

#include "topo_node.hpp"
#include "head_ccl/global_config.hpp"
#include "utils/addr_util.hpp"
#include "utils/device.hpp"
#include "utils/json.hpp"


namespace comm_backend {
namespace head_ccl {

using nlohmann::json;
using utils::DeviceCompany;

class TopoGraph {
public:
  int add_node_from_json(const json& j);
  void remove_node_by_rank(int rank);

private:
  std::unordered_map<int, TopoNode> rank_node;
};

int ConstructGraphFromInitJson(const std::vector<json>& json_data, TopoGraph* graph);

volatile extern bool user_stop;
void signal_handler(int);

int MasterCollectInitJson(TopoGraph* graph);

int WorkerSendInitJson();

} // namespace head_ccl
} // namespace comm_backend
