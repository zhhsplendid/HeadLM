#pragma once

#include <signal.h>
#include <string>
#include <thread>
#include <map>
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
  TopoGraph() = default;
  TopoGraph(const json& json_data);

  int add_node_from_json(const json& j);
  void remove_node_by_rank(int rank);
  json to_json() const;
  
  size_t node_size() const {
    return rank_node_.size();
  }

  std::string get_addr(int rank) const {
    auto iter = rank_node_.upper_bound(rank);
    HEADLM_ASSERT_NE(iter, rank_node_.begin(), "Precondition not met. Topo Graph can not find key for rank " + std::to_string(rank));
    --iter;
    return iter->second.address;
  }

private:
  std::map<int, TopoNode> rank_node_;
};

int ConstructGraphFromInitJson(const std::vector<json>& json_data, TopoGraph* graph);

volatile extern bool user_stop;
void signal_handler(int);

int MasterCollectInitJson(TopoGraph* graph);

int WorkerSendInitJson();

int MasterSendTopo(const TopoGraph& graph);

int WorkerReceiveTopo(TopoGraph* graph);

} // namespace head_ccl
} // namespace comm_backend
