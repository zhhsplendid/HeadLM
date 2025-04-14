#include "CrossBrandBackend.hpp"

#include <cstdlib>
#include <iostream>

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <zmq.hpp>

#include "head_ccl/global_config.hpp"
#include "head_ccl/topo.hpp"
#include "utils/addr_util.hpp"
#include "utils/json.hpp"
#include "utils/logging.h"

namespace comm_backend {

using head_ccl::Communicator;

CrossBrandBackend::CrossBrandBackend(
    const c10::intrusive_ptr<::c10d::Store> &store, int rank, int size,
    const std::chrono::duration<float> &timeout, DeviceType device_type)
    : Backend(rank, size), origin_device_type_(device_type) {

  // TODO: currently just do init here, we should move it to better place
  // when we finish topo analysis

  std::string master_ip = utils::get_master_addr();
  std::string local_ip = utils::get_local_ip();

  topo_graph_ = nullptr;

  if (local_ip != master_ip) {
    head_ccl::WorkerSendInitJson();
    head_ccl::WorkerReceiveTopo(topo_graph_);
  } else if (rank == 0) {
    head_ccl::MasterCollectInitJson(topo_graph_);
    head_ccl::MasterSendTopo(*topo_graph_);
  }

  world_size_ = std::stoi(get_env_variable("WORLD_SIZE"));
  rank_ = rank;

  for (int i = 0; i < world_size_; ++i) {
    comms_.push_back(Communicator("mlx5_bond_1", 1));
    if (i != rank_) {
      int min_rank = std::min(i, rank);
      int max_rank = std::max(i, rank);
      // Can be optimized to fewer ports but we just do quick experiment
      int port = commSendPort(min_rank, max_rank);
      std::string addr = topo_graph_->get_addr(i);
      comms_.back().tcpMetaConnect(addr, port, port);
    }
  }
}

c10::intrusive_ptr<Work>
CrossBrandBackend::send(std::vector<at::Tensor> &tensors, int dstRank,
                        int tag) {
  // Still in expr code, fix here
  auto &t = tensors[0];
  comms_[dstRank].sendTensorAsync(t);

  auto future = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  future->markCompleted(c10::IValue(tensors));
  return c10::make_intrusive<WorkDummy>(OpType::SEND, std::move(future));
}

c10::intrusive_ptr<Work>
CrossBrandBackend::recv(std::vector<at::Tensor> &tensors, int srcRank,
                        int tag) {

  auto future = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::ListType::create(c10::TensorType::get())));

  auto callback = [&](int64_t status_code) {
    future->markCompleted(c10::IValue(tensors));
  };

  // Still in expr code, fix here
  auto &t = tensors[0];
  comms_[srcRank].recvTensorAsync(&t, callback);
  return c10::make_intrusive<WorkDummy>(OpType::RECV, std::move(future));
}

int CrossBrandBackend::commSendPort(int rank_i, int rank_j) const {
  return SOCKET_EXCHANGE_SEND_RECV_PORT + rank_i * world_size_ + rank_j;
}

} // namespace comm_backend