#include "CrossBrandBackend.hpp"

#include <cstdlib>
#include <iostream>

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <zmq.hpp>

#include "head_ccl/topo.hpp"
#include "utils/addr_util.hpp"
#include "utils/json.hpp"
#include "utils/logging.h"

namespace comm_backend {

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
  } else if (rank == 0) {
    head_ccl::MasterCollectInitJson(topo_graph_);
  }
}

c10::intrusive_ptr<Work>
CrossBrandBackend::send(std::vector<at::Tensor> &tensors, int dstRank,
                        int tag) {
  return nullptr;
}

c10::intrusive_ptr<Work>
CrossBrandBackend::recv(std::vector<at::Tensor> &tensors, int srcRank,
                        int tag) {
  return nullptr;
}

} // namespace comm_backend