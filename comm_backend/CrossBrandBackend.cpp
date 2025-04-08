#include "CrossBrandBackend.hpp"

#include <cstdlib>
#include <iostream>

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "utils/addr_util.hpp"

namespace comm_backend {

CrossBrandBackend::CrossBrandBackend(const c10::intrusive_ptr<::c10d::Store> &store, int rank,
                       int size, const std::chrono::duration<float> &timeout,
                       DeviceType device_type)
    : Backend(rank, size), origin_device_type_(device_type) {
  std::cout << "local ip = " << get_local_ip() << std::endl;
}

c10::intrusive_ptr<Work> CrossBrandBackend::send(std::vector<at::Tensor> &tensors,
                                          int dstRank, int tag) {
  return nullptr;
}

c10::intrusive_ptr<Work> CrossBrandBackend::recv(std::vector<at::Tensor> &tensors,
                                          int srcRank, int tag) {
  return nullptr;
}

} // namespace comm_backend