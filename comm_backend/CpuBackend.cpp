#include "CpuBackend.hpp"

#include <cstdlib>

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

namespace comm_backend {

CpuBackend::CpuBackend(const c10::intrusive_ptr<::c10d::Store> &store, int rank,
                       int size, const std::chrono::duration<float> &timeout,
                       DeviceType device_type)
    : Backend(rank, size), origin_device_type_(device_type) {
  auto timeout_millis =
      std::chrono::duration_cast<std::chrono::milliseconds>(timeout);
  auto options = ::c10d::ProcessGroupGloo::Options::create(timeout_millis);
  options->devices.push_back(ProcessGroupGloo::createDefaultDevice());

  cpu_process_group_ =
      c10::make_intrusive<ProcessGroupGloo>(store, rank, size, options);
}

c10::intrusive_ptr<Work> CpuBackend::send(std::vector<at::Tensor> &tensors,
                                          int dstRank, int tag) {
  if (origin_device_type_ != DeviceType::CPU) {
    for (at::Tensor &tensor : tensors) {
      tensor = tensor.cpu();
    }
  }

  return cpu_process_group_->send(tensors, dstRank, tag);
}

c10::intrusive_ptr<Work> CpuBackend::recv(std::vector<at::Tensor> &tensors,
                                          int srcRank, int tag) {
  if (origin_device_type_ == DeviceType::CPU) {
    return cpu_process_group_->recv(tensors, srcRank, tag);
  }

  auto cpu_tensors = std::make_shared<std::vector<at::Tensor>>();
  for (at::Tensor &tensor : tensors) {
    cpu_tensors->push_back(tensor.to("cpu"));
  }

  c10::intrusive_ptr<Work> gloo_recv_work =
      cpu_process_group_->recv(*cpu_tensors, srcRank, tag);

  auto ret = c10::make_intrusive<ToDeviceRecvWork>(
      gloo_recv_work, tensors, cpu_tensors, origin_device_type_);
  return ret;
}

} // namespace comm_backend