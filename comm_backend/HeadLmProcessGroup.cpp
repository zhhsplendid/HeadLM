#include "HeadLmProcessGroup.hpp"

#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

namespace comm_backend {

HeadLmProcessGroup::HeadLmProcessGroup(int rank, int size,
                                       DeviceType device_type)
    : Backend(rank, size), origin_device_type_(device_type) {
  std::time_t cur_time = std::time(nullptr);
  file_store_ = c10::make_intrusive<FileStore>(
      "/tmp/headlm_" + std::to_string(cur_time), size);
  auto options = ::c10d::ProcessGroupGloo::Options::create();
  options->timeout = std::chrono::seconds(20);
  options->devices.push_back(
        ::c10d::ProcessGroupGloo::createDeviceForHostname("127.0.0.1"));
  cpu_process_group_ =
      c10::make_intrusive<ProcessGroupGloo>(file_store_, rank, size, options);
}

c10::intrusive_ptr<Work>
HeadLmProcessGroup::send(std::vector<at::Tensor> &tensors, int dstRank,
                         int tag) {
  if (origin_device_type_ != DeviceType::CPU) {
    for (at::Tensor &tensor : tensors) {
      tensor.cpu();
    }
  }
  return cpu_process_group_->send(tensors, dstRank, tag);
}

c10::intrusive_ptr<Work>
HeadLmProcessGroup::recv(std::vector<at::Tensor> &tensors, int srcRank,
                         int tag) {
  c10::intrusive_ptr<Work> gloo_recv_work =
      cpu_process_group_->recv(tensors, srcRank, tag);
  auto ret = c10::make_intrusive<ToDeviceRecvWork>(gloo_recv_work, tensors, origin_device_type_);
  /*
  if (origin_device_type_ != DeviceType::CPU) {
    for (at::Tensor &tensor : tensors) {
      tensor.to(origin_device_type_);
    }
  }*/
  
  return ret;
}

c10::intrusive_ptr<Backend> HeadLmProcessGroup::createHeadLmProcessGroup(
    const c10::intrusive_ptr<::c10d::Store> &store, int rank, int size,
    const std::chrono::duration<float> &timeout) {
  // TODO: hardcode the CUDA here for fast test. Should change parameter after
  // we know it works
  return c10::make_intrusive<HeadLmProcessGroup>(rank, size, DeviceType::CUDA);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createHeadLmProcessGroup",
        &HeadLmProcessGroup::createHeadLmProcessGroup);
}

} // namespace comm_backend