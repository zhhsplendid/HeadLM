#pragma once

#include <ctime>

#include <c10/core/DeviceType.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>

#include <torch/python.h>

namespace comm_backend {

using c10::DeviceType;

using c10d::Backend;
using c10d::FileStore;
using c10d::ProcessGroupGloo;
using c10d::Work;

class HeadLmProcessGroup : public Backend {
public:
  HeadLmProcessGroup(int rank, int size, DeviceType device_type);

  c10::intrusive_ptr<Work> send(std::vector<at::Tensor> &tensors, int dstRank,
                                int tag) override;

  c10::intrusive_ptr<Work> recv(std::vector<at::Tensor> &tensors, int srcRank,
                                int tag) override;

  static c10::intrusive_ptr<Backend>
  createHeadLmProcessGroup(const c10::intrusive_ptr<::c10d::Store> &store,
                           int rank, int size,
                           const std::chrono::duration<float> &timeout);

  static void BackendDummyConstructor() __attribute__((constructor)) {
    py::object module = py::module::import("torch.distributed");
    py::object register_backend =
        module.attr("Backend").attr("register_backend");
    register_backend("headlm", py::cpp_function(createHeadLmProcessGroup));
  }

private:
  DeviceType origin_device_type_;
  c10::intrusive_ptr<FileStore> file_store_;
  c10::intrusive_ptr<ProcessGroupGloo> cpu_process_group_;
};

} // namespace comm_backend