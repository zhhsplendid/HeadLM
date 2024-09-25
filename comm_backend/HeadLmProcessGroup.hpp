#pragma once

#include <ctime>
#include <memory>

#include <torch/extension.h>

#include <c10/core/DeviceType.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>

#include <torch/python.h>

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>

namespace comm_backend {

class HeadLmProcessGroup {
public:
  static c10::intrusive_ptr<c10d::Backend>
  createHeadLmProcessGroup(const c10::intrusive_ptr<::c10d::Store> &store,
                           int rank, int size,
                           const std::chrono::duration<float> &timeout);

  static void HeadLmProcessGroupConstructor() __attribute__((constructor)) {
    py::object module = py::module::import("torch.distributed");
    py::object register_backend =
        module.attr("Backend").attr("register_backend");
    register_backend("headlm", py::cpp_function(createHeadLmProcessGroup));
  }
};

} // namespace comm_backend