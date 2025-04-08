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

using c10::DeviceType;

using c10d::Backend;
using c10d::FileStore;
using c10d::OpType;
using c10d::PrefixStore;
using c10d::ProcessGroupGloo;
using c10d::Work;

class CrossBrandBackend : public Backend {
public:

  CrossBrandBackend(const c10::intrusive_ptr<::c10d::Store> &store, int rank, int size,
             const std::chrono::duration<float> &timeout,
             DeviceType device_type);

  c10::intrusive_ptr<Work> send(std::vector<at::Tensor> &tensors, int dstRank,
                                int tag) override;

  c10::intrusive_ptr<Work> recv(std::vector<at::Tensor> &tensors, int srcRank,
                                int tag) override;

private:
  DeviceType origin_device_type_;
};

} // namespace comm_backend