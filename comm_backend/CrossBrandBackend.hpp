#pragma once

#include <ctime>
#include <memory>

#include <torch/extension.h>

#include <c10/core/DeviceType.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>

#include <torch/python.h>

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>

#include "head_ccl/topo.hpp"
#include "head_ccl/communicator.hpp"

namespace comm_backend {

using c10::DeviceType;

using c10d::Backend;
using c10d::FileStore;
using c10d::OpType;
using c10d::PrefixStore;
using c10d::ProcessGroupGloo;
using c10d::Work;

class WorkDummy : public Work {
  public:
    WorkDummy(
      OpType opType,
      c10::intrusive_ptr<c10::ivalue::Future> future) // future of the output
      : Work(
          -1, // rank, only used by recvAnySource, irrelevant in this demo
          opType),
      future_(std::move(future)) {}
      bool isCompleted() override;
      bool isSuccess() const override;
      bool wait(std::chrono::milliseconds timeout = c10d::kUnsetTimeout) override;
      virtual c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;
    
  private:
    c10::intrusive_ptr<c10::ivalue::Future> future_;
};

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
  int commSendPort(int rank_i, int rank_j) const;

  DeviceType origin_device_type_;
  head_ccl::TopoGraph* topo_graph_;
  std::vector<head_ccl::Communicator> comms_;
  int world_size_;
  int rank_;
  
};

} // namespace comm_backend