#pragma once

#include <ctime>

#include <torch/extension.h>

#include <c10/core/DeviceType.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>

#include <torch/python.h>

#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>

#include <iostream>

namespace comm_backend {

using c10::DeviceType;

using c10d::Backend;
using c10d::OpType;
using c10d::PrefixStore;
using c10d::FileStore;
using c10d::ProcessGroupGloo;
using c10d::Work;

class HeadLmProcessGroup : public Backend {
public:

  class ToDeviceRecvWork : public Work {
   public:
    ToDeviceRecvWork(c10::intrusive_ptr<Work>& gloo_recv_work,
                     std::vector<at::Tensor> &tensors,
                     DeviceType device_type) : 
                     Work(-1, OpType::RECV, "gloo:revc", std::optional<std::vector<at::Tensor>>(tensors)),
                     gloo_recv_work_(gloo_recv_work),
                     tensors_(tensors),
                     origin_device_type_(device_type) {}

    int sourceRank() const override {
      return gloo_recv_work_->sourceRank();
    }

    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override {
      bool ret = gloo_recv_work_->wait();
     
      if (origin_device_type_ != DeviceType::CPU) {
        for (at::Tensor &tensor : tensors_) {
          tensor.to(origin_device_type_);
        }
      }
      return ret;
    }

    void abort() override {
      return gloo_recv_work_->abort();
    }

    uint64_t getSequencenumber() const override {
      return gloo_recv_work_->getSequencenumber();
    }

    std::vector<at::Tensor> result() override {
      auto ans = gloo_recv_work_->result();
      if (origin_device_type_ != DeviceType::CPU) {
        for (at::Tensor &tensor : tensors_) {
          tensor.to(origin_device_type_);
        }
      }
      return ans;
    }
      
   private:
     c10::intrusive_ptr<Work> gloo_recv_work_;
     std::vector<at::Tensor> &tensors_;
     DeviceType origin_device_type_;
  };

  HeadLmProcessGroup(int rank, int size, DeviceType device_type);

  c10::intrusive_ptr<Work> send(std::vector<at::Tensor> &tensors, int dstRank,
                                int tag) override;

  c10::intrusive_ptr<Work> recv(std::vector<at::Tensor> &tensors, int srcRank,
                                int tag) override;

  static c10::intrusive_ptr<Backend>
  createHeadLmProcessGroup(const c10::intrusive_ptr<::c10d::Store> &store, int rank, int size,
    const std::chrono::duration<float> &timeout);

  static void HeadLmProcessGroupConstructor() __attribute__((constructor)) {
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