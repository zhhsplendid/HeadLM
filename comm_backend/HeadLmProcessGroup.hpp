#pragma once

#include <ctime>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>

namespace comm_backend {

using c10d::Backend;
using c10d::FileStore;
using c10d::ProcessGroupGloo;
using c10d::Work;

class HeadLmProcessGroup : public Backend {
 public:
  HeadLmProcessGroup(int rank, int size) : Backend(rank, size) {
    std::time_t cur_time = std::time(nullptr);
    file_store_ = c10::make_intrusive<FileStore>("/tmp/headlm_" + std::to_string(cur_time), size);
    cpu_process_group_ = c10::make_intrusive<ProcessGroupGloo>(file_store_, rank, size);
  }

  c10::intrusive_ptr<Work> send(std::vector<at::Tensor>& tensors, int dstRank,
                                int tag) override {
                                    for (at::Tensor& tensor : tensors) {
                                      tensor.cpu();
                                    }
                                    return cpu_process_group_->send(tensors, dstRank, tag);
                                }

  c10::intrusive_ptr<Work> recv(std::vector<at::Tensor>& tensors, int srcRank,
                                int tag) override {
                                    c10::intrusive_ptr<Work> ret = cpu_process_group_->recv(tensors, srcRank, tag);
                                    return ret;
                                }
  private:
    c10::intrusive_ptr<FileStore> file_store_;
    c10::intrusive_ptr<ProcessGroupGloo> cpu_process_group_;
  
};


}  // namespace comm_backend