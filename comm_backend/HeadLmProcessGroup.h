#pragma once

#include <torch/csrc/distributed/c10d/Backend.hpp>

namespace comm_backend {

using c10d::Work;

class HeadLmProcessGroup : public Backend {
 public:
  class HeadLmWork : public Work {
    // TODO: current this is a class using for pass compile
  }

  HeadLmProcessGroup(int rank, int size) : Backend(rank, size) {}

  c10::intrusive_ptr<Work> send(std::vector<at::Tensor>& tensors, int dstRank,
                                int tag) override {
                                    return c10::make_intrusive<HeadLmWork>();
                                }

  c10::intrusive_ptr<Work> recv(std::vector<at::Tensor>& tensors, int srcRank,
                                int tag) override {
                                    return c10::make_intrusive<HeadLmWork>();
                                }
  
}


}  // namespace comm_backend