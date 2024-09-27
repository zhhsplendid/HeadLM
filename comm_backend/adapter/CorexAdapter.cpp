#include "adapter/CorexAdapter.hpp"

#include <cstdlib>

#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>

namespace comm_backend {
namespace adapter {

CorexAdapter::CorexAdapter() {}

void CorexAdapter::setUp() const {
  putenv("NCCL_ALGO=RING");
  putenv("IXCCL_MIX_NV=1");
  putenv("IXCCL_FUSED_ENABLE=0");
}

c10::intrusive_ptr<c10d::Backend> CorexAdapter::createProcessGroup(
    const c10::intrusive_ptr<::c10d::Store> &store, int rank, int size,
    const std::chrono::duration<float> &timeout) const {
  return c10::make_intrusive<::c10d::ProcessGroupNCCL>(store, rank, size);
}

} // namespace adapter
} // namespace comm_backend
