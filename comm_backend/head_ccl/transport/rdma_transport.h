#pragma once



#include <cstdint>
#include <functional>
#include <future>
#include <infiniband/verbs.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "head_ccl/transport/memory_pool.h"
#include "head_ccl/transport/rdma_types.h"
#include "utils/json.hpp"

namespace comm_backend {
namespace head_ccl {
namespace transport {

using json = nlohmann::json;

class RdmaContext {
public:
  /*
    A link of rdma QP.
  */
  RdmaContext() {}

  ~RdmaContext();

  /* Initialize */
  int init_rdma_context(const std::string &dev_name, uint8_t ib_port,
                        const std::string &link_type);

  /* RDMA Info Exchange */
  int modify_qp_to_rtsr(const RdmaInfo &remote_rdma_info);

  /* Memory Allocation */
  int register_memory_region(const std::string &mr_key, uintptr_t data_ptr,
                             size_t length) {
    memory_pool_.register_memory_region(mr_key, data_ptr, length);
    return 0;
  }

  int register_remote_memory_region(const std::string &mr_key,
                                    const json &json_mr_info) {
    MrInfo mr_info(json_mr_info);
    memory_pool_.register_remote_memory_region(mr_key, mr_info);
    return 0;
  }

  /* Async RDMA SendRecv */
  int send_async(const std::string &mr_key, uint64_t offset, uint64_t length,
                 const std::function<void(int64_t)> &callback);
  int recv_async(const std::string &mr_key, uint64_t offset, uint64_t length,
                 const std::function<void(int64_t)> &callback);

  /* Async RDMA Read */
  int r_rdma_async(const std::string &mr_key, uint64_t target_offset,
                   uint64_t source_offset, uint64_t length,
                   const std::function<void(int64_t)>& callback);
  int batch_r_rdma_async(const std::string &mr_key,
                         const std::vector<uint64_t> &target_offsets,
                         const std::vector<uint64_t> &source_offsets,
                         uint64_t length,
                         const std::function<void(int64_t)> &callback);

  /* Completion Queue Polling */
  int cq_poll_handle();

  void launch_cq_future();
  void stop_cq_future();

  rdma_info_t get_local_rdma_info() { return local_rdma_info_; }
  rdma_info_t get_remote_rdma_info() { return remote_rdma_info_; }

  json local_info() {
    return json{{"rdma_info", local_rdma_info_.to_json()},
                {"mr_info", memory_pool_.local_mr_info_to_json()}};
  }

private:
  std::string device_name_ = "";

  /* RDMA Configuration */
  struct ibv_context *ib_ctx_ = nullptr;
  struct ibv_pd *pd_ = nullptr;
  struct ibv_comp_channel *comp_channel_ = nullptr;
  struct ibv_cq *cq_ = nullptr;
  struct ibv_qp *qp_ = nullptr;
  uint8_t ib_port_ = -1;

  MemoryPool memory_pool_;

  /* RDMA Exchange Information */
  rdma_info_t remote_rdma_info_;
  rdma_info_t local_rdma_info_;

  /* State Management */
  bool initialized_ = false;
  bool connected_ = false;
  std::atomic<int> outstanding_rdma_reads_{0};
  std::atomic<bool> stop_{false};

  /* Send Mutex */
  std::mutex rdma_post_send_mutex_;

  /* async cq handler */
  std::future<void> cq_future_;

  std::mutex mutex_;
};

} // namespace transport
} // namespace head_ccl
}  // namespace comm_backend