#pragma once

#include <arpa/inet.h>
#include <atomic>
#include <boost/lockfree/spsc_queue.hpp>
#include <condition_variable>
#include <deque>
#include <future>
#include <infiniband/verbs.h>
#include <mutex>
#include <stdexcept>
#include <sys/socket.h>
#include <unordered_map>

#include "info_struct/config.h"

namespace slime {
namespace transfer {

// RDMA send buffer
// because write_cache will be invoked asynchronously,
// so each request will have a standalone send buffer.
struct SendBuffer {
  void *buffer_ = NULL;
  struct ibv_mr *mr_ = NULL;

  SendBuffer(struct ibv_pd *pd, size_t size);
  SendBuffer(const SendBuffer &) = delete;
  ~SendBuffer();
};

class RDMAContext {
public:
  RDMAContext() {}
  ~RDMAContext() {}

  int32_t connect_client(const client_config_t &config);

  int32_t init_rdma_context(const std::string &dev_name, uint8_t ib_port,
                            const std::string &link_type);

  int32_t setup_rdma(const client_config_t &config);

  int32_t register_mr(void *base_ptr, size_t ptr_region_size);

  int32_t create_endpoint(std::string remote_server_addr) {
    throw std::runtime_error("NotImplementedError");
  }

  void openRDMADevice(std::string device, uint8_t port, int gid_index) {
    std::runtime_error("NotImplementedError");
  }

  void construct() { throw std::runtime_error("NotImplementedError"); }

  int32_t register_metadata(std::string metadata_endpoint) {
    throw std::runtime_error("NotImplementedError");
  }

private:
  int32_t exchange_conn_info();

  // Modify Queue Pair (qp) state to Init
  int32_t modify_qp_to_init();

  // Modify Queue Pair (qp) state to Ready To Receive (rtr)
  int32_t modify_qp_to_rtr();

  // Modify Queue Pair (qp) state to Ready to Send (rts)
  int32_t modify_qp_to_rts();

  void cq_handler();

  SendBuffer *get_send_buffer();

  void release_send_buffer(SendBuffer *buffer);

  void post_recv(struct ibv_sge *recv_sge, rdma_info_base *info);

private:
  rdma_conn_info_t local_info_;
  rdma_conn_info_t remote_info_;

  // tcp socket
  int sock_ = 0;

  // rdma connections
  struct ibv_context *ib_ctx_ = NULL;
  struct ibv_pd *pd_ = NULL;
  struct ibv_cq *cq_ = NULL;
  struct ibv_qp *qp_ = NULL;
  int gidx_ = -1;
  int lid_ = -1;
  uint8_t ib_port_ = -1;

  struct ibv_comp_channel *comp_channel_ = NULL;

  // local active_mtu attr, after exchanging with remote, we will use the min of
  // the two for path.mtu
  ibv_mtu active_mtu_;
  /*
    This is MAX_RECV_WR not MAX_SEND_WR,
    because server also has the same number of buffers
    */
  boost::lockfree::spsc_queue<SendBuffer *> send_buffers_{MAX_RECV_WR};

  // this recv buffer is used in
  // 1. allocate rdma
  // 2. recv IMM data, although IMM DATA is not put into recv_buffer,
  // but for compatibility, we still use a zero-length recv_buffer.
  void *recv_buffer_ = NULL;
  struct ibv_mr *recv_mr_ = NULL;

  std::atomic<int> rdma_inflight_count_{0};
  std::atomic<bool> stop_{false};
  std::future<void> cq_future_; // cq thread

  // protect rdma_inflight_count
  std::mutex mutex_;
  std::condition_variable cv_;

  // protect ibv_post_send, outstanding_rdma_writes_queue
  std::mutex rdma_post_send_mutex_;
  std::atomic<int> outstanding_rdma_writes_{0};
  std::deque<std::pair<struct ibv_send_wr *, struct ibv_sge *>>
      outstanding_rdma_writes_queue_;

  std::unordered_map<uintptr_t, struct ibv_mr *> local_mr_;
};

class RDMATransport {
public:
  RDMATransport() {}
  ~RDMATransport() {}
  void init() { std::runtime_error("NotImplementedError"); }
};

} // namespace transfer
} // namespace slime