#pragma once

#include "config.h"

#include <cstdint>
#include <infiniband/verbs.h>
#include <stdexcept>
#include <vector>

namespace slime {

class RDMAContext {
public:
  RDMAContext() {}
  ~RDMAContext() {}
  void openRDMADevice(std::string device, uint8_t port, int gid_index) {
    std::runtime_error("NotImplementedError");
  }
  void construct() { throw std::runtime_error("NotImplementedError"); }

  int64_t modify_qp_to_init();
  int64_t modify_qp_to_rtr() {
    throw std::runtime_error("NotImplementedError");
  }
  int64_t modify_qp_to_rts() {
    throw std::runtime_error("NotImplementedError");
  }
  void modify_qp_to_rtsr(RDMAInfo info);
  int64_t init_rdma_context(std::string dev_name, uint8_t ib_port,
                            std::string link_type);
  int64_t registerMemoryRegion(int64_t addr, size_t length);
  int64_t getRKey(int64_t mr_idx);
  void cq_poll_handle();
  int64_t rdma_exchange();

  int32_t register_metadata(std::string metadata_endpoint) {
    throw std::runtime_error("NotImplementedError");
  }

  void post_recv(struct ibv_sge *recv_sge, uint64_t info);
  int64_t r_rdma_async(uint64_t info, uintptr_t target_addr,
                       uintptr_t source_addr, uint64_t length, int64_t lkey,
                       uintptr_t wid);
  rdma_info_t get_local_rdma_info() { return local_rdma_info_; }

private:
  rdma_info_t remote_rdma_info_;
  rdma_info_t local_rdma_info_;
  std::string device_name_ = "";
  struct ibv_context *ib_ctx_;
  uint8_t ib_port_ = -1;
  int64_t gidx_ = -1;
  uint16_t lid_ = 0;
  ibv_mtu active_mtu_;
  struct ibv_pd *pd_ = nullptr;
  struct ibv_comp_channel *comp_channel_ = nullptr;
  struct ibv_cq *cq_ = nullptr;
  struct ibv_qp *qp_ = nullptr;
  std::vector<ibv_mr *> memory_region_list_;
  bool rdma_connected_ = false;
  uint64_t psn_;
};

class RDMATransport {
public:
  RDMATransport() {}
  ~RDMATransport() {}
  void init() { std::runtime_error("NotImplementedError"); }
};

} // namespace slime