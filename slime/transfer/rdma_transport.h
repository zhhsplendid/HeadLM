#pragma once

#include <infiniband/verbs.h>
#include <stdexcept>


namespace slime {

typedef struct __attribute__((packed)) rdma_conn_info_t {
    uint32_t qpn;
    uint32_t psn;
    union ibv_gid gid;  // RoCE v2
    uint16_t lid;       // IB
    uint32_t mtu;       // peers should have the same mtu
} rdma_conn_info_t;

class RDMAContext {
public:
  RDMAContext() {}
  ~RDMAContext() {}
  void openRDMADevice(std::string device, uint8_t port, int gid_index) {
    std::runtime_error("NotImplementedError");
  }

  void construct() { throw std::runtime_error("NotImplementedError"); }

  // Modify Queue Pair (qp) state to Init
  int32_t modify_qp_to_init();

  // Modify Queue Pair (qp) state to Ready To Receive (rtr) 
  int32_t modify_qp_to_rtr();

  // Modify Queue Pair (qp) state to Ready to Send (rts)
  int32_t modify_qp_to_rts();

  int32_t init_rdma_context(std::string dev_name, uint8_t ib_port,
                            std::string link_type);

  int32_t create_endpoint(std::string remote_server_addr);

  int32_t register_metadata(std::string metadata_endpoint) {
    throw std::runtime_error("NotImplementedError");
  }
private:
  ibv_mtu active_mtu_;

  rdma_conn_info_t local_info_;
  rdma_conn_info_t remote_info_;
};

class RDMATransport {
public:
  RDMATransport() {}
  ~RDMATransport() {}
  void init() { std::runtime_error("NotImplementedError"); }
};

} // namespace slime