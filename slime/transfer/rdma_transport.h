#pragma once

#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <stdexcept>
#include <sys/socket.h>

#include "info_struct/config.h"

namespace slime {
namespace transfer {

class RDMAContext {
public:
  RDMAContext() {}
  ~RDMAContext() {}
  void openRDMADevice(std::string device, uint8_t port, int gid_index) {
    std::runtime_error("NotImplementedError");
  }

  void construct() { throw std::runtime_error("NotImplementedError"); }

  int32_t connect_client(const client_config_t& config);

  //int32_t setup_rdma(const client_config_t& config);

  // Modify Queue Pair (qp) state to Init
  int32_t modify_qp_to_init();

  // Modify Queue Pair (qp) state to Ready To Receive (rtr) 
  int32_t modify_qp_to_rtr();

  // Modify Queue Pair (qp) state to Ready to Send (rts)
  int32_t modify_qp_to_rts();

  int32_t init_rdma_context(const std::string& dev_name,
                            uint8_t ib_port,
                            const std::string& link_type);

  int32_t create_endpoint(std::string remote_server_addr);

  int32_t register_metadata(std::string metadata_endpoint) {
    throw std::runtime_error("NotImplementedError");
  }
private:
  ibv_mtu active_mtu_;

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
};

class RDMATransport {
public:
  RDMATransport() {}
  ~RDMATransport() {}
  void init() { std::runtime_error("NotImplementedError"); }
};

} // namespace transfer
} // namespace slime