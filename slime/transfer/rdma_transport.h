#pragma once

#include <infiniband/verbs.h>
#include <stdexcept>


namespace slime {

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
  int32_t modify_qp_to_rtr() {
    throw std::runtime_error("NotImplementedError");
  }

  // Modify Queue Pair (qp) state to Ready to Send (rts)
  int32_t modify_qp_to_rts();

  int32_t init_rdma_context(std::string dev_name, uint8_t ib_port,
                            std::string link_type);
  int32_t create_endpoint(std::string remote_server_addr);
  int32_t register_metadata(std::string metadata_endpoint) {
    throw std::runtime_error("NotImplementedError");
  }

};

class RDMATransport {
public:
  RDMATransport() {}
  ~RDMATransport() {}
  void init() { std::runtime_error("NotImplementedError"); }
};

} // namespace slime