#pragma once

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

  int64_t modify_qp_to_init();
  int64_t modify_qp_to_rtr() {
    throw std::runtime_error("NotImplementedError");
  }
  int64_t modify_qp_to_rts() {
    throw std::runtime_error("NotImplementedError");
  }
  int64_t init_rdma_context(std::string dev_name, uint8_t ib_port,
                            std::string link_type);
  int64_t registerMemoryRegion(int64_t addr, size_t length);
  void cq_poll_handle();
  int64_t rdma_exchange(uint32_t psn);

  int32_t register_metadata(std::string metadata_endpoint) {
    throw std::runtime_error("NotImplementedError");
  }

private:
};

class RDMATransport {
public:
  RDMATransport() {}
  ~RDMATransport() {}
  void init() { std::runtime_error("NotImplementedError"); }
};

} // namespace slime