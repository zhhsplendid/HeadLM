#pragma once

#include <string>

#include "comm_backend/head_ccl/transport/rdma_transport.h"
#include "comm_backend/head_ccl/transport/rdma_types.h"

namespace comm_backend {
namespace head_ccl {

using transport::RdmaContext;
using transport::RdmaInfo;

/** 
 * A class wrapper basic RDMA communication between.
 * We may change it to support more communication types in the future
 */
class Communicator {
public:
  Communicator(const std::string& dev_name,
               int ib_port,
               const std::string& link_type = "Ethernet");

  void connectTo(const std::string& remote_addr);

 
  
private:
  RdmaContext rdma_ctx_;

};

} // namespace head_ccl
}  // namespace comm_backend
