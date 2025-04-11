#pragma once

#include <functional>
#include <string>

#include <zmq.hpp>

#include <torch/python.h>

#include "head_ccl/transport/rdma_transport.h"
#include "head_ccl/transport/rdma_types.h"

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
  
  ~Communicator();

  void tcpMetaConnect(const std::string &remote_addr, int remote_port, int local_port);

  void sendTensorAsync(const at::Tensor& tensor);

  void recvTensorAsync(at::Tensor* tensor, const std::function<void(int64_t)> &callback);
  
private:
  RdmaContext rdma_ctx_;

  zmq::context_t* zmq_ctx_ = nullptr;
  zmq::socket_t* send_socket_ = nullptr;
  zmq::socket_t* recv_socket_ = nullptr;
  
};

} // namespace head_ccl
}  // namespace comm_backend
