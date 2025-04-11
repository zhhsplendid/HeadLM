#include "communicator.hpp"

#include <functional>
#include <string>

#include <zmq.hpp>

#include <torch/python.h>

#include "head_ccl/transport/memory_pool.h"
#include "head_ccl/transport/rdma_types.h"
#include "utils/json.hpp"
#include "utils/logging.h"

namespace comm_backend {
namespace head_ccl {

using nlohmann::json;
using transport::MrInfo;

comm_backend::head_ccl::Communicator::Communicator(const std::string &dev_name, int ib_port, const std::string &link_type)
{
    rdma_ctx_.init_rdma_context(dev_name, ib_port, link_type);
}

Communicator::~Communicator()
{
    rdma_ctx_.stop_cq_future();
    if (send_socket_ != nullptr) {
        delete send_socket_;
    }
    if (recv_socket_ != nullptr) {
        delete recv_socket_;
    }
}

void Communicator::tcpMetaConnect(const std::string& remote_addr,
                             int remote_port,
                             int local_port) {
    zmq_ctx_ = new zmq::context_t(2);
    send_socket_ = new zmq::socket_t(*zmq_ctx_, ZMQ_PUSH);
    send_socket_->connect("tcp://" + remote_addr + ":" + std::to_string(remote_port));
    recv_socket_ = new zmq::socket_t(*zmq_ctx_, ZMQ_PULL);
    recv_socket_->bind("tcp://*:" + std::to_string(local_port));

    RdmaInfo local_rdma_info = rdma_ctx_.get_local_rdma_info();
    json local_rdma_json = local_rdma_info.to_json();
    
    std::string str_data = local_rdma_json.dump();
    zmq::message_t send_msg(str_data.size());
    memcpy(send_msg.data(), str_data.data(), str_data.size());
    send_socket_->send(send_msg);

    zmq::message_t recv_msg;
    recv_socket_->recv(recv_msg);
    json remote_rdma_json = json::parse(std::string(static_cast<char*>(recv_msg.data()), recv_msg.size()));
    RdmaInfo remote_rdma_info(remote_rdma_json);
    rdma_ctx_.modify_qp_to_rtsr(remote_rdma_info);
}

void Communicator::sendTensorAsync(const at::Tensor &tensor)
{
    HEADLM_ASSERT_NE(zmq_ctx_, nullptr, "Please call tcpMetaConnect before send/recv Tensor");
    uint64_t data_ptr = reinterpret_cast<uint64_t>(tensor.data_ptr());
    size_t length = tensor.numel() * tensor.element_size(); 
    std::string mr_key = std::to_string(data_ptr) + "L" + std::to_string(length);
    MrInfo mr_info = rdma_ctx_.register_memory_region(mr_key, data_ptr, length);

    json mr_info_json = mr_info.to_json();
    std::string str_data = mr_info_json.dump();
    zmq::message_t msg(str_data.size());
    memcpy(msg.data(), str_data.data(), str_data.size());
    send_socket_->send(msg, zmq::send_flags::dontwait);
}

void Communicator::recvTensorAsync(at::Tensor *tensor, const std::function<void(int64_t)> &callback)
{
    HEADLM_ASSERT_NE(zmq_ctx_, nullptr, "Please call tcpMetaConnect before send/recv Tensor");
    uint64_t data_ptr = reinterpret_cast<uint64_t>(tensor->data_ptr());
    size_t length = tensor->numel() * tensor->element_size(); 
    std::string mr_key = std::to_string(data_ptr) + "L" + std::to_string(length);
    rdma_ctx_.register_memory_region(mr_key, data_ptr, length);

    zmq::message_t recv_msg;
    recv_socket_->recv(recv_msg);
    json mr_info_json = json::parse(std::string(static_cast<char*>(recv_msg.data()), recv_msg.size()));
    MrInfo mr_info(mr_info_json);
    rdma_ctx_.register_remote_memory_region(mr_key, mr_info);

    rdma_ctx_.r_rdma_async(mr_key, 0, 0, length, callback);
}

} // namespace head_ccl
}  // namespace comm_backend