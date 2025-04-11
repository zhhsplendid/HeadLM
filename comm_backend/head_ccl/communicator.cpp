#include "communicator.hpp"


namespace comm_backend {
namespace head_ccl {

comm_backend::head_ccl::Communicator::Communicator(const std::string &dev_name, int ib_port, const std::string &link_type)
{
    rdma_ctx_.init_rdma_context(dev_name, ib_port, link_type);
}

void Communicator::connectTo(const std::string& remote_addr)
{
    rdma_ctx_.modify_qp_to_rtsr(remote_rdma_info);
}

} // namespace head_ccl
}  // namespace comm_backend