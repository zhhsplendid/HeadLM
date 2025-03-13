#include "rdma_transport.h"
#include "ibv_helper.h"
#include "logging.h"

#include <bits/socket.h>
#include <infiniband/verbs.h>
#include <stdexcept>

namespace slime {

// this number should be big for lots of RMDA_WRITE requests
#define MAX_SEND_WR 8192

// this is only used for recving RDMA_SEND or IMM data. this should be bigger
// than max layers of model.
#define MAX_RECV_WR 64

struct ibv_context *ib_ctx_;
uint8_t ib_port_ = -1;
int64_t gidx_ = -1;
uint16_t lid_ = 0;
ibv_mtu active_mtu_;
struct ibv_pd *pd_ = nullptr;
struct ibv_comp_channel *comp_channel_ = nullptr;
struct ibv_cq *cq_ = nullptr;
struct ibv_qp *qp_ = nullptr;

int32_t RDMAContext::create_endpoint(std::string remote_server_addr) {
  throw std::runtime_error("NotImplementedError");
}

int32_t RDMAContext::init_rdma_context(std::string dev_name, uint8_t ib_port,
                                       std::string link_type) {
  struct ibv_device **dev_list;
  struct ibv_device *ib_dev;

  int num_devices = 0;
  dev_list = ibv_get_device_list(&num_devices);

  if (!dev_list) {
    throw std::runtime_error("Failed to get RDMA device list");
    return -1;
  }

  for (int i = 0; i < num_devices; i++) {
    char *dev_name_from_list = (char *)ibv_get_device_name(dev_list[i]);
    if (strcmp(dev_name_from_list, dev_name.c_str()) == 0) {
      ib_dev = dev_list[i];
      ib_ctx_ = ibv_open_device(ib_dev);
      break;
    }
  }

  if (!ib_ctx_) {
    SLIME_LOG_INFO(
        "Can't find or failed to open the specified device, try to open "
        "the default device {}"
        << (char *)ibv_get_device_name(dev_list[0]));
    ib_ctx_ = ibv_open_device(dev_list[0]);
    if (!ib_ctx_) {
      SLIME_ABORT("Failed to open the default device");
      return -1;
    }
  }

  struct ibv_port_attr port_attr;
  ib_port_ = ib_port;

  if (ibv_query_port(ib_ctx_, ib_port, &port_attr)) {
    SLIME_ABORT("Unable to query port {} attributes\n" << ib_port);
    return -1;
  }
  if ((port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND &&
       link_type == "Ethernet") ||
      (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET &&
       link_type == "Infiniband")) {
    SLIME_ABORT("port link layer and config link type don't match");
    return -1;
  }

  if (port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
    gidx_ = -1;
  } else {
    gidx_ = ibv_find_sgid_type(ib_ctx_, ib_port, IBV_GID_TYPE_ROCE_V2, AF_INET);
  }

  lid_ = port_attr.lid;
  active_mtu_ = port_attr.active_mtu;

  pd_ = ibv_alloc_pd(ib_ctx_);
  SLIME_ASSERT(pd_, "Failed to allocate PD");

  comp_channel_ = ibv_create_comp_channel(ib_ctx_);
  SLIME_ASSERT(comp_channel_, "Failed to create completion channel");

  cq_ = ibv_create_cq(ib_ctx_, MAX_SEND_WR + MAX_RECV_WR, nullptr,
                      comp_channel_, 0);
  SLIME_ASSERT(cq_, "Failed to create CQ");

  SLIME_ASSERT(!ibv_req_notify_cq(cq_, 0), "Failed to request CQ notification");

  struct ibv_qp_init_attr qp_init_attr = {};
  qp_init_attr.send_cq = cq_;
  qp_init_attr.recv_cq = cq_;
  qp_init_attr.qp_type = IBV_QPT_RC;
  qp_init_attr.cap.max_send_wr = MAX_SEND_WR;
  qp_init_attr.cap.max_recv_wr = MAX_RECV_WR;
  qp_init_attr.cap.max_send_sge = 1;
  qp_init_attr.cap.max_recv_sge = 1;

  qp_ = ibv_create_qp(pd_, &qp_init_attr);
  SLIME_ASSERT(qp_, "Failed to create QP, " << strerror(errno));

  SLIME_ASSERT(!modify_qp_to_init(),
               "Failed to modify QP to INIT, " << strerror(errno));

  return 0;
}

int32_t RDMAContext::modify_qp_to_init() {
  struct ibv_qp_attr attr = {};
  attr.qp_state = IBV_QPS_INIT;
  attr.port_num = ib_port_;
  attr.pkey_index = 0;
  attr.qp_access_flags =
      IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;

  int flags =
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

  int ret = ibv_modify_qp(qp_, &attr, flags);
  if (ret) {
    SLIME_ABORT("Failed to modify QP to INIT");
    return ret;
  }
  return 0;
}

int32_t RDMAContext::modify_qp_to_rtr() {
  struct ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_RTR;

    // update MTU
    if (remote_info_.mtu != active_mtu_) {
      std::string warn_str = "remote MTU: "
          + std::to_string(1 << ((uint32_t)remote_info_.mtu + 7))
          + " , local MTU: {} is not the same, update to minimal MTU"
          + std::to_string(1 << ((uint32_t)active_mtu_ + 7));
       SLIME_WARN(warn_str);
    }
    attr.path_mtu = (enum ibv_mtu) std::min((uint32_t)active_mtu_, (uint32_t)remote_info_.mtu);

    attr.dest_qp_num = remote_info_.qpn;
    attr.rq_psn = remote_info_.psn;
    attr.max_dest_rd_atomic = 4;
    attr.min_rnr_timer = 12;
    attr.ah_attr.dlid = 0;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = ib_port_;

    if (gidx_ == -1) {
        // IB
        attr.ah_attr.dlid = remote_info_.lid;
        attr.ah_attr.is_global = 0;
    }
    else {
        // RoCE v2
        attr.ah_attr.is_global = 1;
        attr.ah_attr.grh.dgid = remote_info_.gid;
        attr.ah_attr.grh.sgid_index = gidx_;  // local gid
        attr.ah_attr.grh.hop_limit = 1;
    }

    int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

    int ret = ibv_modify_qp(qp_, &attr, flags);
    if (ret) {
        std::runtime_error("Failed to modify QP to RTR");
        return ret;
    }
    return 0;
}

int32_t RDMAContext::modify_qp_to_rts() {
  struct ibv_qp_attr attr = {};
  attr.qp_state = IBV_QPS_RTS;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.sq_psn = local_info_.psn; // Use 0 or match with local PSN?
  attr.max_rd_atomic = 1;

  int flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
              IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;

  int ret = ibv_modify_qp(qp_, &attr, flags);
  if (ret) {
    throw std::runtime_error("Failed to modify QP to RTS");
    return ret;
  }
  return 0;
}



} // namespace slime
