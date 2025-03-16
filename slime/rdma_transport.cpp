#include "rdma_transport.h"
#include "ibv_helper.h"
#include "logging.h"
#include "utils.h"

#include <cassert>
#include <unistd.h>
#include <vector>

#include <bits/socket.h>
#include <infiniband/verbs.h>
#include <stdexcept>

namespace slime {

// this number should be big for lots of RMDA_WRITE requests
#define MAX_SEND_WR 8192

// this is only used for recving RDMA_SEND or IMM data. this should be bigger
// than max layers of model.
#define MAX_RECV_WR 64

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

int64_t RDMAContext::rdma_exchange(uint32_t psn) {

  int ret;

  if (rdma_connected_ == true) {
    SLIME_ABORT("already connected");
  }

  assert(ib_ctx_);
  if (!ib_ctx_) {
    std::cout << "!!!" << std::endl;
  }
  std::cout << "ib_ctx: " << (void *)ib_ctx_ << std::endl;
  ;
  comp_channel_ = ibv_create_comp_channel(ib_ctx_);

  // assert(comp_channel_ != NULL);
  std::cout << "???" << std::endl;

  // cq_ = ibv_create_cq(ib_ctx_, MAX_SEND_WR + MAX_RECV_WR, NULL,
  // comp_channel_, 0); assert(!cq_);

  // std::cout << "???" << std::endl;

  // // Create Queue Pair
  // struct ibv_qp_init_attr qp_init_attr = {};
  // qp_init_attr.send_cq = cq_;
  // qp_init_attr.recv_cq = cq_;
  // qp_init_attr.qp_type = IBV_QPT_RC;  // Reliable Connection
  // qp_init_attr.cap.max_send_wr = MAX_SEND_WR;
  // qp_init_attr.cap.max_recv_wr = MAX_RECV_WR;
  // qp_init_attr.cap.max_send_sge = 1;
  // qp_init_attr.cap.max_recv_sge = 1;

  // qp_ = ibv_create_qp(pd_, &qp_init_attr);
  // if (!qp_) {
  //     SLIME_ABORT("Failed to create QP");
  //     return -1;
  // }
  // std::cout << "111???" << std::endl;
  // // Modify QP to INIT state
  // struct ibv_qp_attr attr = {};
  // attr.qp_state = IBV_QPS_INIT;
  // attr.port_num = 1;
  // attr.pkey_index = 0;
  // attr.qp_access_flags =
  //     IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
  //     IBV_ACCESS_LOCAL_WRITE;

  // int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
  // IBV_QP_ACCESS_FLAGS;

  // ret = ibv_modify_qp(qp_, &attr, flags);
  // if (ret) {
  //     SLIME_ABORT("Failed to modify QP to INIT");
  // }
  // std::cout << "???" << std::endl;

  // union ibv_gid gid;
  // // get gid
  // if (gidx_ != -1 && ibv_query_gid(ib_ctx_, 1, gidx_, &gid)) {
  //     SLIME_ABORT("Failed to get GID");
  // }

  // std::cout << "???" << std::endl;

  // // local_info_.qpn = qp_->qp_num;
  // // local_info_.psn = lrand48() & 0xffffff;
  // // local_info_.gid = gid;
  // // local_info_.lid = lid;
  // // local_info_.mtu = (uint32_t)active_mtu;

  // SLIME_LOG_INFO("gid index: {}" << gidx_);
  // // print_rdma_conn_info(&local_info_, false);
  // // print_rdma_conn_info(&remote_info_, true);

  // // Modify QP to RTR state
  // memset(&attr, 0, sizeof(attr));
  // attr.qp_state = IBV_QPS_RTR;
  // attr.path_mtu = (enum ibv_mtu)std::min((uint32_t)active_mtu_,
  // (uint32_t)active_mtu_); attr.dest_qp_num = MAX_SEND_WR + MAX_RECV_WR;
  // attr.rq_psn = psn;
  // attr.max_dest_rd_atomic = 4;
  // attr.min_rnr_timer = 12;
  // attr.ah_attr.dlid = 0;  // RoCE v2 is used.
  // attr.ah_attr.sl = 0;
  // attr.ah_attr.src_path_bits = 0;
  // attr.ah_attr.port_num = 1;

  // if (gidx_ == -1) {
  //     // IB
  //     attr.ah_attr.dlid = lid_;
  //     attr.ah_attr.is_global = 0;
  // }
  // else {
  //     // RoCE v2
  //     attr.ah_attr.is_global = 1;
  //     attr.ah_attr.grh.dgid = gid;
  //     attr.ah_attr.grh.sgid_index = gidx_;
  //     attr.ah_attr.grh.hop_limit = 1;
  // }

  // flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
  // IBV_QP_RQ_PSN |
  //         IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

  // ret = ibv_modify_qp(qp_, &attr, flags);
  // if (ret) {
  //     SLIME_ABORT("Failed to modify QP to RTR: reason: {}" << strerror(ret));
  // }

  // // Modify QP to RTS state
  // memset(&attr, 0, sizeof(attr));
  // attr.qp_state = IBV_QPS_RTS;
  // attr.timeout = 14;
  // attr.retry_cnt = 7;
  // attr.rnr_retry = 7;
  // attr.sq_psn = lrand48() & 0xffffff;
  // attr.max_rd_atomic = 1;

  // flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY
  // | IBV_QP_SQ_PSN |
  //         IBV_QP_MAX_QP_RD_ATOMIC;

  // ret = ibv_modify_qp(qp_, &attr, flags);
  // if (ret) {
  //     SLIME_ABORT("Failed to modify QP to RTS");
  // }
  // SLIME_LOG_INFO("RDMA exchange done");
  // rdma_connected_ = true;

  // if (ibv_req_notify_cq(cq_, 0)) {
  //     SLIME_ABORT("Failed to request notify for CQ");
  // }

  return 0;
}

void RDMAContext::cq_poll_handle() {
  SLIME_LOG_INFO("Polling CQ");

  struct ibv_cq *cq;
  void *cq_context;

  if (ibv_get_cq_event(comp_channel_, &cq, &cq_context) != 0) {
    SLIME_ABORT("Failed to get CQ event");
  }

  if (ibv_req_notify_cq(cq, 0) != 0) {
    SLIME_ABORT("Failed to request CQ notification");
  }

  struct ibv_wc wc = {0};

  while (ibv_poll_cq(cq, 1, &wc) > 0) {
    sleep(5);
  }
}

int64_t RDMAContext::registerMemoryRegion(int64_t addr, size_t length) {
  const static int access_rights =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

  ibv_mr *mr = ibv_reg_mr(pd_, (void *)addr, length, access_rights);
  SLIME_ASSERT(mr, "Failed to register memory" << addr);
  memory_region_list_.push_back(mr);
  SLIME_LOG_INFO("Memory region: "
                 << (void *)addr << " -- " << (void *)((uintptr_t)addr + length)
                 << ", Device name: " << device_name_ << ", Length: " << length
                 << " (" << length / 1024 / 1024 << " MB)"
                 << ", Permission: " << access_rights << std::hex
                 << ", LKey: " << mr->lkey << ", RKey: " << mr->rkey);

  return 0;
}

int64_t RDMAContext::init_rdma_context(std::string dev_name, uint8_t ib_port,
                                       std::string link_type) {
  struct ibv_device **dev_list;
  struct ibv_device *ib_dev;
  int num_devices;
  dev_list = ibv_get_device_list(&num_devices);
  if (!dev_list) {
    SLIME_ABORT("Failed to get RDMA devices list");
    return -1;
  }

  for (int i = 0; i < num_devices; ++i) {
    char *dev_name_from_list = (char *)ibv_get_device_name(dev_list[i]);
    if (strcmp(dev_name_from_list, dev_name.c_str()) == 0) {
      SLIME_LOG_INFO("found device {}" << dev_name_from_list);
      ib_dev = dev_list[i];
      ib_ctx_ = ibv_open_device(ib_dev);
      break;
    }
  }

  if (!ib_ctx_) {
    SLIME_LOG_INFO("Can't find or failed to open the specified device, try to open "
         "the default device {}" <<
         (char *)ibv_get_device_name(dev_list[0]));
    ib_ctx_ = ibv_open_device(dev_list[0]);
    if (!ib_ctx_) {
      SLIME_ABORT("Failed to open the default device");
      return -1;
    }
  }

  struct ibv_port_attr port_attr;
  ib_port_ = ib_port;
  if (ibv_query_port(ib_ctx_, ib_port, &port_attr)) {
    SLIME_ABORT("Unable to query port {} attributes\n" << ib_port_);
    return -1;
  }
  if ((port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND &&
       link_type == "Ethernet") ||
      (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET &&
       link_type == "IB")) {
    SLIME_ABORT("port link layer and config link type don't match");
    return -1;
  }
  if (port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
    gidx_ = -1;
  } else {
    // gidx_ = ibv_find_sgid_type(ib_ctx_, ib_port_, IBV_GID_TYPE_ROCE_V2, AF_INET);
    gidx_ = 3;
    if (gidx_ < 0) {
      SLIME_ABORT("Failed to find GID");
      return -1;
    }
  }

  lid_ = port_attr.lid;
  active_mtu_ = port_attr.active_mtu;

  pd_ = ibv_alloc_pd(ib_ctx_);
  if (!pd_) {
    SLIME_ABORT("Failed to allocate PD");
    return -1;
  }
  comp_channel_ = ibv_create_comp_channel(ib_ctx_);

  return 0;
}
} // namespace slime
