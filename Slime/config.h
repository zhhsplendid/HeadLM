#pragma once

#include "logging.h"
#include <cstdint>
#include <infiniband/verbs.h>
#include <iostream>
#include <string>
#include <tuple>

namespace slime {

typedef struct TransferConfig {
  size_t ib_port;
  size_t gid_index;
} transfer_config_t;

transfer_config_t loadGlobalConfig();

typedef struct RDMAInfo {
  uint32_t qpn;
  union ibv_gid gid;
  int64_t gidx;
  uint16_t lid;
  uint64_t psn;
  uint64_t mtu;
  RDMAInfo() {}
  RDMAInfo(uint32_t qpn, uint64_t gid_subnet_prefix, uint64_t gid_interface_id,
           int64_t gidx, uint16_t lid, uint64_t psn, uint64_t mtu)
      : qpn(qpn), gidx(gidx), lid(lid), psn(psn), mtu(mtu) {
    gid.global = {gid_subnet_prefix, gid_interface_id};
  }
  RDMAInfo(uint32_t qpn, union ibv_gid gid, int64_t gidx, uint16_t lid,
           uint64_t psn, uint64_t mtu)
      : RDMAInfo(qpn, gid.global.interface_id, gid.global.subnet_prefix, gidx,
                 lid, psn, mtu) {}
  std::tuple<uint64_t, uint64_t> get_gid() {
    return {gid.global.subnet_prefix, gid.global.interface_id};
  }
  void set_gid(std::tuple<uint64_t, uint64_t> remote_gid) {
    gid.global = {std::get<0>(remote_gid), std::get<1>(remote_gid)};
  }
  void log() {
    SLIME_LOG_INFO("GID: " << gid.global.subnet_prefix << ", "
                           << gid.global.interface_id);
    SLIME_LOG_INFO("GIDX: " << gidx);
    SLIME_LOG_INFO("LID: " << lid);
    SLIME_LOG_INFO("QPN: " << qpn);
    SLIME_LOG_INFO("PSN: " << psn);
    SLIME_LOG_INFO("MTU: " << mtu);
  }
} rdma_info_t;

}; // namespace slime