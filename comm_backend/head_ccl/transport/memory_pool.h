#pragma once

#include <cstdint>
#include <cstdlib>
#include <infiniband/verbs.h>
#include <string>
#include <sys/types.h>
#include <unordered_map>

#include "head_ccl/transport/rdma_types.h"

namespace comm_backend {
namespace head_ccl {
namespace transport {

using json = nlohmann::json;

/**
 * struct for necessary memory region information which will be used later
 */
struct MrInfo {
public:
  uint64_t addr;
  uint32_t lkey;
  uint32_t rkey;

  MrInfo() = default;

  MrInfo(uint64_t addr, uint32_t lkey, int32_t rkey) {
    this->addr = addr;
    this->lkey = lkey;
    this->rkey = rkey;
  }

  MrInfo(const json& json_mr_info) {
    addr = json_mr_info["addr"];
    lkey = json_mr_info["lkey"];
    rkey = json_mr_info["rkey"];
  }

  json to_json() {
    return json{ {"mr_info", {"addr", addr}, {"lkey", lkey}, {"rkey", rkey} } };
  }
};

class MemoryPool {
public:
    MemoryPool() = default;
    MemoryPool(ibv_pd* pd): pd_(pd) {}
    ~MemoryPool();

    int register_memory_region(const std::string& mr_key,
                               uintptr_t data_ptr,
                               uint64_t length);

    int deregister_memory_region(const std::string& mr_key);

    int register_remote_memory_region(const std::string& mr_key,
                                      const MrInfo& mr_info);

    int deregister_remote_memory_region(const std::string& mr_key);

    MrInfo get_mr_info(const std::string& mr_key)
    {
        return mr_info_.find(mr_key)->second;
    }
    
    MrInfo get_remote_mr_info(const std::string& mr_key)
    {
        return remote_mr_info_.find(mr_key)->second;
    }

    json local_mr_info_to_json();

private:
    ibv_pd*                                       pd_;
    std::unordered_map<uint64_t, struct ibv_mr*>  addrs_to_mrs_;
    std::unordered_map<std::string, MrInfo>       mr_info_;
    std::unordered_map<std::string, MrInfo>       remote_mr_info_;
};

}  // namespace transport
}  // namespace head_ccl
}  // namespace comm_backend