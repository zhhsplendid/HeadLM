#include "head_ccl/transport/memory_pool.h"
#include "utils/logging.h"

#include <cstdint>
#include <cstdlib>
#include <infiniband/verbs.h>
#include <sys/types.h>
#include <unordered_map>
#include "memory_pool.h"

namespace comm_backend {
namespace head_ccl {
namespace transport {

MemoryPool::~MemoryPool(){
    for (auto p : addrs_to_mrs_) {
        ibv_dereg_mr(p.second);
    }
}

MrInfo MemoryPool::register_memory_region(const std::string& mr_key,
    uintptr_t data_ptr,
    uint64_t length)
{
    /* MemoryRegion Access Right = 777 */
    const static int access_rights = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    ibv_mr*          mr            = ibv_reg_mr(pd_, (void*)data_ptr, length, access_rights);

    HEADLM_ASSERT(mr, " Failed to register memory " << data_ptr);

    HEADLM_LOG_INFO("Memory region: " << (void*)data_ptr << " -- " << (void*)(data_ptr + length)
                                     << ", Device name: " << pd_->context->device->dev_name << ", Length: " << length
                                     << " (" << length / 1024 / 1024 << " MB)"
                                     << ", Permission: " << access_rights << ", LKey: " << mr->lkey
                                     << ", RKey: " << mr->rkey);
    uint64_t addr = reinterpret_cast<uint64_t>(mr->addr);
    addrs_to_mrs_[addr] = mr;
    MrInfo info(addr, mr->lkey, mr->rkey);
    mr_info_[mr_key] = info;
    return info;
}

int MemoryPool::deregister_memory_region(const std::string& mr_key)
{
    HEADLM_ASSERT(mr_info_.count(mr_key), " Deregister a non-exist key.");
    MrInfo mr_info = mr_info_[mr_key];
    uint64_t addr = mr_info.addr;
    ibv_dereg_mr(addrs_to_mrs_[addr]);
    addrs_to_mrs_.erase(addr);
    mr_info_.erase(mr_key);
    return 0;
}

int MemoryPool::register_remote_memory_region(const std::string &mr_key, const MrInfo &mr_info)
{
    remote_mr_info_[mr_key] = mr_info;
    return 0;
}

int MemoryPool::deregister_remote_memory_region(const std::string &mr_key)
{
    remote_mr_info_.erase(mr_key);
    return 0;
}

json MemoryPool::local_mr_info_to_json()
{
    json mr_info;
    for (auto& m : mr_info_) {
        mr_info[m.first] = m.second.to_json();
    }
    return mr_info;
}

int MemoryPool::register_remote_mr_info_from_json(const json &j)
{
    for (auto& elem : j.items()) {
        std::string mr_key = elem.key();
        MrInfo info(elem.value());
        register_remote_memory_region(mr_key, info);
    }
    return 0;
}

}  // namespace transport
}  // namespace head_ccl
}  // namespace comm_backend