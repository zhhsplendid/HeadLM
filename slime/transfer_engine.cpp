#include "transfer_engine.h"

namespace slime {
void TransferEngine::initialize(std::string metadata_endpoint,
                                std::string local_server) {
  throw std::runtime_error("NotImplementedError");
}

mr_hash_key_t TransferEngine::registerLocalMemory(void *data_ptr,
                                                  uint64_t length) {
  throw std::runtime_error("NotImplementedError");
}

trans_status_t TransferEngine::transferBatch(
    std::string segment_id, std::vector<mr_hash_key_t> targetKey,
    std::vector<int64_t> targetOffset, std::vector<mr_hash_key_t> sourceKey,
    std::vector<int64_t> sourceOffset, uint64_t length) {
  throw std::runtime_error("NotImplementedError");
}
} // namespace slime
