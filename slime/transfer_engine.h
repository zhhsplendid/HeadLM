#include <stdexcept>
#include <string>
#include <vector>

namespace slime {

using mr_hash_key_t = uint64_t;
typedef enum { SUCCESS } error_code_t;

class TransferEngine {
public:
  TransferEngine() {}

  void initialize(std::string metadata_endpoint, std::string local_server);

  mr_hash_key_t registerLocalMemory(void *data_ptr, uint64_t length);

  error_code_t transferBatch(std::string segment_id,
                             std::vector<mr_hash_key_t> targetKey,
                             std::vector<int64_t> targetOffset,
                             std::vector<mr_hash_key_t> sourceKey,
                             std::vector<int64_t> sourceOffset,
                             uint64_t length);
};

} // namespace slime
