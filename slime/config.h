#pragma once

#include <infiniband/verbs.h>
#include <string>

namespace slime {

typedef struct TransferConfig {
  size_t ib_port;
  size_t gid_index;
} transfer_config_t;

transfer_config_t loadGlobalConfig();
}; // namespace slime
