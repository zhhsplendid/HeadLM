#include "head_ccl/transport/roce.hpp"

#include <format>
#include <string>

namespace comm_backend {
namespace head_ccl {
namespace transport {

int getRoceVersion(const std::string& deviceName,
                   int portNum,
                   int groupIdIndex) {
    std::string roceTypePath = std::format("/sys/class/infiniband/{}/ports/{}/gid_attrs/types/{}",
      deviceName, portNum, groupIdIndex);
    
    int fd = open(roceTypePath, O_RDONLY);
    if (fd == -1) {
        // Error handle
        return -1;
    }

    int ret = read(fd, gidRoceVerStr, 15);
    close(fd);

    if (ret == -1) {
        return -1;
    }

    if (strlen(gidRoceVerStr)) {
        if (strncmp(gidRoceVerStr, "IB/RoCE v1", strlen("IB/RoCE v1")) == 0 
          || strncmp(gidRoceVerStr, "RoCE v1", strlen("RoCE v1")) == 0) {
            return 1
        } else if (strncmp(gidRoceVerStr, "RoCE v2", strlen("RoCE v2")) == 0) {
            return 2
        }
    }

    return -1;
}




} // namespace transport
} // namespace head_ccl
} // namespace comm_backend