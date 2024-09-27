#include "utils/device.hpp"

#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace comm_backend {
namespace utils {

DeviceCompany getDeviceCompany() {
  std::map<std::string, DeviceCompany> smi_to_company = {
      {"nvidia-smi", DeviceCompany::Nvidia},
      {"mx-smi", DeviceCompany::Metax},
      {"ixsmi", DeviceCompany::Corex}};

  for (auto p : smi_to_company) {
    if (system((p.first + " > /dev/null 2>&1").c_str()) == 0) {
      return p.second;
    }
  }
  return DeviceCompany::Unknown;
}

} // namespace utils
} // namespace comm_backend