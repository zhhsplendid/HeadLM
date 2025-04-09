#include "utils/device.hpp"

#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "utils/addr_util.hpp"
#include "utils/logging.h"


namespace comm_backend {
namespace utils {

std::string device_company_to_string(DeviceCompany d) {
  switch (d) {
  case DeviceCompany::Nvidia:
    return "Nvidia";
  case DeviceCompany::Metax:
    return "Metax";
  case DeviceCompany::Corex:
    return "Corex";
  default:
    return "Unknown";
  }
}

DeviceCompany device_company_from_string(const std::string &s) {
  if (s == "Nvidia")
    return DeviceCompany::Nvidia;
  if (s == "Metax")
    return DeviceCompany::Metax;
  if (s == "Corex")
    return DeviceCompany::Corex;
  return DeviceCompany::Unknown;
}

DeviceCompany get_device_company() {
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

int device_count() {
  std::string nproc_str = get_env_variable("NPROC_PER_NODE");
  HEADLM_ASSERT(!nproc_str.empty(), "Please set NPROC_PER_NODE")
  int nproc = std::stoi(nproc_str);
  return nproc;
}

json device_init_info_json()
{
    std::string device_company = utils::device_company_to_string(get_device_company());
    std::string address = utils::get_local_ip();
    int port = utils::get_local_port();
    int rank = std::stoi(get_env_variable("RANK"));
    int nproc = utils::device_count();
    return {
      {"device_brand", device_company},
      {"card_version", "TODO"},
      {"address", address},
      {"port", port},
      {"first_rank", rank},
      {"nproc_in_node", nproc}
    };
}

} // namespace utils
} // namespace comm_backend