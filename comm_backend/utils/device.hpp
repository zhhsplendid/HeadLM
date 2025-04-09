#pragma once

#include <string>

#include "utils/json.hpp"

namespace comm_backend {
namespace utils {

using nlohmann::json;

/**
 * Current supported companies
 */
enum class DeviceCompany {
  Unknown,
  Nvidia,
  Metax, // Mu Xi
  Corex  // Tian Shu
};

std::string device_company_to_string(DeviceCompany d);

DeviceCompany device_company_from_string(const std::string &s);

DeviceCompany get_device_company();

int device_count();

json device_init_info_json();

} // namespace utils
} // namespace comm_backend