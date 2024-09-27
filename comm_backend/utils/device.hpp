#pragma once

namespace comm_backend {
namespace utils {

/**
 * Current supported companies
 */
enum class DeviceCompany {
  Unknown,
  Nvidia,
  Metax, // Mu Xi
  Corex  // Tian Shu
};

DeviceCompany getDeviceCompany();

} // namespace utils
} // namespace comm_backend