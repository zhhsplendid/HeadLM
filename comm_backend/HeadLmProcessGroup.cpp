#include "HeadLmProcessGroup.hpp"

#include <cstdlib>

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>

#include "CpuBackend.hpp"
#include "adapter/CorexAdapter.hpp"
#include "utils/device.hpp"

namespace comm_backend {

using adapter::CorexAdapter;
using utils::DeviceCompany;

c10::intrusive_ptr<c10d::Backend> HeadLmProcessGroup::createHeadLmProcessGroup(
    const c10::intrusive_ptr<::c10d::Store> &store, int rank, int size,
    const std::chrono::duration<float> &timeout) {

  DeviceCompany device_company = utils::getDeviceCompany();

  if (device_company == DeviceCompany::Nvidia) {
    return c10::make_intrusive<::c10d::ProcessGroupNCCL>(store, rank, size);
  } else if (device_company == DeviceCompany::Corex) {
    CorexAdapter corex_adapter;
    corex_adapter.setUp();
    return corex_adapter.createProcessGroup(store, rank, size, timeout);
  }

  // TODO: hardcode the CUDA here for fast test. Should change parameter after
  // we know it works
  return c10::make_intrusive<CpuBackend>(store, rank, size, timeout,
                                         c10::DeviceType::CUDA);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createHeadLmProcessGroup",
        &HeadLmProcessGroup::createHeadLmProcessGroup);
}

} // namespace comm_backend