#include "config.h"
#include "rdma_transport.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_slime_c, m) {
  py::class_<slime::RDMAContext>(m, "rdma_context")
      .def(py::init<>())
      .def("init_rdma_context", &slime::RDMAContext::init_rdma_context)
      .def("register_memory_region", &slime::RDMAContext::registerMemoryRegion)
      .def("cq_poll_handle", &slime::RDMAContext::cq_poll_handle)
      .def("launch_cq_future", &slime::RDMAContext::launch_cq_future)
      .def("stop_cq_future", &slime::RDMAContext::stop_cq_future)
      .def("r_rdma_async", &slime::RDMAContext::r_rdma_async, py::call_guard<py::gil_scoped_release>(), "Read remote memory asynchronously")
      .def("modify_qp_to_rtsr", &slime::RDMAContext::modify_qp_to_rtsr)
      .def("get_local_rdma_info", &slime::RDMAContext::get_local_rdma_info)
      .def("get_r_key", &slime::RDMAContext::getRKey);

  py::class_<slime::RDMAInfo>(m, "rdma_info")
      .def(py::init<>())
      .def(py::init<uint32_t, uint64_t, uint64_t, int64_t, uint16_t, uint64_t,
                    uint64_t>())
      .def("get_gid", &slime::RDMAInfo::get_gid)
      .def("set_gid", &slime::RDMAInfo::set_gid)
      .def("log", &slime::RDMAInfo::log)
      .def_readwrite("qpn", &slime::RDMAInfo::qpn)
      .def_readwrite("lid", &slime::RDMAInfo::lid)
      .def_readwrite("psn", &slime::RDMAInfo::psn)
      .def_readwrite("mtu", &slime::RDMAInfo::mtu)
      .def_readwrite("gidx", &slime::RDMAInfo::gidx);
}