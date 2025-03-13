#include "rdma_transport.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_slime_c, m) {
  py::class_<slime::RDMATransport>(m, "rdma_transport").def(py::init<>());
  py::class_<slime::RDMAContext>(m, "rdma_context")
      .def(py::init<>())
      .def("init_rdma_context", &slime::RDMAContext::init_rdma_context)
      .def("register_metadata", &slime::RDMAContext::register_metadata);
}