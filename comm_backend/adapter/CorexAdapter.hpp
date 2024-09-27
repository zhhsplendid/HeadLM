#pragma once

#include <ctime>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>

namespace comm_backend {
namespace adapter {

/** Adapter used to set heterogeneous device communication flags and create
 * Process Group */
class CorexAdapter {
public:
  /** Consturctor */
  CorexAdapter();

  /**
   * It is called before creating Process Group and initialize the communication
   * library, We set some necessary FLAGs and initial variables in this method.
   */
  void setUp() const;

  /** Creates ProcessGroup for Pytorch */
  c10::intrusive_ptr<c10d::Backend>
  createProcessGroup(const c10::intrusive_ptr<::c10d::Store> &store, int rank,
                     int size,
                     const std::chrono::duration<float> &timeout) const;
};

} // namespace adapter
} // namespace comm_backend