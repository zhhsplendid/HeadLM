#include "HeadLmProcessGroup.hpp"

#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

namespace comm_backend {

HeadLmProcessGroup::HeadLmProcessGroup(int rank, int size,
                                       DeviceType device_type)
    : Backend(rank, size), origin_device_type_(device_type) {
  std::time_t cur_time = std::time(nullptr);
  file_store_ = c10::make_intrusive<FileStore>(
      "/tmp/headlm_" + std::to_string(cur_time), size);
  auto options = ::c10d::ProcessGroupGloo::Options::create();
  options->timeout = std::chrono::seconds(20);
  options->devices.push_back(
        ::c10d::ProcessGroupGloo::createDeviceForHostname("127.0.0.1"));
  cpu_process_group_ =
      c10::make_intrusive<ProcessGroupGloo>(file_store_, rank, size, options);
}

c10::intrusive_ptr<Work>
HeadLmProcessGroup::send(std::vector<at::Tensor> &tensors, int dstRank,
                         int tag) {
  if (origin_device_type_ != DeviceType::CPU) {
    for (at::Tensor &tensor : tensors) {
      tensor = tensor.cpu();
    }
  }

  return cpu_process_group_->send(tensors, dstRank, tag);
}

c10::intrusive_ptr<Work>
HeadLmProcessGroup::recv(std::vector<at::Tensor> &tensors, int srcRank,
                         int tag) {
  if (origin_device_type_ == DeviceType::CPU) {
    return cpu_process_group_->recv(tensors, srcRank, tag);
  }

  /*
  std::vector<at::Tensor> cpu_tensors;
  for (at::Tensor &tensor : tensors) {
    cpu_tensors.push_back(tensor.to("cpu"));
  }*/
  auto cpu_tensors = std::make_shared<std::vector<at::Tensor>>();
  for (at::Tensor &tensor : tensors) {
    cpu_tensors->push_back(tensor.to("cpu"));
  }

  c10::intrusive_ptr<Work> gloo_recv_work = cpu_process_group_->recv(*cpu_tensors, srcRank, tag);

  /*
  for (size_t i = 0; i < tensors.size(); ++i) {
    tensors[i].copy_(copy_tensors[i].to(origin_device_type_));
  }
  */
  /*
  gloo_recv_work.synchronize();
  std::cout << "Huihuang copy to gpu " << std::endl;
  if (origin_device_type_ != DeviceType::CPU) {
    for (at::Tensor &tensor : tensors) {
      tensor = tensor.to(origin_device_type_);
    }
  }*/
   /*
  if (origin_device_type_ != DeviceType::CPU) {
    std::cout << "&(tensor[0]) = " << &(tensors[0]) << std::endl;
    std::cout << "&(copy_tensor[0]) = " << &(copy_tensors[0]) << std::endl;
    for (size_t i = 0; i < tensors.size(); ++i) {
      
      //tensors[i] = copy_tensors[i].to(origin_device_type_);
      //tensors[i] = tensors[i].cpu();

      tensors[i].index_put_({0, 0, 0}, 1192);
      //tensors[i] = tensors[i].to(origin_device_type_);
      std::cout << "gpu_assign, &(tensor[0]) = " << &(tensors[0]) << std::endl;
    }
  }
  return gloo_recv_work;
  */

  
  auto ret = c10::make_intrusive<ToDeviceRecvWork>(gloo_recv_work, tensors, cpu_tensors, origin_device_type_);
  return ret;
}

c10::intrusive_ptr<Backend> HeadLmProcessGroup::createHeadLmProcessGroup(
    const c10::intrusive_ptr<::c10d::Store> &store, int rank, int size,
    const std::chrono::duration<float> &timeout) {
  // TODO: hardcode the CUDA here for fast test. Should change parameter after
  // we know it works
  return c10::make_intrusive<HeadLmProcessGroup>(rank, size, DeviceType::CUDA);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createHeadLmProcessGroup",
        &HeadLmProcessGroup::createHeadLmProcessGroup);
}

} // namespace comm_backend