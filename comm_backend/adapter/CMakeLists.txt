find_package(Gloo REQUIRED)
include_directories(${GLOO_INCLUDE_DIRS})

file(GLOB CPU_BACKEND_SRCS CpuBackend.cpp)
add_library(cpu_backend_adapter SHARED ${CPU_BACKEND_SRCS})
message(STATUS "Torch TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
target_include_directories(cpu_backend_adapter PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(cpu_backend_adapter PUBLIC Python3::Python torch)
