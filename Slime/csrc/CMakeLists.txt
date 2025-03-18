cmake_minimum_required(VERSION 3.10)

project(slime C CXX CUDA)

include_directories(${PROJECT_SOURCE_DIR})

add_library(
    slime
    SHARED
    transfer_engine.cpp
    rdma_transport.cpp
    ibv_helper.cpp
)
target_link_libraries(slime PUBLIC ibverbs)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Python Wrapper
find_package(pybind11 REQUIRED)
pybind11_add_module(
    _slime_c
    bind.cpp
)
target_link_libraries(_slime_c PUBLIC slime)
