cmake_minimum_required(VERSION 3.10)

project(slime C CXX CUDA)

include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(thirdparty/flatbuffers)
add_subdirectory(utils)
add_subdirectory(info_struct)
add_subdirectory(transfer)

add_library(
    slime
    SHARED
    transfer_engine.cpp
)

target_link_libraries(slime PUBLIC ibverbs)
target_link_libraries(slime PUBLIC info_struct)
target_link_libraries(slime PUBLIC rdma_transfer)
target_link_libraries(slime PUBLIC utils)

# Python Wrapper
find_package(pybind11 REQUIRED)
pybind11_add_module(
    _slime_c
    bind.cpp
)

target_link_libraries(_slime_c PUBLIC slime)
