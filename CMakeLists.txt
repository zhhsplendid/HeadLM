cmake_minimum_required(VERSION 3.20)

project(HeadLM)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

find_package(Torch REQUIRED)
message(STATUS "Found Torch Version: ${Torch_VERSION}")

message(STATUS "Torch TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})

add_subdirectory(comm_backend)
