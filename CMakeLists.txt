cmake_minimum_required(VERSION 3.20)

project(HeadLM)
#set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard whose features are requested to build this target.")

set(HEADLM_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(PythonLibs REQUIRED)
message(STATUS "Python Include Dirs: ${PYTHON_INCLUDE_DIRS}")
include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS})


find_package(Torch REQUIRED)
message(STATUS "Found Torch Version: ${Torch_VERSION}")
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})

include_directories(${HEADLM_SOURCE_DIR})

enable_testing()

include(cmake/third_party.cmake)

add_subdirectory(comm_backend)
