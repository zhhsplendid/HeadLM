find_package(Gloo REQUIRED)
include_directories(${GLOO_INCLUDE_DIRS})

file(GLOB HEADLM_COMM_BACKEND_SRCS
  c10dOps.cpp
  HeadLmProcessGroup.cpp
  )
add_library(headlm_comm_backend SHARED ${HEADLM_COMM_BACKEND_SRCS})
message(STATUS "Torch TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
target_include_directories(headlm_comm_backend PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(headlm_comm_backend PUBLIC Python3::Python torch)

file(GLOB HEADLM_COMM_BACKEND_TEST_SRCS HeadLmProcessGroupTest.cpp)
add_executable(headlm_comm_backend_test ${HEADLM_COMM_BACKEND_TEST_SRCS})

#essage(STATUS "GTEST_INCLUDE_DIR: ${GTEST_INCLUDE_DIR}")
#target_include_directories(headlm_comm_backend_test PUBLIC ${GTEST_INCLUDE_DIR})
#message(STATUS "GTEST_LIBRARIES: ${GTEST_LIBRARIES}")
# TODO: Pytorch has conflict with GTest. If we link torch related lib here,
# There will be link error for undefined reference of EqFailure.
# We just put a demo test here and will solve it soon.

target_link_libraries(headlm_comm_backend_test headlm_comm_backend Python3::Python torch)
add_test(headlm_comm_backend_test headlm_comm_backend_test)

add_subdirectory(comm_backend)
