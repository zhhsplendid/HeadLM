set(HEADLM_COMM_BACKEND_SRCS
  c10dOps.cpp
  HeadLmProcessGroup.cpp
  )





add_library(headlm_comm_backend SHARED ${HEADLM_COMM_BACKEND_SRCS})

message(STATUS "Torch TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
target_include_directories(headlm_comm_backend PUBLIC ${TORCH_INCLUDE_DIRS})

target_link_libraries(headlm_comm_backend PRIVATE Python3::Python torch)