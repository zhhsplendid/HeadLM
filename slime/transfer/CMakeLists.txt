file(GLOB RDMA_TRANSFER_SOURCES "*.cpp")

add_library(rdma_transfer SHARED ${RDMA_TRANSFER_SOURCES})
target_link_libraries(rdma_transfer PUBLIC info_struct utils flatbuffers ibverbs)
