file(GLOB RDMA_TRANSFER_SOURCES "*.cpp")

add_library(rdma_transfer SHARED ${RDMA_TRANSFER_SOURCES})