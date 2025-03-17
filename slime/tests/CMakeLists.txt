file(GLOB RDMA_TRANSFER_TEST_SOURCES "rdma_transfer_test.cpp")

find_package(GTest REQUIRED)

add_executable(rdma_transfer_test ${RDMA_TRANSFER_TEST_SOURCES})
target_link_libraries(rdma_transfer_test PUBLIC rdma_transfer)
add_test(NAME rdma_transfer_test COMMAND rdma_transfer_test)