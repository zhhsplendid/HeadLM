file(GLOB RDMA_TRANSFER_SOURCES "*.cpp")

add_library(rdma_transfer SHARED ${RDMA_TRANSFER_SOURCES})
target_link_libraries(rdma_transfer PUBLIC info_struct utils flatbuffers)
target_link_libraries(rdma_transfer PUBLIC ibverbs)

#find_package(libuv REQUIRED)
#find_package(PkgConfig REQUIRED)
#pkg_check_modules(LIBUV REQUIRED libuv)
#message(STATUS ${LIBUV_INCLUDE_DIRS})
#target_include_directories(rdma_transfer PRIVATE ${LIBUV_INCLUDE_DIRS})
target_link_libraries(rdma_transfer PUBLIC uv)