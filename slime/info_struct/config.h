#ifndef LIBCONFIG_H
#define LIBCONFIG_H

#include <functional>
#include <infiniband/verbs.h>
#include <string>
#include <vector>

#include "flatbuffers/flatbuffers.h"

// this number should be big for lots of RMDA_WRITE requests
#define MAX_SEND_WR 8192

// this is only used for recving RDMA_SEND or IMM data. this should be bigger than max layers of
// model.
#define MAX_RECV_WR 64

// how many RDMA write requests can be outstanding, this should be bigger than MAX_WR_BATCH and less
// than MAX_SEND_WR
#define MAX_RDMA_WRITE_WR 4096

// every MAX_WR_BATCH RDMA write requests will have a RDMA_SIGNAL
#define MAX_WR_BATCH 32

#define MAGIC 0xdeadbeef
#define MAGIC_SIZE 4

#define OP_RDMA_EXCHANGE 'E'
#define OP_RDMA_ALLOCATE 'D'
#define OP_RDMA_READ 'A'
#define OP_RDMA_WRITE_COMMIT 'T'
#define OP_CHECK_EXIST 'C'
#define OP_GET_MATCH_LAST_IDX 'M'
#define OP_DELETE_KEYS 'X'
#define OP_SIZE 1
// If you change here, please also add op name in config.cpp

// error code: int
#define INVALID_REQ 400
#define FINISH 200
#define TASK_ACCEPTED 202
#define INTERNAL_ERROR 500
#define KEY_NOT_FOUND 404
#define RETRY 408
#define SYSTEM_ERROR 503
#define OUT_OF_MEMORY 507

#define RETURN_CODE_SIZE sizeof(int)

#define PROTOCOL_BUFFER_SIZE (4 << 20)  // 4M could be enough

typedef struct __attribute__((packed)) {
    unsigned int magic;
    char op;
    unsigned int body_size;
} header_t;

#define FIXED_HEADER_SIZE sizeof(header_t)

typedef struct ServerConfig {
    int service_port;
    std::string log_level;
    std::string dev_name;
    size_t prealloc_size;  // unit: GB
    int ib_port;
    std::string link_type;
    int minimal_allocate_size;  // unit: KB
    int num_stream;             // can only be 1,2,4, number of stream for each client
    bool auto_increase;
} server_config_t;

typedef struct ClientConfig {
    int service_port;
    std::string log_level;
    std::string dev_name;
    std::string host_addr;
    int ib_port;
    std::string link_type;
} client_config_t;

typedef struct __attribute__((packed)) rdma_conn_info_t {
    uint32_t qpn;
    uint32_t psn;
    union ibv_gid gid;  // RoCE v2
    uint16_t lid;       // IB
    uint32_t mtu;       // peers should have the same mtu
} rdma_conn_info_t;

enum class WrType {
    BASE,
    ALLOCATE,
    READ_COMMIT,
    WRITE_ACK,
};

struct rdma_info_base {
   protected:
    WrType wr_type;

   public:
    rdma_info_base(WrType wr_type) : wr_type(wr_type) {}
    virtual ~rdma_info_base() = default;
    WrType get_wr_type() const { return wr_type; }
};

struct rdma_allocate_info : rdma_info_base {
    std::function<void()> callback;
    rdma_allocate_info(std::function<void()> callback)
        : rdma_info_base(WrType::ALLOCATE), callback(callback) {}
};

struct rdma_read_commit_info : rdma_info_base {
    // call back function.
    std::function<void(unsigned int)> callback;
    rdma_read_commit_info(std::function<void(unsigned int)> callback)
        : rdma_info_base(WrType::READ_COMMIT), callback(callback) {}
};

struct rdma_write_commit_info : rdma_info_base {
    // call back function.
    std::function<void()> callback;
    // the number of blocks that have been written.
    std::vector<uintptr_t> remote_addrs;

    rdma_write_commit_info(std::function<void()> callback, int n)
        : rdma_info_base(WrType::WRITE_ACK), callback(callback), remote_addrs() {
        remote_addrs.reserve(n);
    }
};


class FixedBufferAllocator : public flatbuffers::Allocator {
    public:
     FixedBufferAllocator(void* buffer, size_t size) : buffer_(buffer), size_(size), offset_(0) {}
 
     uint8_t* allocate(size_t size) override;
     void deallocate(uint8_t*, size_t) override;
 
    private:
     void* buffer_;
     size_t size_;
     size_t offset_;
 };

// remote_block_t is used to to talk to PYTHON layer. not used in RDMA/TCP layer.
typedef struct {
    uint32_t rkey;
    uintptr_t remote_addr;
} remote_block_t;

// block_t is used to to talk to PYTHON layer. not used in RDMA/TCP layer.
typedef struct {
    std::string key;
    unsigned long offset;
} block_t;



#endif