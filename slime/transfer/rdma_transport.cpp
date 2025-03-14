#include "rdma_transport.h"

#include <arpa/inet.h>
#include <bits/socket.h>
#include <future>
#include <infiniband/verbs.h>
#include <stdexcept>

#include <flatbuffers/flatbuffers.h>

#include "info_struct/config.h"
#include "logging.h"
#include "transfer/allocate_response_generated.h"
#include "transfer/ibv_helper.h"
#include "transfer/meta_request_generated.h"
#include "utils/utils.cpp"

using flatbuffers::FlatBufferBuilder;

namespace slime {
namespace transfer {

bool is_fake_remote_block(remote_block_t &block) {
  return block.remote_addr == 0 && block.rkey == 0;
}

RDMAContext::~RDMAContext() {
  SLIME_LOG_INFO("destroying connection");

  if (!stop_ && cq_future_.valid()) {
    SLIME_LOG_WARN("user should call close() before destroying connection, "
                   "segmenation fault may occur");
    // throw std::runtime_error("user should call close() before destroying
    // connection");
  }

  SendBuffer *buffer;
  while (send_buffers_.pop(buffer)) {
    if (buffer)
      delete buffer;
  }

  for (auto it = local_mr_.begin(); it != local_mr_.end(); it++) {
    ibv_dereg_mr(it->second);
  }
  local_mr_.clear();

  if (recv_mr_) {
    ibv_dereg_mr(recv_mr_);
  }

  if (recv_buffer_) {
    free(recv_buffer_);
  }

  if (qp_) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RESET;
    ibv_modify_qp(qp_, &attr, IBV_QP_STATE);
  }
  if (qp_) {
    ibv_destroy_qp(qp_);
  }
  if (cq_) {
    ibv_destroy_cq(cq_);
  }

  if (comp_channel_) {
    ibv_destroy_comp_channel(comp_channel_);
  }
  if (pd_) {
    ibv_dealloc_pd(pd_);
  }
  if (ib_ctx_) {
    ibv_close_device(ib_ctx_);
  }
}

int32_t RDMAContext::connect_client(const client_config_t &config) {
  signal(SIGSEGV, signal_handler);
  signal(SIGABRT, signal_handler);
  signal(SIGBUS, signal_handler);
  signal(SIGFPE, signal_handler);
  signal(SIGILL, signal_handler);

  struct sockaddr_in serv_addr;
  // create socket
  if ((sock_ = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    SLIME_ERROR("Failed to create socket");
    return -1;
  }

  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(config.service_port);

  // always connect to localhost
  if (inet_pton(AF_INET, config.host_addr.data(), &serv_addr.sin_addr) <= 0) {
    SLIME_ERROR("Invalid address/ Address not supported");
    return -1;
  }

  if (connect(sock_, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
    SLIME_ERROR("Failed to connect to server");
    return -1;
  }
  return 0;
}

int32_t RDMAContext::exchange_conn_info() {
  header_t header = {
      .magic = MAGIC,
      .op = OP_RDMA_EXCHANGE,
      .body_size = sizeof(rdma_conn_info_t),
  };

  struct iovec iov[2];
  struct msghdr msg;

  iov[0].iov_base = &header;
  iov[0].iov_len = FIXED_HEADER_SIZE;
  iov[1].iov_base = &local_info_;
  iov[1].iov_len = sizeof(rdma_conn_info_t);

  memset(&msg, 0, sizeof(msg));
  msg.msg_iov = iov;
  msg.msg_iovlen = 2;

  if (sendmsg(sock_, &msg, 0) < 0) {
    SLIME_ERROR("Failed to send local connection information");
    return -1;
  }

  int return_code = -1;
  if (recv(sock_, &return_code, RETURN_CODE_SIZE, MSG_WAITALL) < 0) {
    SLIME_ERROR("Failed to receive return code");
    return -1;
  }

  if (return_code != FINISH) {
    SLIME_ERROR("Failed to exchange connection information, return code: " +
                std::to_string(return_code));
    return -1;
  }

  if (recv(sock_, &remote_info_, sizeof(rdma_conn_info_t), MSG_WAITALL) !=
      sizeof(rdma_conn_info_t)) {
    SLIME_ERROR("Failed to receive remote connection information");
    return -1;
  }
  return 0;
}

int RDMAContext::setup_rdma(const client_config_t &config) {
  if (init_rdma_context(config.dev_name, config.ib_port, config.link_type) <
      0) {
    SLIME_ERROR("Failed to initialize RDMA resources");
    return -1;
  }

  // Exchange RDMA connection information with the server
  if (exchange_conn_info()) {
    return -1;
  }

  print_rdma_conn_info(&remote_info_, true);

  // Modify QP to RTR state
  if (modify_qp_to_rtr()) {
    SLIME_ERROR("Failed to modify QP to RTR");
    return -1;
  }

  if (modify_qp_to_rts()) {
    SLIME_ERROR("Failed to modify QP to RTS");
    return -1;
  }

  if (posix_memalign(&recv_buffer_, 4096, PROTOCOL_BUFFER_SIZE) != 0) {
    SLIME_ERROR("Failed to allocate recv buffer");
    return -1;
  }
  recv_mr_ = ibv_reg_mr(pd_, recv_buffer_, PROTOCOL_BUFFER_SIZE,
                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  if (!recv_mr_) {
    SLIME_ERROR("Failed to register recv MR");
    return -1;
  }

  /*
  This is MAX_RECV_WR not MAX_SEND_WR,
  because server also has the same number of buffers
  */
  for (int i = 0; i < MAX_RECV_WR; i++) {
    send_buffers_.push(new SendBuffer(pd_, PROTOCOL_BUFFER_SIZE));
  }

  rdma_inflight_count_ = 0;
  stop_ = false;

  cq_future_ = std::async(std::launch::async, [this]() { cq_handler(); });
  return 0;
}

SendBuffer *RDMAContext::get_send_buffer() {
  /*
  if send buffer list is empty,we just report error, and return NULL
  normal user should not have too many inflight requests, so we just report
  error
  */
  SLIME_ASSERT(!send_buffers_.empty(),
               "get_send_buffer when send_buffers_ is empty()");

  SendBuffer *buffer;
  SLIME_ASSERT(send_buffers_.pop(buffer), "pop buffer failed");
  return buffer;
}

void RDMAContext::release_send_buffer(SendBuffer *buffer) {
  send_buffers_.push(buffer);
}

void RDMAContext::post_recv(struct ibv_sge *recv_sge, rdma_info_base *info) {
  struct ibv_recv_wr recv_wr = {0};
  struct ibv_recv_wr *bad_recv_wr = NULL;

  recv_wr.wr_id = (uintptr_t)info;
  if (recv_sge != NULL) {
    recv_wr.next = NULL;
    recv_wr.sg_list = recv_sge;
    recv_wr.num_sge = 1;
  } else {
    recv_wr.next = NULL;
    recv_wr.sg_list = NULL;
    recv_wr.num_sge = 0;
  }

  int ret = ibv_post_recv(qp_, &recv_wr, &bad_recv_wr);
  if (ret) {
    SLIME_ERROR("Failed to post recv wr : " + std::string(strerror(ret)));
  }
}

void RDMAContext::cq_handler() {
  assert(comp_channel_ != NULL);
  while (!stop_) {
    struct ibv_cq *ev_cq;
    void *ev_ctx;
    int ret = ibv_get_cq_event(comp_channel_, &ev_cq, &ev_ctx);
    if (ret == 0) {
      ibv_ack_cq_events(ev_cq, 1);
      if (ibv_req_notify_cq(ev_cq, 0)) {
        SLIME_ERROR("Failed to request CQ notification");
        return;
      }

      struct ibv_wc wc[10] = {};
      int num_completions;
      while ((num_completions = ibv_poll_cq(cq_, 10, wc)) &&
             num_completions > 0) {
        for (int i = 0; i < num_completions; i++) {
          if (wc[i].status != IBV_WC_SUCCESS) {
            // only fake wr will use IBV_WC_SEND
            // we use it to wake up cq thread and exit
            if (wc[i].opcode == IBV_WC_SEND) {
              SLIME_LOG_INFO("cq thread exit");
              return;
            }
            SLIME_ERROR("Failed status: " +
                        std::string(ibv_wc_status_str(wc[i].status)));
            return;
          }

          if (wc[i].opcode ==
              IBV_WC_SEND) { // read cache/allocate msg/commit msg: request sent
            SLIME_LOG_DEBUG("read cache/allocated/commit msg request send" +
                            std::to_string((uintptr_t)wc[i].wr_id));
            release_send_buffer((SendBuffer *)wc[i].wr_id);
          } else if (wc[i].opcode == IBV_WC_RECV) { // allocate msg recved.
            rdma_info_base *ptr =
                reinterpret_cast<rdma_info_base *>(wc[i].wr_id);
            switch (ptr->get_wr_type()) {
            case WrType::ALLOCATE: {
              auto *info = reinterpret_cast<rdma_allocate_info *>(ptr);
              info->callback();
              delete info;
              break;
            }
            case WrType::READ_COMMIT: {
              SLIME_LOG_INFO("read cache done: Received IMM, imm_data: " +
                             wc[i].imm_data);
              auto *info = reinterpret_cast<rdma_read_commit_info *>(ptr);
              info->callback(wc[i].imm_data);
              delete info;
              rdma_inflight_count_--;
              cv_.notify_all();
              break;
            }
            case WrType::WRITE_ACK: {
              SLIME_LOG_INFO("write cache done: Received IMM, imm_data: " +
                             wc[i].imm_data);
              auto *info = reinterpret_cast<rdma_write_commit_info *>(ptr);
              info->callback();
              delete info;
              rdma_inflight_count_--;
              cv_.notify_all();
              break;
            }
            }
          } else if (wc[i].opcode == IBV_WC_RDMA_WRITE) { // write cache done

            SLIME_ASSERT(outstanding_rdma_writes_ >= 0,
                         "Cache expected to be written but actually NOT");

            std::unique_lock<std::mutex> lock(rdma_post_send_mutex_);

            outstanding_rdma_writes_ -= MAX_WR_BATCH;
            std::string debug_str =
                "RDMA_WRITE completed, wr_id: " + std::to_string(wc[i].wr_id) +
                " outstanding_rdma_writes: " +
                std::to_string(outstanding_rdma_writes_.load());
            SLIME_LOG_DEBUG(debug_str);

            // drain the queue
            if (!outstanding_rdma_writes_queue_.empty()) {
              auto item = outstanding_rdma_writes_queue_.front();
              struct ibv_send_wr *wrs = item.first;
              struct ibv_sge *sges = item.second;
              ibv_send_wr *bad_wr = nullptr;
              SLIME_LOG_DEBUG("IBV POST SEND, wr_id: " +
                              std::to_string(wrs[0].wr_id));
              int ret = ibv_post_send(qp_, &wrs[0], &bad_wr);
              if (ret) {
                SLIME_ERROR("Failed to post RDMA write " +
                            std::string(strerror(ret)));
                throw std::runtime_error("Failed to post RDMA write");
              }
              outstanding_rdma_writes_ += MAX_WR_BATCH;
              delete[] wrs;
              delete[] sges;
              outstanding_rdma_writes_queue_.pop_front();
            }

            // If this is the last WR of w_rdma, send RDMA COMMIT msg to server
            if (wc[i].wr_id != 0) {
              SendBuffer *send_buffer = get_send_buffer();
              FixedBufferAllocator allocator(send_buffer->buffer_,
                                             PROTOCOL_BUFFER_SIZE);
              FlatBufferBuilder builder(64 << 10, &allocator);
              auto *info =
                  reinterpret_cast<rdma_write_commit_info *>(wc[i].wr_id);

              auto remote_addrs_offset =
                  builder.CreateVector(info->remote_addrs);

              auto req = CreateRemoteMetaRequest(
                  builder, 0, 0, 0, remote_addrs_offset, OP_RDMA_WRITE_COMMIT);
              builder.Finish(req);

              // recv RDMA COMMIT's ACK from server
              post_recv(NULL, info);

              // send RDMA COMMIT msg to server
              struct ibv_sge sge = {0};
              struct ibv_send_wr wr = {0};
              struct ibv_send_wr *bad_wr = NULL;

              sge.addr = (uintptr_t)builder.GetBufferPointer();
              sge.length = builder.GetSize();
              sge.lkey = send_buffer->mr_->lkey;

              wr.wr_id = (uintptr_t)send_buffer;
              wr.opcode = IBV_WR_SEND;
              wr.sg_list = &sge;
              wr.num_sge = 1;
              wr.send_flags = IBV_SEND_SIGNALED;

              int ret = ibv_post_send(qp_, &wr, &bad_wr);
              if (ret) {
                SLIME_ERROR("Failed to post RDMA send " +
                            std::string(strerror(ret)));
                return;
              }

              // release lock before callback to prevent deadlock
              lock.unlock();
            }
          } else {
            SLIME_ERROR("Unexpected opcode: " +
                        std::to_string((int)wc[i].opcode));
            return;
          }
        }
      }
    } else {
      // TODO: graceful shutdown
      if (errno != EINTR) {
        SLIME_LOG_WARN("Failed to get CQ event " +
                       std::string(strerror(errno)));
        return;
      }
    }
  }
}

int32_t RDMAContext::init_rdma_context(const std::string &dev_name,
                                       uint8_t ib_port,
                                       const std::string &link_type) {
  struct ibv_device **dev_list;
  struct ibv_device *ib_dev;

  int num_devices = 0;
  dev_list = ibv_get_device_list(&num_devices);

  if (!dev_list) {
    throw std::runtime_error("Failed to get RDMA device list");
    return -1;
  }

  for (int i = 0; i < num_devices; i++) {
    char *dev_name_from_list = (char *)ibv_get_device_name(dev_list[i]);
    if (strcmp(dev_name_from_list, dev_name.c_str()) == 0) {
      ib_dev = dev_list[i];
      ib_ctx_ = ibv_open_device(ib_dev);
      break;
    }
  }

  if (!ib_ctx_) {
    SLIME_LOG_INFO(
        "Can't find or failed to open the specified device, try to open "
        "the default device {}"
        << (char *)ibv_get_device_name(dev_list[0]));
    ib_ctx_ = ibv_open_device(dev_list[0]);
    if (!ib_ctx_) {
      SLIME_ABORT("Failed to open the default device");
      return -1;
    }
  }

  struct ibv_port_attr port_attr;
  ib_port_ = ib_port;

  if (ibv_query_port(ib_ctx_, ib_port, &port_attr)) {
    SLIME_ABORT("Unable to query port {} attributes\n" << ib_port);
    return -1;
  }
  if ((port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND &&
       link_type == "Ethernet") ||
      (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET &&
       link_type == "Infiniband")) {
    SLIME_ABORT("port link layer and config link type don't match");
    return -1;
  }

  if (port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
    gidx_ = -1;
  } else {
    gidx_ = ibv_find_sgid_type(ib_ctx_, ib_port, IBV_GID_TYPE_ROCE_V2, AF_INET);
  }

  lid_ = port_attr.lid;
  active_mtu_ = port_attr.active_mtu;

  pd_ = ibv_alloc_pd(ib_ctx_);
  SLIME_ASSERT(pd_, "Failed to allocate PD");

  comp_channel_ = ibv_create_comp_channel(ib_ctx_);
  SLIME_ASSERT(comp_channel_, "Failed to create completion channel");

  cq_ = ibv_create_cq(ib_ctx_, MAX_SEND_WR + MAX_RECV_WR, nullptr,
                      comp_channel_, 0);
  SLIME_ASSERT(cq_, "Failed to create CQ");

  SLIME_ASSERT(!ibv_req_notify_cq(cq_, 0), "Failed to request CQ notification");

  struct ibv_qp_init_attr qp_init_attr = {};
  qp_init_attr.send_cq = cq_;
  qp_init_attr.recv_cq = cq_;
  qp_init_attr.qp_type = IBV_QPT_RC;
  qp_init_attr.cap.max_send_wr = MAX_SEND_WR;
  qp_init_attr.cap.max_recv_wr = MAX_RECV_WR;
  qp_init_attr.cap.max_send_sge = 1;
  qp_init_attr.cap.max_recv_sge = 1;

  qp_ = ibv_create_qp(pd_, &qp_init_attr);
  SLIME_ASSERT(qp_, "Failed to create QP, " << strerror(errno));

  SLIME_ASSERT(!modify_qp_to_init(),
               "Failed to modify QP to INIT, " << strerror(errno));

  return 0;
}

int32_t RDMAContext::modify_qp_to_init() {
  struct ibv_qp_attr attr = {};
  attr.qp_state = IBV_QPS_INIT;
  attr.port_num = ib_port_;
  attr.pkey_index = 0;
  attr.qp_access_flags =
      IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;

  int flags =
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

  int ret = ibv_modify_qp(qp_, &attr, flags);
  if (ret) {
    SLIME_ABORT("Failed to modify QP to INIT");
    return ret;
  }
  return 0;
}

int32_t RDMAContext::modify_qp_to_rtr() {
  struct ibv_qp_attr attr = {};
  attr.qp_state = IBV_QPS_RTR;

  // update MTU
  if (remote_info_.mtu != active_mtu_) {
    std::string warn_str =
        "remote MTU: " + std::to_string(1 << ((uint32_t)remote_info_.mtu + 7)) +
        " , local MTU: {} is not the same, update to minimal MTU" +
        std::to_string(1 << ((uint32_t)active_mtu_ + 7));
    SLIME_LOG_WARN(warn_str);
  }
  attr.path_mtu =
      (enum ibv_mtu)std::min((uint32_t)active_mtu_, (uint32_t)remote_info_.mtu);

  attr.dest_qp_num = remote_info_.qpn;
  attr.rq_psn = remote_info_.psn;
  attr.max_dest_rd_atomic = 4;
  attr.min_rnr_timer = 12;
  attr.ah_attr.dlid = 0;
  attr.ah_attr.sl = 0;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num = ib_port_;

  if (gidx_ == -1) {
    // IB
    attr.ah_attr.dlid = remote_info_.lid;
    attr.ah_attr.is_global = 0;
  } else {
    // RoCE v2
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid = remote_info_.gid;
    attr.ah_attr.grh.sgid_index = gidx_; // local gid
    attr.ah_attr.grh.hop_limit = 1;
  }

  int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
              IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

  int ret = ibv_modify_qp(qp_, &attr, flags);
  if (ret) {
    std::runtime_error("Failed to modify QP to RTR");
    return ret;
  }
  return 0;
}

int32_t RDMAContext::modify_qp_to_rts() {
  struct ibv_qp_attr attr = {};
  attr.qp_state = IBV_QPS_RTS;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.sq_psn = local_info_.psn; // Use 0 or match with local PSN?
  attr.max_rd_atomic = 1;

  int flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
              IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;

  int ret = ibv_modify_qp(qp_, &attr, flags);
  if (ret) {
    throw std::runtime_error("Failed to modify QP to RTS");
    return ret;
  }
  return 0;
}

int RDMAContext::register_mr(void *base_ptr, size_t ptr_region_size) {
  assert(base_ptr != NULL);
  if (local_mr_.count((uintptr_t)base_ptr)) {
    SLIME_LOG_WARN("this memory address is already registered!");
    ibv_dereg_mr(local_mr_[(uintptr_t)base_ptr]);
  }
  struct ibv_mr *mr;
  mr = ibv_reg_mr(pd_, base_ptr, ptr_region_size,
                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                      IBV_ACCESS_REMOTE_READ);
  if (!mr) {
    SLIME_ERROR("Failed to register memory regions, size: " + ptr_region_size);
    return -1;
  }
  std::string info_str =
      "register mr done for base_ptr: " + std::to_string((uintptr_t)base_ptr) +
      ", size: " + std::to_string(ptr_region_size);
  SLIME_LOG_INFO(info_str);
  local_mr_[(uintptr_t)base_ptr] = mr;
  return 0;
}

std::vector<remote_block_t> *
RDMAContext::allocate_rdma(std::vector<std::string> &keys, int block_size) {
  // convert allocate_rdma_async to sync version
  std::promise<void> promise;
  auto future = promise.get_future();
  std::vector<remote_block_t> *ret_blocks;
  allocate_rdma_async(
      keys, block_size,
      [&promise, &ret_blocks](std::vector<remote_block_t> *blocks,
                              unsigned int error_code) {
        ret_blocks = blocks;
        if (error_code != FINISH) {
          SLIME_ERROR("allocate_rdma failed, error_code: " + error_code);
        }
        promise.set_value();
      });

  auto status = future.wait_for(std::chrono::seconds(5)); // timeout 5s
  if (status == std::future_status::timeout) {
    SLIME_ERROR("allocate_rdma timeout");
    return nullptr;
  } else {
    future.get();
  }
  return ret_blocks;
}

int32_t RDMAContext::allocate_rdma_async(
    std::vector<std::string> &keys, int block_size,
    std::function<void(std::vector<remote_block_t> *, unsigned int error_code)>
        callback) {
  /*
  ENCODING
  remote_meta_request req = {
      .keys = keys,
      .block_size = block_size,
      .op = OP_RDMA_ALLOCATE,
  }
  */
  int ret;

  // post recv msg first
  struct ibv_sge recv_sge = {0};
  struct ibv_recv_wr *bad_recv_wr = NULL;
  struct ibv_recv_wr recv_wr = {0};

  // recv all remote addresses
  recv_sge.addr = (uintptr_t)recv_buffer_;
  recv_sge.length = PROTOCOL_BUFFER_SIZE;
  recv_sge.lkey = recv_mr_->lkey;

  auto *info = new rdma_allocate_info([this, callback]() {
    const RdmaAllocateResponse *resp = GetRdmaAllocateResponse(recv_buffer_);
    SLIME_LOG_INFO("Received allocate response, #keys: " +
                   resp->blocks()->size());

    std::vector<remote_block_t> *blocks = new std::vector<remote_block_t>();
    blocks->reserve(resp->blocks()->size());
    for (const auto *block : *resp->blocks()) {
      remote_block_t remote_block = {
          .rkey = block->rkey(),
          .remote_addr = block->remote_addr(),
      };
      blocks->push_back(remote_block);
    }
    callback(blocks, resp->error_code());
  });
  // build a new callback function:

  {
    std::unique_lock<std::mutex> lock(rdma_post_send_mutex_);
    post_recv(&recv_sge, info);
  }

  // Send RDMA request
  SendBuffer *send_buffer = get_send_buffer();

  FixedBufferAllocator allocator(send_buffer->buffer_, PROTOCOL_BUFFER_SIZE);
  FlatBufferBuilder builder(64 << 10, &allocator);
  auto keys_offset = builder.CreateVectorOfStrings(keys);

  auto req = CreateRemoteMetaRequest(builder, keys_offset, block_size, 0, 0,
                                     OP_RDMA_ALLOCATE);

  builder.Finish(req);

  struct ibv_sge sge = {0};
  struct ibv_send_wr wr = {0};
  struct ibv_send_wr *bad_wr = NULL;

  sge.addr = (uintptr_t)builder.GetBufferPointer();
  sge.length = builder.GetSize();
  sge.lkey = send_buffer->mr_->lkey;

  wr.wr_id = (uintptr_t)send_buffer;
  wr.opcode = IBV_WR_SEND;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.send_flags = IBV_SEND_SIGNALED;
  {
    std::unique_lock<std::mutex> lock(rdma_post_send_mutex_);
    ret = ibv_post_send(qp_, &wr, &bad_wr);
  }
  if (ret) {
    SLIME_ERROR("Failed to post RDMA send : " + std::string(strerror(ret)));
    return -1;
  }
  return 0;
}

int32_t RDMAContext::write_rdma(unsigned long *p_offsets, size_t offsets_len,
                                int block_size, remote_block_t *p_remote_blocks,
                                size_t remote_blocks_len, void *base_ptr) {
  return write_rdma_async(p_offsets, offsets_len, block_size, p_remote_blocks,
                          remote_blocks_len, base_ptr, []() {});
}

int32_t RDMAContext::write_rdma_async(unsigned long *p_offsets,
                                      size_t offsets_len, int block_size,
                                      remote_block_t *p_remote_blocks,
                                      size_t remote_blocks_len, void *base_ptr,
                                      std::function<void()> callback) {
  assert(base_ptr != NULL);
  assert(p_remote_blocks != NULL);
  assert(offsets_len == remote_blocks_len);

  std::stringstream strm;
  strm << "write_rdma, block_size: " << block_size
       << ", base_ptr: " << base_ptr;
  SLIME_LOG_INFO(strm.str());

  if (!local_mr_.count((uintptr_t)base_ptr)) {
    SLIME_ERROR("Please register memory first " +
                std::to_string((uint64_t)base_ptr));
    return -1;
  }

  struct ibv_mr *mr = local_mr_[(uintptr_t)base_ptr];

  std::unique_lock<std::mutex> lock(rdma_post_send_mutex_);

  const size_t max_wr = MAX_WR_BATCH;

  struct ibv_send_wr local_wrs[max_wr];
  struct ibv_sge local_sges[max_wr];

  struct ibv_send_wr *wrs = local_wrs;
  struct ibv_sge *sges = local_sges;

  size_t num_wr = 0;

  bool wr_full = false;

  auto *info = new rdma_write_commit_info([callback]() { callback(); },
                                          remote_blocks_len);

  if (outstanding_rdma_writes_ + max_wr > MAX_RDMA_WRITE_WR) {
    wr_full = true;
    wrs = new struct ibv_send_wr[max_wr];
    sges = new struct ibv_sge[max_wr];
  }

  size_t skipped = 0;
  for (size_t i = 0; i < remote_blocks_len; i++) {
    // skip duplicated remote blocks
    if (is_fake_remote_block(p_remote_blocks[i])) {
      skipped++;
      continue;
    }

    sges[num_wr].addr = (uintptr_t)(base_ptr + p_offsets[i]);
    sges[num_wr].length = block_size;
    sges[num_wr].lkey = mr->lkey;

    wrs[num_wr].opcode = IBV_WR_RDMA_WRITE;
    if (i == remote_blocks_len - 1) {
      // save all the remote addresses for committing keys
      for (size_t j = 0; j < remote_blocks_len; j++) {
        info->remote_addrs.push_back(p_remote_blocks[j].remote_addr);
      }

      wrs[num_wr].wr_id = reinterpret_cast<uint64_t>(info);
    } else {
      wrs[num_wr].wr_id = 0;
    }

    wrs[num_wr].sg_list = &sges[num_wr];
    wrs[num_wr].num_sge = 1;
    wrs[num_wr].send_flags =
        (num_wr == max_wr - 1 || i == remote_blocks_len - 1) ? IBV_SEND_SIGNALED
                                                             : 0;

    wrs[num_wr].wr.rdma.remote_addr = p_remote_blocks[i].remote_addr;
    wrs[num_wr].wr.rdma.rkey = p_remote_blocks[i].rkey;
    wrs[num_wr].next = (num_wr == max_wr - 1 || i == remote_blocks_len - 1)
                           ? nullptr
                           : &wrs[num_wr + 1];
    num_wr++;

    if (num_wr == max_wr || i == remote_blocks_len - 1) {
      if (!wr_full) {
        struct ibv_send_wr *bad_wr = nullptr;
        int ret = ibv_post_send(qp_, &wrs[0], &bad_wr);
        if (ret) {
          SLIME_ERROR("Failed to post RDMA write " +
                      std::string(strerror(ret)));
          return -1;
        }
        outstanding_rdma_writes_ += max_wr;

        // check if next iteration will exceed the limit
        if (outstanding_rdma_writes_ + max_wr > MAX_RDMA_WRITE_WR) {
          wr_full = true;
        }
      } else {
        // if WR queue is full, we need to put them into queue
        std::stringstream strm;
        strm << "WR queue full: push into temp queue, len: " << num_wr
             << ", first wr_id: " << wrs[0].wr_id
             << ", last wr_id: " << wrs[num_wr - 1].wr_id;
        SLIME_LOG_DEBUG(strm.str());
        outstanding_rdma_writes_queue_.push_back({&wrs[0], &sges[0]});
      }

      if (wr_full) {
        wrs = new struct ibv_send_wr[max_wr];
        sges = new struct ibv_sge[max_wr];
      }
      num_wr = 0; // Reset the counter for the next batch
    }
  }

  // Check if there are remaining WRs to be sent
  if (num_wr > 0) {
    if (wr_full) {
      std::stringstream strm;
      strm << "WR queue full: push into temp queue, len: " << num_wr
           << ", first wr_id: " << wrs[0].wr_id
           << ", last wr_id: " << wrs[num_wr - 1].wr_id;
      SLIME_LOG_DEBUG(strm.str());
      outstanding_rdma_writes_queue_.push_back({&wrs[0], &sges[0]});
    } else {
      struct ibv_send_wr *bad_wr = nullptr;
      int ret = ibv_post_send(qp_, &wrs[0], &bad_wr);
      if (ret) {
        SLIME_ERROR("Failed to post RDMA write " + std::string(strerror(ret)));
        return -1;
      }
    }
  }

  if (skipped > 0) {
    SLIME_LOG_WARN("Skipped " + std::to_string(skipped) + " duplicated keys");
    if (skipped == remote_blocks_len) {
      // All keys are duplicated, skip RDMA write
      lock.unlock();
      info->callback();
      delete info;
      return 0;
    }
  }
  rdma_inflight_count_++;
  SLIME_LOG_DEBUG("rdma_inflight_count: " + rdma_inflight_count_.load());

  return 0;
}

int32_t RDMAContext::read_rdma(std::vector<block_t> &blocks, int block_size,
                               void *base_ptr) {
  return read_rdma_async(blocks, block_size, base_ptr, [](unsigned int code) {
    if (code != FINISH) {
      SLIME_ERROR("Failed to read cache, error code: ", code);
    }
  });
}

int32_t
RDMAContext::read_rdma_async(std::vector<block_t> &blocks, int block_size,
                             void *base_ptr,
                             std::function<void(unsigned int)> callback) {
  assert(base_ptr != NULL);

  if (!local_mr_.count((uintptr_t)base_ptr)) {
    SLIME_ERROR("Please register memory first");
    return -1;
  }

  SLIME_LOG_INFO("r_rdma, block_size: ", block_size, ", base_ptr: ", base_ptr);
  struct ibv_mr *mr = local_mr_[(uintptr_t)base_ptr];
  assert(mr != NULL);

  auto *info = new rdma_read_commit_info(
      [callback](unsigned int code) { callback(code); });
  {
    std::unique_lock<std::mutex> lock(rdma_post_send_mutex_);
    post_recv(NULL, info);
  }

  std::vector<std::string> keys;
  std::vector<uintptr_t> remote_addrs;
  for (auto &block : blocks) {
    keys.push_back(block.key);
    remote_addrs.push_back((uintptr_t)(base_ptr + block.offset));
  }
  /*
  remote_meta_req = {
      .keys = keys,
      .block_size = block_size,
      .rkey = mr->rkey,
      .remote_addrs = remote_addrs,
      .op = OP_RDMA_READ,
  }
  */
  SendBuffer *send_buffer = get_send_buffer();
  FixedBufferAllocator allocator(send_buffer->buffer_, PROTOCOL_BUFFER_SIZE);
  FlatBufferBuilder builder(64 << 10, &allocator);

  auto keys_offset = builder.CreateVectorOfStrings(keys);
  auto remote_addrs_offset = builder.CreateVector(remote_addrs);
  auto req = CreateRemoteMetaRequest(builder, keys_offset, block_size, mr->rkey,
                                     remote_addrs_offset, OP_RDMA_READ);

  builder.Finish(req);

  // send RDMA request
  struct ibv_sge sge = {0};
  sge.addr = (uintptr_t)builder.GetBufferPointer();
  sge.length = builder.GetSize();
  sge.lkey = send_buffer->mr_->lkey;

  struct ibv_send_wr wr = {0};
  struct ibv_send_wr *bad_wr = NULL;

  wr.wr_id = (uintptr_t)send_buffer;
  wr.opcode = IBV_WR_SEND;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.send_flags = IBV_SEND_SIGNALED;

  int ret;
  {
    std::unique_lock<std::mutex> lock(rdma_post_send_mutex_);
    ret = ibv_post_send(qp_, &wr, &bad_wr);
  }

  if (ret) {
    SLIME_ERROR("Failed to post RDMA send: ", strerror(ret));
    return -1;
  }
  rdma_inflight_count_++;

  return 0;
}

int32_t RDMAContext::sync_rdma() {
  std::unique_lock<std::mutex> lock(mutex_);
  bool ret = cv_.wait_for(lock, std::chrono::seconds(10),
                          [this] { return rdma_inflight_count_ == 0; });

  if (!ret) {
    SLIME_ERROR("timeout to sync RDMA");
    return -1;
  }
  return 0;
}

void RDMAContext::close_conn() {
  if (!stop_ && cq_future_.valid()) {
    stop_ = true;

    // create fake wr to wake up cq thread
    ibv_req_notify_cq(cq_, 0);
    struct ibv_sge sge;
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uintptr_t)this;
    sge.length = sizeof(*this);
    sge.lkey = 0;

    struct ibv_send_wr send_wr;
    memset(&send_wr, 0, sizeof(send_wr));
    send_wr.wr_id = (uintptr_t)this;
    send_wr.sg_list = &sge;
    send_wr.num_sge = 1;
    send_wr.opcode = IBV_WR_SEND;
    send_wr.send_flags = IBV_SEND_SIGNALED;

    struct ibv_send_wr *bad_send_wr;
    {
      std::unique_lock<std::mutex> lock(rdma_post_send_mutex_);
      ibv_post_send(qp_, &send_wr, &bad_send_wr);
    }
    // wait thread done
    cq_future_.get();
  }

  if (sock_) {
    close(sock_);
  }
}

} // namespace transfer
} // namespace slime
