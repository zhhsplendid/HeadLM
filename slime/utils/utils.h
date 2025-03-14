#ifndef UTILS_H
#define UTILS_H

#include "info_struct/config.h"

void signal_handler(int signum);

std::string human_readable_gid(rdma_conn_info_t* info);

void print_rdma_conn_info(rdma_conn_info_t* info, bool is_remote);

#endif  // UTILS_H