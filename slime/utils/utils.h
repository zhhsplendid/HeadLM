#ifndef UTILS_H
#define UTILS_H

#include <atomic>

#include "info_struct/config.h"

void signal_handler(int signum);

std::string human_readable_gid(rdma_conn_info_t* info);

void print_rdma_conn_info(rdma_conn_info_t* info, bool is_remote);

class IntrusivePtrTarget {
    public:
     IntrusivePtrTarget() : ref_count(0) {}
 
     virtual ~IntrusivePtrTarget() = default;
 
     IntrusivePtrTarget(const IntrusivePtrTarget &) = delete;
     IntrusivePtrTarget &operator=(const IntrusivePtrTarget &) = delete;
 
     friend void intrusive_ptr_add_ref(IntrusivePtrTarget *p) {
         p->ref_count.fetch_add(1, std::memory_order_relaxed);
     }
 
     friend void intrusive_ptr_release(IntrusivePtrTarget *p) {
         if (p->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
             delete p;
         }
     }
 
    private:
     mutable std::atomic<int> ref_count;
 };

#endif  // UTILS_H