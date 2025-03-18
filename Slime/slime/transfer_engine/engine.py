from typing import Tuple

import torch

from slime import _slime_c


class TransferEngine:
    def __init__(self, dev_name: str, ib_port:int=1, link_type:str="Ethernet"):
        self._rdma_context_c = _slime_c.rdma_context()
        self.init_rdma_context(dev_name, ib_port, link_type)
        # 1G Memory Pool
        self.memory_pool = torch.zeros([1024, 1024, 1024], dtype=torch.half, device="cuda")
        self.mr_key = "remote_kv"
        self._rdma_context_c.register_memory_region(
            self.mr_key,
            self.memory_pool.data_ptr(),
            self.memory_pool.numel() * self.memory_pool.itemsize)
    
    def init_rdma_context(self, dev_name: str, ib_port:int=1, link_type:str="Ethernet") -> int:
        return self._rdma_context_c.init_rdma_context(dev_name, ib_port, link_type)
    
    def construct(self, gid: Tuple[int, int], gidx: int, lid: int, qpn: int, psn: int, mtu: int):
        remote_rdma_info = _slime_c.rdma_info(
            qpn, gid[0], gid[1], gidx, lid, psn, mtu
        )
        self._rdma_context_c.modify_qp_to_rtsr(remote_rdma_info)
        self._rdma_context_c.launch_cq_future()

    async def r_rdma_async(self, target_addr, length, rkey):
        self._rdma_context_c.r_rdma_async(
            0, target_addr, target_addr, length, self.mr_key, rkey, 1)
