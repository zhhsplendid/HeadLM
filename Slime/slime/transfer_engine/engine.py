from typing import Dict, Tuple

from .context import RDMAContext

from slime.config import RDMAInfo


class TransferEngine:
    def __init__(self, dev_name: str, ib_port: int, link_type="Ethernet"):
        self.dev_name = dev_name
        self.ib_port = ib_port
        self.link_type=link_type
        self.links:Dict[int, RDMAContext] = {}

    def init_link(self, session_id: int):
        if session_id in self.links:
            raise KeyError(f"session_id {session_id} already in links")
        self.links[session_id] = RDMAContext(dev_name=self.dev_name, ib_port=self.ib_port, link_type=self.link_type)

    def register_mr(self, session_id, mr_key, length, device="cpu"):
        if session_id not in self.links:
            raise KeyError(f"session_id {id} not in links")
        self.links[session_id].register_mr(mr_key, length, device=device)
    
    def register_torch(self, session_id, mr_key, t):
        if session_id not in self.links:
            raise KeyError(f"session_id {id} not in links")
        self.links[session_id].register_torch(mr_key, t)

    def construct(self, id, local_info: RDMAInfo):
        if id not in self.links:
            raise KeyError(f"session_id {id} not in links")
        self.links[id].construct(local_info)
    
    def get_local_info(self, session_id: int) -> RDMAInfo:
        if session_id not in self.links:
            raise KeyError(f"session_id {session_id} not in links")
        local_info = self.links[session_id].get_local_info()
        return local_info
    
    async def r_rdma_async(self, session_id, mr_key, target_offset, source_offset, length, rkey):
        if session_id not in self.links:
            raise KeyError(f"session_id {session_id} not in links")
        await self.links[session_id].r_rdma_async(mr_key, target_offset, source_offset, length, rkey)
        
    def stop_link(self, session_id: int):
        if session_id not in self.links:
            raise KeyError(f"session_id {id} not in links")
        self.links[session_id]._rdma_context_c.stop_cq_future()
        del self.links[session_id]