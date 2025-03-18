from typing import Dict, Tuple

from .context import RDMAContext


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

    def construct(self, id, gid: Tuple[int, int], gidx: int, lid: int, qpn: int, psn: int, mtu: int):
        if id not in self.links:
            raise KeyError(f"session_id {id} not in links")
        self.links[id].construct(gid, gidx, lid, qpn, psn, mtu)
    
    def get_local_info(self, session_id: int):
        if session_id not in self.links:
            raise KeyError(f"session_id {session_id} not in links")
        return self.links[session_id].get_local_info()
    
    async def r_rdma_async(self, session_id, target_addr, offset, length, rkey):
        if session_id not in self.links:
            raise KeyError(f"session_id {session_id} not in links")
        await self.links[session_id].r_rdma_async(target_addr, offset, length, rkey)
        
