from typing import Tuple
from pydantic import BaseModel


class MemoryRegionInfo(BaseModel):
    addr: int
    r_key: int 

class RDMAInfo(BaseModel):
    gid: Tuple[int, int]
    gidx: int
    lid: int
    qpn: int
    psn: int
    mtu: int
