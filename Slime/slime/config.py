from typing import Tuple
from pydantic import BaseModel


class RDMAInfo(BaseModel):
    gid: Tuple[int, int]
    gidx: int
    lid: int
    qpn: int
    psn: int
    mtu: int
