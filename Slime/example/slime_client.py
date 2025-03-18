import time

import requests

import torch
from slime import _slime_c

id = remote_info = requests.get(
    "http://10.130.8.139:4469/init_link"
).json()["id"]

ctx = _slime_c.rdma_context()

# Init RDMA
ctx.init_rdma_context("mlx5_bond_0", 1, "Ethernet")

# Init Memory Region
mr_key = "local_kv"
local_memory = torch.ones([1024, 1024, 1024, 16], dtype=torch.int8)
ctx.register_memory_region(
    mr_key,
    local_memory.data_ptr(),
    local_memory.numel() * local_memory.itemsize
)

# memory key
local_rkey = ctx.get_r_key(mr_key)
# rdma info
local_rdma_info = ctx.get_local_rdma_info()
local_rdma_info.log()

info = remote_info = requests.get(
    "http://10.130.8.139:4469/get_local_info", json={"id": id}
).json()

ctx.modify_qp_to_rtsr(
    _slime_c.rdma_info(
        info["qpn"], info["gid"][0], info["gid"][1], info["gidx"],
        info["lid"],
        info["psn"], info["mtu"]
    )
)

requests.post(
    "http://10.130.8.139:4469/exchange_info", 
    json={
        "id": id,
        "gid": local_rdma_info.get_gid(),
        "gidx": local_rdma_info.gidx,
        "lid": local_rdma_info.lid,
        "qpn": local_rdma_info.qpn,
        "psn": local_rdma_info.psn,
        "mtu": local_rdma_info.mtu,
    }
)

psum = requests.post(
    "http://10.130.8.139:4469/rdma_read", 
    json={
        "id":id,
        "remote_rkey": ctx.get_r_key(mr_key),
        "length": 128 * 64 * 1 * 2 * local_memory.itemsize,
        "offset": 0,
        "remote_addr": local_memory.data_ptr()
    }
)
print(psum.json())

psum = requests.post(
    "http://10.130.8.139:4469/rdma_read", 
    json={
        "id": id,
        "remote_rkey": ctx.get_r_key(mr_key),
        "length": 1024 * 1024 * 1024 * local_memory.itemsize,
        "offset": 1024 * 1024 * 512 * local_memory.itemsize,
        "remote_addr": local_memory.data_ptr()
    }
)
print(psum.json())
