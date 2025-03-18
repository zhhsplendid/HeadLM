import asyncio

import time
from typing import Tuple

import torch

import uvicorn
import uvloop

from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from Slime.slime.transfer_engine.engine import TransferEngine

from .server_args import ServerArgs

import argparse


asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


session_id = 0


app = FastAPI()
app.add_middleware(CORSMiddleware)


transfer_engine: TransferEngine = None


@app.get("/health")
async def health() -> Response:
    return JSONResponse({"health": True})

@app.post("/exchange_info")
async def exchange_info(raw_request: Request) -> Response:
    raw_request = await raw_request.json()
    transfer_engine.construct(**raw_request)
    return JSONResponse({"status": True})


@app.get("/init_link")
async def create_link() -> Response:
    global session_id
    id = session_id
    session_id += 1
    transfer_engine.init_link(id)

    return JSONResponse({"status": "Success", "id": id})

@app.post("/rdma_read")
async def rdma_read(raw_request: Request) -> Response:
    raw_request = await raw_request.json()
    id = raw_request["id"]
    length = raw_request["length"]
    rkey = raw_request["remote_rkey"]
    target_addr = raw_request["remote_addr"]
    offset = raw_request["offset"]

    begin = time.time()
    await transfer_engine.r_rdma_async(id, target_addr, offset, length, rkey)
    end = time.time()
    print(f"latency: {end - begin}, bw: {(length) / (end - begin) / (1e9)} GBps")
    return JSONResponse({"psum": int(torch.sum(transfer_engine.links[id].memory_pool[0]))})


@app.get("/get_local_info")
async def get_local_info(raw_request: Request) -> Response:
    raw_request = await raw_request.json()
    id = raw_request["id"]
    info = transfer_engine.get_local_info(id)
    return JSONResponse({
        "gid": info.get_gid(),
        "gidx": info.gidx,
        "lid": info.lid,
        "qpn": info.qpn,
        "psn": info.psn,
        "mtu": info.mtu,})


def launch_server(server_args, dev_name, ib_port, link_type):
    global transfer_engine
    transfer_engine = TransferEngine(dev_name, ib_port, link_type)
    uvicorn.run(app, host=server_args.host, port=server_args.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-host", type=str,
                        help="--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int,
                        help="--server-port", default=4469)

    parser.add_argument("--dev-name", type=str, default="mlx5_bond_0")
    parser.add_argument("--ib-port", type=int, help="--ib-port", default=1)
    parser.add_argument("--link-type", type=str, help="--link-type", choices=["Ethernet", "Infiniband"], default="Ethernet")

    args = parser.parse_args()

    server_args = ServerArgs(host=args.server_host, port=args.server_port)

    launch_server(server_args=server_args, dev_name=args.dev_name, ib_port=args.ib_port, link_type=args.link_type)
