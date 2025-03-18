import asyncio

from typing import Tuple

import uvicorn
import uvloop

from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from slime.transfer_engine.engine import TransferEngine

from .server_args import ServerArgs

import argparse


asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


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


@app.post("/rdma_read")
async def rdma_write(raw_request: Request) -> Response:
    raw_request = await raw_request.json()
    length = raw_request["length"]
    rkey = raw_request["remote_rkey"]
    target_addr = raw_request["remote_addr"]

    await transfer_engine.r_rdma_async(target_addr, length, rkey)
    return {"psum": transfer_engine.memory_pool.sum()}


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
