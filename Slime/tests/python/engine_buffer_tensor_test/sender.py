import asyncio
import torch

from slime.transfer_engine.engine import TransferEngine

import argparse


parser = argparse.ArgumentParser("sender")


async def main(args):
    test_shape = args.shape
    send_indices = args.indices
    test_tensor = torch.arange(torch.prod(torch.tensor(test_shape)), dtype=torch.float32, device="cuda").view(test_shape)

    engine = TransferEngine(args.device)
    session_id = 0
    engine.init_link(session_id)
    future = engine.buffered_send_tensor(session_id, test_tensor, send_indices, args.host, args.host, args.remote_port, args.port)
    await future
    print(test_tensor)
    engine.stop_link(session_id)


if __name__ == "__main__":
    parser.add_argument("--device", type=str, default="mlx5_bond_0")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=3344)
    parser.add_argument("--remote-host", type=str, default="localhost")
    parser.add_argument("--remote-port", type=str, default=4433)
    parser.add_argument("--shape", type=int, nargs="+", default=[5, 4])
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 2, 3])
    parser.add_argument("--mode", type=str, choices=["batch-send", "send"], default="batch-send")
    args = parser.parse_args()
    asyncio.run(main(args))
