import asyncio
import torch

from slime.transfer_engine.engine import TransferEngine

async def main():
    test_shape = [5, 4]
    recv_indices = [0, 1, 4]
    test_tensor = torch.zeros(test_shape, device="cuda", dtype=torch.float32)

    engine = TransferEngine("mlx5_bond_0", 1, "Ethernet")
    session_id = 0
    engine.init_link(session_id)
    future = engine.buffered_receive_tensor(session_id, test_tensor, recv_indices, "localhost", 3344, 4433)
    await future
    print(test_tensor)
    engine.stop_link(session_id)


if __name__ == "__main__":
    asyncio.run(main())