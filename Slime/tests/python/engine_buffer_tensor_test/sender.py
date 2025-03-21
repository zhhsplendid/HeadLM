import asyncio
import torch

from slime.transfer_engine.engine import TransferEngine

async def main():
    test_shape = [5, 4]
    send_indices = [0, 2, 3]
    test_tensor = torch.arange(torch.prod(torch.tensor(test_shape)), dtype=torch.float32, device="cuda").view(test_shape)

    engine = TransferEngine("mlx5_bond_0", 1, "Ethernet")
    session_id = 0
    engine.init_link(session_id)
    future = engine.buffered_send_tensor(session_id, test_tensor, send_indices, "localhost", 4433, 3344)
    await future
    print(test_tensor)
    engine.stop_link(session_id)


if __name__ == "__main__":
    asyncio.run(main())