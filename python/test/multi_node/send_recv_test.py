import os
import argparse

import torch
import torch.distributed as dist

# importing makes torch.distributed recognize `headlm_comm` as a valid backend.
import headlm_comm

# Example command to run this python script
# python -m torch.distributed.launch \
#   --nproc_per_node=8 --nnodes=2 --node_rank=TODO \
#   --master_addr=TODO --master_port=TODO \
#   send_recv_test.py \
#   --backend=headlm --device=cpu


def run(backend, device, world_rank, world_size):
    dist.init_process_group(backend, rank=world_rank, world_size=world_size)
    tensor = torch.ones([2, 3], device=device, dtype="float32") * world_rank

    if world_rank == 0:
        for dst_rank in range(1, world_size):
            dist.send(tensor=tensor, dst=dst_rank)
            print(f"Rank 0 sent data to Rank {dst_rank}")
    else:
        dist.recv(tensor=tensor, src=0)
        print(f"Rank {world_rank} has received data from Rank 0")
    
    print(f"Rank {world_rank} After send/recv, tensor.device = {tensor.device}")
    print(f"Rank {world_rank} has data {tensor}")

    assert torch.allclose(torch.zeros([2, 3], device=device), tensor), "After send/recv, tensor values should be zeros"

    dist.destroy_process_group()

def test_base(backend, device):
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['WORLD_RANK'])
    run(backend, device, WORLD_RANK, WORLD_SIZE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", choices=['cuda', 'cpu'])
    parser.add_argument("--backend", type=str, default="headlm", choices=['headlm', 'nccl', 'gloo'])
    args = parser.parse_args()

    test_base(args.backend, args.device)
