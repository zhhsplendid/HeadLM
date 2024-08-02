import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# importing makes torch.distributed recognize `headlm_comm` as a valid backend.
import headlm_comm

"""Blocking point-to-point communication."""
def run(rank, size):
    tensor = torch.ones([1, 2, 3], device="cuda") * rank
    #tensor.cuda()
    print(f"Before send/recv, tensor.device = {tensor.device}") 
    if rank == 0:
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print(f"After send/recv, tensor.device = {tensor.device}")
    tensor.to("cpu")
    print('Rank ', rank, ' has data ', tensor[0], '\n')

def init_process(rank, size, fn, backend='headlm'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1' # 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def main():
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    #for symbol in dir(headlm_comm):
    #    print(symbol)
    #print(headlm_comm.__file__)
    main()
