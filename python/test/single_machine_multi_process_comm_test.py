import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# importing makes torch.distributed recognize `headlm_comm` as a valid backend.
import headlm_comm


class TestSingleMachineMultiProcessSendRecv(unittest.TestCase):
    """Blocking point-to-point communication."""

    def send_recv_func(self, rank, device):

        tensor = torch.ones([2, 3], device=device) * rank

        if rank == 0:
            # Send the tensor to process 1
            dist.send(tensor=tensor, dst=1)
        else:
            # Receive tensor from process 0
            dist.recv(tensor=tensor, src=0)
        print(f"Rank {rank} After send/recv, tensor.device = {tensor.device}")
        print('Rank ', rank, ' has data ', tensor, '\n')
        self.assertTrue(
            torch.allclose(torch.zeros([2, 3], device=device), tensor))

    def init_process(self, rank, size, fn, device, backend='headlm'):
        """ Initialize the distributed environment. """
        os.environ['MASTER_ADDR'] = '127.0.0.1'  # 'localhost'
        os.environ['MASTER_PORT'] = '29512'
        dist.init_process_group(backend, rank=rank, world_size=size)
        fn(rank, device)
        dist.destroy_process_group()

    def _test_base(self, device):
        size = 2
        processes = []
        for rank in range(size):
            class_instance = TestSingleMachineMultiProcessSendRecv()
            p = mp.Process(target=class_instance.init_process,
                           args=(rank, size, class_instance.send_recv_func,
                                 device))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    def test_cpu(self):
        self._test_base("cpu")

    def test_cuda(self):
        self._test_base("cuda")


if __name__ == "__main__":
    unittest.main()
