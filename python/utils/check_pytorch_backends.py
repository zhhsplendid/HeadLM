import os
import torch
import torch.distributed as dist


def is_backend_supported(backend_name: str) -> bool:
    supported = False
    old_master_addr = os.environ[
        'MASTER_ADDR'] if 'MASTER_ADDR' in os.environ else None
    old_master_port = os.environ[
        'MASTER_PORT'] if 'MASTER_PORT' in os.environ else None
    if old_master_addr is None:
        os.environ['MASTER_ADDR'] = 'localhost'
    if old_master_port is None:
        os.environ['MASTER_PORT'] = '29500'  # random value
    try:

        dist.init_process_group(backend=backend_name, rank=0, world_size=1)
        dist.destroy_process_group()
        supported = True
    except:
        supported = False
    finally:
        if old_master_addr is None:
            del os.environ['MASTER_ADDR']
        if old_master_port is None:
            del os.environ['MASTER_PORT']
        return supported


def is_nccl_supported() -> bool:
    return is_backend_supported('nccl')


def is_gloo_supported() -> bool:
    return is_backend_supported('gloo')


if __name__ == '__main__':
    print(f"Is nccl supported: {is_nccl_supported()}")
    print(f"Is gloo supported: {is_gloo_supported()}")
    print(f"Is mpi supported: {is_backend_supported('mpi')}")
