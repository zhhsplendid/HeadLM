# This setup.py adds HeadLm Backend as Custom Extension into Pytorch
# Reference: https://pytorch.org/tutorials/intermediate/process_group_cpp_extension_tutorial.html#step-3-build-the-custom-extension

import os
import sys
import torch
from setuptools import setup
from torch.utils import cpp_extension

import torch.utils


torch_install_path = os.path.dirname(os.path.dirname(torch.utils.cmake_prefix_path))
torch_lib_path = os.path.join(torch_install_path, 'lib')

sources = [
    "comm_backend/HeadLmProcessGroup.cpp",
    "comm_backend/adapter/CpuBackend.cpp"
    ]
library_dirs = [torch_lib_path] 
include_dirs = [
    f"{os.path.dirname(os.path.abspath(__file__))}/comm_backend/",
    f"{os.path.dirname(os.path.abspath(__file__))}/comm_backend/adapter",
    f"{torch_install_path}/include/"
]

if torch.cuda.is_available():
    module = cpp_extension.CUDAExtension(
        name="headlm_comm",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=['-DUSE_C10D_GLOO=1', '-DUSE_C10D_NCCL=1', '-DUSE_GLOG']
    )
else:
    module = cpp_extension.CppExtension(
        name="headlm_comm",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=['-DUSE_C10D_GLOO=1', '-DUSE_C10D_NCCL=1', '-DUSE_GLOG']
    )

'''
module = cpp_extension.CppExtension(
    name="headlm_comm",
    sources=sources,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    extra_compile_args=['-DUSE_C10D_GLOO=1']
)
'''

setup(name="Headlm-Communication",
      version="0.0.1",
      ext_modules=[module],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
