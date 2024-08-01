import os
import sys
import torch
from setuptools import setup
from torch.utils import cpp_extension

sources = ["common_backend/HeadLmProcessGroup.cpp"]
include_dirs = [
    f"{os.path.dirname(os.path.abspath(__file__))}/common_backend/"
]

if torch.cuda.is_available():
    module = cpp_extension.CUDAExtension(
        name="headlm_commn",
        sources=sources,
        include_dirs=include_dirs,
    )
else:
    module = cpp_extension.CppExtension(
        name="eadlm_commn",
        sources=sources,
        include_dirs=include_dirs,
    )

setup(name="Headlm-Communication",
      version="0.0.1",
      ext_modules=[module],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
