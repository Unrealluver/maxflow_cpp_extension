from setuptools import setup, Extension
import torch
from torch.utils import cpp_extension
import pybind11

setup(name='maxflow_dgcn_cpp',
      ext_modules=[cpp_extension.CppExtension(name='maxflow_dgcn_cpp',
                                              sources=[
                                                    'maxflow_dgcn.cpp',
                                              ])],
      include_dirs=[
          pybind11.get_include(),
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension}
      )