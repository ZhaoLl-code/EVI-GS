#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

glm_include = "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")
nvcc_args = [glm_include]
if os.name == "nt":
    nvcc_args = [
        "-allow-unsupported-compiler",
        "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
        "-Xcompiler",
        "/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
    ] + nvcc_args

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": nvcc_args})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
