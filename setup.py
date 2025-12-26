"""
Light-MoE CUDA Extension Build Script

Uses PyTorch's cpp_extension directly (not CMake) for simpler builds.
"""

import os
import sys
from pathlib import Path

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Project root
ROOT = Path(__file__).parent.absolute()

# Find CUDA includes
def get_cuda_include_dirs():
    """Find CUDA include directories."""
    dirs = []
    
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        # Standard conda CUDA location
        cuda_inc = Path(conda_prefix) / "targets" / "x86_64-linux" / "include"
        if cuda_inc.exists():
            dirs.append(str(cuda_inc))
        
        # nvidia pip package location
        nvidia_inc = Path(conda_prefix) / "lib" / "python3.10" / "site-packages" / "nvidia" / "cuda_runtime" / "include"
        if nvidia_inc.exists():
            dirs.append(str(nvidia_inc))
    
    return dirs


# Check if CUDA is available
try:
    from torch.utils.cpp_extension import CUDA_HOME
    cuda_available = CUDA_HOME is not None or len(get_cuda_include_dirs()) > 0
except:
    cuda_available = False


# Build extension only if CUDA is available
ext_modules = []
cmdclass = {}

if cuda_available:
    cuda_include_dirs = get_cuda_include_dirs()
    print(f"CUDA include dirs: {cuda_include_dirs}")
    
    nvcc_flags = [
        "-O3",
        "--use_fast_math",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
    ]
    
    # Add architecture flags
    # SM75 = Turing (RTX 2080 Ti), SM80 = Ampere (A100), SM86 = Ampere (RTX 3090)
    for arch in ["75", "80", "86"]:
        nvcc_flags.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")
    
    # Library paths
    cuda_lib_dirs = []
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        cuda_lib = Path(conda_prefix) / "targets" / "x86_64-linux" / "lib"
        if cuda_lib.exists():
            cuda_lib_dirs.append(str(cuda_lib))
    
    ext_modules = [
        CUDAExtension(
            name="light_moe._C",
            sources=[
                str(ROOT / "csrc" / "grouped_gemm.cu"),
                str(ROOT / "csrc" / "fused_moe_ops.cu"),
                str(ROOT / "csrc" / "bindings.cpp"),
            ],
            include_dirs=[str(ROOT / "include")] + cuda_include_dirs,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": nvcc_flags,
            },
        )
    ]
    cmdclass = {"build_ext": BuildExtension}
else:
    print("WARNING: CUDA not available, building without CUDA extensions")


setup(
    name="light-moe",
    version="0.1.0",
    description="High-performance MoE inference engine",
    author="Light-MoE Team",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "numpy",
    ],
)
