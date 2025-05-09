#!/usr/bin/env python
# Copyright (c) 2023, Tri Dao.
# Modified 2025-05-08: head-dim-64-only build & progress feedback.

import os
import sys
import re
import ast
import shutil
import subprocess
import functools
import warnings
import platform
import urllib.request
import urllib.error
from pathlib import Path
from packaging.version import parse, Version

from setuptools import setup, find_packages
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    ROCM_HOME,
    CUDA_HOME,
    IS_HIP_EXTENSION,
)

# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------
PACKAGE_NAME         = "flash_attn"
BASE_WHEEL_URL       = "https://github.com/Dao-AILab/flash-attention/releases/download/{tag}/{fname}"
FORCE_BUILD          = os.getenv("FLASH_ATTENTION_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD      = os.getenv("FLASH_ATTENTION_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
FORCE_CXX11_ABI      = os.getenv("FLASH_ATTENTION_FORCE_CXX11_ABI", "FALSE") == "TRUE"
this_dir             = Path(__file__).resolve().parent
IS_ROCM              = IS_HIP_EXTENSION or os.getenv("BUILD_TARGET", "auto") == "rocm"

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def get_platform_tag() -> str:
    if sys.platform.startswith("linux"):
        return f"linux_{platform.uname().machine}"
    if sys.platform == "darwin":
        maj, min_, *_ = platform.mac_ver()[0].split(".")
        return f"macosx_{maj}.{min_}_x86_64"
    if sys.platform == "win32":
        return "win_amd64"
    raise RuntimeError(f"Unsupported platform: {sys.platform}")

def get_version() -> str:
    text = (this_dir / "flash_attn" / "__init__.py").read_text()
    return ast.literal_eval(re.search(r"^__version__\s*=\s*(.*)$", text, re.M).group(1))

def progress_hook(count: int, block_size: int, total_size: int):
    pct = int(count * block_size * 100 / (total_size or 1))
    sys.stdout.write(f"\rDownloading pre-built wheel … {pct:3d}%")
    sys.stdout.flush()

def guessed_wheel() -> tuple[str, str]:
    ver            = get_version()
    torch_v        = parse(torch.__version__)
    py_tag         = f"cp{sys.version_info.major}{sys.version_info.minor}"
    plat_tag       = get_platform_tag()
    abi_flag       = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
    cuda_tag       = f"cu{parse(torch.version.cuda).major}" if not IS_ROCM else f"rocm{parse(torch.version.hip).major}{parse(torch.version.hip).minor}"
    fname          = (f"{PACKAGE_NAME}-{ver}+{cuda_tag}"
                      f"torch{torch_v.major}.{torch_v.minor}"
                      f"cxx11abi{abi_flag}-{py_tag}-{py_tag}-{plat_tag}.whl")
    url            = BASE_WHEEL_URL.format(tag=f"v{ver}", fname=fname)
    return url, fname

# -----------------------------------------------------------------------------
# Custom bdist_wheel that attempts to fetch a cached wheel first
# -----------------------------------------------------------------------------
class CachedWheelsCommand(_bdist_wheel):
    def run(self):
        if FORCE_BUILD:
            return super().run()

        url, fname = guessed_wheel()
        print(f"Attempting cached wheel:\n  {url}")
        try:
            urllib.request.urlretrieve(url, fname, reporthook=progress_hook)
            print(f"\nDownloaded {fname}")
            os.makedirs(self.dist_dir, exist_ok=True)
            impl, abi, plat = self.get_tag()
            dest = Path(self.dist_dir) / f"{self.wheel_dist_name}-{impl}-{abi}-{plat}.whl"
            Path(fname).rename(dest)
            print(f"Placed wheel at {dest}")
        except (urllib.error.HTTPError, urllib.error.URLError):
            print("\nNo cached wheel available — compiling from source.")
            super().run()

# -----------------------------------------------------------------------------
# Ninja-enabled parallel compiler with memory-aware job count
# -----------------------------------------------------------------------------
class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs):
        if "MAX_JOBS" not in os.environ:
            import psutil
            cores = max(1, os.cpu_count() // 2)
            mem_gb = psutil.virtual_memory().available / (1024 ** 3)
            jobs = max(1, min(cores, int(mem_gb / 9)))
            os.environ["MAX_JOBS"] = str(jobs)
        super().__init__(*args, **kwargs)

# -----------------------------------------------------------------------------
# CUDA extension sources (head-dim = 64 only)
# -----------------------------------------------------------------------------
def cuda_sources() -> list[str]:
    base = "csrc/flash_attn/src"
    return [
        "csrc/flash_attn/flash_api.cpp",
        # fused kernels
        f"{base}/flash_fwd_hdim64_fp16_sm80.cu",
        f"{base}/flash_fwd_hdim64_bf16_sm80.cu",
        f"{base}/flash_fwd_hdim64_fp16_causal_sm80.cu",
        f"{base}/flash_fwd_hdim64_bf16_causal_sm80.cu",
        f"{base}/flash_bwd_hdim64_fp16_sm80.cu",
        f"{base}/flash_bwd_hdim64_bf16_sm80.cu",
        f"{base}/flash_bwd_hdim64_fp16_causal_sm80.cu",
        f"{base}/flash_bwd_hdim64_bf16_causal_sm80.cu",
        # split‑KV forward kernels
        f"{base}/flash_fwd_split_hdim64_fp16_sm80.cu",
        f"{base}/flash_fwd_split_hdim64_bf16_sm80.cu",
        f"{base}/flash_fwd_split_hdim64_fp16_causal_sm80.cu",
        f"{base}/flash_fwd_split_hdim64_bf16_causal_sm80.cu",
    ]

def rocm_sources() -> list[str]:
    """
    Generates head-dim-64 ROCm sources via Composable-Kernel codegen
    and returns the resulting .cu list.
    """
    ck_dir = this_dir / "csrc" / "composable_kernel"
    build_dir = this_dir / "build"
    build_dir.mkdir(exist_ok=True)

    # Generate only necessary kernels (hdim64 fwd/bwd, causal & non-causal)
    gen_script = ck_dir / "example" / "ck_tile" / "01_fmha" / "generate.py"
    cmds = [
        ["fwd"], ["fwd_causal"], ["bwd"], ["bwd_causal"],
        ["fwd_splitkv"], ["fwd_splitkv_causal"]
    ]
    for args in cmds:
        subprocess.run([sys.executable, str(gen_script), "-d", *args,
                        "--receipt", "2", "--output_dir", str(build_dir)],
                       check=True)

    # rename .cpp -> .cu because HIP toolchain expects .cu
    renamed = []
    for cpp in build_dir.glob("fmha_*hdim64*wd*.cpp"):
        cu = cpp.with_suffix(".cu")
        shutil.copy(cpp, cu)
        renamed.append(str(cu))

    return ([
        "csrc/flash_attn_ck/flash_api.cpp",
        "csrc/flash_attn_ck/flash_common.cpp",
    ] + renamed)

# -----------------------------------------------------------------------------
# Build extension list
# -----------------------------------------------------------------------------
ext_modules = []
if not SKIP_CUDA_BUILD and not IS_ROCM:
    extra_nvcc = [
        "-O3", "-std=c++17",
        "--expt-relaxed-constexpr", "--expt-extended-lambda",
        "--use_fast_math",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ]
    ext_modules.append(
        CUDAExtension(
            name="flash_attn_2_cuda",
            sources=cuda_sources(),
            extra_compile_args={"cxx": ["-O3", "-std=c++17"],
                                "nvcc": extra_nvcc},
            include_dirs=[
                this_dir / "csrc" / "flash_attn",
                this_dir / "csrc" / "flash_attn" / "src",
                this_dir / "csrc" / "cutlass" / "include",
            ],
        )
    )
elif not SKIP_CUDA_BUILD and IS_ROCM:
    # HIP / ROCm build — codegen then compile
    archs = os.getenv("GPU_ARCHS", "gfx90a").split(";")
    ck_flags = [f"--offload-arch={a}" for a in archs]
    extra_hip = ["-O3", "-std=c++17", *ck_flags,
                 "-DCK_TILE_FMHA_FWD_FAST_EXP2=1",
                 "-fgpu-flush-denormals-to-zero",
                 "-DCK_ENABLE_BF16", "-DCK_ENABLE_FP16", "-DCK_ENABLE_FP8",
                 "-D__HIP_PLATFORM_HCC__=1"]
    ext_modules.append(
        CUDAExtension(
            name="flash_attn_2_cuda",
            sources=rocm_sources(),
            extra_compile_args={"cxx": ["-O3", "-std=c++17"],
                                "nvcc": extra_hip},
            include_dirs=[
                this_dir / "csrc" / "composable_kernel" / "include",
                this_dir / "csrc" / "composable_kernel" / "library" / "include",
                this_dir / "build",  # generated kernels
            ],
        )
    )

# -----------------------------------------------------------------------------
# setup()
# -----------------------------------------------------------------------------
setup(
    name                 = PACKAGE_NAME,
    version              = get_version(),
    description          = "FlashAttention — head-dim = 64-only build",
    long_description     = (this_dir / "README.md").read_text(encoding="utf-8"),
    long_description_content_type = "text/markdown",
    author               = "Tri Dao",
    author_email         = "tri@tridao.me",
    url                  = "https://github.com/Dao-AILab/flash-attention",
    python_requires      = ">=3.9",
    packages             = find_packages(exclude=("build", "dist", "tests", "docs",
                                                  "benchmarks", f"{PACKAGE_NAME}.egg-info")),
    install_requires     = ["torch", "einops"],
    setup_requires       = ["packaging", "psutil", "ninja"],
    cmdclass             = {
        "bdist_wheel": CachedWheelsCommand,
        "build_ext"  : NinjaBuildExtension,
    },
    ext_modules          = ext_modules,
    zip_safe             = False,
)
