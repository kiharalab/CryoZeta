# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Modifications Copyright 2026 KiharaLab, Purdue University.
#
# This file is included in a GPLv3-licensed project. The original
# code remains under Apache 2.0; the combined work is distributed
# under GPLv3.

import os
import platform

import torch
from torch.utils.cpp_extension import CUDA_HOME, load


def _get_extra_cuda_include_paths():
    if CUDA_HOME is None:
        return []
    target_include = os.path.join(
        CUDA_HOME, "targets", f"{platform.machine()}-linux", "include"
    )
    if os.path.isdir(target_include):
        return [target_include]
    return []


def _get_cuda_archs():
    cuda_version = tuple(int(x) for x in torch.version.cuda.split(".")[:2])

    if cuda_version < (13, 0):
        archs = ["7.0", "8.0", "8.6", "9.0"]
    else:
        archs = ["8.0", "8.6", "9.0", "10.0", "12.0"]

    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        cap_str = f"{cap[0]}.{cap[1]}"
        if cap_str not in archs:
            archs.append(cap_str)

    return archs


def compile(name, sources, extra_include_paths, build_directory):
    cuda_version = tuple(int(x) for x in torch.version.cuda.split(".")[:2])
    cuda_env = f"cu{cuda_version[0]}{cuda_version[1]}"
    env_build_dir = os.path.join(build_directory, cuda_env)
    os.makedirs(env_build_dir, exist_ok=True)

    archs = _get_cuda_archs()
    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(archs)
    gencode_flags = []
    for arch in archs:
        cc = arch.replace(".", "")
        gencode_flags.extend(["-gencode", f"arch=compute_{cc},code=sm_{cc}"])
    cuda_include_flags = []
    for p in _get_extra_cuda_include_paths():
        cuda_include_flags.extend(["-I", p])
    return load(
        name=name,
        sources=sources,
        extra_include_paths=extra_include_paths + _get_extra_cuda_include_paths(),
        extra_cflags=[
            "-O3",
            "-DVERSION_GE_1_1",
            "-DVERSION_GE_1_3",
            "-DVERSION_GE_1_5",
        ],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-DVERSION_GE_1_1",
            "-DVERSION_GE_1_3",
            "-DVERSION_GE_1_5",
            "-std=c++17",
            "-maxrregcount=50",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            *cuda_include_flags,
            *gencode_flags,
        ],
        verbose=True,
        buildDirectory=env_build_dir,
    )
