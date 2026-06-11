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

_CRYOZETA_EXTENSION_CACHE_ENV = "CRYOZETA_TORCH_EXTENSIONS_DIR"
_CRYOZETA_MODEL_CACHE_ENV = "CRYOZETA_MODEL_CACHE_DIR"


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


def _expand_cache_path(path):
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def _get_cryozeta_extension_cache_root():
    for env_var in (_CRYOZETA_EXTENSION_CACHE_ENV, _CRYOZETA_MODEL_CACHE_ENV):
        path = os.environ.get(env_var)
        if path:
            return _expand_cache_path(path)
    return None


def _get_build_directory(name, cuda_env, build_directory=None):
    cache_root = _get_cryozeta_extension_cache_root()
    if cache_root is not None:
        return os.path.join(cache_root, name, cuda_env)

    if build_directory is None:
        return None

    return os.path.join(build_directory, cuda_env)


def compile(name, sources, extra_include_paths, build_directory=None):
    cuda_version = tuple(int(x) for x in torch.version.cuda.split(".")[:2])
    cuda_env = f"cu{cuda_version[0]}{cuda_version[1]}"
    env_build_dir = _get_build_directory(name, cuda_env, build_directory)
    if env_build_dir is not None:
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
    load_kwargs = {
        "name": name,
        "sources": sources,
        "extra_include_paths": extra_include_paths + _get_extra_cuda_include_paths(),
        "extra_cflags": [
            "-O3",
            "-DVERSION_GE_1_1",
            "-DVERSION_GE_1_3",
            "-DVERSION_GE_1_5",
        ],
        "extra_cuda_cflags": [
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
        "verbose": True,
    }
    if env_build_dir is not None:
        load_kwargs["build_directory"] = env_build_dir

    return load(**load_kwargs)
