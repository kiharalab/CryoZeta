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
# code remains under Apache-2.0; the combined work is distributed
# under GPLv3.

import os
import random

import numpy as np
import torch


def seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic=True applies to CUDA convolution operations, and nothing else.
        torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True) affects all the normally-nondeterministic operations listed here https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html?highlight=use_deterministic#torch.use_deterministic_algorithms
        torch.use_deterministic_algorithms(True)
        # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
