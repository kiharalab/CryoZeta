"""Inference-only GeGLU for contiguous, packed linear projections."""

import logging
import os

import torch
import triton
import triton.language as tl
from triton.errors import TritonError


_unavailable_devices = set()


@triton.jit
def _geglu_kernel(packed, output, count, width: tl.constexpr, block: tl.constexpr):
    i = tl.program_id(0) * block + tl.arange(0, block)
    offset = (i // width) * (2 * width) + i % width
    value = tl.load(packed + offset, mask=i < count, other=0)
    gate = tl.load(packed + offset + width, mask=i < count, other=0)
    g = gate.to(tl.float32)
    # Match the intermediate dtype of eager GELU before multiplying.
    activation = (0.5 * g * (1.0 + tl.erf(g * 0.7071067811865476))).to(gate.dtype)
    tl.store(output + i, value * activation, mask=i < count)


def try_geglu(packed):
    if (
        packed.device in _unavailable_devices
        or packed.dtype not in (torch.float16, torch.bfloat16)
        or not packed.is_contiguous()
        or not 2**20 <= packed.numel() < 2**31
        or torch.cuda.get_device_capability(packed.device)[0] < 8
        or os.getenv("CRYOZETA_TRITON_GEGLU", "1") == "0"
    ):
        return None
    width = packed.shape[-1] // 2
    output = packed.new_empty((*packed.shape[:-1], width))
    count = output.numel()
    try:
        with torch.cuda.device(packed.device):
            _geglu_kernel[(triton.cdiv(count, 512),)](packed, output, count, width, 512)
    except (TritonError, FileNotFoundError, RuntimeError) as exc:
        if isinstance(exc, RuntimeError) and "ptxas" not in str(exc).lower():
            raise
        _unavailable_devices.add(packed.device)
        logging.getLogger(__name__).warning(
            "GeGLU using PyTorch on %s: %s", packed.device, exc
        )
        return None
    return output
