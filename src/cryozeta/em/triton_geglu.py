"""Inference-only GeGLU for contiguous, packed linear projections."""

import logging
import os
from functools import cache

import torch
import triton
import triton.language as tl
from triton.errors import TritonError


_unavailable_devices = set()


@cache
def _make_gelu_table(device, dtype):
    # FP16 and BF16 each have 65,536 bit patterns. Using the eager GELU for
    # the table preserves its rounding across CUDA and PyTorch versions.
    codes = torch.arange(1 << 16, device=device, dtype=torch.int32).to(torch.int16)
    table = torch.nn.functional.gelu(codes.view(dtype))
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream(device))
    return table, ready


@triton.jit(do_not_specialize=["count"])
def _geglu_kernel(
    packed, gelu_table, output, count, width: tl.constexpr, block: tl.constexpr
):
    i = tl.program_id(0) * block + tl.arange(0, block)
    offset = (i // width) * (2 * width) + i % width
    value = tl.load(packed + offset, mask=i < count, other=0)
    gate = tl.load(packed + offset + width, mask=i < count, other=0)
    code = gate.to(tl.uint16, bitcast=True).to(tl.int32)
    activation = tl.load(gelu_table + code, mask=i < count, other=0)
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
            table, ready = _make_gelu_table(packed.device, packed.dtype)
            torch.cuda.current_stream(packed.device).wait_event(ready)
            _geglu_kernel[(triton.cdiv(count, 512),)](
                packed, table, output, count, width, 512, enable_fp_fusion=False
            )
    except (TritonError, FileNotFoundError, RuntimeError) as exc:
        if isinstance(exc, RuntimeError) and "ptxas" not in str(exc).lower():
            raise
        _unavailable_devices.add(packed.device)
        logging.getLogger(__name__).warning(
            "GeGLU using PyTorch on %s: %s", packed.device, exc
        )
        return None
    return output
