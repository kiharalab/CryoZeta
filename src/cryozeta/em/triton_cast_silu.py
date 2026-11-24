"""Fuse the low-precision cast after detector GroupNorm with SiLU."""

import logging

import torch
import triton
import triton.language as tl
from triton.errors import TritonError


@triton.jit(do_not_specialize=["count"])
def _cast_silu_kernel(source, dest, count, block: tl.constexpr):
    index = tl.program_id(0) * block + tl.arange(0, block)
    value = tl.load(source + index, index < count, other=0)
    # Eager SiLU receives values already rounded to the activation dtype.
    x = value.to(dest.dtype.element_ty).to(tl.float32)
    tl.store(dest + index, x / (1.0 + tl.exp(-x)), index < count)


_unavailable_devices = set()


def cast_silu(value, dtype):
    if (
        value.device not in _unavailable_devices
        and value.dtype == torch.float32
        and (
            value.is_contiguous()
            or value.is_contiguous(memory_format=torch.channels_last_3d)
        )
    ):
        out = torch.empty_like(value, dtype=dtype)
        count = value.numel()
        try:
            with torch.cuda.device(value.device):
                _cast_silu_kernel[(triton.cdiv(count, 512),)](
                    value, out, count, 512, enable_fp_fusion=False
                )
            return out
        except (TritonError, FileNotFoundError, RuntimeError) as exc:
            if isinstance(exc, RuntimeError) and "ptxas" not in str(exc).lower():
                raise
            _unavailable_devices.add(value.device)
            logging.getLogger(__name__).warning("Detector SiLU using PyTorch: %s", exc)
    return torch.nn.functional.silu(value.to(dtype))
