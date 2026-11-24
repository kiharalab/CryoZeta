"""Streaming CUDA inference fallback for square, pair-biased triangle attention."""

import logging

import torch
import triton
import triton.language as tl
from triton.errors import TritonError


@triton.jit
def _triangle(
    Q,
    K,
    V,
    MASK,
    BIAS,
    OUT,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    qs0: tl.constexpr,
    qs1: tl.constexpr,
    qs2: tl.constexpr,
    ks0: tl.constexpr,
    ks1: tl.constexpr,
    ks2: tl.constexpr,
    vs0: tl.constexpr,
    vs1: tl.constexpr,
    vs2: tl.constexpr,
    ms0: tl.constexpr,
    ms3: tl.constexpr,
    bs0: tl.constexpr,
    bs1: tl.constexpr,
    bs2: tl.constexpr,
    bs3: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    i = tl.program_id(0) * BM + tl.arange(0, BM)
    row = tl.program_id(1).to(tl.int64)
    head = tl.program_id(2)
    c = tl.arange(0, D)
    j = tl.arange(0, BN)
    q = tl.load(
        Q + row * qs0 + head * qs1 + i[:, None] * qs2 + c[None, :], i[:, None] < N, 0
    )
    # Maintain an online softmax for each query while streaming key tiles.
    running_max = tl.full((BM,), -float("inf"), tl.float32)
    normalizer = tl.full((BM,), 0.0, tl.float32)
    acc = tl.full((BM, D), 0.0, tl.float32)
    for tile in range(tl.cdiv(N, BN)):
        kpos = tile * BN + j
        k = tl.load(
            K + row * ks0 + head * ks1 + kpos[:, None] * ks2 + c[None, :],
            kpos[:, None] < N,
            0,
        )
        v = tl.load(
            V + row * vs0 + head * vs1 + kpos[:, None] * vs2 + c[None, :],
            kpos[:, None] < N,
            0,
        )
        bias = tl.load(
            BIAS + row * bs0 + head * bs1 + i[:, None] * bs2 + kpos[None, :] * bs3,
            (i[:, None] < N) & (kpos[None, :] < N),
            0,
        )
        mask = tl.load(MASK + row * ms0 + kpos * ms3, kpos < N, 0)
        score = tl.dot(q, tl.trans(k)).to(q.dtype)
        score = (score + mask[None, :]).to(q.dtype)
        score = (score + bias).to(q.dtype).to(tl.float32)
        score = tl.where(kpos[None, :] < N, score, -float("inf"))
        maximum = tl.maximum(running_max, tl.max(score, 1))
        # An entirely masked first tile contributes zero until a finite key arrives.
        shift = tl.where(maximum == -float("inf"), 0.0, maximum)
        alpha = tl.exp(running_max - shift)
        prob = tl.exp(score - shift[:, None])
        normalizer = normalizer * alpha + tl.sum(prob, 1)
        acc = acc * alpha[:, None] + tl.dot(prob.to(v.dtype), v)
        running_max = maximum
    output = acc / normalizer[:, None]
    if V.dtype.element_ty == tl.float16:
        # Rounding the tensor-core weights can move a finite weighted mean
        # just outside the FP16 range before the final cast.
        output = tl.where(
            tl.abs(output) < float("inf"),
            tl.minimum(tl.maximum(output, -65504.0), 65504.0),
            output,
        )
    tl.store(
        OUT + ((row * H + head) * N + i[:, None]) * D + c[None, :],
        output,
        i[:, None] < N,
    )


_failed_devices = set()


def try_triangle(q, k, v, biases):
    if (
        not q.is_cuda
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
        or q.ndim != 4
        or q.dtype not in (torch.float16, torch.bfloat16)
        or q.shape != k.shape
        or q.shape != v.shape
        or len(biases) != 2
        or q.device in _failed_devices
    ):
        return None
    rows, heads, length, width = q.shape
    if not (
        0 < rows <= 65535
        and 0 < heads <= 65535
        and 32 <= length <= 4096
        and width in (16, 32, 64)
    ):
        return None
    if torch.cuda.get_device_capability(q.device)[0] < 8:
        return None
    if any(
        t.dtype != q.dtype or t.device != q.device or t.stride(-1) != 1
        for t in (q, k, v)
    ):
        return None
    mask, bias = biases
    if (
        mask.ndim != 4
        or mask.shape[0] not in (1, rows)
        or mask.shape[1:] != (1, 1, length)
        or bias.ndim != 4
        or bias.shape[0] not in (1, rows)
        or bias.shape[1:] != (heads, length, length)
        or any(
            t.device != q.device or t.dtype not in (q.dtype, torch.float32)
            for t in biases
        )
    ):
        return None
    # Tile and head offsets use 32-bit arithmetic. Large-stride views retain
    # the reference implementation even if their logical shapes are small.
    for tensor in (q, k, v, mask, bias):
        if (
            sum(
                (size - 1) * stride
                for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
            )
            >= 2**31
        ):
            return None
    mask = mask.expand(rows, 1, 1, length)
    bias = bias.expand(rows, heads, length, length)
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    try:
        with torch.cuda.device(q.device):
            _triangle[(triton.cdiv(length, 32), rows, heads)](
                q,
                k,
                v,
                mask,
                bias,
                out,
                length,
                heads,
                width,
                *q.stride()[:3],
                *k.stride()[:3],
                *v.stride()[:3],
                mask.stride(0),
                mask.stride(3),
                *bias.stride(),
                32,
                32,
            )
    except (TritonError, FileNotFoundError, RuntimeError) as exc:
        if (
            isinstance(exc, RuntimeError)
            and not isinstance(exc, TritonError)
            and "ptxas" not in str(exc).lower()
        ):
            raise
        _failed_devices.add(q.device)
        logging.getLogger(__name__).warning(
            "Triangle attention using PyTorch on %s: %s", q.device, exc
        )
        return None
    return out
