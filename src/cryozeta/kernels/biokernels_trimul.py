"""Inference TriMul adapted from BioKernels (GPL-3.0).

The tiled composition and masked gate come from BioKernels 37e3d6e. The
LayerNorm forward kernel includes its strided-input refinement. Here both
projections support checkpoint biases and the output supports a residual.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _norm_forward(
    x,
    weight,
    bias,
    out,
    rows,
    row_stride,
    col_stride,
    eps,
    WIDTH: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col = tl.arange(0, WIDTH)
    valid = row[:, None] < rows
    values = tl.load(
        x + row[:, None] * row_stride + col[None, :] * col_stride, valid, 0
    ).to(tl.float32)
    centered = values - (tl.sum(values, axis=1) / WIDTH)[:, None]
    variance = tl.sum(centered * centered, axis=1) / WIDTH
    scale = tl.load(weight + col).to(tl.float32)
    offset = tl.load(bias + col).to(tl.float32)
    result = centered * tl.rsqrt(variance + eps)[:, None]
    tl.store(out + row[:, None] * WIDTH + col[None, :], result * scale + offset, valid)


@triton.jit(do_not_specialize=["count"])
def _gate_forward(
    value,
    gate,
    mask,
    out,
    count,
    WIDTH: tl.constexpr,
    MASKED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = index < count
    p = tl.load(value + index, valid, 0).to(tl.float32)
    g = tl.load(gate + index, valid, 0).to(tl.float32)
    sigmoid = (1 / (1 + tl.exp(-g))).to(gate.dtype.element_ty).to(tl.float32)
    result = p * sigmoid
    if MASKED:
        keep = tl.load(mask + index // WIDTH, valid, 0).to(tl.float32)
        result = result.to(value.dtype.element_ty).to(tl.float32) * keep
    tl.store(out + index, result, valid)


def _normalize(x, norm, affine):
    rows, width = x.shape
    out = x.new_empty((rows, width))
    _norm_forward[(triton.cdiv(rows, 16),)](
        x,
        *affine,
        out,
        rows,
        *x.stride(),
        norm.eps,
        width,
        16,
    )
    return out


def _gate(value, gate, mask, out):
    _gate_forward[(triton.cdiv(value.numel(), 512),)](
        value,
        gate,
        value if mask is None else mask,
        out,
        value.numel(),
        value.shape[-1],
        mask is not None,
        512,
        enable_fp_fusion=False,
    )


def _contract(a, b, outgoing):
    # Each contraction consumes its destination before overwriting it. The full
    # reduction dimension stays in every GEMM, including the last partial tile.
    width, length, _ = a.shape
    scratch = min(
        256 * 2**20, torch.cuda.get_device_properties(a.device).total_memory // 64
    )
    limit = max(1, scratch // (width * length * a.element_size()))
    block = 1 << (limit.bit_length() - 1)
    if outgoing:
        right = b.transpose(-1, -2)
        for start in range(0, length, block):
            dest = a[:, start : start + block, :]
            dest.copy_(torch.bmm(dest, right))
        return a
    left = a.transpose(-1, -2)
    for start in range(0, length, block):
        dest = b[:, :, start : start + block]
        dest.copy_(torch.bmm(left, dest))
    return b


def _trimul(module, z, mask, *, residual):
    length, _, width = z.shape[-3:]
    rows = length * length
    x = z.view(rows, width)
    mask = None if mask is None else mask.reshape(-1).contiguous()
    affine_in = tuple(
        p.to(z.dtype).contiguous()
        for p in (module.layer_norm_in.weight, module.layer_norm_in.bias)
    )
    affine_out = tuple(
        p.to(z.dtype).contiguous()
        for p in (module.layer_norm_out.weight, module.layer_norm_out.bias)
    )
    # Project the two operands together, then copy each into its own contraction buffer.
    p_weight = torch.cat((module.linear_a_p.weight, module.linear_b_p.weight))
    p_bias = torch.cat((module.linear_a_p.bias, module.linear_b_p.bias))
    g_weight = torch.cat((module.linear_a_g.weight, module.linear_b_g.weight))
    g_bias = torch.cat((module.linear_a_g.bias, module.linear_b_g.bias))
    a, b = (z.new_empty((width, rows)) for _ in range(2))
    tile_rows = (32 * 2**20) // (2 * width * z.element_size())
    for start in range(0, rows, tile_rows):
        tile = slice(start, start + tile_rows)
        norm = _normalize(x[tile], module.layer_norm_in, affine_in)
        value = F.linear(norm, p_weight, p_bias)
        gate = F.linear(norm, g_weight, g_bias)
        _gate(value, gate, None if mask is None else mask[tile], value)
        a[:, tile].copy_(value[:, :width].T)
        b[:, tile].copy_(value[:, width:].T)
        del norm, value, gate

    contracted = _contract(
        a.view(width, length, length), b.view(width, length, length), module._outgoing
    )
    del a, b
    contracted = contracted.view(width, rows).T
    out = torch.empty_like(x)
    for start in range(0, rows, 2 * tile_rows):
        tile = slice(start, start + 2 * tile_rows)
        value = module.linear_z(
            _normalize(contracted[tile], module.layer_norm_out, affine_out)
        )
        gate = module.linear_g(_normalize(x[tile], module.layer_norm_in, affine_in))
        _gate(value, gate, None, out[tile])
        del value, gate
    if residual:
        out.add_(x)
    return out.view_as(z)


def try_trimul(module, z, mask, *, residual):
    """Try the BF16 forward path for a single contiguous pair activation."""
    if (
        z.ndim < 3
        or z.shape[-3] != z.shape[-2]
        or z.numel() != z.shape[-2] ** 2 * 128
        or not 128 <= z.shape[-2] <= 4096
        or z.shape[-1] != 128
        or module.c_hidden != 128
        or not z.is_cuda
        or not z.is_contiguous()
        or z.dtype != torch.bfloat16
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
        or not torch.is_autocast_enabled("cuda")
        or torch.get_autocast_dtype("cuda") != torch.bfloat16
        or torch.__version__ != "2.15.0+cu132"
        or torch.cuda.get_device_name(z.device) != "NVIDIA RTX A6000"
        or (
            mask is not None and (mask.shape != z.shape[:-1] or mask.device != z.device)
        )
    ):
        return None
    with torch.cuda.device(z.device):
        return _trimul(module, z, mask, residual=residual)
