"""Inference-only fused LayerNorm + Transition (AF3 Algorithm 11) for CUDA rows.

One kernel computes, per row tile, the LayerNorm, both c_in -> n*c_in
projections, the silu(a) * b gate, and the n*c_in -> c_in output projection,
accumulating in FP32 with TF32x3 inputs on SM86. Only the output is stored,
so the two hidden projection tiles stay inside the kernel.

Weights are read in their original contiguous layouts (``wa``/``wb`` are
[H, CIN], ``wo`` is [CIN, H]); tiles load in native memory order and are
transposed in-register with ``tl.trans``, so no repacking copies are needed.

Only inference-shaped inputs are eligible: CUDA FP32 residency everywhere,
no autograd, and dimensions the FP32 dots support. Anything else returns
``None`` and the caller falls back to the reference PyTorch implementation.
"""

import logging
import math

import torch

try:
    import triton
    import triton.language as tl
    from triton.errors import TritonError
except ImportError:  # pragma: no cover - triton ships with CUDA torch builds
    triton = None
    tl = None

    class TritonError(Exception):
        pass


_logger = logging.getLogger(__name__)

# Compilation depends on the channel dimensions as well as the tile settings.
_failed_configs = set()

if triton is not None:

    @triton.jit(do_not_specialize=["rows"])
    def _fused_transition_kernel(
        x_ptr,
        ln_w_ptr,
        ln_b_ptr,
        wa_ptr,
        wb_ptr,
        wo_ptr,
        out_ptr,
        rows,
        eps,
        CIN: tl.constexpr,
        KP2: tl.constexpr,
        H: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
    ):
        # One CTA owns BM rows and the full CIN output width: each row's
        # LayerNorm and every hidden projection are computed exactly once.
        pid = tl.program_id(0)
        offs_m = pid * BM + tl.arange(0, BM)
        offs_k = tl.arange(0, KP2)
        mask_m = offs_m < rows
        mask_k = offs_k < CIN

        x = tl.load(
            x_ptr + offs_m[:, None] * CIN + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )

        # LayerNorm over the full row: the reduction dim CIN fits in one tile.
        mean = tl.sum(x, axis=1) / CIN
        xc = tl.where(mask_k[None, :], x - mean[:, None], 0.0)
        var = tl.sum(xc * xc, axis=1) / CIN
        rstd = 1.0 / tl.sqrt(var + eps)
        ln_w = tl.load(ln_w_ptr + offs_k, mask=mask_k, other=0.0)
        ln_b = tl.load(ln_b_ptr + offs_k, mask=mask_k, other=0.0)
        xn = xc * rstd[:, None] * ln_w[None, :] + ln_b[None, :]
        xn = tl.where(mask_k[None, :], xn, 0.0)

        offs_n = tl.arange(0, BN)

        acc = tl.zeros((BM, KP2), dtype=tl.float32)
        for n0 in range(0, H, BN):
            rows_h = n0 + offs_n
            mask_h = rows_h < H
            # Serialize the up-projection live ranges: the wa tile is consumed
            # by its dot and dead before the wb tile is loaded, halving peak
            # shared memory / register pressure inside the loop. Both weights
            # are [H, CIN] row-major, so these native-order loads are coalesced
            # along CIN; tl.trans costs no memory traffic.
            wa_tile = tl.load(
                wa_ptr + rows_h[:, None] * CIN + offs_k[None, :],
                mask=mask_h[:, None] & mask_k[None, :],
                other=0.0,
            )
            a = tl.dot(xn, tl.trans(wa_tile), input_precision="tf32x3")
            wb_tile = tl.load(
                wb_ptr + rows_h[:, None] * CIN + offs_k[None, :],
                mask=mask_h[:, None] & mask_k[None, :],
                other=0.0,
            )
            b = tl.dot(xn, tl.trans(wb_tile), input_precision="tf32x3")
            g = a * tl.sigmoid(a) * b
            # wo is [CIN, H] row-major; the native-order tile is [KP2, BN]
            # (coalesced along H) and transposes to the [BN, KP2] operand.
            wo_tile = tl.load(
                wo_ptr + offs_k[:, None] * H + rows_h[None, :],
                mask=mask_k[:, None] & mask_h[None, :],
                other=0.0,
            )
            acc = tl.dot(g, tl.trans(wo_tile), acc, input_precision="tf32x3")

        tl.store(
            out_ptr + offs_m[:, None] * CIN + offs_k[None, :],
            acc,
            mask=mask_m[:, None] & mask_k[None, :],
        )


# This tile avoided register spills on the 9BH4 checkpoint replay.
_SM86_TF32X3_TILE = (16, 32, 4)
_NUM_STAGES = 1


def try_fused_transition(
    x: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    wa: torch.Tensor,
    wb: torch.Tensor,
    wo: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    """Run the fused FP32 Transition on contiguous CUDA rows, or decline.

    Returns ``None`` for anything outside the supported envelope (wrong
    device/dtype/layout, autograd or compile activity, unsupported sizes, or
    a tile configuration that previously failed on this device); the caller
    then uses the reference PyTorch path. Raises ``ValueError`` only for a
    non-finite or negative ``eps``.
    """
    if not math.isfinite(eps) or eps < 0:
        raise ValueError(f"LayerNorm eps must be finite and >= 0, got {eps}")
    if (
        triton is None
        or not x.is_cuda
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
        or torch.is_autocast_enabled("cuda")
        or x.dim() != 2
        or x.dtype != torch.float32
        or not x.is_contiguous()
    ):
        return None
    rows, c_in = x.shape
    if (
        rows == 0
        or not 0 < c_in <= 256
        or rows * c_in >= 2**31  # int32 row indexing inside the kernel
        or ln_weight.shape != (c_in,)
        or ln_bias.shape != (c_in,)
        or wa.dim() != 2
        or wa.shape[1] != c_in
        or wb.shape != wa.shape
        or wo.dim() != 2
        or wo.shape[0] != c_in
        or wo.shape[1] != wa.shape[0]
    ):
        return None
    h = wa.shape[0]
    if h == 0 or h * c_in >= 2**31:
        return None
    for weight in (ln_weight, ln_bias, wa, wb, wo):
        if (
            weight.dtype != torch.float32
            or weight.device != x.device
            or not weight.is_contiguous()
        ):
            return None
    if torch.cuda.get_device_capability(x.device) != (8, 6):
        return None

    kp2 = max(16, triton.next_power_of_2(c_in))
    bm, bn, num_warps = _SM86_TF32X3_TILE
    config_key = (x.device.index, c_in, h)
    if config_key in _failed_configs:
        return None

    output = torch.empty_like(x)
    grid = (triton.cdiv(rows, bm),)
    try:
        with torch.cuda.device(x.device):
            _fused_transition_kernel[grid](
                x,
                ln_weight,
                ln_bias,
                wa,
                wb,
                wo,
                output,
                rows,
                eps,
                CIN=c_in,
                KP2=kp2,
                H=h,
                BM=bm,
                BN=bn,
                num_warps=num_warps,
                num_stages=_NUM_STAGES,
            )
    except (TritonError, FileNotFoundError, RuntimeError) as exc:
        if isinstance(exc, RuntimeError) and "ptxas" not in str(exc).lower():
            raise
        _failed_configs.add(config_key)
        _logger.warning(
            "Fused transition tile BM=%d BN=%d for CIN=%d H=%d unavailable "
            "on %s, using PyTorch: %s",
            bm,
            bn,
            c_in,
            h,
            x.device,
            exc,
        )
        return None
    return output
