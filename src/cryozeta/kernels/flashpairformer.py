"""Inference adapters for the vendored FlashPairformer kernels."""

from functools import cache
import logging

import torch

logger = logging.getLogger(__name__)


def _qualified(tensor):
    return (
        tensor.is_cuda
        and tensor.dtype == torch.bfloat16
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and torch.__version__ == "2.15.0+cu132"
        and torch.cuda.get_device_name(tensor.device) == "NVIDIA RTX A6000"
    )


@cache
def _warn_fallback(kernel, reason):
    logger.warning("%s uses the regular inference backend: %s", kernel, reason)


def try_k2b(q, k, v, bias, mask):
    if q.shape[-3:] != (4, 1980, 32) or not _qualified(q):
        return None
    try:
        from cryozeta.third_party.fpf.opt_core.kernels.fpf_triatt_k2b import triatt_k2b

        mask = None if mask is None else mask.bool()
        supported, reason = triatt_k2b.flash_supported(q, k, v, bias, mask)
        if supported:
            return triatt_k2b.flash_triangle_attention(q, k, v, bias, mask)
        _warn_fallback("FlashPairformer K2B", reason)
    except (ImportError, OSError, RuntimeError, TypeError) as exc:
        _warn_fallback("FlashPairformer K2B", str(exc))
    return None


def _trimul_params(module):
    return {
        "ln_in_w": module.layer_norm_in.weight,
        "ln_in_b": module.layer_norm_in.bias,
        "ln_out_w": module.layer_norm_out.weight,
        "ln_out_b": module.layer_norm_out.bias,
        "w_ag": module.linear_a_g.weight,
        "w_ap": module.linear_a_p.weight,
        "w_bg": module.linear_b_g.weight,
        "w_bp": module.linear_b_p.weight,
        "b_ag": module.linear_a_g.bias,
        "b_ap": module.linear_a_p.bias,
        "b_bg": module.linear_b_g.bias,
        "b_bp": module.linear_b_p.bias,
        "w_o": module.linear_z.weight,
        "b_o": module.linear_z.bias,
        "w_og": module.linear_g.weight,
        "b_og": module.linear_g.bias,
    }


def _packed_weights(module, pack_weights):
    params = _trimul_params(module)
    if any(param.is_inference() for param in params.values()):
        return pack_weights(**params)
    stamp = tuple(
        (param.data_ptr(), param._version, param.device, param.dtype)
        for param in params.values()
    )
    if getattr(module, "_fpf_weight_stamp", None) != stamp:
        packed = pack_weights(**params)
        for name, value in packed.items():
            if torch.is_tensor(value):
                module.register_buffer(f"_fpf_{name}", value, persistent=False)
        module._fpf_weight_stamp = stamp
        module._fpf_weights = {
            name: None if torch.is_tensor(value) else value
            for name, value in packed.items()
        }
    return {
        name: getattr(module, f"_fpf_{name}", value)
        for name, value in module._fpf_weights.items()
    }


def try_trimul(module, z, mask, *, residual):
    if z.shape[-3:] != (1980, 1980, 128) or module.c_hidden != 128 or not _qualified(z):
        return None
    try:
        from cryozeta.third_party.fpf.opt_core.kernels.fpf_trimul_v4 import generic

        supported, reason = generic.supported(
            z, mask, c_z=module.c_z, d_hidden=module.c_hidden
        )
        if supported:
            weights = _packed_weights(module, generic.pack_weights)
            return generic.trimul(
                z,
                mask,
                direction="outgoing" if module._outgoing else "incoming",
                weights=weights,
                residual=residual,
            )
        _warn_fallback("FlashPairformer TriMul", reason)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _warn_fallback("FlashPairformer TriMul", str(exc))
    return None
