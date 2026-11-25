"""Kernel-bench candidate providers: narrow adapters over vendored FlashPairformer kernels,
plus derived-weight caches (C-CAT). Imported lazily, only when a variant requests them.

Each adapter checks support first and raises ``Ineligible(reason)`` with a
kernel_routing reason; the caller decides between UNSUPPORTED (strict, in scope) and the V0 route.
"""

from __future__ import annotations

import os

import torch

_KERNELS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "third_party", "fpf", "opt_core", "kernels")
_MODS: dict = {}


class Ineligible(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def _reason(fpf_reason: str) -> str:
    """Map a FlashPairformer refusal word to a kernel_routing reason."""
    r = str(fpf_reason)
    if r in ("not_cuda", "device", "cc<8", "no-cell", "none", "refused") or r.startswith("none:"):  # none/refused: lever cannot build/run
        return "ineligible_hw"
    if r.startswith("dtype"):
        return "ineligible_dtype"
    if r == "probe-failed":
        return "runtime_error"
    return "ineligible_shape"


def _load(name: str):
    if name not in _MODS:
        try:
            if name == "k2b":
                # K2B reads its launch-cell table at import; pin it to the vendored copy (recorded in counters env).
                os.environ.setdefault("PF_TRIATTN_TABLE", os.path.join(_KERNELS, "fpf_triatt_k2b", "K2B_CELLS.json"))
                from cryozeta.third_party.fpf.opt_core.kernels.fpf_triatt_k2b import triatt_k2b as mod
            else:
                from cryozeta.third_party.fpf.opt_core.kernels.fpf_trimul_v4 import generic as mod
            _MODS[name] = mod
        except Exception as e:  # noqa: BLE001 - any import failure is the provider's import_error
            _MODS[name] = e
    mod = _MODS[name]
    if isinstance(mod, Exception):
        raise Ineligible("import_error", repr(mod)) from mod
    return mod


def k2b_attention(q, k, v, bias, mask=None):
    """cueq ``triangle_attention`` signature: q/k/v [B,N,H,S,D], bias [B,1,H,S,S] fp32, mask [B,N,1,1,S]."""
    k2b = _load("k2b")
    mask = None if mask is None else mask.bool()
    ok, why = k2b.flash_supported(q, k, v, bias, mask)
    if not ok:
        raise Ineligible(_reason(why), why)
    return k2b.flash_triangle_attention(q, k, v, bias, mask)


def cached_buffers(mod: torch.nn.Module, name: str, sources, build) -> dict:
    """Derived tensors cached as non-persistent buffers on ``mod``.

    Buffers follow ``mod.to()`` (trunk offload) and stay out of the state_dict; the cache is
    rebuilt when any source tensor's storage or version changes. Non-tensor values live in __dict__.
    """
    stamp = tuple((t.data_ptr(), t._version, t.device, t.dtype) for t in sources)
    key = f"_kb_{name}_meta"
    meta = mod.__dict__.get(key)
    if meta is None or meta[0] != stamp:
        with torch.no_grad():
            out = build()
        tkeys = [k for k, v in out.items() if torch.is_tensor(v)]
        for k in tkeys:
            mod.register_buffer(f"_kb_{name}_{k}", out[k].detach(), persistent=False)
        meta = (stamp, {k: v for k, v in out.items() if k not in tkeys}, tkeys)
        mod.__dict__[key] = meta
    _, plain, tkeys = meta
    return {**plain, **{k: getattr(mod, f"_kb_{name}_{k}") for k in tkeys}}


def _trimul_params(m):
    return [m.linear_a_p.weight, m.linear_b_p.weight, m.linear_a_g.weight, m.linear_b_g.weight,
            m.linear_a_p.bias, m.linear_b_p.bias, m.linear_a_g.bias, m.linear_b_g.bias]


def cat_weights(m) -> dict:
    """C-CAT: the four concatenations the cueq TriMul path rebuilds on every call, cached."""
    def build():
        ap, bp, ag, bg, apb, bpb, agb, bgb = _trimul_params(m)
        return {
            "p_in_weight": torch.cat([ap, bp], dim=0),
            "g_in_weight": torch.cat([ag, bg], dim=0),
            "p_in_bias": torch.cat([apb, bpb], dim=0),
            "g_in_bias": torch.cat([agb, bgb], dim=0),
        }

    return cached_buffers(m, "ccat", _trimul_params(m), build)


def fpf_weight_kwargs(m) -> dict:
    """CryoZeta TriMul parameters under FlashPairformer v4 names (``pack_weights``/``reference_torch``)."""
    return dict(
        ln_in_w=m.layer_norm_in.weight, ln_in_b=m.layer_norm_in.bias,
        ln_out_w=m.layer_norm_out.weight, ln_out_b=m.layer_norm_out.bias,
        w_ag=m.linear_a_g.weight, w_ap=m.linear_a_p.weight, w_bg=m.linear_b_g.weight, w_bp=m.linear_b_p.weight,
        b_ag=m.linear_a_g.bias, b_ap=m.linear_a_p.bias, b_bg=m.linear_b_g.bias, b_bp=m.linear_b_p.bias,
        w_o=m.linear_z.weight, b_o=m.linear_z.bias, w_og=m.linear_g.weight, b_og=m.linear_g.bias,
    )


def fpf_trimul(m, z, mask, residual: bool):
    """FlashPairformer TriMul v4 (generic) for a CryoZeta TriangleMultiplicativeUpdate module."""
    G = _load("trimul")
    ok, why = G.supported(z, mask, c_z=m.c_z, d_hidden=m.c_hidden)
    if not ok:
        raise Ineligible(_reason(why), why)
    srcs = _trimul_params(m) + [m.layer_norm_in.weight, m.layer_norm_in.bias, m.layer_norm_out.weight,
                                m.layer_norm_out.bias, m.linear_z.weight, m.linear_z.bias, m.linear_g.weight,
                                m.linear_g.bias]

    w = cached_buffers(m, "fpf_v4", srcs, lambda: G.pack_weights(**fpf_weight_kwargs(m)))
    try:
        return G.trimul(z, mask, direction="outgoing" if m._outgoing else "incoming", weights=w, residual=residual)
    except G.TrimulUnsupported as e:
        raise Ineligible(_reason(e.reason), str(e)) from e
