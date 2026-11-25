"""Kernel backend routing, backend counters, the M-CLEAR control and stage clocks.

Inert by default: with no ``CRYOZETA_KERNEL_VARIANT`` every op takes its c544c6b route,
block cache clearing follows the legacy ``N > 2000`` rule and nothing is recorded.

Environment:
  CRYOZETA_KERNEL_VARIANT   path to a variant YAML/JSON (``load_variant``)
  CRYOZETA_KERNEL_COUNTERS  path of the counters JSON written after each forward and at exit
  CRYOZETA_SYNC_STAGE_TIMER 1 -> stage clocks synchronize CUDA and use perf_counter (T2)
  CRYOZETA_STAGE_PEAKS      1 -> per-stage peak memory with resets (instrumented, diagnosis only)
  CRYOZETA_TRUNK_DUMP       path -> trunk s and z saved as CPU fp32 after the trunk (G-trunk, diagnosis only)
"""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import json
import os
import time
from dataclasses import dataclass, field

import torch

OPS = {
    "triatt": {"cueq", "ds4sci", "lma", "torch", "fpf_k2b"},
    "trimul": {"cueq", "cueq_cached_cat", "torch", "fpf_trimul_v4"},
    "opm": {"tilelang", "chunked", "einsum"},
    "apb": {"cueq", "torch"},
    "rowcol": {"ds4sci", "lma", "torch"},
}
SITES = {"empairformer", "msa", "confidence", "diffusion", "atom", "other"}
CLEAR_POLICIES = {"legacy", "always", "never"}
# Frozen V0 composite policy: first-choice backend -> substitutes it may legitimately end on.
V0_ALLOWED = {("triatt", "cueq"): {"ds4sci"}, ("apb", "cueq"): {"torch"}}
VARIANT_KEYS = {
    "variant_id", "policy_id", "baseline_id", "layer", "description",
    "strict", "replace", "clear_cache", "cli_args", "env",
    "status", "sample_paired", "deferred_reason",  # harness metadata; no runtime effect
}
VARIANT_STATUS = {"active", "deferred"}


class KernelUnsupported(RuntimeError):
    """A requested in-scope replacement is ineligible (strict mode)."""


class KernelFallback(RuntimeError):
    """A requested in-scope replacement was served by another provider (strict mode)."""


@dataclass
class Variant:
    variant_id: str = "V0"
    policy_id: str = "V0"
    baseline_id: str | None = None
    layer: str | None = None
    description: str = ""
    strict: bool = False
    replace: dict = field(default_factory=dict)  # (op, call_site) -> provider
    clear_cache: str = "legacy"
    cli_args: list = field(default_factory=list)
    env: dict = field(default_factory=dict)


def load_variant(path: str | None) -> Variant:
    """Parse and validate a variant file; ``None`` gives the inert V0 variant."""
    if not path:
        return Variant()
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: variant must be a mapping")
    unknown = set(raw) - VARIANT_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    if "variant_id" not in raw:
        raise ValueError(f"{path}: variant_id is required")
    replace = {}
    for r in raw.get("replace") or []:
        op, site, prov = r.get("op"), r.get("call_site"), r.get("provider")
        if op not in OPS:
            raise ValueError(f"{path}: unknown op {op!r}")
        if site not in SITES:
            raise ValueError(f"{path}: unknown call_site {site!r}")
        if prov not in OPS[op]:
            raise ValueError(f"{path}: provider {prov!r} not valid for {op}")
        if (op, site) in replace:
            raise ValueError(f"{path}: duplicate replacement ({op}, {site})")
        replace[(op, site)] = prov
    clear = raw.get("clear_cache", "legacy")
    if clear not in CLEAR_POLICIES:
        raise ValueError(f"{path}: clear_cache must be one of {sorted(CLEAR_POLICIES)}")
    if not isinstance(raw.get("cli_args") or [], list):
        raise ValueError(f"{path}: cli_args must be a list of flag/value strings")
    cli_args = [str(a) for a in raw.get("cli_args") or []]
    env = {str(k): None if v is None else str(v) for k, v in (raw.get("env") or {}).items()}  # None = unset
    for k in ("strict", "sample_paired"):
        if not isinstance(raw.get(k, False), bool):
            raise ValueError(f"{path}: {k} must be a boolean")
    if raw.get("status", "active") not in VARIANT_STATUS:
        raise ValueError(f"{path}: status must be one of {sorted(VARIANT_STATUS)}")
    if raw.get("status") == "deferred" and not raw.get("deferred_reason"):
        raise ValueError(f"{path}: a deferred variant needs deferred_reason")
    return Variant(
        variant_id=str(raw["variant_id"]),
        policy_id=str(raw.get("policy_id", "V0")),
        baseline_id=raw.get("baseline_id"),
        layer=raw.get("layer"),
        description=str(raw.get("description", "")),
        strict=raw.get("strict", False),
        replace=replace,
        clear_cache=clear,
        cli_args=cli_args,
        env=env,
    )


VARIANT = load_variant(os.environ.get("CRYOZETA_KERNEL_VARIANT"))
_COUNTERS_PATH = os.environ.get("CRYOZETA_KERNEL_COUNTERS")
_SYNC = os.environ.get("CRYOZETA_SYNC_STAGE_TIMER") == "1"
_PEAKS = os.environ.get("CRYOZETA_STAGE_PEAKS") == "1"
_TRUNK = os.environ.get("CRYOZETA_TRUNK_DUMP")

_SITE: contextvars.ContextVar[str] = contextvars.ContextVar("kb_site", default="other")
_CALL: contextvars.ContextVar[_Call | None] = contextvars.ContextVar("kb_call", default=None)
_REC = {"counters": {}, "block_clears": {}, "other_clears": 0, "t3": [], "stages": [], "inputs": []}


@dataclass
class _Call:
    op: str
    site: str
    requested: str
    v0: str
    in_scope: bool
    shape: list
    dtype: str
    served: set = field(default_factory=set)
    reasons: set = field(default_factory=set)


def reset() -> None:
    """Clear recorded counters, clears, clocks and T3 (op_bench between backends, Mode B iterations)."""
    _REC["counters"].clear()
    _REC["block_clears"].clear()
    _REC["other_clears"] = 0
    _REC["t3"].clear()
    _REC["stages"].clear()
    _REC["inputs"].clear()


def requested(op_name: str) -> str | None:
    """Provider requested for ``op_name`` at the current call site, or None (V0 route)."""
    return VARIANT.replace.get((op_name, _SITE.get()))


def current() -> _Call | None:
    return _CALL.get()


def _route_allowed(c: _Call, served: str) -> bool:
    if c.in_scope:
        return served == c.requested
    return served == c.v0 or served in V0_ALLOWED.get((c.op, c.v0), set())


def _record(c: _Call) -> None:
    served = "+".join(sorted(c.served)) or "none"
    reason = "+".join(sorted(c.reasons)) or ("served" if c.served else "none")
    allowed = bool(c.served) and all(_route_allowed(c, s) for s in c.served)
    key = (c.op, c.site, c.requested, served, reason, c.in_scope, allowed)
    rec = _REC["counters"].get(key)
    if rec is None:
        _REC["counters"][key] = {"count": 1, "first_shape": c.shape, "dtype": c.dtype}
    else:
        rec["count"] += 1


def is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


@contextlib.contextmanager
def op(name: str, v0: str, x: torch.Tensor):
    """One logical op call (all its chunks). Yields the call record, or None when inert."""
    if not (_COUNTERS_PATH or VARIANT.replace or VARIANT.strict):
        yield None
        return
    site = _SITE.get()
    req = VARIANT.replace.get((name, site))
    c = _Call(name, site, req or v0, v0, req is not None, list(x.shape), str(x.dtype).replace("torch.", ""))
    tok = _CALL.set(c)
    try:
        yield c
    except BaseException as e:
        if not isinstance(e, (KernelUnsupported, KernelFallback)):
            c.reasons.add("oom" if is_oom(e) else "runtime_error")
        _record(c)
        raise
    finally:
        _CALL.reset(tok)
    _record(c)
    if VARIANT.strict and c.in_scope and c.served != {c.requested}:
        raise KernelFallback(f"{name}@{site}: requested {c.requested}, served {sorted(c.served)}")


def served(backend: str, reason: str | None = None) -> None:
    """Record the backend that actually executed (plus an optional non-'served' reason)."""
    c = _CALL.get()
    if c is not None:
        c.served.add(backend)
        if reason:
            c.reasons.add(reason)


def ineligible(reason: str) -> None:
    """The requested provider rejected this call. Strict + in scope: UNSUPPORTED; else caller reroutes."""
    c = _CALL.get()
    if c is None:
        return
    c.reasons.add(reason)
    if VARIANT.strict and c.in_scope:
        raise KernelUnsupported(f"{c.op}@{c.site}: {c.requested} {reason}")


def rescue_allowed() -> bool:
    """Whether a provider failure may be rescued by another backend (never inside strict scope)."""
    c = _CALL.get()
    return not (c is not None and c.in_scope and VARIANT.strict)


# --- M-CLEAR: block-level cache clearing ------------------------------------------------------


def clear_between_blocks(n_token: int, training: bool) -> bool:
    if training:
        return False
    if VARIANT.clear_cache == "always":
        return True
    if VARIANT.clear_cache == "never":
        return False
    return n_token > 2000  # legacy c544c6b rule


def run_block(site: str, clear: bool, block, *args, **kwargs):
    """Run one stack block with its call-site label; clear the allocator cache first if ``clear``."""
    tok = _SITE.set(site)
    try:
        if clear:
            _REC["block_clears"][site] = _REC["block_clears"].get(site, 0) + 1
            _real_empty_cache()
        return block(*args, **kwargs)
    finally:
        _SITE.reset(tok)


_real_empty_cache = torch.cuda.empty_cache


def _counted_empty_cache():
    _REC["other_clears"] += 1
    _real_empty_cache()


if _COUNTERS_PATH:  # count non-block empty_cache calls separately
    torch.cuda.empty_cache = _counted_empty_cache


# --- clocks (T2/T3) and the counters file -------------------------------------------------------


def clock(stage: str | None = None) -> float:
    """Stage clock: time.time() as at c544c6b, or synchronized perf_counter with SYNC_STAGE_TIMER."""
    if _SYNC and torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.perf_counter() if _SYNC else time.time()
    if stage and _COUNTERS_PATH:
        rec = {"stage": stage, "t": t, "synchronized": _SYNC}
        if _PEAKS and torch.cuda.is_available():
            rec["m1"] = torch.cuda.max_memory_allocated()
            rec["m2"] = torch.cuda.max_memory_reserved()
            torch.cuda.reset_peak_memory_stats()
        _REC["stages"].append(rec)
    return t


def record_input(name, feats: dict) -> None:
    """Actual model-input N_token and P: a matrix cell counts only if they match its request.
    The pipeline has no point padding/mask, so valid points are the em_support_points rows with finite xyz."""
    if not _COUNTERS_PATH:
        return
    sp = feats["em_support_points"]
    _REC["inputs"].append({
        "name": name, "n_token": int(feats["restype"].shape[-2]), "n_point": int(sp.shape[-2]),
        "valid_points": int(torch.isfinite(sp).all(-1).sum()),
        "shapes": {k: list(feats[k].shape) for k in ("restype", "em_support_points", "p", "pz") if k in feats},
    })


def save_trunk(**tensors) -> None:
    """G-trunk input: trunk outputs to CPU fp32. Instrumented runs only (adds to T3)."""
    if _TRUNK:
        torch.save({k: v.detach().float().cpu() for k, v in tensors.items()}, _TRUNK)


@contextlib.contextmanager
def timed_forward():
    """T3: one synchronized complete model.forward (only when counters are on)."""
    if not _COUNTERS_PATH:
        yield
        return
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    ok = False
    try:
        yield
        ok = True
    finally:
        if cuda and ok:
            torch.cuda.synchronize()
        _REC["t3"].append({"seconds": time.perf_counter() - t0, "ok": ok})
        dump()


def snapshot() -> dict:
    out = {
        "variant_id": VARIANT.variant_id,
        "policy_id": VARIANT.policy_id,
        "strict": VARIANT.strict,
        "clear_cache": VARIANT.clear_cache,
        "replacement_scope": [
            {"op": o, "call_site": s, "provider": p} for (o, s), p in sorted(VARIANT.replace.items())
        ],
        "counters": [
            {
                "op": o, "call_site": s, "requested_backend": r, "served_backend": sv,
                "policy_id": VARIANT.policy_id, "in_replacement_scope": scope,
                "route_allowed": allowed, "reason": reason, **v,
            }
            for (o, s, r, sv, reason, scope, allowed), v in sorted(_REC["counters"].items(), key=str)
        ],
        "block_clears": dict(_REC["block_clears"]),
        "other_empty_cache_calls": _REC["other_clears"],
        "t3": list(_REC["t3"]),
        "stages": list(_REC["stages"]),
        "inputs": list(_REC["inputs"]),
        "env": {
            k: os.environ.get(k)
            for k in ("CUEQ_TRIATTN_FALLBACK_THRESHOLD", "USE_OPM_CHUNKED", "LAYERNORM_TYPE",
                      "PF_TRIATTN_TABLE", "CRYOZETA_SYNC_STAGE_TIMER", "CRYOZETA_STAGE_PEAKS",
                      "CRYOZETA_TRUNK_DUMP")
        },
    }
    try:
        from cryozeta.openfold_local.model import primitives

        out["cueq_triattn_fallback_threshold"] = primitives.CUEQ_TRIATTN_FALLBACK_THRESHOLD
    except Exception as e:  # noqa: BLE001 - provenance only
        out["cueq_triattn_fallback_threshold"] = f"unavailable: {e}"
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        out["m1_invocation"] = max(
            [torch.cuda.max_memory_allocated()] + [s.get("m1", 0) for s in _REC["stages"]]
        )
        out["m2_invocation"] = max(
            [torch.cuda.max_memory_reserved()] + [s.get("m2", 0) for s in _REC["stages"]]
        )
        out["compute_capability"] = list(torch.cuda.get_device_capability())
        out["gpu_name"] = torch.cuda.get_device_name()
    return out


def dump() -> None:
    if not _COUNTERS_PATH:
        return
    tmp = _COUNTERS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(snapshot(), f, indent=1, default=str)
    os.replace(tmp, _COUNTERS_PATH)


atexit.register(dump)
