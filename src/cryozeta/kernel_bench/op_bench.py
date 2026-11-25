"""L1 per-op harness: one real CryoZeta module, baseline vs. candidate provider,
through the same kernel_routing seam the model uses.

Weights come from the checkpoint by module key (strict load) or a seeded init. Inputs are synthetic at
(N, P, S) or captured activations (``capture_activations``). Each backend gets: outcome class, first
call (compile included) time and peak-memory delta, CUDA-event median/IQR over repeats x iters with
inputs cloned outside the timed region, determinism, served counters, and errors vs. an fp64 reference
(small N) and vs. the baseline. Records append to ``--out`` (JSONL).

  python -m cryozeta.kernel_bench.op_bench --module empairformer_stack.blocks.0.tri_att_start \
      --checkpoint ASSETS/cryozeta-v0.0.1.safetensors --n 256 --backend fpf_k2b --out l1.jsonl
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import statistics
import sys
import time

REF = {"triatt": "torch", "trimul": "torch", "opm": "einsum", "apb": "torch", "rowcol": "torch"}
# Demo (V0) flags as the model passes them (inference_demo.sh).
V0_FLAGS = {"use_deepspeed_evo_attention": True, "use_cuequivariance_attention": True,
            "use_cuequivariance_multiplicative_update": True, "use_cuequivariance_attention_pair_bias": True}


def site_of(key: str) -> str:
    if ".atom_transformer." in key:
        return "atom"
    for prefix, site in (("empairformer_stack.", "empairformer"), ("msa_module.", "msa"),
                         ("confidence_head.", "confidence"), ("diffusion_module.diffusion_transformer.", "diffusion")):
        if key.startswith(prefix):
            return site
    return "other"


def op_of(mod) -> str:
    names = {c.__name__ for c in type(mod).__mro__}
    for cls, op in (("TriangleAttention", "triatt"), ("TriangleMultiplicativeUpdate", "trimul"),
                    ("OuterProductMean", "opm"), ("AttentionPairBias", "apb"), ("MSAAttention", "rowcol"),
                    ("PointColumnAttentionWithPairBias", "rowcol")):
        if cls in names:
            return op
    raise SystemExit(f"{type(mod).__name__}: not a routed op module")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--module", required=True, help="checkpoint module key, e.g. empairformer_stack.blocks.0.tri_mul_out")
    p.add_argument("--checkpoint", help="safetensors; omitted -> seeded init (contract checks only)")
    p.add_argument("--interpolate", action="store_true", help="build the interpolate model (p_dim 114)")
    p.add_argument("--n", type=int, help="N_token (synthetic inputs)")
    p.add_argument("--n-point", type=int, help="N_point / MSA depth for opm and rowcol")
    p.add_argument("--n-sample", type=int, default=5, help="diffusion samples for diffusion APB")
    p.add_argument("--inputs", help="capture_activations dir (index.json) instead of synthetic inputs")
    p.add_argument("--call", type=int, default=0, help="captured call index")
    p.add_argument("--backend", required=True, help="candidate provider")
    p.add_argument("--baseline-backend", default="v0", help="'v0' (demo flags) or a provider")
    p.add_argument("--strict", action="store_true", help="strict routing for the candidate")
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--chunk-size", type=int)
    p.add_argument("--fp64-max-n", type=int, default=256)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True, help="JSONL to append records to")
    a = p.parse_args(argv)
    if not a.inputs and not a.n:
        p.error("--n is required without --inputs")
    if a.n is not None and a.n < 1 or a.warmup < 0 or a.iters < 1 or a.repeats < 1:
        p.error("sizes and counts must be positive")
    if a.checkpoint and not os.path.isfile(a.checkpoint):
        p.error(f"checkpoint not found: {a.checkpoint}")
    if a.inputs and not os.path.isfile(os.path.join(a.inputs, "index.json")):
        p.error(f"{a.inputs}/index.json not found")
    return a


def build_module(a, torch):
    from cryozeta.model.cryozeta import CryoZeta
    from cryozeta.runner.inference import build_configs

    argv, sys.argv = sys.argv, ["op_bench", "--use_interpolation", "true" if a.interpolate else "false"]
    try:
        configs = build_configs()
    finally:
        sys.argv = argv
    with torch.device("meta"):
        model = CryoZeta(configs)
    mod = model.get_submodule(a.module)
    del model
    if list(mod.buffers()):
        raise SystemExit(f"{a.module} has buffers; op_bench only initializes parameters")
    mod.to_empty(device=a.device)
    if a.checkpoint:
        from safetensors import safe_open

        with safe_open(a.checkpoint, "pt", device="cpu") as f:
            pre = a.module + "."
            sd = {k[len(pre):]: f.get_tensor(k) for k in f.keys() if k.startswith(pre)}
        mod.load_state_dict(sd, strict=True)
    else:
        g = torch.Generator().manual_seed(a.seed)
        with torch.no_grad():
            for n, prm in mod.named_parameters():
                if prm.dim() == 1:
                    prm.copy_(torch.ones_like(prm) if n.endswith("weight") else torch.zeros_like(prm))
                else:
                    prm.copy_(torch.randn(prm.shape, generator=g) * 0.02)
    return mod.eval()


def synth_inputs(a, mod, op, torch):
    """(args, kwargs) shaped as the model calls the op at this site; flags are the demo V0 flags."""
    g = torch.Generator().manual_seed(a.seed + 1)
    dev = a.device

    def rnd(*shape):
        return torch.randn(shape, generator=g).to(dev)

    def dim(name):  # channel width from a LayerNorm weight
        return mod.get_parameter(name).shape[0]

    n, pts = a.n, a.n_point
    key = a.module
    if op == "trimul":
        c = dim("layer_norm_in.weight")
        return [rnd(n, n, c)], {"mask": torch.ones(n, n, device=dev), "inplace_safe": True,
                                "_add_with_inplace": True, "use_cuequivariance_multiplicative_update": True}
    if op == "triatt":
        c = dim("layer_norm.weight")
        return [rnd(n, n, c)], {"mask": torch.ones(n, n, device=dev), "chunk_size": a.chunk_size,
                                "use_memory_efficient_kernel": False, "use_deepspeed_evo_attention": True,
                                "use_cuequivariance_attention": True, "use_lma": False, "inplace_safe": True}
    if not pts and op in ("opm", "rowcol"):
        raise SystemExit("--n-point is required for opm/rowcol")
    if op == "opm":
        c = dim("layer_norm.weight")
        m = rnd(n, pts, c) if "outter_column" in key else rnd(pts, n, c)
        return [m], {"mask": None, "chunk_size": a.chunk_size, "inplace_safe": True}
    if op == "apb":
        c_a = mod.attention.linear_q.weight.shape[1]
        c_z = dim("layernorm_z.weight")
        if mod.has_s:
            c_s = mod.linear_a_last.weight.shape[1]
            # diffusion.py expand_at_dim: z is a stride-0 [N_sample, N, N, c_z] view, not a copy
            z = rnd(n, n, c_z).expand(a.n_sample, n, n, c_z)
            return [rnd(a.n_sample, n, c_a), rnd(a.n_sample, n, c_s), z], {
                "use_cuequivariance_attention_pair_bias": True}
        return [rnd(n, c_a), None, rnd(n, n, c_z)], {"use_cuequivariance_attention_pair_bias": True}
    flags = {"chunk_size": a.chunk_size, "use_deepspeed_evo_attention": True, "use_lma": False}
    if type(mod).__name__ == "PointColumnAttentionWithPairBias":
        c_m = dim("column_att.layer_norm_m.weight")
        c_p = dim("column_att.layer_norm_z.weight")
        return [rnd(pts, n, c_m), rnd(pts, pts, c_p)], flags
    c_m, c_z = dim("layer_norm_m.weight"), dim("layer_norm_z.weight")
    return [rnd(pts, n, c_m), rnd(n, n, c_z)], {**flags, "use_memory_efficient_kernel": False}


def captured_inputs(a, torch):
    from safetensors.torch import load_file

    idx = json.load(open(os.path.join(a.inputs, "index.json")))
    rec = next((r for r in idx["calls"] if r["module"] == a.module and r["call"] == a.call), None)
    if rec is None:
        raise SystemExit(f"no captured call {a.call} for {a.module}")
    t = load_file(os.path.join(a.inputs, rec["file"]), device=a.device)
    args = [t.get(f"x{i}") for i in range(rec["n_args"])]
    kwargs = {**rec["kwargs_json"], **{k[3:]: v for k, v in t.items() if k.startswith("kw.")}}
    return args, kwargs, rec


def clone(x, dtype=None):
    import torch

    if torch.is_tensor(x):
        bcast = [i for i, (d, s) in enumerate(zip(x.shape, x.stride())) if s == 0 and d > 1]
        if bcast:  # keep expand() views as views: a dense copy would overstate peak memory
            base = x[tuple(slice(0, 1) if i in bcast else slice(None) for i in range(x.dim()))]
            return clone(base, dtype).expand(x.shape)
        y = x.detach().clone()
        return y.to(dtype) if dtype is not None and y.is_floating_point() else y
    return x


def call_args(args, kwargs, dtype=None):
    return [clone(x, dtype) for x in args], {k: clone(v, dtype) for k, v in kwargs.items()}


def errors(out, ref):
    import torch

    d = (out.double() - ref.double())
    return {"max_abs": d.abs().max().item(), "rel_fro": (d.norm() / ref.double().norm().clamp_min(1e-30)).item(),
            "nonfinite": int((~torch.isfinite(out)).sum().item())}


def main(argv=None):
    a = parse_args(argv)
    import torch

    from cryozeta.model import kernel_routing as kr

    torch.manual_seed(a.seed)
    mod = build_module(a, torch)
    op, site = op_of(mod), site_of(a.module)
    if a.backend not in kr.OPS[op] or a.baseline_backend not in kr.OPS[op] | {"v0"}:
        raise SystemExit(f"providers for {op}: {sorted(kr.OPS[op])}")
    if a.inputs:
        args, kwargs, src = captured_inputs(a, torch)
        src = {"source": "captured", **{k: src[k] for k in ("file", "sha256", "shapes")}}
    else:
        args, kwargs = synth_inputs(a, mod, op, torch)
        src = {"source": "synthetic", "seed": a.seed}
    # One calling convention: everything by name (the model passes most module inputs as keywords).
    kwargs = dict(inspect.signature(mod.forward).bind(*args, **kwargs).arguments)
    args, primary = [], next(iter(kwargs))
    residual = op == "trimul" and kwargs.get("inplace_safe") and kwargs.get("_add_with_inplace")
    n = a.n or kwargs[primary].shape[-3 if "outter_column" in a.module else -2]
    cuda = torch.device(a.device).type == "cuda"
    kr._COUNTERS_PATH = os.path.abspath(a.out) + ".counters.json"
    kr._SITE.set(site)
    autocast = (lambda: torch.autocast(device_type=torch.device(a.device).type, dtype=torch.bfloat16)) \
        if a.dtype == "bf16" else (lambda: torch.autocast(device_type=torch.device(a.device).type, enabled=False))

    def sync():
        if cuda:
            torch.cuda.synchronize()

    def run(m, backend, strict, x_args, x_kwargs, amp=True):
        kr.VARIANT = kr.Variant(variant_id=f"op_bench:{backend}", strict=strict,
                                replace={} if backend == "v0" else {(op, site): backend})
        x_in = x_kwargs[primary].detach().clone() if residual else None
        with torch.no_grad(), (autocast() if amp else torch.autocast(device_type="cpu", enabled=False)):
            out = m(*x_args, **x_kwargs)
        out = out.float() if out.dtype != torch.float64 else out
        return out - x_in.to(out.dtype) if residual else out

    def measure(backend, strict):
        kr.reset()
        rec = {"backend": backend, "strict": strict}
        try:
            sync()
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated() if cuda else 0
            x_args, x_kwargs = call_args(args, kwargs)
            t0 = time.perf_counter()
            out = run(mod, backend, strict, x_args, x_kwargs)
            sync()
            rec["first_call_s"] = time.perf_counter() - t0
            rec["peak_delta_bytes"] = torch.cuda.max_memory_allocated() - base if cuda else None
            del x_args, x_kwargs
            reps = [run(mod, backend, strict, *call_args(args, kwargs)) for _ in range(2)]
            rec["deterministic"] = all(torch.equal(out, r) for r in reps)
            del reps
            times = []
            for _ in range(a.repeats):
                for i in range(a.warmup + a.iters):
                    x_args, x_kwargs = call_args(args, kwargs)  # outside the timed region
                    if cuda:
                        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        s.record()
                        run(mod, backend, strict, x_args, x_kwargs)
                        e.record()
                        e.synchronize()
                        ms = s.elapsed_time(e)
                    else:
                        t = time.perf_counter()
                        run(mod, backend, strict, x_args, x_kwargs)
                        ms = (time.perf_counter() - t) * 1e3
                    if i >= a.warmup:
                        times.append(ms)
            q = statistics.quantiles(times, n=4) if len(times) > 1 else [times[0]] * 3
            rec.update(outcome="SUCCESS", median_ms=statistics.median(times), iqr_ms=q[2] - q[0],
                       n_timed=len(times))
        except kr.KernelUnsupported as e:
            rec.update(outcome="UNSUPPORTED", detail=str(e))
            out = None
        except kr.KernelFallback as e:
            rec.update(outcome="FALLBACK", detail=str(e))
            out = None
        except Exception as e:  # noqa: BLE001 - classified below
            rec.update(outcome="GPU_OOM" if kr.is_oom(e) else "CRASH", detail=repr(e)[:500])
            out = None
        snap = kr.snapshot()
        rec["counters"] = snap["counters"]
        if out is not None and any(not c["route_allowed"] for c in snap["counters"]):
            rec["outcome"] = "FALLBACK"
            rec["detail"] = "served backend outside the requested route"
        if out is not None and not torch.isfinite(out).all():
            rec["outcome"] = "NUMERICAL"
        return rec, (out if rec["outcome"] == "SUCCESS" else None)  # no errors/gates from a non-SUCCESS arm

    common = {"module": a.module, "op": op, "call_site": site, "N": n, "N_point": a.n_point,
              "dtype": a.dtype, "device": a.device, "inputs": src, "checkpoint": a.checkpoint,
              "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name() if cuda else None,
              "compute_capability": list(torch.cuda.get_device_capability()) if cuda else None}
    base_rec, base_out = measure(a.baseline_backend, False)
    cand_rec, cand_out = measure(a.backend, a.strict)

    ref = None
    if n <= a.fp64_max_n:  # fp64 reference: REF provider, OpenFold LayerNorm, no autocast
        from cryozeta.openfold_local.model.primitives import OpenFoldLayerNorm

        m64 = copy.deepcopy(mod)
        for name, sub in list(m64.named_modules()):
            if type(sub).__name__ == "FusedLayerNorm":
                ln = OpenFoldLayerNorm(sub.weight.shape[0], sub.eps).to(sub.weight.device)
                ln.load_state_dict(sub.state_dict())
                parent = m64.get_submodule(name.rsplit(".", 1)[0]) if "." in name else m64
                setattr(parent, name.rsplit(".", 1)[-1], ln)
        m64 = m64.double()
        kr.reset()
        x64, kw64 = call_args(args, kwargs, torch.float64)
        try:
            ref = run(m64, REF[op], False, x64, kw64, amp=False)
        except Exception as e:  # noqa: BLE001 - recorded, reference just unavailable
            common["fp64_reference_error"] = repr(e)[:300]
        del m64
    for rec, out in ((base_rec, base_out), (cand_rec, cand_out)):
        if out is not None and ref is not None:
            rec["vs_fp64"] = errors(out, ref)
    if cand_out is not None and base_out is not None:
        cand_rec["vs_baseline"] = errors(cand_out, base_out)
    gate = {"G-op": None}
    if "vs_fp64" in cand_rec and "vs_fp64" in base_rec:
        gate["G-op"] = cand_rec["vs_fp64"]["rel_fro"] <= 2 * base_rec["vs_fp64"]["rel_fro"]
        gate["rule"] = "candidate rel_fro vs fp64 <= 2 x baseline rel_fro vs fp64 (G-op)"
    if base_rec["outcome"] == cand_rec["outcome"] == "SUCCESS":
        gate["speed_ratio"] = cand_rec["median_ms"] / base_rec["median_ms"]
    with open(a.out, "a") as f:
        for role, rec in (("baseline", base_rec), ("candidate", cand_rec)):
            f.write(json.dumps({**common, "role": role, **rec}, default=str) + "\n")
        f.write(json.dumps({**common, "role": "gate", "baseline": a.baseline_backend, "candidate": a.backend,
                            **gate}, default=str) + "\n")
    print(json.dumps({"op": op, "site": site, "N": n, "baseline": base_rec["outcome"],
                      "candidate": cand_rec["outcome"], **gate}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
