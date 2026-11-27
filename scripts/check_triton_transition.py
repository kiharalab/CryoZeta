"""Standalone correctness + memory + speed check for the fused Triton transition.

FP32 only: the fused kernel supports CUDA float32 tensors exclusively. The
fp64 reference is computed in bounded 4096-row chunks so even checkpoint-scale
row counts cannot OOM the GPU.

Run: python scripts/check_triton_transition.py [--rows N] [--bench]
"""

import argparse
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "src")

from cryozeta.model.modules.triton_transition import try_fused_transition

# Hard cap on rows materialized for the fp64 reference in one chunk.
REF_CHUNK = 4096


def reference(x, ln_w, ln_b, wa, wb, wo, eps, dtype=torch.float64):
    xf = x.to(dtype)
    mean = xf.mean(-1, keepdim=True)
    var = xf.var(-1, unbiased=False, keepdim=True)
    xn = (xf - mean) / torch.sqrt(var + eps) * ln_w.to(dtype) + ln_b.to(dtype)
    a = xn @ wa.to(dtype).t()
    b = xn @ wb.to(dtype).t()
    return (F.silu(a) * b) @ wo.to(dtype).t()


def make_inputs(rows, c_in=128, n=4, scale=1.0, device="cuda", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(rows, c_in, generator=g).mul(scale).to(device=device)
    ln_w = torch.randn(c_in, generator=g).abs().add(0.5).to(device)
    ln_b = torch.randn(c_in, generator=g).mul(0.1).to(device)
    wa = torch.randn(n * c_in, c_in, generator=g).mul(0.05 * scale).to(device)
    wb = torch.randn(n * c_in, c_in, generator=g).mul(0.05 * scale).to(device)
    wo = torch.randn(c_in, n * c_in, generator=g).mul(0.05 * scale).to(device)
    return x, ln_w, ln_b, wa, wb, wo


def check_case(x, ln_w, ln_b, wa, wb, wo, eps, label, limit):
    with torch.no_grad():
        out = try_fused_transition(x, ln_w, ln_b, wa, wb, wo, eps)
    assert out is not None, f"kernel declined {label}"
    assert torch.isfinite(out).all(), f"non-finite output for {label}"
    err_vs64 = err32_vs64 = scale = 0.0
    error_sq = reference_sq = 0.0
    for x_chunk, out_chunk in zip(
        x.split(REF_CHUNK), out.split(REF_CHUNK), strict=True
    ):
        # fp64 reference lives on the CPU in bounded chunks: no GPU OOM and
        # no unbounded host allocation regardless of the row count.
        ref64 = reference(
            x_chunk.cpu(), ln_w.cpu(), ln_b.cpu(), wa.cpu(), wb.cpu(), wo.cpu(), eps
        )
        ref32 = reference(
            x_chunk.cpu(),
            ln_w.cpu(),
            ln_b.cpu(),
            wa.cpu(),
            wb.cpu(),
            wo.cpu(),
            eps,
            torch.float32,
        )
        got = out_chunk.cpu().to(torch.float64)
        error = got - ref64
        err_vs64 = max(err_vs64, error.abs().max().item())
        err32_vs64 = max(err32_vs64, (ref32 - ref64).abs().max().item())
        scale = max(scale, ref64.abs().max().item())
        error_sq += error.square().sum().item()
        reference_sq += ref64.square().sum().item()
    rel_l2 = (error_sq / reference_sq) ** 0.5 if reference_sq else error_sq**0.5
    assert rel_l2 <= limit, (label, rel_l2)
    print(
        f"{label:>34}: max|err| vs fp64 = {err_vs64:.3e}"
        f" (torch fp32 ref: {err32_vs64:.3e}, out scale {scale:.2e}, rel L2 {rel_l2:.3e})"
    )
    assert out.dtype == torch.float32 and out.shape == x.shape
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=4096)
    p.add_argument("--bench", action="store_true")
    args = p.parse_args()

    eps = 1e-5
    print(
        f"device: {torch.cuda.get_device_name(0)}, cc {torch.cuda.get_device_capability(0)}"
    )

    # ---- correctness across row counts, fp32 only ----
    for rows in (1, 7, 33, 1025, 2048, args.rows):
        x, ln_w, ln_b, wa, wb, wo = make_inputs(rows)
        check_case(x, ln_w, ln_b, wa, wb, wo, eps, f"rows={rows} fp32", 1e-4)
        del x

    # ---- adversarial finite magnitudes: large and tiny, never inf/NaN ----
    for label, scale in (
        ("adversarial large x1e2", 1e2),
        ("adversarial tiny x1e-4", 1e-4),
    ):
        x, ln_w, ln_b, wa, wb, wo = make_inputs(517, scale=scale, seed=1)
        assert torch.isfinite(x).all() and torch.isfinite(wa).all()
        # Error scales with the data; bound the relative error only.
        check_case(x, ln_w, ln_b, wa, wb, wo, eps, f"{label} fp32", 1e-3)
        del x

    # ---- near-zero variance rows (constant rows stress the eps path) ----
    x, ln_w, ln_b, wa, wb, wo = make_inputs(64, seed=2)
    x[:8] = 3.25  # zero-variance rows: output must stay finite
    check_case(x, ln_w, ln_b, wa, wb, wo, eps, "zero-variance rows fp32", 1e-4)
    del x

    # ---- ragged channel count (generality check) ----
    x, ln_w, ln_b, wa, wb, wo = make_inputs(129, c_in=96, n=3)
    check_case(x, ln_w, ln_b, wa, wb, wo, eps, "c_in=96 n=3 generality", 1e-4)
    del x

    # ---- helper must decline non-FP32 / non-CUDA / bad shapes, not crash ----
    x, ln_w, ln_b, wa, wb, wo = make_inputs(16)
    declined = [
        ("fp16 x", x.half(), ln_w, ln_b, wa, wb, wo),
        ("fp16 weights", x, ln_w.half(), ln_b, wa, wb, wo),
        ("cpu tensors", x.cpu(), ln_w.cpu(), ln_b.cpu(), wa.cpu(), wb.cpu(), wo.cpu()),
        ("mismatched wo", x, ln_w, ln_b, wa, wb, wo[:, :-1].contiguous()),
    ]
    for label, *t in declined:
        with torch.no_grad():
            assert try_fused_transition(*t, eps) is None, f"not declined: {label}"
    print("decline guards: fp16/cpu/mismatched shapes all declined cleanly")
    for bad_eps in (-1.0, float("nan"), float("inf")):
        try:
            try_fused_transition(x, ln_w, ln_b, wa, wb, wo, bad_eps)
        except ValueError:
            pass
        else:
            raise AssertionError(f"eps={bad_eps} not rejected")
    print("eps validation: negative/NaN/inf eps rejected")
    del x

    if not args.bench:
        return

    # ---- memory + speed at checkpoint scale (1980^2 rows) ----
    rows = 1980 * 1980
    x, ln_w, ln_b, wa, wb, wo = make_inputs(rows)

    def pytorch_chunked(x, rows_per_chunk):
        outs = torch.empty_like(x)
        for s in range(0, x.shape[0], rows_per_chunk):
            c = x[s : s + rows_per_chunk]
            xf = c.float()
            mean = xf.mean(-1, keepdim=True)
            var = xf.var(-1, unbiased=False, keepdim=True)
            xn = (xf - mean) / torch.sqrt(var + eps) * ln_w + ln_b
            a = F.silu(xn @ wa.t())
            b = xn @ wb.t()
            outs[s : s + rows_per_chunk] = a * b @ wo.t()
        return outs

    def measure(fn, label, warmup=3, iters=10):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / iters
        peak = torch.cuda.max_memory_allocated() - base
        print(f"{label:>28}: {ms:8.2f} ms  peak-extra {peak / 2**30:6.2f} GiB")
        return fn()

    def triton_fn():
        with torch.no_grad():
            return try_fused_transition(x, ln_w, ln_b, wa, wb, wo, eps)

    r_tri = measure(triton_fn, "triton fused")
    r_pt = measure(lambda: pytorch_chunked(x, 262144), "torch chunked 256k rows")
    err = max(
        (a - b).abs().max().item()
        for a, b in zip(r_tri.split(REF_CHUNK), r_pt.split(REF_CHUNK), strict=True)
    )
    print(f"triton vs torch chunked max|diff|: {err:.3e}")


if __name__ == "__main__":
    main()
