"""Benchmark PyTorch and Triton triangle attention with matched inputs."""

import argparse
import json
import math
from pathlib import Path
from statistics import median

import torch

from cryozeta.openfold_local.model import primitives
from cryozeta.openfold_local.model.triangular_attention import TriangleAttention


def load_module(path, key, device):
    from safetensors import safe_open

    prefix = key + "."
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        state = {
            name.removeprefix(prefix): checkpoint.get_tensor(name)
            for name in checkpoint.keys()
            if name.startswith(prefix)
        }
    if not state:
        raise ValueError(f"checkpoint has no module {key!r}")
    heads, channels = state["linear.weight"].shape
    width = state["mha.linear_q.weight"].shape[0] // heads
    ending = key.endswith(".tri_att_end")
    module = TriangleAttention(channels, width, heads, starting=not ending).eval()
    module.load_state_dict(state, strict=True)
    return module.to(device), channels, width


def measure(routes, device, warmup, repeats):
    for fn in routes.values():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize(device)
    times = {name: [] for name in routes}
    peaks = {name: 0 for name in routes}
    names = list(routes)
    for i in range(repeats):
        for name in names[:: -1 if i % 2 else 1]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            base = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            start.record()
            output = routes[name]()
            end.record()
            end.synchronize()
            times[name].append(start.elapsed_time(end))
            peaks[name] = max(
                peaks[name], torch.cuda.max_memory_allocated(device) - base
            )
            del output
    return {
        name: {
            "median_ms": median(times[name]),
            "peak_extra_allocated_bytes": peaks[name],
        }
        for name in names
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--module", required=True, help="triangle attention checkpoint prefix"
    )
    parser.add_argument("--length", type=int, default=257)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if (
        min(args.length, args.rows, args.warmup, args.repeats) < 1
        or args.rows > args.length
    ):
        parser.error(
            "length, rows, warmup, and repeats must be positive, with rows <= length"
        )
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("a CUDA device is required")

    dtype = torch.bfloat16
    with (
        torch.cuda.device(device),
        torch.inference_mode(),
        torch.autocast("cuda", dtype=dtype),
    ):
        module, channels, width = load_module(args.checkpoint, args.module, device)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        x = torch.randn(
            args.length,
            args.length,
            channels,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        x = module.layer_norm(x)
        bias = module.linear(x).permute(2, 0, 1).unsqueeze(0)
        q, k, v = module.mha._prep_qkv(
            x[: args.rows], x[: args.rows], apply_scale=False
        )
        keep = (
            torch.rand(args.rows, 1, 1, args.length, generator=generator, device=device)
            >= 0.1
        )
        keep[..., 0] = True
        mask = keep.to(dtype)
        additive_mask = (mask - 1) * module.inf
        del x, module
        scaled = q / math.sqrt(width)
        biases = [additive_mask, bias]
        from cryozeta.kernels.triton_triangle import try_triangle

        if try_triangle(scaled, k, v, biases) is None:
            parser.error(
                "Triton triangle attention does not support this input or CUDA stack"
            )
        routes = {
            "pytorch": lambda: primitives._bounded_attention(scaled, k, v, biases),
            "triton": lambda: try_triangle(scaled, k, v, biases),
        }
        timings = measure(routes, device, args.warmup, args.repeats)
        ref, cand = (fn() for fn in routes.values())
        delta = cand.float() - ref.float()
        if not torch.isfinite(ref).all() or not torch.isfinite(cand).all():
            raise RuntimeError("a benchmark backend returned nonfinite values")
        result = {
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "module": args.module,
            "shape": list(q.shape),
            "inputs": "seeded synthetic activations with checkpoint weights",
            "seed": args.seed,
            "dtype": str(dtype),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "backends": timings,
            "triton_vs_pytorch": {
                "max_abs": delta.abs().max().item(),
                "relative_l2": (
                    delta.norm() / ref.float().norm().clamp_min(1e-30)
                ).item(),
                "bit_mismatches": (
                    ref.contiguous().view(torch.int16)
                    != cand.contiguous().view(torch.int16)
                )
                .sum()
                .item(),
                "elements": ref.numel(),
            },
        }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
