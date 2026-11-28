"""Compare the detector's eager and fused GeGLU on the current CUDA device."""

from contextlib import nullcontext
from statistics import median
from unittest.mock import patch

import torch

from cryozeta.em.blocks import GeGLU, try_geglu


def measure(model, x):
    for _ in range(4):
        model(x)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(9):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(50):
            model(x)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 50)
    peak = torch.cuda.max_memory_allocated() - baseline
    return model(x), median(samples), peak


@torch.inference_mode()
def main():
    torch.manual_seed(42)
    print(torch.cuda.get_device_name(), torch.__version__, torch.version.cuda)
    for dtype in (torch.float16, torch.bfloat16):
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            continue
        model = GeGLU(128, 512).cuda().to(dtype).eval()
        x = torch.randn(4, 4096, 128, device="cuda", dtype=dtype)
        for fused in (False, True):
            if fused and (try_geglu is None or try_geglu(model.proj(x)) is None):
                print(f"{dtype} fused GeGLU unavailable on this device; skipping")
                continue
            route = (
                nullcontext() if fused else patch("cryozeta.em.blocks.try_geglu", None)
            )
            with route:
                output, elapsed, peak = measure(model, x)
            if not fused:
                reference = output
            else:
                torch.testing.assert_close(output, reference, atol=0, rtol=0)
            del output
            backend = "Triton" if fused else "PyTorch"
            print(f"{dtype} {backend} {elapsed:.4f} ms, {peak / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
