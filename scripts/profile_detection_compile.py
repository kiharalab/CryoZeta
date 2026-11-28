"""Time detection on changing window sizes using the same checkpoint and inputs."""

import argparse
import statistics
import time
from unittest.mock import patch

import torch

from cryozeta.em.munet import get_detection_model


def measure(model, x, repeats):
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = model(x)
    torch.cuda.synchronize()
    first = time.perf_counter() - start
    samples = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        model(x)
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    return output, first, statistics.median(samples)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--whole-model", action="store_true")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    def load(compile_model):
        return (
            get_detection_model(
                checkpoint_path=args.checkpoint, compile=compile_model, device="cuda"
            )
            .bfloat16()
            .eval()
        )

    baseline = load(False)
    candidate = load(args.whole_model)
    dtype = (
        torch.float16 if torch.cuda.get_device_capability()[0] >= 12 else torch.bfloat16
    )
    print(torch.cuda.get_device_name(), torch.__version__, torch.version.cuda, dtype)
    print(
        "batch roi eager_ms first_candidate_s candidate_ms max_abs_error mean_abs_error"
    )
    generator = torch.Generator(device="cuda").manual_seed(42)
    with torch.autocast("cuda", dtype=dtype):
        for batch, roi in [(4, 64), (2, 48), (1, 32), (3, 64), (4, 64)]:
            x = torch.randn(
                batch, 1, roi, roi, roi, dtype=dtype, device="cuda", generator=generator
            )
            with patch("cryozeta.em.blocks.cast_silu", None):
                ref, _, eager_ms = measure(baseline, x, args.repeats)
            out, first, candidate_ms = measure(candidate, x, args.repeats)
            error = (ref - out).float().abs()
            print(
                batch,
                roi,
                eager_ms,
                first,
                candidate_ms,
                error.max().item(),
                error.mean().item(),
                flush=True,
            )


if __name__ == "__main__":
    main()
