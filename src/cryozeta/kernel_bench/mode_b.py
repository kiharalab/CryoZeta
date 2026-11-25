"""Mode B in-process repeat driver: one loaded model, >= 1 untimed complete warmup
forward, then timed complete forwards through the unchanged ``inference.main``.

Each iteration rebuilds features on CPU through the dataloader, reseeds (``main`` does), writes to a
fresh ``iter_<i>`` dir (so no output-skip), and releases inputs/outputs when ``main`` returns. An
iteration counts as warmed only if it follows the warmups and saw no Triton compile and no new
compile-cache files. Launched by ``run_trial --mode B``; usable directly:

  python -m cryozeta.kernel_bench.mode_b --warmup 1 --timed 1 --iter-root DIR --record rec.json \
      -- <cryozeta-inference args>
"""

from __future__ import annotations

import argparse
import gc
import os
import sys

from cryozeta.kernel_bench import common as C


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--timed", type=int, default=1)
    p.add_argument("--iter-root", required=True)
    p.add_argument("--record", required=True)
    p.add_argument("cli_args", nargs=argparse.REMAINDER, help="-- then cryozeta-inference arguments")
    a = p.parse_args(argv)
    a.cli_args = a.cli_args[1:] if a.cli_args[:1] == ["--"] else a.cli_args
    if a.warmup < 1 or a.timed < 1:
        p.error("need >= 1 warmup and >= 1 timed forward")
    if not a.cli_args:
        p.error("missing cryozeta-inference arguments after --")
    return a


class CompileCounter:
    """Counts in-process Triton compiles via ``JITFunction.compiled_hook`` (triton 3.3)."""

    def __init__(self):
        self.n = 0
        try:
            from triton.runtime.jit import JITFunction

            prev = JITFunction.compiled_hook

            def hook(*args, **kwargs):
                self.n += 1
                return prev(*args, **kwargs) if prev else False

            JITFunction.compiled_hook = hook
            self.available = True
        except Exception:  # noqa: BLE001 - no triton: cache deltas still apply
            self.available = False


def main(argv=None):
    a = parse_args(argv)
    sys.argv = ["cryozeta-inference", *a.cli_args]
    import torch

    from cryozeta.model import kernel_routing as kr
    from cryozeta.runner.inference import InferenceRunner, build_configs, main as infer

    configs = build_configs()
    compiles = CompileCounter()
    runner = InferenceRunner(configs)
    rec = {"warmup": a.warmup, "timed": a.timed, "compile_hook": compiles.available, "iterations": []}
    for i in range(a.warmup + a.timed):
        d = os.path.join(a.iter_root, f"iter_{i}")
        configs.dump_dir = runner.dump_dir = runner.dumper.base_dir = d
        os.makedirs(d, exist_ok=True)
        kr.reset()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        n0, c0 = compiles.n, C.cache_entries(os.environ)
        infer(configs, runner)
        snap = kr.snapshot()
        c1 = C.cache_entries(os.environ)
        new_files = sum(c1[k] - c0[k] for k in c1)
        t3 = [t["seconds"] for t in snap["t3"] if t["ok"]]
        rec["iterations"].append({
            "index": i, "dir": os.path.relpath(d, a.iter_root), "timed": i >= a.warmup,
            "compile_events": compiles.n - n0, "cache_new_files": new_files,
            "warmed": i >= a.warmup and compiles.n == n0 and new_files == 0 and len(t3) == 1,
            "t3": t3[0] if len(t3) == 1 else None, "t3_all": snap["t3"], "stages": snap["stages"],
            "counters": snap["counters"], "inputs": snap["inputs"], "block_clears": snap["block_clears"],
            "other_empty_cache_calls": snap["other_empty_cache_calls"],
            "m1_diagnostic": snap.get("m1_invocation"), "m2_diagnostic": snap.get("m2_invocation"),
            "err_files": C.err_files(d),
        })
        C.write_json(a.record, rec)
        gc.collect()  # release references; the allocator cache is kept (no between-iteration clear)
    return 0


if __name__ == "__main__":
    sys.exit(main())
