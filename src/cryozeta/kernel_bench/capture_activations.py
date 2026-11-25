"""Capture real module inputs from one CryoZeta inference run for ``op_bench --inputs``.

Forward-pre-hooks on the selected modules (checkpoint keys or fnmatch patterns) copy the first
``--calls`` invocations to CPU safetensors (tensors as ``x<i>`` / ``kw.<name>``; non-tensor kwargs in
``index.json``), each with its sha256 and shapes. Calls above ``--max-bytes`` are recorded as skipped.

  python -m cryozeta.kernel_bench.capture_activations --out DIR \
      --modules 'empairformer_stack.blocks.0.*' -- <cryozeta-inference args>
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys

from cryozeta.kernel_bench import common as C


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="new capture directory (must not exist)")
    p.add_argument("--modules", nargs="+", required=True, help="module keys or fnmatch patterns")
    p.add_argument("--calls", type=int, default=1, help="invocations captured per module")
    p.add_argument("--max-bytes", type=float, default=4e9, help="per-call tensor byte cap")
    p.add_argument("cli_args", nargs=argparse.REMAINDER, help="-- then cryozeta-inference arguments")
    a = p.parse_args(argv)
    a.cli_args = a.cli_args[1:] if a.cli_args[:1] == ["--"] else a.cli_args
    if not a.cli_args:
        p.error("missing cryozeta-inference arguments after --")
    if a.calls < 1 or a.max_bytes <= 0:
        p.error("--calls and --max-bytes must be positive")
    if os.path.exists(a.out):
        p.error(f"--out exists: {a.out}")
    return a


def jsonable(v):
    try:
        json.dumps(v)
        return True
    except TypeError:
        return False


def main(argv=None):
    a = parse_args(argv)
    sys.argv = ["cryozeta-inference", *a.cli_args]
    import torch
    from safetensors.torch import save_file

    from cryozeta.runner.inference import InferenceRunner, build_configs, main as infer

    configs = build_configs()
    runner = InferenceRunner(configs)
    os.makedirs(a.out)
    index = {"cli_args": a.cli_args, "patterns": a.modules, "calls": [], "skipped": []}
    seen: dict[str, int] = {}

    def hook(key):
        def pre(mod, args, kwargs):
            i = seen.get(key, 0)
            if i >= a.calls:
                return
            seen[key] = i + 1
            tensors = {f"x{j}": x for j, x in enumerate(args) if torch.is_tensor(x)}
            tensors.update({f"kw.{k}": v for k, v in kwargs.items() if torch.is_tensor(v)})
            size = sum(t.numel() * t.element_size() for t in tensors.values())
            if size > a.max_bytes:
                index["skipped"].append({"module": key, "call": i, "bytes": size, "why": "max-bytes"})
                return
            rest = {k: v for k, v in kwargs.items() if not torch.is_tensor(v)}
            bad = [k for k, v in rest.items() if not jsonable(v)]
            if bad or any(x is not None and not torch.is_tensor(x) for x in args):
                index["skipped"].append({"module": key, "call": i, "why": f"non-JSON inputs {bad}"})
                return
            f = f"{key}.{i}.safetensors"
            path = os.path.join(a.out, f)
            save_file({k: v.detach().to("cpu").contiguous().clone() for k, v in tensors.items()}, path)
            index["calls"].append({
                "module": key, "call": i, "file": f, "sha256": C.sha256_file(path), "bytes": size,
                "n_args": len(args), "kwargs_json": rest,
                "shapes": {k: [list(v.shape), str(v.dtype)] for k, v in tensors.items()},
            })
        return pre

    hooked = [k for k, _ in runner.model.named_modules() if any(fnmatch.fnmatchcase(k, p) for p in a.modules)]
    if not hooked:
        raise SystemExit(f"no module matches {a.modules}")
    for k in hooked:
        runner.model.get_submodule(k).register_forward_pre_hook(hook(k), with_kwargs=True)
    index["hooked"] = hooked
    try:
        infer(configs, runner)
    finally:
        C.write_json(os.path.join(a.out, "index.json"), index)
    print(json.dumps({"captured": len(index["calls"]), "skipped": len(index["skipped"]), "out": a.out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
