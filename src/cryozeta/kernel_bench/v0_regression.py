"""V0 numerical regression: this working tree's inference vs a baseline git revision under identical
weights, inputs, settings, seeds and environment. CPU or GPU: the device, flags and kernels are whatever the
inference args and the caller's env select, identical for every arm.

The baseline is ``git archive <rev> src`` extracted into OUT/baseline; the working tree is never touched.
Every arm runs ``cryozeta.runner.inference`` in a child whose PYTHONPATH puts that arm's src first (the
imported ``cryozeta.__file__`` is checked), with cwd OUT/<arm>, a relative ``--dump_dir`` and one shared
task-scoped kernel cache (OUT/cache, so autotuned GPU kernels are the same for every arm). Arms:

  baseline  <rev> tree, kernel-bench env stripped
  patched   working tree, kernel-bench env stripped (routing inert)
  v0trial   working tree under variants/V0.yaml with counters and the sync stage timer, as run_trial sets them
  repeat    patched again: the run-to-run noise floor (a difference here means the bitwise check is void)
  --route   one strict arm per op, replacing it at every call site the v0trial counters saw with the first
            provider in ALT that V0 did not serve there, else ALT's first (opm einsum, rowcol lma,
            apb torch identity, trimul C-CAT)

Every dumped file is compared with baseline (routing arms: with v0trial): bytes first, then .json numbers,
.pt tensors and .npz arrays as max |diff|. An .npz counts as identical when every member's bytes are (zip
entries carry timestamps); object members are compared as raw bytes, never unpickled. Exit 0 iff patched, v0trial
and repeat preserve all baseline files bitwise and any new seed score tables match the original scores and
coordinate ranks, and every routing arm ran with its counters showing the
requested provider served at every replaced site (an arm whose every site keeps V0's served provider, or
swaps cueq trimul for C-CAT, must also be bitwise equal);
report "v0_ok" is the V0 part alone. Without CUDA the child shims Module.to(cuda) to CPU (see CHILD).

  CUDA_VISIBLE_DEVICES= pixi run -e default --frozen python -m cryozeta.kernel_bench.v0_regression \\
      --out NEW_DIR [--rev REV] [--route] -- <cryozeta-inference args>
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import pickle
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from cryozeta.kernel_bench import common as C

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BASELINE_REV = "c544c6bbd0366951d8b4e0bc4c0da547836f7b8f"
V0_YAML = os.path.join(REPO, "docs", "kernel-bench", "variants", "V0.yaml")
STRIP = ("CRYOZETA_KERNEL_VARIANT", "CRYOZETA_KERNEL_COUNTERS", "CRYOZETA_SYNC_STAGE_TIMER",
         "CRYOZETA_STAGE_PEAKS", "CRYOZETA_TRUNK_DUMP")
# apb: torch only. cueq APB is ineligible at every site of this model (c544c6b gate c_hidden % 32: pairformer
# 384/16 = 24, diffusion 768/16 = 48) and atom attention is local with no alternative, so a strict cueq request
# raises KernelFallback (run2/route_apb). Its arm is identity routing: strict, in scope, must be bitwise.
ALT = {"opm": ("einsum", "chunked"), "rowcol": ("lma", "torch"), "apb": ("torch",), "trimul": ("cueq_cached_cat",)}
# CPU shim, identical in every arm and only without CUDA: c544c6b cryozeta.py:467-476 takes a CPU-resident trunk
# for the leftover of an OOM and moves it to torch.device("cuda"), so a CPU-only run cannot finish. Module.to(cuda)
# then stays on CPU (a no-op, as every module already is). Source and V0 behaviour are untouched.
CHILD = """import sys, torch, cryozeta, numpy
print("CRYOZETA_FILE=" + cryozeta.__file__, flush=True)
print("NUMPY_FILE=" + numpy.__file__, flush=True)
if not torch.cuda.is_available():
    _to = torch.nn.Module.to
    def _cpu_to(self, *a, **k):
        if a and str(a[0]).startswith("cuda"):
            a = ("cpu",) + a[1:]
        if str(k.get("device", "")).startswith("cuda"):
            k["device"] = "cpu"
        return _to(self, *a, **k)
    torch.nn.Module.to = _cpu_to
    print("CPU_SHIM=Module.to(cuda)->cpu", flush=True)
from cryozeta.runner.inference import cli
sys.argv[0] = "cryozeta-inference"
cli()
"""


def _leaves(x, key=""):
    """Flatten parsed JSON / tensors / arrays to {path: leaf}."""
    if isinstance(x, dict):
        return {k2: v for k, val in x.items() for k2, v in _leaves(val, f"{key}/{k}").items()}
    if isinstance(x, (list, tuple)):
        return {k2: v for i, val in enumerate(x) for k2, v in _leaves(val, f"{key}/{i}").items()}
    return {key: x}


def _load(path):
    if path.endswith(".json"):
        with open(path) as f:
            return json.load(f)
    if path.endswith(".npz"):  # object members (atom names etc.) stay raw .npy bytes: never unpickled
        import numpy as np

        out = {}
        with np.load(path, allow_pickle=False) as z:
            for k in z.files:
                try:
                    out[k] = z[k]
                except ValueError:
                    out[k] = z.zip.read(k + ".npy")
        return out
    if path.endswith(".pt"):
        import torch

        return torch.load(path, map_location="cpu", weights_only=False)
    return None


def _max_diff(a, b):
    """max |a-b| for numbers/arrays/tensors; 0.0 for equal non-numeric leaves; None if incomparable."""
    import numpy as np

    def to(v):  # numpy has no bfloat16 (GPU dumps carry bf16 logits): upcast, which is exact
        if not hasattr(v, "detach"):
            return np.asarray(v)
        v = v.detach().cpu()
        return (v.float() if str(v.dtype) == "torch.bfloat16" else v).numpy()

    try:
        x, y = to(a), to(b)
    except Exception:  # noqa: BLE001
        return 0.0 if a == b else None
    if x.shape != y.shape or x.dtype.kind != y.dtype.kind:
        return None
    if x.dtype.kind not in "biufc":
        return 0.0 if np.array_equal(x, y) else None
    if x.size == 0:
        return 0.0
    if np.array_equal(x, y, equal_nan=x.dtype.kind in "fc"):
        return 0.0
    d = np.abs(x.astype(np.float64) - y.astype(np.float64))
    return float(np.nanmax(d)) if not np.isnan(d).all() else float("nan")


def _npz_members(path):
    import zipfile

    with zipfile.ZipFile(path) as z:
        return {n: z.read(n) for n in z.namelist()}


def compare_file(a: str, b: str) -> dict:
    with open(a, "rb") as f, open(b, "rb") as g:
        ba, bb = f.read(), g.read()
    if ba == bb:
        return {"identical": True}
    if a.endswith(".npz") and _npz_members(a) == _npz_members(b):  # same arrays, zip header times differ
        return {"identical": True, "npz_members_identical": True}
    pa = _load(a)
    if pa is not None:
        la, lb = _leaves(pa), _leaves(_load(b))
        if la.keys() != lb.keys():
            return {"identical": False, "structure": "differs",
                    "keys_only_ref": sorted(la.keys() - lb.keys())[:5], "keys_only_cand": sorted(lb.keys() - la.keys())[:5]}
        diffs = {k: _max_diff(la[k], lb[k]) for k in la}
        bad = sorted(k for k, d in diffs.items() if d is None)
        num = {k: d for k, d in diffs.items() if d}
        top = sorted(num, key=lambda k: -num[k] if num[k] == num[k] else float("-inf"))[:5]
        return {"identical": False, "value_equal": not bad and not num, "incomparable": bad[:5],
                "max_abs_diff": max(num.values(), default=0.0), "worst": {k: num[k] for k in top}}
    la, lb = ba.decode(errors="replace").splitlines(), bb.decode(errors="replace").splitlines()
    i = next((i for i, (x, y) in enumerate(zip(la, lb)) if x != y), min(len(la), len(lb)))
    return {"identical": False, "first_diff_line": i + 1, "ref": la[i][:200] if i < len(la) else None,
            "cand": lb[i][:200] if i < len(lb) else None}


def compare_dirs(ref: str, cand: str) -> dict:
    files = lambda d: {os.path.relpath(os.path.join(r, f), d) for r, _, fs in os.walk(d) for f in fs}
    fa, fb = files(ref), files(cand)
    res = {f: compare_file(os.path.join(ref, f), os.path.join(cand, f)) for f in sorted(fa & fb)}
    diff = {f: r for f, r in res.items() if not r["identical"]}
    return {"bitwise": not diff and fa == fb, "n_files": len(fa | fb), "n_identical": len(res) - len(diff),
            "only_ref": sorted(fa - fb), "only_cand": sorted(fb - fa), "differing": diff}


def compare_baseline(ref: str, cand: str) -> dict:
    """Check all original artifacts and independently reconstruct seed-local rank tables for older dumps."""
    result = compare_dirs(ref, cand)
    base = Path(ref)
    tables = {}
    for predictions in base.glob("*/*/seed_*/predictions"):
        seed = predictions.parent
        stage = seed.parent
        table = seed / "scores.csv"
        if table.exists() or stage.name not in ("CryoZeta", "CryoZeta-Interpolate"):
            continue
        rel = str(table.relative_to(base))
        try:
            from cryozeta.runner.combine import _legacy_ranks

            name = stage.parent.name
            ranks = _legacy_ranks(stage, seed, name)
            with (stage / "saved_data" / "scores.csv").open(newline="") as f:
                expected = [
                    (name, r["sample_idx"], r["method"], r["recall_ccmask_ca"], str(ranks[int(r["sample_idx"])]))
                    for r in csv.DictReader(f) if r["pdb_id"] == name
                ]
            with (Path(cand) / rel).open(newline="") as f:
                rows = csv.reader(f)
                header = next(rows, [])
                correct = header == ["pdb_id", "sample_idx", "method", "recall_ccmask_ca", "cif_rank"]
                tables[rel] = correct and list(map(tuple, rows)) == expected
        except (OSError, ValueError, KeyError, RuntimeError, pickle.UnpicklingError) as err:
            tables[rel] = str(err)
    result["seed_score_tables"] = tables
    result["outputs_match"] = (not result["differing"] and not result["only_ref"]
                               and set(result["only_cand"]) == set(tables)
                               and all(v is True for v in tables.values()))
    return result


def _err_tail(dump: str) -> str:
    """Last line of the first ERR/*.txt inference wrote for a skipped sample."""
    for root, _, fs in sorted(os.walk(dump)):
        for f in sorted(fs):
            if os.path.basename(root) == "ERR" and f.endswith(".txt"):
                with open(os.path.join(root, f), errors="replace") as h:
                    lines = h.read().strip().splitlines()
                return lines[-1] if lines else "empty ERR file"
    return "no ERR file"


def run_arm(out: str, name: str, src: str, args: list, env: dict) -> dict:
    d = os.path.join(out, name)
    os.makedirs(d)
    log = os.path.join(d, "run.log")
    t = time.time()
    with open(log, "w") as f:
        r = subprocess.run([sys.executable, "-P", "-c", CHILD, *args, "--dump_dir", "dump"], cwd=d, env=env,
                           stdout=f, stderr=subprocess.STDOUT)
    with open(log, errors="replace") as f:
        text = f.read()
    m = re.search(r"^CRYOZETA_FILE=(.*)$", text, re.M)
    got = m.group(1) if m else None
    npf = (re.search(r"^NUMPY_FILE=(.*)$", text, re.M) or [None, None])[1]
    dump = os.path.join(d, "dump")
    cifs = [f for _, _, fs in os.walk(dump) for f in fs if f.endswith(".cif")]
    why = ("exit %d" % r.returncode if r.returncode else
           "imported %s, not from %s" % (got, src) if not (got or "").startswith(src + os.sep) else
           "numpy %s is not the frozen env's (%s)" % (npf, sys.prefix) if not (npf or "").startswith(sys.prefix + os.sep) else
           "no .cif dumped (inference skips a failed sample with exit 0): " + _err_tail(dump) if not cifs else None)
    rec = {"tree": src, "cryozeta_file": got, "numpy_file": npf, "cpu_shim": "CPU_SHIM=" in text, "exit": r.returncode, "seconds": round(time.time() - t, 1),
           "n_cif": len(cifs), "ok": why is None, "why": why, "log": log}
    print(f"[{name}] {'ok' if why is None else 'FAILED: ' + why} ({rec['seconds']} s)", file=sys.stderr, flush=True)
    return rec


def route_variants(counters: dict) -> dict:
    """{op: variant dict} replacing each op at every site V0 reached with the first ALT provider V0 did not serve
    there, else ALT[op][0] (identity routing when that is all V0 served). expect_bitwise: V0 served exactly the
    replacement at every site, or only cueq trimul that C-CAT replaces (same math, cached weights)."""
    sites = {}  # op -> site -> every backend V0 served there (counter rows also split by reason/allowed)
    for c in counters["counters"]:
        if c["op"] in ALT:
            sites.setdefault(c["op"], {}).setdefault(c["call_site"], set()).add(c["served_backend"])
    out = {}
    for op, by_site in sites.items():
        rep = out[op] = {"variant_id": f"ROUTE-{op}", "policy_id": "V0", "strict": True, "replace": [],
                         "v0_served": {}, "expect_bitwise": True}
        for site, served in by_site.items():
            alt = next((p for p in ALT[op] if p not in served), ALT[op][0])
            rep["v0_served"][site] = sorted(served)
            rep["replace"].append({"op": op, "call_site": site, "provider": alt})
            rep["expect_bitwise"] &= served == {alt} or (op, served, alt) == ("trimul", {"cueq"}, "cueq_cached_cat")
    return out


def check_routing(variant: dict, counters: dict) -> list:
    """Problems with a strict routing arm's counters: every replaced site must serve exactly its provider."""
    want = {(r["op"], r["call_site"]): r["provider"] for r in variant["replace"]}
    seen, bad = set(), []
    for c in counters["counters"]:
        k = (c["op"], c["call_site"])
        if k in want:
            seen.add(k)
            if not (c["in_replacement_scope"] and c["requested_backend"] == want[k] == c["served_backend"]
                    and c["route_allowed"] and c["count"] > 0):
                bad.append(c)
    return bad + [f"not reached: {k}" for k in sorted(want.keys() - seen)]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="new directory (must not exist)")
    p.add_argument("--rev", default=BASELINE_REV, help="baseline git revision (default: %(default)s)")
    p.add_argument("--route", action="store_true", help="add the strict per-op routing arms")
    argv = sys.argv[1:] if argv is None else list(argv)
    cut = argv.index("--") if "--" in argv else len(argv)
    a = p.parse_args(argv[:cut])
    a.infer_args = argv[cut + 1:]
    if not a.infer_args:
        p.error("give the cryozeta-inference args after --")
    if os.path.exists(a.out):
        p.error(f"--out exists: {a.out}")
    return a


def main(argv=None):
    a = parse_args(argv)
    out = os.path.abspath(a.out)
    sha = subprocess.run(["git", "-C", REPO, "rev-parse", "--verify", a.rev + "^{commit}"], capture_output=True,
                         text=True, check=True).stdout.strip()
    tar = subprocess.run(["git", "-C", REPO, "archive", "--format=tar", sha, "src"], capture_output=True,
                         check=True).stdout
    base_dir = os.path.join(out, "baseline_tree")
    os.makedirs(base_dir)
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(base_dir, filter="data")
    env0 = {k: v for k, v in os.environ.items() if k not in STRIP}
    env0["PYTHONNOUSERSITE"] = "1"  # ~/.local packages would otherwise shadow the frozen env (numpy 2.3.3 vs 1.26.4)
    env0.update({k: os.path.join(out, "cache", v) for k, v in C.CACHE_ENVS.items()})

    def env(src, **kw):
        pp = env0.get("PYTHONPATH")
        return dict(env0, PYTHONPATH=src + (os.pathsep + pp if pp else ""), **kw)

    def trial_env(name, variant):
        return env(here, CRYOZETA_KERNEL_VARIANT=variant, CRYOZETA_SYNC_STAGE_TIMER="1",
                   CRYOZETA_KERNEL_COUNTERS=os.path.join(out, name, "counters.json"))

    here, base = os.path.join(REPO, "src"), os.path.join(base_dir, "src")
    rep = {"rev": sha, "repo": REPO, "infer_args": a.infer_args, "python": sys.executable,
           "env": {k: env0.get(k) for k in ("CUDA_VISIBLE_DEVICES", "LAYERNORM_TYPE", "USE_OPM_CHUNKED",
                                            "CRYOZETA_ASSETS_DIR", "OMP_NUM_THREADS")},
           "git_status": subprocess.run(["git", "-C", REPO, "status", "--porcelain"], capture_output=True,
                                        text=True).stdout.splitlines(),
           "arms": {}, "compare": {}, "routing": {}}
    arms = rep["arms"]
    arms["baseline"] = run_arm(out, "baseline", base, a.infer_args, env(base))
    arms["patched"] = run_arm(out, "patched", here, a.infer_args, env(here))
    arms["v0trial"] = run_arm(out, "v0trial", here, a.infer_args, trial_env("v0trial", V0_YAML))
    arms["repeat"] = run_arm(out, "repeat", here, a.infer_args, env(here))
    dump = lambda n: os.path.join(out, n, "dump")
    ok = all(r["ok"] for r in arms.values())
    if ok:
        for n in ("patched", "v0trial", "repeat"):
            rep["compare"][f"{n}_vs_baseline"] = cmp = compare_baseline(dump("baseline"), dump(n))
            ok &= cmp["outputs_match"]
    rep["v0_ok"] = ok
    if a.route and arms["v0trial"]["ok"]:
        with open(os.path.join(out, "v0trial", "counters.json")) as f:
            v0c = json.load(f)
        os.makedirs(os.path.join(out, "variants"))
        routes = route_variants(v0c)
        rep["route_missing"] = sorted(set(ALT) - set(routes))  # an op V0 never reached would pass vacuously
        ok &= not rep["route_missing"]
        for op, var in routes.items():
            name = f"route_{op}"
            r = rep["routing"][name] = {"variant": var, "expect_bitwise": var["expect_bitwise"]}
            path = os.path.join(out, "variants", f"{name}.yaml")
            C.write_json(path, {k: v for k, v in var.items() if k not in ("v0_served", "expect_bitwise")})  # JSON is YAML
            arms[name] = run_arm(out, name, here, a.infer_args, trial_env(name, path))
            if not arms[name]["ok"]:
                ok = False
                continue
            with open(os.path.join(out, name, "counters.json")) as f:
                r["problems"] = check_routing(var, json.load(f))
            r["compare_vs_v0trial"] = cmp = compare_dirs(dump("v0trial"), dump(name))
            ok &= not r["problems"] and (cmp["bitwise"] or not r["expect_bitwise"])
    rep["ok"] = ok
    C.write_json(os.path.join(out, "report.json"), rep)
    summary = {"ok": ok, "v0_ok": rep["v0_ok"], "cpu_shim": arms["baseline"]["cpu_shim"],
               "route_missing": rep.get("route_missing"), **{k: v["bitwise"] for k, v in rep["compare"].items()},
               **{k: {"ran": arms.get(k, {}).get("ok"), "problems": len(v.get("problems", [])),
                      "bitwise": v.get("compare_vs_v0trial", {}).get("bitwise"),
                      "max_abs_diff": max([d.get("max_abs_diff", 0.0) for d in
                                           v.get("compare_vs_v0trial", {}).get("differing", {}).values()],
                                          default=0.0)}
                  for k, v in rep["routing"].items()}}
    print(json.dumps(summary, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
