"""One kernel-bench trial: real CLI(s) (Mode A) or the in-process repeat
driver (Mode B) under ``/usr/bin/time -v``, an NVML sampler and one hard deadline; then outputs,
counters and the outcome class go to ``<out>/trial.json``.

Run inside CryoZeta's Pixi env (it launches that env's console scripts), e.g.
  pixi run -e default --frozen python -m cryozeta.kernel_bench.run_trial --protocol P-CZ \
    --entry entries/9b0l.json --frozen-det DET --assets ASSETS --variant variants/V0.yaml \
    --gpu 0 --timeout 7200 --cache warm --cache-root CACHE --mode A --out TRIALS/9b0l-V0-r1
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from cryozeta.kernel_bench import common as C

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CKPT = {"cz": "cryozeta-v0.0.1.safetensors", "czi": "cryozeta-interpolate-v0.0.1.safetensors",
        "det": "cryozeta-detection-v0.0.1.safetensors"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--protocol", required=True, choices=C.PROTOCOLS)
    p.add_argument("--entry", required=True, help="input JSON (P-LC: the cycle base JSON with absolute EM .pt path)")
    p.add_argument("--frozen-det", help="frozen detection tree used as --em_file_dir (not P-LC)")
    p.add_argument("--assets", required=True, help="dir holding the three checkpoints")
    p.add_argument("--variant", help="variant YAML (default: inert V0)")
    p.add_argument("--gpu", required=True, help="one physical GPU index (PCI bus order)")
    p.add_argument("--timeout", type=float, required=True, help="predeclared hard cap H in seconds")
    p.add_argument("--baseline-t4", type=float, help="T4 of the same-cell baseline (Mode A)")
    p.add_argument("--smaller", help="N_smaller:T4_smaller of the nearest smaller baseline cell")
    p.add_argument("--n-token", type=int, help="N_token of this cell (needed with --smaller)")
    p.add_argument("--cache", required=True, choices=("cold", "warm"))
    p.add_argument("--cache-root", help="pinned compile-cache dir for --cache warm")
    p.add_argument("--mode", required=True, choices=("A", "B"))
    p.add_argument("--warmup-forwards", type=int, default=1)
    p.add_argument("--timed-forwards", type=int, default=1)
    p.add_argument("--out", required=True, help="fresh trial dir (must not exist)")
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--n-sample", type=int, default=5)
    p.add_argument("--n-step", type=int, default=20)
    p.add_argument("--n-cycle", type=int, default=10)
    p.add_argument("--registration", choices=("auto", "svd_only", "teaser_only"), default="auto")
    p.add_argument("--sync-stage-timer", type=int, choices=(0, 1), default=1)
    p.add_argument("--instrument", action="append", default=[], choices=("stage_peaks", "trunk_dump"),
                   help="diagnosis-only instrumentation (not for speed claims)")
    p.add_argument("--synthetic", action="store_true", help="resource-stress input; quality metrics never computed")
    p.add_argument("--ballast-free-mib", type=int,
                   help="emulate a smaller card: after the idle check a helper process holds device memory "
                        "until this many MiB stay free; it is killed when the steps end")
    p.add_argument("--allow-foreign-pid", type=int, action="append", default=[],
                   help="ignore this PID's idle compute app in the idle check; "
                        "its MiB are recorded as foreign_background")
    p.add_argument("--dry-run", action="store_true", help="validate and print the resolved plan; run nothing")
    a = p.parse_args(argv)
    if a.timeout <= 0:
        p.error("--timeout must be positive")
    if a.mode == "B" and a.protocol not in C.MODE_B_PROTOCOLS:
        p.error(f"Mode B supports only {C.MODE_B_PROTOCOLS}")
    if a.mode == "B" and (a.warmup_forwards < 1 or a.timed_forwards < 1):
        p.error("Mode B needs >= 1 warmup and >= 1 timed forward")
    if a.protocol != "P-LC" and not a.frozen_det:
        p.error("--frozen-det is required (kernel A/B uses frozen detection)")
    if a.cache == "warm" and not a.cache_root:
        p.error("--cache warm needs a pinned --cache-root")
    if a.smaller and not a.n_token:
        p.error("--smaller needs --n-token")
    if not re.fullmatch(r"\d+", a.gpu):
        p.error("--gpu must be one physical index")
    if os.path.exists(a.out):
        p.error(f"--out {a.out} exists; every trial needs a fresh directory")
    for path in [a.entry, a.assets] + [x for x in (a.frozen_det, a.variant) if x]:
        if not os.path.exists(path):
            p.error(f"not found: {path}")
    return a


# --- commands ------------------------------------------------------------------------------------


def _bin(name):
    return os.path.join(os.path.dirname(sys.executable), name)


def _flags(v_args):
    base = ["--use_deepspeed_evo_attention", "true", "--use_cuequivariance_attention", "true",
            "--use_cuequivariance_multiplicative_update", "true",
            "--use_cuequivariance_attention_pair_bias", "true", "--use_opm_tilelang", "false"]
    over = dict(zip(v_args[::2], v_args[1::2]))  # variant cli_args override same-named demo flags
    out = []
    for k, v in zip(base[::2], base[1::2]):
        out += [k, over.pop(k, v)]
    for k, v in over.items():
        out += [k, v]
    return out


def build_steps(a, dump, variant_cli):
    """[(step_name, argv)] for the protocol; Mode B swaps the CLI for the repeat driver."""
    ck = {k: os.path.join(a.assets, v) for k, v in CKPT.items()}
    entry = os.path.abspath(a.entry)

    def infer(interp):
        args = ["--seeds", str(a.seed), "--load_checkpoint_path", ck["czi" if interp else "cz"],
                "--em_file_dir", os.path.abspath(a.frozen_det), "--dump_dir", dump,
                "--input_json_path", entry, *_flags(variant_cli), "--model.N_cycle", str(a.n_cycle),
                "--sample_diffusion.N_sample", str(a.n_sample), "--sample_diffusion.N_step", str(a.n_step),
                "--data.num_dl_workers", "1", "--use_interpolation", "true" if interp else "false",
                "--overwrite", "false"]
        if a.mode == "B":
            return [sys.executable, "-m", "cryozeta.kernel_bench.mode_b", "--warmup", str(a.warmup_forwards),
                    "--timed", str(a.timed_forwards), "--iter-root", dump, "--record",
                    os.path.join(a.out, "mode_b.json"), "--", *args]
        return [_bin("cryozeta-inference"), *args]

    if a.protocol in ("P-CZ", "P-LC-stage"):
        return [("inference", infer(False))]
    if a.protocol == "P-CZI":
        return [("inference_interp", infer(True))]
    if a.protocol == "P-COMB":
        return [("inference", infer(False)), ("inference_interp", infer(True)),
                ("combine", [_bin("cryozeta-combine"), "--dump-dir", dump, "--input-json-path", entry,
                             "--seeds", str(a.seed), "--num-select", "5"])]
    reg = [] if a.registration == "auto" else [f"--{a.registration}"]
    fl = _flags(variant_cli)
    native, extra = fl[:10], fl[10:]  # _flags puts the five use_* pairs (cycle_predict's own options) first
    return [("cycle_predict", [_bin("cryozeta-cycle-predict"), "--base_json", entry, "--load_checkpoint_path",
                               ck["cz"], "--dump_root", dump, "--threshold", "5.0", "--seeds", str(a.seed),
                               "--sample.N_sample", str(a.n_sample), "--sample.N_step", str(a.n_step),
                               "--model.N_cycle", str(a.n_cycle), *native, "--num_dl_workers", "1", *reg,
                               *(["--", *extra] if extra else [])]),  # every stage's inference configs
            ("combine_stages", [_bin("cryozeta-combine-stages"), "--dump_root", dump, "--output",
                                os.path.join(dump, "combined.cif")])]


BALLAST = """import sys, time, torch
free = torch.cuda.mem_get_info()[0]
x = torch.empty(max(free - (int(sys.argv[1]) << 20), 0), dtype=torch.uint8, device="cuda")
print(x.numel(), torch.cuda.mem_get_info()[0], flush=True)
time.sleep(10 ** 7)
"""


def build_env(a, variant, caches):
    env = dict(os.environ)
    prefix = sys.prefix
    # PYTHONNOUSERSITE: ~/.local packages (numpy 2.3.3 vs the lock's 1.26.4) otherwise shadow the frozen env
    env.update(PYTHONNOUSERSITE="1", LAYERNORM_TYPE="fast_layernorm", CUDA_DEVICE_ORDER="PCI_BUS_ID", USE_OPM_CHUNKED="1",
               CUDA_HOME=prefix, CUDA_VISIBLE_DEVICES=a.gpu, CRYOZETA_SYNC_STAGE_TIMER=str(a.sync_stage_timer))
    if os.access(os.path.join(prefix, "bin", "ptxas"), os.X_OK):
        env["TRITON_PTXAS_PATH"] = env["TRITON_PTXAS_BLACKWELL_PATH"] = os.path.join(prefix, "bin", "ptxas")
    env.update(caches)
    for k, v in variant.get("env", {}).items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    if a.variant:
        env["CRYOZETA_KERNEL_VARIANT"] = os.path.abspath(a.variant)
    else:
        env.pop("CRYOZETA_KERNEL_VARIANT", None)
    if "stage_peaks" in a.instrument:
        env["CRYOZETA_STAGE_PEAKS"] = "1"
    if "trunk_dump" in a.instrument:
        env["CRYOZETA_TRUNK_DUMP"] = os.path.join(a.out, "trunk.pt")
    return env


_PREFLIGHT = """
import sys
kind, argv = sys.argv[1], sys.argv[2:]
if kind == "cycle_predict":
    from pathlib import Path
    from cryozeta.runner.cycle_predict import parse_args, stage_configs
    a = parse_args(argv)
    c = stage_configs(temp_json=Path(a.base_json), dump_dir=Path(a.dump_root), args=a)
else:
    from cryozeta.runner.inference import build_configs
    sys.argv = ["cryozeta-inference", *argv]
    c = build_configs()
from cryozeta.runner.cycle_predict import attention_conflict
why = attention_conflict(c)
if why:
    sys.exit("error: " + why)
"""


def preflight(steps, env):
    """Every model step's argv through the real parsers (P-LC: the stage config of every stage) in a CPU-only
    child with throwaway caches, before any GPU work. Returns an error string or None."""
    with tempfile.TemporaryDirectory() as tmp:
        cenv = dict(env, CUDA_VISIBLE_DEVICES="", **{k: os.path.join(tmp, v) for k, v in C.CACHE_ENVS.items()})
        cenv.pop("LAYERNORM_TYPE", None)  # fast_layernorm JIT-builds its CUDA extension at import (primitives.py)
        for name, argv in steps:
            if argv[0] == _bin("cryozeta-cycle-predict"):
                kind, args = "cycle_predict", argv[1:]
            elif argv[0] == _bin("cryozeta-inference"):
                kind, args = "inference", argv[1:]
            elif "cryozeta.kernel_bench.mode_b" in argv:
                kind, args = "inference", argv[argv.index("--") + 1:]
            else:
                continue  # combine steps take no variant flags
            r = subprocess.run([sys.executable, "-P", "-c", _PREFLIGHT, kind, *args], env=cenv, capture_output=True,
                               text=True, timeout=1800)
            if r.returncode:  # the parser's error line, else the traceback tail
                lines = r.stderr.strip().splitlines()
                err = [s for s in lines if "error: " in s]
                return f"preflight {name}: " + (err[-1] if err else "\n".join(lines[-5:]))
    return None


# set by the harness per trial; a variant may not override them (seed, inputs, checkpoint, sampler sizes, device)
RESERVED_FLAGS = {"--seeds", "--load_checkpoint_path", "--em_file_dir", "--dump_dir", "--input_json_path",
                  "--overwrite", "--use_interpolation", "--model.N_cycle", "--sample_diffusion.N_sample",
                  "--sample_diffusion.N_step", "--data.num_dl_workers", "--base_json", "--dump_root", "--threshold",
                  "--sample.N_sample", "--sample.N_step", "--num_dl_workers",
                  "--num_workers",  # the DataLoader's workers (data.num_dl_workers is never read): changes the P draw
                  "--model.input_embedder.p_dim"}  # set from use_interpolation after parsing: a variant value is ignored
RESERVED_ENV = {"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CRYOZETA_KERNEL_COUNTERS", "CRYOZETA_KERNEL_VARIANT",
                "CRYOZETA_SYNC_STAGE_TIMER", "CRYOZETA_STAGE_PEAKS", "CRYOZETA_TRUNK_DUMP", *C.CACHE_ENVS}
CHUNK_FLAG = "--infer_setting.sample_diffusion_chunk_size"


def read_variant(path):
    """Variant fields run_trial needs, without importing torch (kernel_routing validates in the child)."""
    if not path:
        return {"variant_id": "V0", "replace": [], "cli_args": [], "env": {}, "strict": False,
                "status": "active", "sample_paired": True}
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if "variant_id" not in raw:
        raise SystemExit(f"{path}: variant_id is required")
    raw.setdefault("status", "active")
    raw.setdefault("sample_paired", raw["status"] == "active")
    raw.setdefault("replace", [])
    if not isinstance(raw.get("cli_args") or [], list):
        raise SystemExit(f"{path}: cli_args must be a list of flag/value strings")
    toks = " ".join(str(x) for x in raw.get("cli_args") or []).split()  # the CLI re-splits on spaces
    if len(toks) % 2 or not all(f.startswith("--") and "=" not in f for f in toks[::2]) or any(
            v.startswith("--") for v in toks[1::2]):
        raise SystemExit(f"{path}: cli_args must be '--flag value' pairs (no '=' form, no spaces inside a value)")
    raw["cli_args"] = toks
    raw["env"] = {str(k): None if v is None else str(v) for k, v in (raw.get("env") or {}).items()}
    flags = dict(zip(toks[::2], toks[1::2]))
    abbrev = {f for f in flags for g in RESERVED_FLAGS | {CHUNK_FLAG} if g != f and g.startswith(f)}  # argparse prefixes
    bad = sorted((RESERVED_FLAGS & set(flags)) | abbrev | (RESERVED_ENV & set(raw["env"])))
    if bad:
        raise SystemExit(f"{path}: harness-controlled settings (or abbreviations of them) cannot be set by a "
                         f"variant: {bad}")
    if flags.get(CHUNK_FLAG, "1") != "1" and raw["sample_paired"] is not False:
        raise SystemExit(f"{path}: sample_diffusion_chunk_size != 1 regroups the RNG draws; it cannot be "
                         "sample_paired")
    return raw


# --- machine state -------------------------------------------------------------------------------


def _smi(*args):
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=60)


def gpu_state(gpu):
    q = _smi("-i", gpu, "--query-gpu=name,driver_version,memory.total,memory.used,compute_cap",
             "--format=csv,noheader,nounits")
    apps = _smi("-i", gpu, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits")
    if q.returncode:
        return {"error": q.stderr.strip()}
    name, drv, tot, used, cc = [x.strip() for x in q.stdout.strip().split(",")]
    return {"name": name, "driver": drv, "memory_total_mib": int(tot), "memory_used_mib": int(used),
            "compute_cap": cc, "compute_apps": [x for x in apps.stdout.splitlines() if x.strip()]}


def _cgroup_dir():
    try:
        for line in open("/proc/self/cgroup"):
            if line.startswith("0::"):
                return "/sys/fs/cgroup" + line.strip()[3:]
    except OSError:
        pass
    return None


def cgroup_state():
    d = _cgroup_dir()
    out = {"cgroup": d}
    if d:
        for f in ("memory.peak", "memory.max"):
            try:
                out[f] = open(os.path.join(d, f)).read().strip()
            except OSError:
                out[f] = None
        try:
            ev = dict(line.split() for line in open(os.path.join(d, "memory.events")))
            out["oom_kill"] = int(ev.get("oom_kill", 0))
        except OSError:
            out["oom_kill"] = None
    return out


def parse_time_v(path):
    out = {}
    try:
        text = open(path).read()
    except OSError:
        return out
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    if m:
        out["max_rss_kib"] = int(m[1])
    m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)", text)
    if m:
        secs = 0.0
        for part in m[1].split(":"):
            secs = secs * 60 + float(part)
        out["elapsed_s"] = secs
    m = re.search(r"Command terminated by signal (\d+)", text)
    if m:
        out["signal"] = int(m[1])
    return out


def nvml_peak(path):
    peak = None
    try:
        for line in open(path):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2 and parts[1].isdigit():
                peak = max(peak or 0, int(parts[1]))
    except OSError:
        pass
    return peak


def provenance(a, entry_json):
    def git(*args):
        r = subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True)
        return r.stdout if r.returncode == 0 else None

    diff = git("diff", "HEAD") or ""
    msa = {}
    for e in entry_json if isinstance(entry_json, list) else [entry_json]:
        for s in e.get("sequences", []):
            for v in s.values():
                d = (v.get("msa") or {}).get("precomputed_msa_dir") if isinstance(v, dict) else None
                if d and d not in msa:
                    msa[d] = C.sha256_tree(d) if os.path.isdir(d) else "missing"
    vers = {}
    for pkg in ("torch", "triton", "cuequivariance", "cuequivariance-torch", "cuequivariance-ops-torch-cu12",
                "deepspeed", "tilelang"):
        try:
            vers[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            vers[pkg] = None
    ckpts = {k: os.path.join(a.assets, v) for k, v in CKPT.items()}
    new = (git("ls-files", "--others", "--exclude-standard", "src") or "").split()  # not in the diff
    used = {"P-CZ": ["cz"], "P-LC-stage": ["cz"], "P-LC": ["cz"], "P-CZI": ["czi"], "P-COMB": ["cz", "czi"]}
    return {
        "code_revision": (git("rev-parse", "HEAD") or "").strip(),
        "baseline_revision": "c544c6bbd0366951d8b4e0bc4c0da547836f7b8f",
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "untracked_sha256": {f: C.sha256_file(os.path.join(REPO, f)) for f in new},
        "pixi_lock_sha256": C.sha256_file(os.path.join(REPO, "pixi.lock")),
        "checkpoint_sha256": {k: C.sha256_file(ckpts[k]) for k in used[a.protocol]},
        "input_json_sha256": C.sha256_file(a.entry),
        "msa_sha256": msa,
        "variant_sha256": C.sha256_file(a.variant) if a.variant else None,
        "pixi_env": os.path.basename(sys.prefix),
        "python": sys.version.split()[0],
        "packages": vers,
        "cutlass_path": os.environ.get("CUTLASS_PATH"),
        "host": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def det_manifest(frozen_det):
    """Manifest written by ``provenance freeze-det`` (sha + N_point per entry), if present."""
    path = os.path.abspath(frozen_det).rstrip("/") + ".manifest.json"
    return json.load(open(path)) if os.path.isfile(path) else None


# --- running -------------------------------------------------------------------------------------


def run_step(argv, env, log_path, time_path, deadline):
    """Run one process group until exit or the shared deadline. Returns (exit_code, timed_out, wall_s)."""
    timev = shutil.which("time", path="/usr/bin") or None
    cmd = ([timev, "-v", "-o", time_path] if timev else []) + argv
    t0 = time.perf_counter()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=max(1.0, deadline - time.monotonic()))
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            code = proc.returncode
    return code, timed_out, time.perf_counter() - t0


_LOGURU_TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3}).*Finish loading checkpoint", re.M)


def model_ready_s(log_text, start_epoch):
    """T0 (process start -> model ready) from the loguru timestamp of 'Finish loading checkpoint.'."""
    m = _LOGURU_TS.search(log_text)
    if not m:
        return None
    return dt.datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S.%f").timestamp() - start_epoch


def main(argv=None):
    a = parse_args(argv)
    variant = read_variant(a.variant)
    if variant["status"] != "active":  # deferred variants are not benchmark arms
        print(f"refusing: {variant['variant_id']} is {variant['status']}: {variant.get('deferred_reason', '')}",
              file=sys.stderr)
        return 2
    smaller = tuple(float(x) for x in a.smaller.split(":")) if a.smaller else None
    eff, derivation = C.effective_timeout(a.timeout, a.mode, a.n_token, a.baseline_t4, smaller)
    out = os.path.abspath(a.out)
    dump = os.path.join(out, "dump")
    cache_root = os.path.join(out, "cache") if a.cache == "cold" else os.path.abspath(a.cache_root)
    caches = {k: os.path.join(cache_root, sub) for k, sub in C.CACHE_ENVS.items()}
    a.out = out
    steps = build_steps(a, dump, variant["cli_args"])
    env = build_env(a, variant, caches)
    with open(a.entry) as f:
        entry_json = json.load(f)
    entries = entry_json if isinstance(entry_json, list) else [entry_json]
    names = [e["name"] for e in entries]
    synthetic = (a.synthetic or any(re.search(r"_syn\d+_(real|single)$", n) for n in names)
                 or os.path.isfile(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.entry))),
                                                "synthetic.json")))  # synth_inputs layout: OUT/entries/<name>.json
    manifest = det_manifest(a.frozen_det) if a.frozen_det else None
    expected, why = C.expected_cells(manifest, names)
    det_before = C.sha256_tree(a.frozen_det) if a.frozen_det else None
    if a.protocol != "P-LC":  # P-LC detects per stage; no frozen expectation exists
        if not expected:
            why = f"cannot verify the requested N_token/P ({why}); run provenance freeze-det"
        elif manifest.get("tree_sha256") != det_before:
            why = "--frozen-det differs from its freeze-det manifest (tree_sha256); re-freeze or use the frozen tree"
        elif any("em_file" in e for e in entries):
            why = "an entry sets em_file, which overrides --em_file_dir and bypasses the frozen tree"
        else:
            why = ""
        if why:
            print(f"refusing: {why}", file=sys.stderr)
            return 2
    why = preflight(steps, env)
    if why:
        print(f"refusing: {why}", file=sys.stderr)
        return 2
    if a.dry_run:
        print(json.dumps({"effective_timeout": eff, "derivation": derivation, "steps": steps,
                          "expected_cells": expected, "synthetic": synthetic,
                          "env": {k: env.get(k) for k in sorted(set(caches) | set(variant["env"]) | {
                              "CRYOZETA_KERNEL_VARIANT", "CUDA_VISIBLE_DEVICES", "LAYERNORM_TYPE",
                              "USE_OPM_CHUNKED", "CRYOZETA_SYNC_STAGE_TIMER"})}}, indent=1))
        return 0

    state = gpu_state(a.gpu)
    apps = [[int(v) if v.strip().isdigit() else 0 for v in x.split(",")] for x in state.get("compute_apps", [])]
    bg = sum(m for pid, m in apps if pid in a.allow_foreign_pid)
    if "error" in state or any(pid not in a.allow_foreign_pid for pid, _ in apps) or state["memory_used_mib"] >= 1024 + bg:
        print(f"GPU {a.gpu} is not idle or not visible: {state}", file=sys.stderr)
        return 3
    os.makedirs(out)
    os.makedirs(dump)
    for d in caches.values():
        os.makedirs(d, exist_ok=True)
    rec = {
        "trial_id": os.path.basename(out), "protocol": a.protocol, "mode": a.mode, "cache": a.cache,
        "variant_id": variant["variant_id"], "baseline_id": variant.get("baseline_id"),
        "policy_id": variant.get("policy_id", "V0"), "replacement_scope": variant["replace"],
        "clear_cache": variant.get("clear_cache", "legacy"), "strict": variant.get("strict", False),
        "variant_status": variant["status"], "sample_paired": variant["sample_paired"],
        "entries": names, "synthetic": synthetic, "frozen_det": a.frozen_det,
        "frozen_det_sha256_before": det_before, "det_manifest": manifest,
        "settings": {"seed": a.seed, "n_sample": a.n_sample, "n_step": a.n_step, "n_cycle": a.n_cycle,
                     "warmup_forwards": a.warmup_forwards if a.mode == "B" else 0,
                     "timed_forwards": a.timed_forwards if a.mode == "B" else 1,
                     "registration": a.registration, "sync_stage_timer": a.sync_stage_timer,
                     "instrument": a.instrument},
        "timeout": {"hard_cap": a.timeout, "effective": eff, "derivation": derivation},
        "gpu_before": state, "cgroup_before": cgroup_state(), "provenance": provenance(a, entry_json),
        "cache_entries_before": C.cache_entries(caches), "steps": [],
    }
    if a.allow_foreign_pid:
        rec["foreign_background"] = {"pids": a.allow_foreign_pid, "mib": bg}
    ballast = None
    if a.ballast_free_mib:
        # ponytail: one up-front allocation; free memory is matched once (±2 MiB allocator rounding), not tracked
        ballast = subprocess.Popen([sys.executable, "-c", BALLAST, str(a.ballast_free_mib)], env=env,
                                   stdout=subprocess.PIPE, text=True)
        held, free_after = map(int, ballast.stdout.readline().split() or (-1, -1))
        state_b = gpu_state(a.gpu)
        rec["ballast"] = {"free_mib_target": a.ballast_free_mib, "pid": ballast.pid, "held_bytes": held,
                          "free_bytes_after": free_after, "gpu_with_ballast": state_b,
                          "used_mib": state_b.get("memory_used_mib", 0) - state["memory_used_mib"]}
        if held < 0 or ballast.poll() is not None:
            ballast.kill()
            print(f"ballast failed: {rec['ballast']}", file=sys.stderr)
            return 3
    C.write_json(os.path.join(out, "trial.json"), rec)  # provisional record before launch

    sampler = subprocess.Popen(
        ["nvidia-smi", "-i", a.gpu, "--query-gpu=timestamp,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits", "-lms", "100"],
        stdout=open(os.path.join(out, "nvml.csv"), "w"), stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + eff
    t5_0 = time.perf_counter()
    timed_out, last_code, counters, inputs, log_text = False, 0, [], [], ""
    try:
        for name, argv_ in steps:
            senv = dict(env, CRYOZETA_KERNEL_COUNTERS=os.path.join(out, f"counters_{name}.json"))
            log_path = os.path.join(out, f"{name}.log")
            start = time.time()
            code, to, wall = run_step(argv_, senv, log_path, os.path.join(out, f"time_{name}.txt"), deadline)
            text = open(log_path, errors="replace").read()
            log_text += text
            cpath = senv["CRYOZETA_KERNEL_COUNTERS"]
            cnt = json.load(open(cpath)) if os.path.isfile(cpath) else None
            if cnt and a.mode == "A":  # Mode B: per-iteration counters come from mode_b.json
                counters += cnt["counters"]
                inputs += cnt.get("inputs", [])
            rec["steps"].append({"step": name, "argv": argv_, "exit_code": code, "timed_out": to,
                                 "T4_s" if a.mode == "A" else "process_wall_s": wall,  # Mode B wall is not T4
                                 "T0_model_ready_s": model_ready_s(text, start),
                                 "time_v": parse_time_v(os.path.join(out, f"time_{name}.txt")),
                                 "counters": cnt})
            last_code, timed_out = code, to
            if code != 0 or to:
                break
    finally:
        sampler.terminate()
        sampler.wait()
        if ballast:
            ballast.kill()
            ballast.wait()
    total = time.perf_counter() - t5_0
    if a.mode == "A":
        rec["T5_s"] = total
        rec["T5_scope"] = ("all steps, per-stage detection included" if a.protocol == "P-LC" else
                           "the steps above only; detection is frozen and excluded")
    else:
        rec["process_wall_total_s"] = total  # warmup + timed forwards in one process: not a user-facing T5

    mode_b = os.path.join(out, "mode_b.json")
    if os.path.isfile(mode_b):
        rec["mode_b"] = json.load(open(mode_b))
        counters += [c for it in rec["mode_b"]["iterations"] for c in (it.get("counters") or [])]
        inputs += [r for it in rec["mode_b"]["iterations"] for r in (it.get("inputs") or [])]
    cg = cgroup_state()
    rec["cgroup_after"] = cg
    oom_before, oom_after = rec["cgroup_before"].get("oom_kill"), cg.get("oom_kill")
    killed9 = any(s["exit_code"] in (-9, 137) or s["time_v"].get("signal") == 9 for s in rec["steps"])
    oom_killed = bool(killed9 and oom_before is not None and oom_after is not None and oom_after > oom_before)

    dumps = [os.path.join(dump, it["dir"]) for it in rec.get("mode_b", {}).get("iterations", [])] or [dump]
    missing, produced = [], []
    for d in dumps:
        for n in names:
            exp = C.expected_outputs(d, a.protocol, n, a.seed, a.n_sample)
            ok = [p for p in exp if os.path.isfile(p) or (os.path.isdir(p) and os.listdir(p))]  # no empty dirs
            missing += [p for p in exp if p not in ok]
            produced += ok
    errs = [open(p, errors="replace").read() for p in C.err_files(dump)]
    det_after = C.sha256_tree(a.frozen_det) if a.frozen_det else None
    model_runs = (rec["mode_b"]["iterations"] if "mode_b" in rec else
                  [s["counters"] or {} for s in rec["steps"] if s["step"].startswith("inference")
                   or s["step"] == "cycle_predict"])
    checks = [C.check_scope(variant["replace"], r.get("counters") or [], r.get("inputs") or [])
              for r in model_runs or [{}]]
    scope_detail = next((f"run {i}: {why}" for i, (_, why) in enumerate(checks) if why), "")
    coverage = min((frac for frac, _ in checks if frac != "N/A"), default="N/A")
    cls, detail = C.classify(exit_code=last_code, timed_out=timed_out, oom_killed=oom_killed, err_texts=errs,
                             log_text=log_text, counters=counters, missing=missing,
                             nonfinite=C.scan_nonfinite(produced), protocol=a.protocol, scope_detail=scope_detail)
    if det_after != det_before:
        cls, detail = "CRASH", "frozen detection tree modified during the trial"
    cell = {"expected": expected, "inputs": inputs, "status": "n/a (P-LC)", "detail": ""}
    if a.protocol != "P-LC":
        runs = [r.get("inputs") or [] for r in model_runs]
        cell["status"], cell["detail"] = C.check_runs(expected, runs)
        cell["outcome_before_check"] = [cls, detail]
        cls, detail = C.cell_outcome(cls, detail, cell["status"], cell["detail"])
    parsed = C.parse_log(log_text)
    rec.update({
        "outcome": cls, "outcome_detail": detail, "exit_code": last_code,
        "gates": "not evaluated by run_trial (outcome is the runtime class): compare_outputs + sealed tolerances",
        "err_sha256": hashlib.sha256("\n".join(errs).encode()).hexdigest() if errs else None,
        "err_files": C.err_files(dump), "missing_outputs": missing[:50], "n_missing": len(missing),
        "frozen_det_sha256_after": det_after, "log": parsed, "cell": cell,
        "scope_coverage": coverage,
        "baseline_rescues": C.rescue_counts(counters),
        "M3_nvml_peak_mib": nvml_peak(os.path.join(out, "nvml.csv")),
        "M4_max_rss_kib": max([s["time_v"].get("max_rss_kib", 0) for s in rec["steps"]] or [0]) or None,
        "M4_cgroup_peak_bytes_highwater": cg.get("memory.peak"),
        "cache_entries_after": C.cache_entries(caches),
        "gpu_after": gpu_state(a.gpu),
    })
    if a.mode == "A":
        t3s = [s["counters"]["t3"] for s in rec["steps"] if s["counters"] and s["counters"]["t3"]]
        rec["T3_first_s"] = [t[0]["seconds"] for t in t3s if t[0]["ok"]]  # first forward of each process
        rec["T3_subsequent_s"] = [x["seconds"] for t in t3s for x in t[1:] if x["ok"]]  # later entries/stages
        rec["M1_invocation"] = [s["counters"].get("m1_invocation") for s in rec["steps"] if s["counters"]]
        rec["M2_invocation"] = [s["counters"].get("m2_invocation") for s in rec["steps"] if s["counters"]]
    else:
        warm = [it for it in rec["mode_b"]["iterations"] if it["warmed"]] if "mode_b" in rec else []
        rec["T3_warm_s"] = [it["t3"] for it in warm]
        rec["throughput_per_s"] = [1 / t for t in rec["T3_warm_s"] if t]
        rec["warmed_status"] = bool(warm)
    C.write_json(os.path.join(out, "trial.json"), rec)
    print(f"{rec['trial_id']}: {cls} {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
