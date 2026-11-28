"""Shared trial logic: hashing, the timeout rule, expected outputs, log parsing and outcome classes.

Stdlib only, so it is importable (and testable) without torch.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re

PROTOCOLS = ("P-CZ", "P-CZI", "P-COMB", "P-LC", "P-LC-stage")
MODE_B_PROTOCOLS = ("P-CZ", "P-CZI", "P-LC-stage")
# SKIPPED is not an outcome class: an entry the pipeline skipped (interpolation pre-check, existing outputs)
# produced no measurement, so it can never count as SUCCESS.
# INVALID_CELL: the model saw a different N_token/P than the cell requested, or it could
# not be verified, so the run is not a measurement of that cell.
OUTCOMES = ("SUCCESS", "GPU_OOM", "HOST_OOM", "UNSUPPORTED", "FALLBACK", "TIMEOUT", "NUMERICAL", "CRASH", "SKIPPED",
            "INVALID_CELL")
# compile/tuning caches -> subdir of a trial's cache root; unset, cueq and CUDA fall back to ~/.cache and ~/.nv
CACHE_ENVS = {"TRITON_CACHE_DIR": "triton", "TORCH_EXTENSIONS_DIR": "torch_extensions",
              "TILELANG_CACHE_DIR": "tilelang", "CUEQ_TRITON_CACHE_DIR": "cueq_triton", "CUDA_CACHE_PATH": "cuda"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha256_tree(root: str) -> dict:
    """Sorted per-file sha256 list of a directory tree, hashed once more."""
    lines = []
    for dirpath, dirnames, files in os.walk(root):
        dirnames.sort()
        for f in sorted(files):
            p = os.path.join(dirpath, f)
            lines.append(f"{os.path.relpath(p, root)} {sha256_file(p)}")
    return {"sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest(), "files": len(lines)}


def write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str, sort_keys=True)
    os.replace(tmp, path)


def cache_entries(env: dict) -> dict:
    """File count per compile cache; the change across a run is the compiled-variant count (T0)."""
    out = {}
    for k in CACHE_ENVS:
        d = env.get(k)
        out[k] = sum(len(f) for _, _, f in os.walk(d)) if d and os.path.isdir(d) else 0
    return out


# --- timeout rule --------------------------------------------------------------------------------


def effective_timeout(hard_cap: float, mode: str, n_token: int | None = None,
                      same_cell_t4: float | None = None, smaller: tuple | None = None) -> tuple[float, dict]:
    """Effective timeout fixed before launch. ``smaller`` is (N_smaller, T4_smaller) of the nearest
    smaller successful baseline cell. Returns (seconds, derivation record)."""
    if not hard_cap or hard_cap <= 0:
        raise ValueError("a positive hard cap H (--timeout) is required")
    if mode == "B":
        return float(hard_cap), {"rule": "H", "why": "Mode B uses H directly", "H": hard_cap}
    if same_cell_t4:
        t = min(hard_cap, max(3 * same_cell_t4, 1800.0))
        return t, {"rule": "min(H, max(3*T4, 1800))", "H": hard_cap, "T4": same_cell_t4}
    if smaller:
        if not n_token:
            raise ValueError("cubic extrapolation needs the cell's N_token")
        n_s, t4_s = smaller
        t_est = t4_s * (n_token / n_s) ** 3
        t = min(hard_cap, max(3 * t_est, 1800.0))
        return t, {"rule": "min(H, max(3*T4_smaller*(N/N_smaller)^3, 1800))", "H": hard_cap,
                   "N": n_token, "N_smaller": n_s, "T4_smaller": t4_s, "T_est": t_est}
    return float(hard_cap), {"rule": "H", "why": "no same-cell or smaller baseline", "H": hard_cap}


# --- expected outputs, logs ------------------------------------------------------------------------


def stage_dirs(protocol: str) -> list[str]:
    return {"P-CZ": ["CryoZeta"], "P-CZI": ["CryoZeta-Interpolate"], "P-LC-stage": ["CryoZeta"],
            "P-COMB": ["CryoZeta", "CryoZeta-Interpolate"]}.get(protocol, [])


def expected_outputs(dump: str, protocol: str, name: str, seed: int, n_sample: int) -> list[str]:
    """Files a SUCCESS must leave. P-LC reads its stage list from ``stage_names.txt``."""
    def preds(entry_dir, ename):
        d = os.path.join(entry_dir, f"seed_{seed}", "predictions")
        return [os.path.join(d, f"{ename}_{kind}_{k}.{ext}") for k in range(n_sample)
                for kind, ext in (("sample", "cif"), ("summary_confidence_sample", "json"))]

    if protocol == "P-LC":
        names_file = os.path.join(dump, "stage_names.txt")
        out = [names_file, os.path.join(dump, "combined.cif")]
        if os.path.isfile(names_file):
            for st in open(names_file).read().split():
                out += preds(os.path.join(dump, st, "CryoZeta"), st)
        return out
    out = []
    for st in stage_dirs(protocol):
        out += preds(os.path.join(dump, name, st), name)
    if protocol == "P-COMB":
        out.append(os.path.join(dump, name, "CryoZeta-Final"))
    return out


def err_files(dump: str) -> list[str]:
    return sorted(glob.glob(os.path.join(dump, "**", "ERR", "*.txt"), recursive=True))


_SIZE_RE = re.compile(r"\] (\S+): N_asym (\d+), N_token (\d+), N_atom (\d+), N_msa (\d+)")
_STAGE_RE = re.compile(r"(EMPairformer done in|Total model forward:) ([\d.]+)s")


def parse_log(text: str) -> dict:
    """Entry sizes and the pipeline's own stage-time lines (T2 comes from the counters' stage clocks)."""
    entries = {m[0]: {"N_asym": int(m[1]), "N_token": int(m[2]), "N_atom": int(m[3]), "N_msa": int(m[4])}
               for m in _SIZE_RE.findall(text)}
    return {
        "entries": entries,
        "log_stage_seconds": [(k, float(v)) for k, v in _STAGE_RE.findall(text)],
        "interp_skipped": sorted(set(re.findall(r"\[pre-check\] (\S+): no interpolation features", text))),
        "output_skipped": sorted(set(re.findall(r"\] (\S+) seed_\d+ already exists, skipping", text))),
    }


def scan_nonfinite(paths: list[str]) -> list[str]:
    """Prediction files containing NaN/Inf coordinates or confidence values."""
    bad = []
    tok = re.compile(r"(?<![A-Za-z])(nan|inf)(?![A-Za-z])", re.I)
    for p in paths:
        if p.endswith(".json") and os.path.isfile(p):
            def walk(v):
                if isinstance(v, float):
                    return not math.isfinite(v)
                if isinstance(v, dict):
                    return any(walk(x) for x in v.values())
                if isinstance(v, list):
                    return any(walk(x) for x in v)
                return False
            try:
                with open(p) as f:
                    v = json.load(f)
            except ValueError:  # truncated/corrupt output must not crash the harness or pass
                bad.append(f"{p} (unparseable)")
                continue
            if walk(v):
                bad.append(p)
        elif p.endswith(".cif") and os.path.isfile(p):
            for line in open(p):
                if line.startswith(("ATOM", "HETATM")) and tok.search(line):
                    bad.append(p)
                    break
    return bad


# --- outcome classification ----------------------------------------------------------------------


_OOM = re.compile(r"out of memory|OutOfMemoryError", re.I)


def classify(*, exit_code: int | None, timed_out: bool, oom_killed: bool, err_texts: list[str],
             log_text: str, counters: list[dict], missing: list[str], nonfinite: list[str],
             gates_failed: list[str] | None = None, protocol: str = "P-CZ",
             scope: list | None = None, scope_detail: str | None = None) -> tuple[str, str]:
    """(outcome class, detail). Order: budget, host kill, strict-routing errors, OOM, crash,
    skips, routing policy, outputs, numerics. An ERR file fails the trial even when the CLI exits 0
    (``inference.main`` catches per-entry exceptions)."""
    errs = "\n".join(err_texts)
    text = errs + "\n" + log_text[-20000:]
    if timed_out:
        return "TIMEOUT", "effective timeout exceeded"
    if oom_killed:
        return "HOST_OOM", "SIGKILL with a cgroup/kernel oom-kill event"
    if "KernelUnsupported" in text:
        return "UNSUPPORTED", _first_line(text, "KernelUnsupported")
    if "KernelFallback" in text:
        return "FALLBACK", _first_line(text, "KernelFallback")
    scoped = [c for c in counters if c.get("in_replacement_scope")]
    if any("import_error" in c["reason"] or "ineligible" in c["reason"] for c in scoped):
        bad = next(c for c in scoped if "import_error" in c["reason"] or "ineligible" in c["reason"])
        return "UNSUPPORTED", f"{bad['op']}@{bad['call_site']}: {bad['reason']}"
    if _OOM.search(errs) or (exit_code not in (0, None) and _OOM.search(text)):
        return "GPU_OOM", _first_line(text, "out of memory") or "CUDA OOM"
    if any("oom" in c["reason"] for c in scoped):
        return "GPU_OOM", "OOM inside the replacement scope"
    if exit_code not in (0, None):
        return "CRASH", f"exit code {exit_code}"
    if errs.strip():
        return "CRASH", "entry failure: ERR file written ({})".format(errs.strip().splitlines()[0][:200])
    parsed = parse_log(log_text)
    if parsed["interp_skipped"] or parsed["output_skipped"]:
        return "SKIPPED", "pipeline skipped entries: " + ",".join(parsed["interp_skipped"] + parsed["output_skipped"])
    for c in counters:
        if not c.get("route_allowed", True) or (
            c.get("in_replacement_scope") and c["served_backend"] != c["requested_backend"]
        ):
            where = "requested replacement" if c.get("in_replacement_scope") else "frozen V0 policy"
            return "FALLBACK", f"{c['op']}@{c['call_site']} served {c['served_backend']} outside the {where}"
    if missing:
        return "CRASH", f"{len(missing)} expected outputs missing, first {missing[0]}"
    if nonfinite:
        return "NUMERICAL", f"NaN/Inf or unparseable: {nonfinite[0]}"
    if scope_detail is None:
        _, scope_detail = check_scope(scope or [], counters)
    if scope_detail:
        return "UNSUPPORTED", scope_detail
    if gates_failed:
        return "NUMERICAL", "gates failed: " + ",".join(gates_failed)
    return "SUCCESS", ""


def expected_cells(manifest: dict | None, names: list[str]) -> tuple[dict, str]:
    """{name: (N_token, N_point)} from a ``freeze-det`` manifest, or ({}, why it cannot be verified)."""
    if manifest is None:
        return {}, "no freeze-det manifest next to --frozen-det"
    out = {}
    for n in names:
        e = manifest.get("entries", {}).get(n)
        if not e or e.get("N_token") is None or e.get("N_point") is None:
            return {}, f"{n}: manifest has no N_token/N_point"
        out[n] = (e["N_token"], e["N_point"])
    return out, ""


def check_cell(expected: dict, inputs: list[dict]) -> tuple[str, str]:
    """('match'|'mismatch'|'unverified', detail): every recorded model input against its expected
    (N_token, P); valid points and both point axes of ``p`` must equal P too."""
    seen = {n: [] for n in expected}
    for r in inputs:
        if r.get("name") not in seen:
            return "mismatch", f"unexpected model input {r.get('name')!r}"
        seen[r["name"]].append(r)
    for n, (n_tok, n_pt) in expected.items():
        if not seen[n]:
            return "unverified", f"{n}: no model input recorded"
        for r in seen[n]:
            pp = r.get("shapes", {}).get("p", [])[-3:-1]
            got = (r["n_token"], r["n_point"], r["valid_points"], pp)
            if got != (n_tok, n_pt, n_pt, [n_pt, n_pt]):
                return "mismatch", (f"{n}: requested N_token={n_tok} P={n_pt}; model got N_token={got[0]} "
                                    f"P={got[1]} valid={got[2]} p_axes={got[3]}")
    return "match", ""


def check_runs(expected: dict, runs: list[list[dict]]) -> tuple[str, str]:
    """check_cell per run (a Mode A inference process or a Mode B iteration): every run must record and
    match every entry, so one verified run cannot vouch for another."""
    res = [check_cell(expected, r) for r in runs] or [("unverified", "no run recorded")]
    for want in ("mismatch", "unverified"):
        for i, (st, d) in enumerate(res):
            if st == want:
                return st, f"run {i}: {d}"
    return "match", ""


def cell_outcome(cls: str, detail: str, status: str, cell_detail: str) -> tuple[str, str]:
    """A wrong shape invalidates any class (an OOM at another P says nothing about this cell); an
    unverified shape only blocks SUCCESS (a failure before the forward keeps its own class)."""
    if status == "mismatch" or (cls == "SUCCESS" and status != "match"):
        return "INVALID_CELL", f"{status}: {cell_detail}"
    return cls, detail


def _first_line(text: str, needle: str) -> str:
    for line in text.splitlines():
        if needle.lower() in line.lower():
            return line.strip()[:300]
    return ""


def check_scope(scope: list[dict], counters: list[dict], inputs=()) -> tuple[float | str, str]:
    """Verify each declared site in one inference process or Mode B iteration."""
    total = served = 0
    missing = []
    sizes = [i["n_token"] for i in inputs]
    for rule in scope:
        op, site, provider = rule["op"], rule["call_site"], rule["provider"]
        floor = rule.get("min_tokens")
        at_site = [c for c in counters if (c["op"], c["call_site"]) == (op, site) and c["count"] > 0]
        scoped = []
        seen_short = False
        for c in at_site:
            shape = c.get("first_shape") or []
            below = floor is not None and len(shape) >= 3 and 0 < shape[-3] == shape[-2] < floor
            if below and not c.get("in_replacement_scope") and c.get("route_allowed") and (
                c["reason"] == "below_min_tokens" and c["requested_backend"] == c["served_backend"] in ("torch", "cueq")
            ):
                seen_short = True
            elif c.get("in_replacement_scope") and c["requested_backend"] == provider and (
                floor is None or (len(shape) >= 3 and shape[-3] == shape[-2] >= floor)
            ):
                scoped.append(c)
            else:
                missing.append(f"{op}@{site} (unexpected route or shape)")
                break
        ok = sum(c["count"] for c in scoped if c["served_backend"] == provider and c.get("route_allowed"))
        served += ok
        total += sum(c["count"] for c in scoped)
        if not ok and (not seen_short or any(n >= floor for n in sizes)):
            missing.append(f"{op}@{site} ({provider})")
        elif floor is not None and any(n < floor for n in sizes) and not seen_short:
            missing.append(f"{op}@{site} (below-floor baseline)")
    if missing:
        return 0.0, "invalid or missing replacement route evidence: " + ", ".join(missing)
    return (served / total if total else "N/A"), ""


def scope_coverage(variant_scope: list[dict], counters: list[dict]):
    return check_scope(variant_scope, counters)[0]


def rescue_counts(counters: list[dict]) -> dict:
    """Allowed V0 rescues (cuEq -> DS4Sci etc.) outside the scope: metadata on a composite SUCCESS."""
    out = {}
    for c in counters:
        if not c.get("in_replacement_scope") and c["served_backend"] != c["requested_backend"]:
            k = f"{c['op']}@{c['call_site']}:{c['requested_backend']}->{c['served_backend']}:{c['reason']}"
            out[k] = out.get(k, 0) + c["count"]
    return out


# --- input entries ---------------------------------------------------------------------------------

POLYMERS = ("proteinChain", "dnaSequence", "rnaSequence")


def absolute_entry(entry: dict, root: str) -> dict:
    """Copy of an input-JSON entry whose map/MSA paths are absolute (examples use repo-relative
    ``assets/...`` paths; ``root`` is the directory those paths are relative to)."""
    e = json.loads(json.dumps(entry))

    def fix(p):
        return p if not p or os.path.isabs(p) else os.path.normpath(os.path.join(root, p))

    if "map_path" in e:
        e["map_path"] = fix(e["map_path"])
    for s in e.get("sequences", []):
        for v in s.values():
            msa = v.get("msa") if isinstance(v, dict) else None
            if msa and msa.get("precomputed_msa_dir"):
                msa["precomputed_msa_dir"] = fix(msa["precomputed_msa_dir"])
    return e


def n_polymer_tokens(entry: dict) -> int:
    """Token count of a polymer-only entry (one token per residue/nucleotide)."""
    n = 0
    for s in entry["sequences"]:
        (kind, v), = s.items()
        if kind not in POLYMERS:
            raise ValueError(f"{entry.get('name')}: {kind} tokens are not counted (polymer chains only)")
        n += len(v["sequence"]) * int(v.get("count", 1))
    return n
