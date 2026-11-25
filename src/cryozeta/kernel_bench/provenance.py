"""Provenance tools.

  snapshot    git revision + diff vs. the audited baseline, lock/manifest hashes vs. the expected ones, pixi
              package list, nvidia-smi, asset hashes vs. a sha256 list. Writes a new directory.
  freeze-det  manifest (per-file sha256, n_detected, N_point) for a detection tree, written next to it
              as ``<tree>.manifest.json``, then the tree is made read-only.
  entries     one input JSON per entry with absolute map/MSA paths.

  python -m cryozeta.kernel_bench.provenance snapshot --repo . --assets ASSETS --assets-sha LIST --out DIR
  python -m cryozeta.kernel_bench.provenance freeze-det --det DET --entries ENTRIES_DIR
  python -m cryozeta.kernel_bench.provenance entries --base-json ASSETS/examples/example.json --root ROOT --out DIR
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import stat
import subprocess
import sys

from cryozeta.kernel_bench import common as C

BASELINE_REV = "c544c6bbd0366951d8b4e0bc4c0da547836f7b8f"
PLAN_HASHES = {
    "pixi.lock": "842c5b7d813948af28becababda384192213e777d2ac426fd49a14a17a170ac1",
    "uv.lock": "78209fc9530160e5bce9eee3445f5e7b0a4660c321fb1bc995eb3ae3b9de8f19",
    "pyproject.toml": "6e324885b4dff82eca25e73eb9efe3d50f3ea32c66c0bb5600a4f0bb4ef7fdda",
}
MAX_POINTS = 2000


def sh(cmd, cwd=None):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300)
        return r.stdout if r.returncode == 0 else f"[exit {r.returncode}] {r.stderr.strip()[:500]}"
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"[unavailable] {e}"


def snapshot(a):
    os.makedirs(a.out)
    git = ["git", "-C", a.repo]
    diff = sh(git + ["diff", BASELINE_REV, "--", "."])
    with open(os.path.join(a.out, "diff_vs_baseline.patch"), "w") as f:
        f.write(diff)
    rec = {
        "host": platform.node(),
        "head": sh(git + ["rev-parse", "HEAD"]).strip(),
        "baseline": BASELINE_REV,
        "branch": sh(git + ["rev-parse", "--abbrev-ref", "HEAD"]).strip(),
        "diff_stat_vs_baseline": sh(git + ["diff", "--stat", BASELINE_REV, "--", "."]).strip().splitlines()[-1:],
        "diff_sha256": C.sha256_file(os.path.join(a.out, "diff_vs_baseline.patch")),
        "untracked_sha256": {f: C.sha256_file(os.path.join(a.repo, f))  # new files are not in the diff
                             for f in sh(git + ["ls-files", "--others", "--exclude-standard"]).split()},
        "lock_hashes": {},
    }
    for f, want in PLAN_HASHES.items():
        p = os.path.join(a.repo, f)
        got = C.sha256_file(p) if os.path.isfile(p) else None
        rec["lock_hashes"][f] = {"sha256": got, "plan": want, "match": got == want}
    for name, cmd in (("pixi_list.txt", ["pixi", "list", "-e", "default", "--frozen"]),
                      ("nvidia_smi.txt", ["nvidia-smi"])):
        with open(os.path.join(a.out, name), "w") as f:
            f.write(sh(cmd, cwd=a.repo))
    if a.assets and a.assets_sha:
        bad, n = [], 0
        for line in open(a.assets_sha):
            want, rel = line.split(maxsplit=1)
            rel = rel.strip()
            p = os.path.join(a.assets, rel)
            n += 1
            if not os.path.isfile(p) or C.sha256_file(p) != want:
                bad.append(rel)
        rec["assets"] = {"root": os.path.abspath(a.assets), "list": os.path.abspath(a.assets_sha),
                         "checked": n, "mismatched_or_missing": bad}
    C.write_json(os.path.join(a.out, "snapshot.json"), rec)
    print(json.dumps({k: rec[k] for k in ("head", "lock_hashes")}, indent=1))
    return 0 if all(v["match"] for v in rec["lock_hashes"].values()) and not rec.get("assets", {}).get(
        "mismatched_or_missing") else 1


def freeze_det(a):
    import torch

    det = os.path.abspath(a.det).rstrip("/")
    man_path = det + ".manifest.json"
    if os.path.exists(man_path):
        raise SystemExit(f"already frozen: {man_path}")
    tokens = {}
    for f in glob.glob(os.path.join(a.entries, "*.json")) if a.entries else []:
        for e in json.load(open(f)):
            try:
                tokens[e["name"]] = C.n_polymer_tokens(e)
            except ValueError:
                tokens[e["name"]] = None  # ligands: N_token from the trial log instead
    entries = {}
    for pt in sorted(glob.glob(os.path.join(det, "*", "CryoZeta-Detection", "*.pt"))):
        name = os.path.basename(pt)[:-3]
        d = torch.load(pt, map_location="cpu", weights_only=False)
        n_det = len(d["main_atom_coords"])
        n_tok = tokens.get(name)
        entries[name] = {
            "pt": os.path.relpath(pt, det), "sha256": C.sha256_file(pt), "n_detected": n_det,
            "has_interpolation": d.get("interpolate_confidence") is not None,
            "N_token": n_tok, "N_point": None if n_tok is None else min(n_det, n_tok, MAX_POINTS),
        }
    if not entries:
        raise SystemExit(f"no <name>/CryoZeta-Detection/<name>.pt under {det}")
    man = {"tree": det, "tree_sha256": C.sha256_tree(det), "entries": entries,
           "rule": "N_point = min(n_detected, N_token, 2000) (json_to_feature + sample_support_points_v2)"}
    C.write_json(man_path, man)
    for root, dirs, files in os.walk(det):
        for x in files + dirs:
            p = os.path.join(root, x)
            os.chmod(p, os.stat(p).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    os.chmod(det, os.stat(det).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    print(json.dumps({"manifest": man_path, "entries": len(entries)}))
    return 0


def entries(a):
    os.makedirs(a.out, exist_ok=True)
    for e in json.load(open(a.base_json)):
        path = os.path.join(a.out, f"{e['name']}.json")
        if os.path.exists(path):
            raise SystemExit(f"exists: {path}")
        with open(path, "w") as f:
            json.dump([C.absolute_entry(e, a.root)], f, indent=1)
        print(path)
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--repo", default=".")
    s.add_argument("--assets")
    s.add_argument("--assets-sha", help="sha256sum-style list relative to --assets")
    s.add_argument("--out", required=True)
    f = sub.add_parser("freeze-det")
    f.add_argument("--det", required=True, help="detection output dir (<name>/CryoZeta-Detection/<name>.pt)")
    f.add_argument("--entries", help="per-entry JSON dir, for N_token/N_point")
    e = sub.add_parser("entries")
    e.add_argument("--base-json", required=True)
    e.add_argument("--root", required=True, help="directory the JSON's relative paths resolve against")
    e.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.cmd == "snapshot" and os.path.exists(a.out):
        p.error(f"--out exists: {a.out}")
    if a.cmd == "freeze-det" and not os.path.isdir(a.det):
        p.error(f"not a directory: {a.det}")
    if a.cmd == "entries" and not os.path.isfile(a.base_json):
        p.error(f"not found: {a.base_json}")
    return a


def main(argv=None):
    a = parse_args(argv)
    return {"snapshot": snapshot, "freeze-det": freeze_det, "entries": entries}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
