"""Baseline vs. candidate output comparison for one entry/stage, with gates.

Inputs are two stage directories (``<dump>/<name>/<CryoZeta|CryoZeta-Interpolate>``) of the same entry,
seed and N_sample. Reports:
  * coordinates: per-sample Ca RMSD from ``saved_data/output_dict_<name>.pt`` (index-aligned), raw
    (same frame) and after Kabsch superposition;
  * confidence: |delta| of every numeric field of the rank-ordered summary JSONs;
  * fitting/ranking: top-1 (method, sample) agreement and Kendall tau of ``recall_ccmask_ca`` over common
    (method, sample) rows of ``scores.csv``, method sets, recall and ``num_dist_over_4`` deltas;
  * trunk: relative Frobenius error of s/z from ``--instrument trunk_dump`` files, if given;
  * lDDT/TM: not computed here (needs deposited reference models).
Gates are evaluated only against a sealed tolerances file (``sealed: true``, every gate filled, calibration
trials listed). They are sample-paired, so ``--tolerances`` needs ``--variant`` (the candidate's YAML), which
must match the candidate's ``trial.json`` (found above ``--cand``), and is refused for a variant marked
``sample_paired: false`` (M-SDC-5) or for synthetic trials: no other thresholds are substituted.

  python -m cryozeta.kernel_bench.compare_outputs --ref R/9b0l/CryoZeta --cand C/9b0l/CryoZeta \
      --name 9b0l --seed 101 --out cmp.json [--ref-trunk R/trunk.pt --cand-trunk C/trunk.pt] \
      [--tolerances TOLERANCES.yaml --variant VARIANT.yaml]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys

from cryozeta.kernel_bench import common as C
from cryozeta.kernel_bench.run_trial import read_variant


def kendall_tau(x: list, y: list) -> float | None:
    """Tau-b over paired values (O(n^2); n is N_sample x methods)."""
    n, conc, disc, tx, ty = len(x), 0, 0, 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = (x[i] > x[j]) - (x[i] < x[j]), (y[i] > y[j]) - (y[i] < y[j])
            if dx and dy:
                conc += dx == dy
                disc += dx != dy
            elif dx:
                ty += 1
            elif dy:
                tx += 1
    denom = ((conc + disc + tx) * (conc + disc + ty)) ** 0.5
    return (conc - disc) / denom if denom else None


def read_scores(stage: str, name: str) -> dict:
    path = os.path.join(stage, "saved_data", "scores.csv")
    if not os.path.isfile(path):
        return {}
    return {(r["method"], int(r["sample_idx"])): r for r in csv.DictReader(open(path)) if r["pdb_id"] == name}


def compare_scores(ref: dict, cand: dict) -> dict:
    out = {"methods_ref": sorted({m for m, _ in ref}), "methods_cand": sorted({m for m, _ in cand})}
    if not ref or not cand:
        out["status"] = "scores.csv missing"
        return out
    common = sorted(set(ref) & set(cand))
    key = "recall_ccmask_ca"
    out["top1_ref"] = max(ref, key=lambda k: float(ref[k][key]))
    out["top1_cand"] = max(cand, key=lambda k: float(cand[k][key]))
    out["top1_match"] = out["top1_ref"] == out["top1_cand"]
    out["kendall_tau"] = kendall_tau([float(ref[k][key]) for k in common], [float(cand[k][key]) for k in common])
    out["recall_abs_max"] = max((abs(float(ref[k]["recall"]) - float(cand[k]["recall"])) for k in common),
                                default=None)
    out["num_dist_over_4"] = {f"{m}:{s}": [int(float(ref[(m, s)]["num_dist_over_4"])),
                                           int(float(cand[(m, s)]["num_dist_over_4"]))] for m, s in common}
    return out


def compare_confidence(ref_stage: str, cand_stage: str, name: str, seed: int) -> dict:
    def read(p):
        with open(p) as f:
            return json.load(f)

    def load(stage):
        pat = os.path.join(stage, f"seed_{seed}", "predictions", f"{name}_summary_confidence_sample_*.json")
        return {int(p.rsplit("_", 1)[1][:-5]): read(p) for p in glob.glob(pat)}

    r, c = load(ref_stage), load(cand_stage)
    deltas = {}
    for rank in sorted(set(r) & set(c)):
        for k, v in r[rank].items():
            w = c[rank].get(k)
            if isinstance(v, (int, float)) and isinstance(w, (int, float)) and not isinstance(v, bool):
                x, prev = abs(v - w), deltas.get(k, 0.0)
                deltas[k] = x if math.isnan(x) or x > prev else prev  # max() would drop a NaN
    return {"ranks": len(set(r) & set(c)), "abs_max": deltas, "note": "summary files are rank-ordered"}


def compare_coords(ref_stage: str, cand_stage: str, name: str) -> dict:
    import torch

    f = os.path.join("saved_data", f"output_dict_{name}.pt")
    if not all(os.path.isfile(os.path.join(s, f)) for s in (ref_stage, cand_stage)):
        return {"status": f"{f} missing"}
    r, c = (torch.load(os.path.join(s, f), map_location="cpu", weights_only=False) for s in (ref_stage, cand_stage))
    mask = torch.as_tensor(r["ca_mask"]).bool()
    if not torch.equal(mask, torch.as_tensor(c["ca_mask"]).bool()):
        return {"status": "different Ca masks (different inputs?)"}
    x = r["pred_all_coordinate"][..., mask, :].double()
    y = c["pred_all_coordinate"][..., mask, :].double()
    raw = ((x - y) ** 2).sum(-1).mean(-1).sqrt()
    return {"ca_rmsd_raw": raw.tolist(), "ca_rmsd_kabsch": kabsch_rmsd(x, y).tolist(), "n_ca": int(mask.sum())}


def kabsch_rmsd(x, y):
    """RMSD of y onto x after optimal rigid superposition; x, y: [..., n, 3]."""
    import torch

    xc, yc = x - x.mean(-2, keepdim=True), y - y.mean(-2, keepdim=True)
    u, _, vt = torch.linalg.svd(yc.transpose(-1, -2) @ xc)
    d = torch.sign(torch.linalg.det(u @ vt))
    fix = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    rot = u @ fix @ vt
    return (((yc @ rot) - xc) ** 2).sum(-1).mean(-1).sqrt()


def compare_trunk(ref: str, cand: str) -> dict:
    import torch

    r, c = torch.load(ref, map_location="cpu"), torch.load(cand, map_location="cpu")
    return {k: ((c[k].double() - r[k].double()).norm() / r[k].double().norm().clamp_min(1e-30)).item()
            for k in sorted(set(r) & set(c))}


def gates(result: dict, tol: dict) -> dict:
    """Pass/fail per gate against a sealed tolerances file; unknown inputs stay None (not a pass)."""

    def worst(xs):  # a NaN/Inf anywhere fails the gate; plain max() would skip a NaN
        xs = list(xs)
        return max(xs) if all(math.isfinite(x) for x in xs) else math.inf

    g = {}
    if "ca_rmsd_max" in tol:
        v = result["coordinates"].get("ca_rmsd_kabsch")
        g["G-coord"] = None if v is None else worst(v) <= tol["ca_rmsd_max"]
    if "confidence_abs_max" in tol:
        d = result["confidence"]["abs_max"]
        g["G-conf"] = None if not d else worst(d.values()) <= tol["confidence_abs_max"]
    if tol.get("top1_must_match"):
        g["G-rank-top1"] = result["scores"].get("top1_match")
    if "kendall_tau_min" in tol:
        t = result["scores"].get("kendall_tau")
        g["G-rank-tau"] = None if t is None else t >= tol["kendall_tau_min"]
    if "trunk_rel_max" in tol:
        t = result.get("trunk")
        g["G-trunk"] = None if not t else worst(t.values()) <= tol["trunk_rel_max"]
    g["all_pass"] = bool(g) and all(v is True for v in g.values())  # None (missing evidence) is not a pass
    return g


GATE_KEYS = ("ca_rmsd_max", "confidence_abs_max", "top1_must_match", "kendall_tau_min", "trunk_rel_max")


def trial_record(stage: str) -> dict | None:
    """The run_trial ``trial.json`` of the trial that produced ``stage`` (nearest parent holding one)."""
    d = os.path.abspath(stage)
    while d != os.path.dirname(d):
        path = os.path.join(d, "trial.json")
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
        d = os.path.dirname(d)
    return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True, help="baseline stage dir")
    p.add_argument("--cand", required=True, help="candidate stage dir")
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--ref-trunk")
    p.add_argument("--cand-trunk")
    p.add_argument("--tolerances", help="tolerances.yaml; gates are evaluated only if sealed: true")
    p.add_argument("--variant", help="candidate variant YAML (required with --tolerances)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    for d in (a.ref, a.cand):
        if not os.path.isdir(d):
            p.error(f"not a directory: {d}")
    if bool(a.ref_trunk) != bool(a.cand_trunk):
        p.error("--ref-trunk and --cand-trunk go together")
    for f in (a.ref_trunk, a.cand_trunk, a.tolerances, a.variant):
        if f and not os.path.isfile(f):
            p.error(f"not found: {f}")
    if os.path.exists(a.out):
        p.error(f"--out {a.out} exists; results are never overwritten")
    if a.tolerances:
        if not a.variant:
            p.error("--tolerances needs --variant: the gates are sample-paired")
        v = read_variant(a.variant)
        if v["sample_paired"] is not True:
            p.error(f"{v['variant_id']} is not sample-paired (status {v['status']}): the sample-paired gates do "
                    f"not apply and no substitute thresholds are used. {v.get('deferred_reason', '')}")
        recs = {r: trial_record(getattr(a, r)) for r in ("ref", "cand")}
        if None in recs.values():
            p.error("--tolerances needs run_trial outputs: no trial.json above --ref/--cand")
        sha = recs["cand"]["provenance"].get("variant_sha256")
        if recs["cand"]["variant_id"] != v["variant_id"] or sha not in (None, C.sha256_file(a.variant)):
            p.error(f"--variant {v['variant_id']} is not the variant the candidate trial ran "
                    f"({recs['cand']['variant_id']}, sha256 {sha})")
        if any(r.get("synthetic") for r in recs.values()):
            p.error("synthetic trial: resource stress only, accuracy gates do not apply")
        failed = {k: r.get("outcome") for k, r in recs.items() if r.get("outcome") != "SUCCESS"}
        if failed:
            p.error(f"gates need two SUCCESS trials; outcome {failed}")

        def pairing(r):
            s = r.get("settings") or {}
            return (r.get("protocol"), r.get("entries"), r.get("frozen_det_sha256_before"),
                    [s.get(k) for k in ("seed", "n_sample", "n_step", "n_cycle")])

        if pairing(recs["ref"]) != pairing(recs["cand"]):
            p.error(f"ref and cand trials are not paired (protocol, entries, frozen det, seed/n_sample/n_step/"
                    f"n_cycle): {pairing(recs['ref'])} vs {pairing(recs['cand'])}")
        want = recs["cand"].get("baseline_id") or "V0"
        if recs["ref"].get("variant_id") != want:
            p.error(f"--ref ran {recs['ref'].get('variant_id')}, not the candidate's baseline {want}")
        if (recs["cand"].get("settings") or {}).get("seed") != a.seed or a.name not in (recs["cand"].get("entries")
                                                                                       or []):
            p.error(f"--seed {a.seed} / --name {a.name} are not the trials' seed and entries")
        import yaml

        with open(a.tolerances) as f:
            a.tol = yaml.safe_load(f) or {}
        unfilled = [k for k in GATE_KEYS if a.tol.get(k) is None]
        if a.tol.get("sealed") is True and (unfilled or not (a.tol.get("calibration") or {}).get("trials")):
            p.error(f"sealed tolerances need every gate and calibration.trials; unfilled: {unfilled}")
    return a


def main(argv=None):
    a = parse_args(argv)
    res = {
        "ref": os.path.abspath(a.ref), "cand": os.path.abspath(a.cand), "name": a.name, "seed": a.seed,
        "coordinates": compare_coords(a.ref, a.cand, a.name),
        "confidence": compare_confidence(a.ref, a.cand, a.name, a.seed),
        "scores": compare_scores(read_scores(a.ref, a.name), read_scores(a.cand, a.name)),
        "trunk": compare_trunk(a.ref_trunk, a.cand_trunk) if a.ref_trunk else None,
        "lddt_tm": "not computed (prerequisite: deposited reference models)",
    }
    if a.tolerances:
        tol = a.tol
        res["tolerances"] = {"path": os.path.abspath(a.tolerances), "sha256": C.sha256_file(a.tolerances)}
        res["variant"] = {"path": os.path.abspath(a.variant), "sha256": C.sha256_file(a.variant)}
        res["gates"] = gates(res, tol) if tol.get("sealed") is True else "not evaluated (tolerances unsealed)"
    C.write_json(a.out, res)
    print(json.dumps({"gates": res.get("gates"), "coordinates": res["coordinates"].get("ca_rmsd_kabsch")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
