"""Synthetic capacity inputs: scale a real polymer entry to a target
N_token and give it a detection file with enough points. Everything written is labeled synthetic;
results on these inputs are capacity/latency evidence only, never accuracy evidence.

Chains are repeated (cyclically, keeping counts) and the last one truncated to hit N_token exactly; a
truncated chain loses its MSA. ``--msa single`` drops all MSAs. A protein/RNA chain without an MSA gets a
query-only (single-sequence) one written under OUT/msa: CryoZeta's featurizer needs an MSA for every such
chain, and without one the sample fails featurization (inference skips it and still exits 0). The detection file gets exactly
N_point points: a seeded random subset when it has more (every point-aligned field and both axes of
interpolation confidence take the same sorted indices), else translated copies (clusters offset,
interpolation confidence block-diagonal) cut to N_point. N_point must be <= min(N_token, 2000), the
unchanged V0 support-point cap (json_to_feature); larger requests are refused, never clamped. With
n_detected == N_point, ``sample_support_points_v2`` prunes nothing, so the model sees P = N_point.

  python -m cryozeta.kernel_bench.synth_inputs --base-json assets/examples/example.json --entry 9b0l \
      --root ASSETS_PARENT --n-token 3000 --det DET/9b0l/CryoZeta-Detection/9b0l.pt --out OUT
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys

from cryozeta.kernel_bench import common as C

DET_KEYS = ("main_atom_coords", "main_atom_probs", "res_features", "cluster_ids")
MAX_POINTS = 2000  # json_to_feature caps support points at min(N_token, 2000)
# files msa_featurizer reads from a precomputed_msa_dir (non-pairing, plus uniref100 pairing for protein heteromers)
QUERY_ONLY_MSA = {"proteinChain": ("mmseqs_other_hits.a3m", "uniref100_hits.a3m"), "rnaSequence": ("rnacentral.a3m",)}


def scale_sequences(seqs: list, n_token: int) -> tuple[list, list]:
    """Repeat chains cyclically (keeping counts) and truncate the last to exactly n_token tokens."""
    out, notes, left, i = [], [], n_token, 0
    while left > 0:
        (kind, v), = copy.deepcopy(seqs[i % len(seqs)]).items()
        length, count = len(v["sequence"]), int(v.get("count", 1))
        take = min(count, left // length)
        if take:
            out.append({kind: {**v, "count": take}})
            left -= take * length
        if take < count and left:
            tail = {k: x for k, x in v.items() if k != "msa"}
            out.append({kind: {**tail, "sequence": v["sequence"][:left], "count": 1}})
            notes.append(f"truncated {kind} #{i % len(seqs)} to {left} residues (query-only MSA)")
            left = 0
        i += 1
    return out, notes


def _point_keys(det: dict, n0: int) -> list:
    """Point-aligned fields: the known detection keys plus any other array with a leading n0 axis."""
    extra = [k for k, v in det.items() if k not in DET_KEYS and k != "interpolate_confidence"
             and getattr(v, "shape", None) and v.shape[0] == n0]
    return list(DET_KEYS) + extra


def tile_detection(det: dict, n_point: int, gap: float = 10.0, seed: int = 0) -> tuple[dict, dict]:
    """Exactly n_point detection points: a seeded sorted subset if more, translated copies if fewer."""
    import hashlib

    import torch

    missing = [k for k in DET_KEYS if k not in det]
    if missing:
        raise ValueError(f"detection file lacks {missing}")
    n0 = len(det["main_atom_coords"])
    keys = _point_keys(det, n0)
    t = {k: torch.as_tensor(det[k]) for k in keys}
    ic = det.get("interpolate_confidence")
    ic = None if ic is None else torch.as_tensor(ic)
    if ic is not None and tuple(ic.shape[:2]) != (n0, n0):
        raise ValueError(f"interpolate_confidence {tuple(ic.shape)} is not [{n0}, {n0}, ...]")
    out = dict(det)
    if n0 == n_point:
        return out, {"mode": "unchanged", "n_detected": n0}
    if n0 > n_point:
        idx = torch.randperm(n0, generator=torch.Generator().manual_seed(seed))[:n_point].sort().values
        for k in keys:
            out[k] = t[k][idx].clone()
        if ic is not None:
            out["interpolate_confidence"] = ic[idx][:, idx].contiguous()
        return out, {"mode": "subsampled", "n_detected": n0, "seed": seed, "point_keys": keys,
                     "indices_sha256": hashlib.sha256(idx.numpy().tobytes()).hexdigest()}
    k = math.ceil(n_point / n0)
    xyz = t["main_atom_coords"].float()
    shift = float(xyz[:, 0].max() - xyz[:, 0].min()) + gap
    for key in keys:
        out[key] = torch.cat([t[key]] * k)[:n_point]
    out["main_atom_coords"] = torch.cat([xyz + torch.tensor([j * shift, 0.0, 0.0]) for j in range(k)])[:n_point]
    out["main_atom_coords"] = out["main_atom_coords"].to(t["main_atom_coords"].dtype)
    cid = t["cluster_ids"]
    out["cluster_ids"] = torch.cat([cid + j * (int(cid.max()) + 1) for j in range(k)])[:n_point]
    if ic is not None:
        big = torch.zeros((n0 * k, n0 * k, *ic.shape[2:]), dtype=ic.dtype)
        for j in range(k):
            big[j * n0:(j + 1) * n0, j * n0:(j + 1) * n0] = ic
        out["interpolate_confidence"] = big[:n_point, :n_point].contiguous()
    return out, {"mode": "tiled", "n_detected": n0, "copies": k, "x_shift_angstrom": shift, "point_keys": keys}


def max_points(n_token: int) -> int:
    """V0 support-point request for N_token tokens (json_to_feature): min(N_token, 2000)."""
    return min(n_token, MAX_POINTS)


def capacity_cells(n_tokens, levels: dict) -> tuple[list, list]:
    """L3 matrix: one cell per distinct realizable (N_token, P), naming every P level that maps
    to it; a level above min(N_token, 2000) is ineligible under V0 (listed, never clamped)."""
    cells, ineligible = {}, []
    for n in n_tokens:
        for name, pt in levels.items():
            if pt > max_points(n):
                ineligible.append({"n_token": n, "level": name, "n_point": pt, "cap": max_points(n)})
            else:
                cells.setdefault((n, pt), []).append(name)
    return [{"n_token": n, "n_point": pt, "levels": ls} for (n, pt), ls in cells.items()], ineligible


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-json", required=True)
    p.add_argument("--entry", required=True, help="entry name in --base-json")
    p.add_argument("--root", required=True, help="directory the entry's relative paths resolve against")
    p.add_argument("--n-token", type=int, required=True)
    p.add_argument("--msa", choices=("real", "single"), default="real")
    p.add_argument("--det", help="the base entry's detection .pt (omit to write the entry JSON only)")
    p.add_argument("--n-point", type=int, help="detection points to provide, <= min(N_token, 2000) (default)")
    p.add_argument("--seed", type=int, default=0, help="subsampling seed (detection with more points)")
    p.add_argument("--out", required=True, help="new output directory (must not exist)")
    a = p.parse_args(argv)
    if a.n_token < 1 or (a.n_point is not None and a.n_point < 1):
        p.error("--n-token/--n-point must be positive")
    if a.n_point is not None and a.n_point > max_points(a.n_token):
        p.error(f"--n-point {a.n_point} is not realizable: V0 caps support points at min(N_token, 2000) = "
                f"{max_points(a.n_token)}; exceeding it is a separate policy change, not this baseline")
    if os.path.exists(a.out):
        p.error(f"--out exists: {a.out}")
    for f in (a.base_json, a.det):
        if f and not os.path.isfile(f):
            p.error(f"not found: {f}")
    return a


def main(argv=None):
    a = parse_args(argv)
    with open(a.base_json) as f:
        base = next((e for e in json.load(f) if e.get("name") == a.entry), None)
    if base is None:
        raise SystemExit(f"{a.entry} not in {a.base_json}")
    C.n_polymer_tokens(base)  # polymer-only check
    entry = C.absolute_entry(base, a.root)
    entry.pop("em_file", None)  # would override --em_file_dir: the synthetic detection tree must be used
    name = f"{a.entry}_syn{a.n_token}_{a.msa}"
    entry["name"] = name
    entry["sequences"], notes = scale_sequences(entry["sequences"], a.n_token)
    if a.msa == "single":
        for s in entry["sequences"]:
            next(iter(s.values())).pop("msa", None)
    assert C.n_polymer_tokens(entry) == a.n_token
    for i, s in enumerate(entry["sequences"]):
        (kind, v), = s.items()
        if kind in QUERY_ONLY_MSA and "msa" not in v:
            d = os.path.abspath(os.path.join(a.out, "msa", str(i)))
            os.makedirs(d)
            for f in QUERY_ONLY_MSA[kind]:
                with open(os.path.join(d, f), "w") as fh:
                    fh.write(f">query\n{v['sequence']}\n")
            v["msa"] = {"precomputed_msa_dir": d, "pairing_db": "uniref100"}
    os.makedirs(os.path.join(a.out, "entries"))
    entry_path = os.path.join(a.out, "entries", f"{name}.json")
    with open(entry_path, "w") as f:
        json.dump([entry], f, indent=1)
    label = {"synthetic": True, "use": "capacity/latency only, never accuracy", "name": name,
             "base_json": os.path.abspath(a.base_json), "base_json_sha256": C.sha256_file(a.base_json),
             "base_entry": a.entry, "base_n_token": C.n_polymer_tokens(base), "n_token": a.n_token,
             "msa": a.msa, "notes": notes, "entry_json": entry_path}
    if a.det:
        import torch

        n_point = a.n_point or max_points(a.n_token)
        det, info = tile_detection(torch.load(a.det, weights_only=False), n_point, seed=a.seed)
        expected = min(len(det["main_atom_coords"]), max_points(a.n_token))
        assert expected == n_point, (expected, n_point)  # parse_args refused unrealizable requests
        det_dir = os.path.join(a.out, "det", name, "CryoZeta-Detection")
        os.makedirs(det_dir)
        torch.save(det, os.path.join(det_dir, f"{name}.pt"))
        label.update(det_source=os.path.abspath(a.det), det_source_sha256=C.sha256_file(a.det),
                     det_tree=os.path.join(a.out, "det"), n_point_requested=n_point,
                     n_point_written=len(det["main_atom_coords"]), n_point_expected=expected, **info)
    C.write_json(os.path.join(a.out, "synthetic.json"), label)
    print(json.dumps({k: label[k] for k in ("name", "n_token", "entry_json") if k in label}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
