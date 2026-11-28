"""Compare stage outputs from two runs of the same entry, seed, and sample order."""

import argparse
import csv
import json
import math
from pathlib import Path


def kendall_tau(x, y):
    """Calculate tau-b for paired fitting scores."""
    concordant = discordant = ties_x = ties_y = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = (x[i] > x[j]) - (x[i] < x[j])
            dy = (y[i] > y[j]) - (y[i] < y[j])
            if dx and dy:
                concordant += dx == dy
                discordant += dx != dy
            elif dx:
                ties_y += 1
            elif dy:
                ties_x += 1
    pairs = concordant + discordant
    denominator = math.sqrt((pairs + ties_x) * (pairs + ties_y))
    return (concordant - discordant) / denominator if denominator else None


def read_scores(stage, name):
    path = stage / "saved_data" / "scores.csv"
    if not path.is_file():
        return {}
    scores = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["pdb_id"] != name:
                continue
            key = row["method"], int(row["sample_idx"])
            if key in scores:
                raise ValueError(f"{path} contains duplicate fitting row {key}")
            scores[key] = row
    return scores


def compare_scores(ref, cand):
    common = sorted(ref.keys() & cand.keys())
    result = {
        "common_rows": len(common),
        "reference_only": sorted(ref.keys() - cand.keys()),
        "candidate_only": sorted(cand.keys() - ref.keys()),
    }
    if not ref or not cand:
        return {**result, "status": "one or both runs have no fitting scores"}

    key = "recall_ccmask_ca"
    ref_ranking = {pair: float(row[key]) for pair, row in ref.items()}
    cand_ranking = {pair: float(row[key]) for pair, row in cand.items()}
    if not all(map(math.isfinite, (*ref_ranking.values(), *cand_ranking.values()))):
        return {**result, "status": "nonfinite fitting score"}
    # Preserve all ties rather than relying on CSV row order to pick a winner.
    ref_best = sorted(
        k for k, v in ref_ranking.items() if v == max(ref_ranking.values())
    )
    cand_best = sorted(
        k for k, v in cand_ranking.items() if v == max(cand_ranking.values())
    )
    recall_deltas = [
        abs(float(ref[pair]["recall"]) - float(cand[pair]["recall"])) for pair in common
    ]
    return {
        **result,
        "top1_ref": ref_best,
        "top1_cand": cand_best,
        "top1_match": ref_best == cand_best,
        "kendall_tau": kendall_tau(
            [ref_ranking[pair] for pair in common],
            [cand_ranking[pair] for pair in common],
        ),
        "recall_abs_max": max(recall_deltas, default=None)
        if all(map(math.isfinite, recall_deltas))
        else None,
    }


def read_confidence(stage, name, seed):
    directory = stage / f"seed_{seed}" / "predictions"
    prefix = f"{name}_summary_confidence_sample_"
    summaries = {}
    for path in directory.glob(f"{prefix}*.json"):
        rank = int(path.stem.removeprefix(prefix))
        with path.open() as stream:
            summaries[rank] = json.load(stream)
    return summaries


def compare_confidence(ref, cand):
    deltas = {}
    nonfinite = set()
    for rank in sorted(ref.keys() & cand.keys()):
        for key, value in ref[rank].items():
            other = cand[rank].get(key)
            if type(value) not in (int, float) or type(other) not in (int, float):
                continue
            delta = abs(value - other)
            if math.isfinite(delta):
                deltas[key] = max(delta, deltas.get(key, 0))
            else:
                nonfinite.add(key)
    return {
        "common_ranks": sorted(ref.keys() & cand.keys()),
        "reference_only": sorted(ref.keys() - cand.keys()),
        "candidate_only": sorted(cand.keys() - ref.keys()),
        "abs_max_by_export_rank": deltas,
        "nonfinite_fields": sorted(nonfinite),
    }


def kabsch_rmsd(x, y):
    import torch

    xc = x - x.mean(-2, keepdim=True)
    yc = y - y.mean(-2, keepdim=True)
    u, _, vt = torch.linalg.svd(yc.transpose(-1, -2) @ xc)
    sign = torch.linalg.det(u @ vt).sign()
    one = torch.ones_like(sign)
    rotation = u @ torch.diag_embed(torch.stack((one, one, sign), -1)) @ vt
    return ((yc @ rotation - xc).square().sum(-1).mean(-1)).sqrt()


def compare_coords(ref, cand, name):
    import numpy as np
    import torch

    paths = [stage / "saved_data" / f"output_dict_{name}.pt" for stage in (ref, cand)]
    if not all(path.is_file() for path in paths):
        return {"status": "one or both runs have no saved coordinate tensors"}
    # Stage outputs include NumPy arrays for the map support points.
    reconstruct = np.core.multiarray._reconstruct
    numpy_types = [
        (reconstruct, "numpy.core.multiarray._reconstruct"),
        (reconstruct, "numpy._core.multiarray._reconstruct"),
        np.ndarray,
        np.dtype,
        *(
            type(np.dtype(dtype))
            for dtype in ("bool", "int32", "int64", "float16", "float32", "float64")
        ),
    ]
    with torch.serialization.safe_globals(numpy_types):
        a, b = (
            torch.load(path, map_location="cpu", weights_only=True) for path in paths
        )
    mask = torch.as_tensor(a["ca_mask"]).bool()
    if not torch.equal(mask, torch.as_tensor(b["ca_mask"]).bool()):
        return {"status": "different C-alpha masks"}
    x, y = a["pred_all_coordinate"], b["pred_all_coordinate"]
    if x.shape != y.shape or x.shape[-2] != mask.numel():
        return {"status": "different coordinate dimensions"}
    if not mask.any():
        return {"status": "no C-alpha atoms"}
    x, y = x[..., mask, :].double(), y[..., mask, :].double()
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        return {"status": "nonfinite coordinates"}
    return {
        "ca_rmsd_raw": (x - y).square().sum(-1).mean(-1).sqrt().tolist(),
        "ca_rmsd_kabsch": kabsch_rmsd(x, y).tolist(),
        "n_ca": int(mask.sum()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ref", required=True, type=Path, help="reference stage directory"
    )
    parser.add_argument(
        "--cand", required=True, type=Path, help="candidate stage directory"
    )
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--seed", type=int, default=101, help="seed for confidence JSONs"
    )
    args = parser.parse_args(argv)
    for stage in (args.ref, args.cand):
        if not stage.is_dir():
            parser.error(f"not a stage directory: {stage}")
    try:
        report = {
            "coordinates": compare_coords(args.ref, args.cand, args.name),
            "confidence": compare_confidence(
                read_confidence(args.ref, args.name, args.seed),
                read_confidence(args.cand, args.name, args.seed),
            ),
            "scores": compare_scores(
                read_scores(args.ref, args.name), read_scores(args.cand, args.name)
            ),
        }
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
