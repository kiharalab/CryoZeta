# Copyright (C) 2026 KiharaLab, Purdue University
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import csv
import json
import shutil
from pathlib import Path

import typer

app = typer.Typer(help="Select best results from CryoZeta and CryoZeta-Interpolate")


def _legacy_ranks(stage: Path, seed_dir: Path, name: str) -> dict[int, int]:
    """Match raw diffusion samples to ranked, unfitted AtomArray coordinates."""
    import numpy as np
    import torch
    from numpy.core.multiarray import _reconstruct

    if any(d != seed_dir for d in stage.glob("seed_*") if d.is_dir()):
        raise ValueError(f"{stage}: legacy scores.csv contains no seed identifiers and several seed directories exist")
    archive = stage / "saved_data" / f"output_dict_{name}.pt"
    if not archive.is_file():
        raise FileNotFoundError(f"{archive}: raw coordinates required to recover legacy CIF ranks")
    # The legacy archive includes a NumPy bool CA mask. Read only tensor/array
    # data and memory-map the unrelated, potentially large point residue logits.
    safe_globals = [np.ndarray, np.dtype, type(np.dtype(bool)), _reconstruct]
    if _reconstruct.__module__ != "numpy.core.multiarray":
        safe_globals.append((_reconstruct, "numpy.core.multiarray._reconstruct"))
    with torch.serialization.safe_globals(safe_globals):
        raw = torch.load(archive, map_location="cpu", weights_only=True, mmap=True)["pred_all_coordinate"]
    # Biotite stores the unfitted coordinates as float32 in the NPZ.
    raw = raw.float().numpy()
    npz_dir = seed_dir / "predictions" / "atom_arrays"
    # Internal model seeds append scores but overwrite the legacy raw archive.
    # A truncated archive cannot identify ranks in the concatenated prediction.
    if len(list(npz_dir.glob(f"{name}_sample_*.npz"))) > len(raw):
        raise ValueError(f"{npz_dir}: more ranked coordinates than raw samples in {archive}")
    ranks = {}
    for rank in range(len(raw)):
        npz = npz_dir / f"{name}_sample_{rank}.npz"
        if not npz.is_file():
            raise FileNotFoundError(f"{npz}: unfitted coordinates required to recover legacy CIF ranks")
        with np.load(npz) as data:
            coord = data["coord"]
        matches = [idx for idx, sample in enumerate(raw) if np.array_equal(coord, sample)]
        if len(matches) != 1 or matches[0] in ranks:
            raise ValueError(f"{npz}: cannot uniquely match ranked coordinates to a raw sample in {archive}")
        ranks[matches[0]] = rank
    return ranks


def _scored_cifs(stage: Path, seed: str, name: str) -> list[tuple[float, Path]]:
    seed_dir = stage / f"seed_{seed}"
    if not seed_dir.is_dir():
        return []
    scores = seed_dir / "scores.csv"
    legacy = not scores.exists()
    if legacy:
        scores = stage / "saved_data" / "scores.csv"
        if not scores.exists():
            return []
    with scores.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"pdb_id", "method", "sample_idx", "recall_ccmask_ca"}
        if not legacy:
            required.add("cif_rank")
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{scores}: incomplete or invalid score table")
        rows = [row for row in reader if row["pdb_id"] == name]
    if not rows:
        return []

    keys = [(r["method"], int(r["sample_idx"])) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{scores}: repeated (method, sample_idx) rows cannot be assigned to one prediction")
    ranks = _legacy_ranks(stage, seed_dir, name) if legacy else {}
    results = []
    for (method, idx), row in zip(keys, rows, strict=True):
        if legacy and idx not in ranks:
            raise ValueError(f"{scores}: no ranked coordinates for raw sample {idx}")
        rank = ranks[idx] if legacy else int(row["cif_rank"])
        path = seed_dir / f"predictions_{method}" / f"{name}_sample_{rank}.cif"
        results.append((float(row["recall_ccmask_ca"]), path))
    return results


@app.command()
def main(
    dump_dir: str = typer.Option(
        ...,
        help="Base output directory (contains per-entry subdirectories)",
    ),
    input_json_path: str = typer.Option(..., help="Path to input JSON file"),
    num_select: int = typer.Option(
        ..., help="Number of top results to select per entry"
    ),
    seeds: str = typer.Option(..., help="Seed value used during inference"),
):
    """Select the best results from CryoZeta and CryoZeta-Interpolate models.

    Each model's seed-local score table records the CIF rank for each fit.
    For older dumps, recover ranks from their saved raw and ranked coordinates.
    """
    with open(input_json_path) as f:
        data = json.load(f)
    for item in data:
        name = item["name"]
        entry = Path(dump_dir) / name
        candidates = []
        for model in ("CryoZeta", "CryoZeta-Interpolate"):
            candidates.extend(_scored_cifs(entry / model, seeds, name))
        if not candidates:
            typer.echo(f"Skipping {name}: no scores for seed {seeds}")
            continue

        candidates.sort(key=lambda pair: pair[0], reverse=True)
        selected = candidates[:num_select]
        for _, path in selected:
            if not path.is_file():
                raise FileNotFoundError(f"Scored CIF missing: {path}")
        final = entry / "CryoZeta-Final"
        final.mkdir(exist_ok=True)
        for i, (_, path) in enumerate(selected):
            shutil.copy(path, final / f"{name}_sample_{i}.cif")

    typer.echo(f"Ensemble selection completed. Results saved to {dump_dir}")


if __name__ == "__main__":
    app()
