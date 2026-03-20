#!/usr/bin/env python3
"""
Combine stage outputs by loading predicted AtomArray NPZ files and writing a single merged CIF.

The script reads NPZ file paths from stage_npz_paths.txt (written by cycle_predict.py),
which records the best registration method (teaser or svd) chosen for each stage.
Merges their chains with unique chain labels and writes a single CIF.

Can auto-discover NPZ paths from stage_npz_paths.txt written by cycle_predict.py.
"""

from __future__ import annotations

import argparse
import string
from pathlib import Path

import numpy as np
from biotite.structure import AtomArray

from cryozeta.data.utils import load_atom_array_npz, save_atoms_to_cif


def _chain_name_generator(used: set[str]):
    """Generate unique chain names (A-Z, then AA-ZZ)."""
    for c in string.ascii_uppercase:
        if c not in used:
            yield c
    for a in string.ascii_uppercase:
        for b in string.ascii_uppercase:
            nm = a + b
            if nm not in used:
                yield nm


def read_npz_paths_from_file(dump_root: Path) -> list[Path]:
    """Read NPZ file paths from stage_npz_paths.txt in dump_root."""
    npz_paths_file = dump_root / "stage_npz_paths.txt"
    if not npz_paths_file.is_file():
        raise FileNotFoundError(f"stage_npz_paths.txt not found in {dump_root}")
    with open(npz_paths_file) as f:
        return [Path(line.strip()) for line in f if line.strip()]


def read_stage_names_from_file(dump_root: Path) -> list[str]:
    """Read stage names from stage_names.txt in dump_root."""
    stage_names_file = dump_root / "stage_names.txt"
    if not stage_names_file.is_file():
        raise FileNotFoundError(f"stage_names.txt not found in {dump_root}")
    with open(stage_names_file) as f:
        return [line.strip() for line in f if line.strip()]


def _rename_chains_inplace(atom_array: AtomArray, mapping: dict[str, str]) -> None:
    """Rename chain IDs in-place according to mapping."""
    for field in ["label_asym_id", "chain_id", "auth_asym_id"]:
        vals = getattr(atom_array, field, None)
        if vals is None:
            continue
        new_vals = np.asarray(vals, dtype=str).copy()
        for old, new in mapping.items():
            new_vals[new_vals == old] = new
        atom_array.set_annotation(field, new_vals)


def combine_npz_to_cif(
    *,
    dump_root: Path,
    npz_paths: list[Path] | None = None,
    output_cif: Path,
) -> Path:
    """Combine multiple stage NPZ files into a single CIF.

    Args:
        dump_root: Root directory from cycle_predict
        npz_paths: List of NPZ file paths (auto-discovered from stage_npz_paths.txt if not provided)
        output_cif: Output CIF file path

    Returns:
        Path to the output CIF file
    """
    # Auto-discover NPZ paths if not provided
    if npz_paths is None or len(npz_paths) == 0:
        npz_paths = read_npz_paths_from_file(dump_root)

    # Load all AtomArrays
    for p in npz_paths:
        if not p.is_file():
            raise FileNotFoundError(f"NPZ not found: {p}")

    arrays = [load_atom_array_npz(str(p)) for p in npz_paths]
    print(f"Loaded {len(arrays)} arrays from: {[p.name for p in npz_paths]}")
    print(f"  Atom counts: {[arr.coord.shape[0] for arr in arrays]}")

    # Assign globally unique chain labels
    used: set[str] = set()
    name_gen = _chain_name_generator(used)

    for idx, arr in enumerate(arrays, start=1):
        # Get existing chain labels
        chain_ids = getattr(arr, "chain_id", None)
        if chain_ids is not None:
            present = np.unique(chain_ids)
            mapping = {}
            for old in present:
                new_label = next(name_gen)
                used.add(new_label)
                mapping[str(old)] = new_label
            _rename_chains_inplace(arr, mapping)
            # Sync all chain fields
            arr.set_annotation("label_asym_id", np.asarray(arr.chain_id, dtype=str))
            arr.set_annotation("auth_asym_id", np.asarray(arr.chain_id, dtype=str))
        else:
            # No chain labels: assign a fresh one
            assigned = next(name_gen)
            used.add(assigned)
            n_atoms = arr.coord.shape[0]
            for f in ["label_asym_id", "chain_id", "auth_asym_id"]:
                arr.set_annotation(f, np.array([assigned] * n_atoms, dtype=str))

        # Set unique entity ID per stage
        arr.set_annotation(
            "label_entity_id", np.array([str(idx)] * arr.coord.shape[0], dtype=str)
        )
        arr.set_annotation("copy_id", np.ones(arr.coord.shape[0], dtype=np.int32))

    # Concatenate all arrays
    coords = np.vstack([a.coord for a in arrays])
    combined = AtomArray(coords.shape[0])
    combined.coord = coords

    # Copy annotations
    fields = [
        "res_id",
        "res_name",
        "atom_name",
        "element",
        "label_asym_id",
        "auth_asym_id",
        "chain_id",
        "label_entity_id",
        "hetero",
        "is_resolved",
        "copy_id",
    ]
    string_fields = {
        "res_name",
        "atom_name",
        "element",
        "label_asym_id",
        "auth_asym_id",
        "chain_id",
        "label_entity_id",
    }

    for field in fields:
        vals = []
        for arr in arrays:
            v = getattr(arr, field, None)
            if v is None:
                vals = []
                break
            v = np.asarray(v)
            if v.ndim == 0:
                v = np.repeat(v.reshape(1), arr.coord.shape[0])
            if field in string_fields and v.dtype.kind != "U":
                v = v.astype(str)
            vals.append(v)
        if vals:
            combined.set_annotation(field, np.concatenate(vals))

    # Write CIF
    output_cif.parent.mkdir(parents=True, exist_ok=True)
    save_atoms_to_cif(
        output_cif_file=str(output_cif),
        atom_array=combined,
        entity_poly_type=None,
        pdb_id="combined",
    )
    print(f"Combined CIF written: {output_cif} ({combined.coord.shape[0]} atoms)")
    return output_cif


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Combine stage outputs by merging AtomArray NPZs into one CIF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dump_root", required=True, help="Root dir from cycle_predict")
    p.add_argument(
        "--npz_paths",
        nargs="+",
        help="NPZ file paths (auto-discovered from stage_npz_paths.txt if not provided)",
    )
    p.add_argument(
        "--output", help="Output CIF path (default: <dump_root>/combined.cif)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dump_root = Path(args.dump_root).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else dump_root / "combined.cif"
    )

    # Parse NPZ paths if provided via CLI
    npz_paths = None
    if args.npz_paths:
        npz_paths = [Path(p).expanduser().resolve() for p in args.npz_paths]

    combine_npz_to_cif(
        dump_root=dump_root,
        npz_paths=npz_paths,
        output_cif=output,
    )


if __name__ == "__main__":
    main()
