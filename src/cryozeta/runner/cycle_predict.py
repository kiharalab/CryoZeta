#!/usr/bin/env python3
"""
Cycle/staged inference orchestrator with EM-point filtering between stages.

This script runs multiple inference stages sequentially. For each stage:
  1) It auto-generates stage JSONs from chains in base_json (sorted by length, descending).
  2) It builds configs like runner/inference.py and calls its main(configs).
  3) It loads the predicted structure and removes EM points near predicted CA atoms.
  4) It writes a new EM .pt and uses it for the next stage.

The pipeline automatically:
  - Extracts all chains from base_json sequences
  - Expands chains by their "count" value (e.g., count=2 becomes 2 separate stages)
  - Sorts all chain copies by sequence length (longest first)
  - Creates one stage per chain copy (stage names: <base_name>_stage_1, _stage_2, ...)
  - Stores intermediate EM files in <dump_root>/em/
  - Stores all AtomArray NPZ files in <dump_root>/atom_arrays/ (flat structure)

Output directory structure:
  <dump_root>/
    em/                  # Intermediate EM .pt files
    atom_arrays/         # All AtomArray .npz files (flat, for easy access)
    stage_jsons/         # Generated stage JSON files
    stage_names.txt      # List of stage names for combine_stages.py
    <stage_name>/        # Per-stage output (CIF files, confidence JSON)
      seed_<seed>/
        predictions/         # Base predictions (no registration)
        predictions_teaser/  # TEASER++ + GICP registered predictions
        predictions_svd/     # SVD-based registered predictions

Registration methods (mutually exclusive via CLI flags):
  - TEASER++ (--teaser_only): Uses support points and descriptor-based correspondences
  - SVD (--svd_only): Uses predicted point-residue correspondences
  - VESPER (--vesper_only): Uses the original MRC map directly (requires --vesper_map_path and --vesper_contour_level)
  - Auto (default): Compares TEASER++ and SVD, picks best by query recall

For example, if base_json has:
  - Chain A with count=2, length=600
  - Chain B with count=2, length=280

The pipeline will create 4 stages (sorted by length):
  - stage_1: Chain A (copy 1)
  - stage_2: Chain A (copy 2)
  - stage_3: Chain B (copy 1)
  - stage_4: Chain B (copy 2)

Example:
  uv run cryozeta-cycle-predict \
    --base_json examples/large_inference_example/8orj.json \
    --load_checkpoint_path /path/to/checkpoint.pt \
    --dump_root output/cycle_8orj \
    --threshold 5.0 \
    --seeds 101 \
    --sample.N_sample 2 \
    --sample.N_step 20 \
    --model.N_cycle 10 \
    --use_deepspeed_evo_attention true

Example with VESPER registration:
  uv run cryozeta-cycle-predict \
    --base_json examples/large_inference_example/8orj.json \
    --load_checkpoint_path /path/to/checkpoint.pt \
    --dump_root output/cycle_8orj \
    --vesper_only \
    --vesper_map_path /path/to/map.mrc \
    --vesper_contour_level 0.1 \
    --vesper_resolution 3.0

Notes:
  - If --use_deepspeed_evo_attention true, ensure CUTLASS_PATH is set.
  - Optionally set LAYERNORM_TYPE=fast_layernorm to compile kernels on demand.
  - VESPER requires --vesper_map_path and --vesper_contour_level when using --vesper_only.
  - VESPER uses the original MRC map directly instead of support points.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cryozeta.configs.configs_base import configs as configs_base
from cryozeta.configs.configs_data import data_configs
from cryozeta.configs.configs_inference import inference_configs
from cryozeta.configs import parse_configs
from cryozeta.data.utils import load_atom_array_npz
from cryozeta.model.modules.fitting import calculate_query_recall
from cryozeta.runner.inference import main as inference_main

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    # src/cryozeta/runner/cycle_predict.py -> repo root (4 levels up)
    return Path(__file__).resolve().parent.parent.parent.parent


def read_base_json(base_json: Path) -> dict:
    """Read and validate base JSON, return the first entry."""
    with open(base_json) as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Invalid base JSON content: {base_json}")
    return data[0]


def read_em_path_from_base_json(base_json: Path) -> Path:
    """Extract the EM file path from base JSON."""
    entry = read_base_json(base_json)
    em_path_str = entry.get("em_file")
    if not isinstance(em_path_str, str):
        raise ValueError("base_json must include an absolute 'em_file' path string")
    em_path = Path(em_path_str).expanduser().resolve()
    if not (em_path.suffix == ".pt" and em_path.is_file()):
        raise ValueError(f"EM file not found or not .pt: {em_path}")
    return em_path


def get_chain_type(seq_entry: dict) -> str | None:
    """Get the chain type key from a sequence entry."""
    for chain_type in ["proteinChain", "dnaSequence", "rnaSequence", "ligand", "ion"]:
        if chain_type in seq_entry:
            return chain_type
    return None


def get_sequence_length(seq_entry: dict) -> int:
    """Get the sequence length from a sequence entry."""
    for chain_type in ["proteinChain", "dnaSequence", "rnaSequence"]:
        if chain_type in seq_entry:
            seq = seq_entry[chain_type].get("sequence", "")
            return len(seq)
    # For ligand/ion, return 0 (they don't have meaningful length)
    return 0


def get_chain_count(seq_entry: dict) -> int:
    """Get the count (number of copies) from a sequence entry."""
    chain_type = get_chain_type(seq_entry)
    if chain_type is None:
        return 1
    return seq_entry[chain_type].get("count", 1)


def make_single_copy_entry(seq_entry: dict) -> dict:
    """Create a copy of seq_entry with count=1."""
    import copy

    new_entry = copy.deepcopy(seq_entry)
    chain_type = get_chain_type(new_entry)
    if chain_type is not None:
        new_entry[chain_type]["count"] = 1
    return new_entry


def extract_chains_sorted_by_length(base_json: Path) -> list[tuple[int, int, dict]]:
    """
    Extract all chain entries from base_json sequences, expanding by count.

    Each chain with count=N becomes N separate entries (one per copy).
    Returns list of (original_seq_index, copy_index, seq_entry_with_count_1)
    sorted by sequence length descending.

    For example, if sequences has:
      - seq 0 with count=2, length=600
      - seq 1 with count=2, length=280

    Returns 4 entries (sorted by length desc):
      - (0, 0, seq_entry_count_1)  # first copy of seq 0
      - (0, 1, seq_entry_count_1)  # second copy of seq 0
      - (1, 0, seq_entry_count_1)  # first copy of seq 1
      - (1, 1, seq_entry_count_1)  # second copy of seq 1
    """
    entry = read_base_json(base_json)
    sequences = entry.get("sequences", [])

    # Expand chains by count
    # Each tuple: (original_seq_index, copy_index, seq_entry_with_count_1, length)
    expanded_chains: list[tuple[int, int, dict, int]] = []

    for seq_idx, seq_entry in enumerate(sequences):
        length = get_sequence_length(seq_entry)
        count = get_chain_count(seq_entry)

        for copy_idx in range(count):
            single_copy_entry = make_single_copy_entry(seq_entry)
            expanded_chains.append((seq_idx, copy_idx, single_copy_entry, length))

    # Sort by length descending
    expanded_chains.sort(key=lambda x: -x[3])

    # Return (seq_idx, copy_idx, seq_entry) tuples
    return [
        (seq_idx, copy_idx, seq_entry)
        for seq_idx, copy_idx, seq_entry, _ in expanded_chains
    ]


def generate_stage_json(
    *,
    base_json: Path,
    seq_entry: dict,
    stage_index: int,
    out_dir: Path,
    em_pt: Path,
) -> tuple[Path, str]:
    """
    Generate a stage JSON with a single chain (count=1).
    Returns (json_path, stage_name).

    Note: seq_entry should already have count=1 (via make_single_copy_entry).
    """
    entry = read_base_json(base_json)
    base_name = entry.get("name", "unnamed")
    stage_name = f"{base_name}_stage_{stage_index}"

    stage_entry = {
        "name": stage_name,
        "em_file": str(em_pt.resolve()),
        "modelSeeds": entry.get("modelSeeds", []),
        "assembly_id": entry.get("assembly_id", "1"),
        "sequences": [seq_entry],
    }

    for key in ("map_path", "resolution", "contour_level"):
        if key in entry:
            stage_entry[key] = entry[key]

    # Preserve covalent_bonds if present and relevant to this chain
    if "covalent_bonds" in entry:
        stage_entry["covalent_bonds"] = entry["covalent_bonds"]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stage_name}.json"
    with open(out_path, "w") as f:
        json.dump([stage_entry], f, indent=2)

    return out_path, stage_name


def build_arg_str(
    *,
    seeds: str,
    checkpoint: Path,
    dump_dir: Path,
    input_json: Path,
    use_deepspeed_evo_attention: str,
    use_cuequivariance_attention: str,
    use_cuequivariance_multiplicative_update: str,
    use_cuequivariance_attention_pair_bias: str,
    use_opm_tilelang: str,
    n_cycle: int,
    n_sample: int,
    n_step: int,
    num_dl_workers: int,
) -> str:
    parts: list[str] = []
    parts += ["--seeds", str(seeds)]
    parts += ["--load_checkpoint_path", str(checkpoint)]
    parts += ["--dump_dir", str(dump_dir)]
    parts += ["--input_json_path", str(input_json)]
    parts += ["--use_deepspeed_evo_attention", str(use_deepspeed_evo_attention)]
    parts += ["--use_cuequivariance_attention", str(use_cuequivariance_attention)]
    parts += ["--use_cuequivariance_multiplicative_update", str(use_cuequivariance_multiplicative_update)]
    parts += ["--use_cuequivariance_attention_pair_bias", str(use_cuequivariance_attention_pair_bias)]
    parts += ["--use_opm_tilelang", str(use_opm_tilelang)]
    parts += ["--model.N_cycle", str(n_cycle)]
    parts += ["--sample_diffusion.N_sample", str(n_sample)]
    parts += ["--sample_diffusion.N_step", str(n_step)]
    parts += ["--data.num_dl_workers", str(num_dl_workers)]
    parts += ["--use_interpolation", "false"]
    parts += ["--overwrite", "true"]
    return " ".join(parts)


def run_stage(
    *,
    stage_name: str,
    temp_json: Path,
    dump_dir: Path,
    atom_arrays_dir: Path,
    args: argparse.Namespace,
    vesper_configs: dict | None = None,
) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    atom_arrays_dir.mkdir(parents=True, exist_ok=True)
    configs: Any = {**configs_base, **{"data": data_configs}, **inference_configs}
    arg_str = build_arg_str(
        seeds=args.seeds,
        checkpoint=Path(args.load_checkpoint_path).expanduser().resolve(),
        dump_dir=dump_dir,
        input_json=temp_json,
        use_deepspeed_evo_attention=str(args.use_deepspeed_evo_attention).lower(),
        use_cuequivariance_attention=str(args.use_cuequivariance_attention).lower(),
        use_cuequivariance_multiplicative_update=str(args.use_cuequivariance_multiplicative_update).lower(),
        use_cuequivariance_attention_pair_bias=str(args.use_cuequivariance_attention_pair_bias).lower(),
        use_opm_tilelang=str(args.use_opm_tilelang).lower(),
        n_cycle=int(args.model_N_cycle),
        n_sample=int(args.sample_N_sample),
        n_step=int(args.sample_N_step),
        num_dl_workers=int(args.num_dl_workers),
    )
    configs = parse_configs(
        configs=configs,
        arg_str=arg_str,
        fill_required_with_null=True,
    )
    # Sync the module-level configs_base dict so the data pipeline
    # (data_transforms_em.PointPairRepresentationEmbedding) produces
    # features with the correct p_dim.
    configs_base["use_interpolation"] = configs.use_interpolation
    if not configs.use_interpolation:
        configs.model.input_embedder.p_dim = 99
    configs.atom_arrays_dir = str(atom_arrays_dir)
    configs.vesper = vesper_configs
    logger.info(f"[Stage {stage_name}] Starting inference: {temp_json}")
    inference_main(configs)
    logger.info(f"[Stage {stage_name}] Inference done. Dump: {dump_dir}")


def get_stage_atomarray_npz(
    *,
    atom_arrays_dir: Path,
    stage_name: str,
    seed: str | int,
    sample_index: int,
    coord_type: str | None = None,
) -> Path:
    """Get the path to a stage's AtomArray NPZ file.

    With the flat atom_arrays directory structure, the path is deterministic:
      atom_arrays_dir / {stage_name}_seed_{seed}_sample_{sample_index}[_{coord_type}].npz

    Args:
        atom_arrays_dir: Directory containing atom array NPZ files
        stage_name: Name of the stage
        seed: Random seed used
        sample_index: Sample index (0-based)
        coord_type: Coordinate type suffix - None for base, "teaser", or "svd"
    """
    suffix = f"_{coord_type}" if coord_type else ""
    return (
        atom_arrays_dir / f"{stage_name}_seed_{seed}_sample_{sample_index}{suffix}.npz"
    )


def extract_pick_mask_coords(arr) -> np.ndarray | None:
    """Extract CA/C4'/P coordinates from an AtomArray for recall calculation.

    Prefers protein CA atoms; falls back to C4' or P if no CA (nucleic acids).

    Args:
        arr: AtomArray loaded from NPZ

    Returns:
        Coordinates of picked atoms as numpy array, or None if invalid
    """
    atom_name = getattr(arr, "atom_name", None)
    hetero = getattr(arr, "hetero", None)
    if atom_name is None or arr.coord is None:
        return None

    name_arr = np.array(atom_name)
    if hetero is not None:
        polymer_mask = np.logical_not(np.array(hetero))
    else:
        polymer_mask = np.ones(len(name_arr), dtype=bool)

    ca_mask = (name_arr == "CA") & polymer_mask
    if not np.any(ca_mask):
        c4_mask = (name_arr == "C4'") & polymer_mask
        p_mask = (name_arr == "P") & polymer_mask
        pick_mask = c4_mask if np.any(c4_mask) else p_mask
    else:
        pick_mask = ca_mask

    return arr.coord[pick_mask]


def _score_npz(
    npz_path: Path,
    support_points: np.ndarray,
    distance_threshold: float,
) -> float:
    """Calculate query recall for a single NPZ file."""
    arr = load_atom_array_npz(str(npz_path))
    coords = extract_pick_mask_coords(arr)
    if coords is not None and len(coords) > 0:
        return calculate_query_recall(support_points, coords, distance_threshold)
    return 0.0


def compare_registration_methods(
    *,
    atom_arrays_dir: Path,
    stage_name: str,
    seed: str | int,
    sample_index: int,
    support_points: np.ndarray,
    distance_threshold: float = 3.0,
    registration_method: str = "auto",
) -> tuple[str, Path, dict]:
    """Compare TEASER++ and SVD registration methods and choose the best one.

    Uses query recall (percentage of support points covered) to rank methods.

    Args:
        atom_arrays_dir: Directory containing atom array NPZ files
        stage_name: Name of the stage
        seed: Random seed used
        sample_index: Sample index
        support_points: Reference EM support points for recall calculation
        distance_threshold: Distance threshold for recall calculation
        registration_method: "auto" to compare both and pick best,
                             "teaser" or "svd" to use one only

    Returns:
        Tuple of (best_method, best_npz_path, metrics_dict)
    """
    import sys

    metrics: dict = {}

    teaser_path = get_stage_atomarray_npz(
        atom_arrays_dir=atom_arrays_dir,
        stage_name=stage_name,
        seed=seed,
        sample_index=sample_index,
        coord_type="teaser",
    )
    svd_path = get_stage_atomarray_npz(
        atom_arrays_dir=atom_arrays_dir,
        stage_name=stage_name,
        seed=seed,
        sample_index=sample_index,
        coord_type="svd",
    )

    if registration_method == "teaser":
        if not teaser_path.is_file():
            logger.error(f"TEASER++ NPZ file not found: {teaser_path}")
            sys.exit(1)
        recall = _score_npz(teaser_path, support_points, distance_threshold)
        metrics["teaser_query_recall"] = recall
        metrics["best_method"] = "teaser"
        metrics["best_recall"] = recall
        return "teaser", teaser_path, metrics

    if registration_method == "svd":
        if not svd_path.is_file():
            logger.error(f"SVD NPZ file not found: {svd_path}")
            sys.exit(1)
        recall = _score_npz(svd_path, support_points, distance_threshold)
        metrics["svd_query_recall"] = recall
        metrics["best_method"] = "svd"
        metrics["best_recall"] = recall
        return "svd", svd_path, metrics

    # Auto mode: compare both methods
    if not teaser_path.is_file():
        logger.error(f"TEASER++ NPZ file not found: {teaser_path}")
        sys.exit(1)

    svd_exists = svd_path.is_file()
    if not svd_exists:
        logger.warning(
            f"SVD NPZ not found: {svd_path} (using TEASER++ only)"
        )

    best_method = "teaser"
    best_path = teaser_path
    best_recall = -1.0

    teaser_recall = _score_npz(teaser_path, support_points, distance_threshold)
    metrics["teaser_query_recall"] = teaser_recall
    if teaser_recall > best_recall:
        best_recall = teaser_recall
        best_method = "teaser"
        best_path = teaser_path

    if svd_exists:
        svd_recall = _score_npz(svd_path, support_points, distance_threshold)
        metrics["svd_query_recall"] = svd_recall
        if svd_recall > best_recall:
            best_recall = svd_recall
            best_method = "svd"
            best_path = svd_path
    else:
        metrics["svd_query_recall"] = None

    metrics["best_method"] = best_method
    metrics["best_recall"] = best_recall

    return best_method, best_path, metrics


def crop_em_points(
    *,
    prev_em_pt: Path,
    threshold: float,
    out_em_pt: Path,
    best_npz_path: Path,
) -> Path:
    """Remove EM support points within 'threshold' Å from predicted CA atoms.

    Uses the provided NPZ file (already selected by compare_registration_methods)
    to determine which EM points to remove.

    Args:
        prev_em_pt: Previous EM .pt file path
        threshold: Distance threshold for cropping EM points
        out_em_pt: Output EM .pt file path
        best_npz_path: Path to the selected NPZ file (from compare_registration_methods)

    Returns:
        Path to the cropped EM .pt file
    """
    # Load EM data
    em = torch.load(prev_em_pt, weights_only=False, map_location="cpu")

    def _as_tensor(x):
        return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)

    coords = _as_tensor(em["main_atom_coords"]).to(torch.float32)  # [L, 3]

    # Load the NPZ for cropping
    if not best_npz_path.is_file():
        raise FileNotFoundError(f"AtomArray NPZ not found: {best_npz_path}")

    arr = load_atom_array_npz(str(best_npz_path))
    pick_coords = extract_pick_mask_coords(arr)
    if pick_coords is None:
        raise ValueError(f"Invalid AtomArray NPZ content: {best_npz_path}")
    ca_coordinate = torch.as_tensor(pick_coords, dtype=torch.float32)

    # Extract EM tensors for cropping
    m_probs = _as_tensor(em["main_atom_probs"])  # [L, ...]
    r_feats = _as_tensor(em["res_features"])  # [L, F]
    clusters = _as_tensor(em["cluster_ids"])  # [L]

    # Compute min distance from each EM point to any CA coord
    with torch.no_grad():
        dists = torch.cdist(coords, ca_coordinate.to(coords.dtype))
        keep = (dists >= float(threshold)).all(dim=1)  # [L]
        keep_idx = torch.where(keep)[0]

    if keep_idx.numel() == 0:
        # Fallback: keep first 17 points to satisfy downstream min size
        n = min(17, coords.shape[0])
        keep_idx = torch.arange(n)

    new_em = {
        "main_atom_coords": coords[keep_idx].cpu(),
        "main_atom_probs": m_probs[keep_idx].cpu(),
        "res_features": r_feats[keep_idx].cpu(),
        "cluster_ids": clusters[keep_idx].cpu(),
    }

    out_em_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_em, out_em_pt)
    logger.info(
        f"Cropped EM saved: {out_em_pt} (kept {keep_idx.numel()} / {coords.shape[0]} points)"
    )
    return out_em_pt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run staged inference with EM filtering between stages (auto-generates stages from chains)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--base_json",
        required=True,
        help="Full JSON with absolute EM .pt path and all chains",
    )
    p.add_argument("--load_checkpoint_path", required=True)
    p.add_argument("--dump_root", default=str(_repo_root() / "output" / "cycle"))
    p.add_argument("--threshold", type=float, default=5.0)
    p.add_argument("--seeds", default="101")
    p.add_argument("--sample.N_sample", dest="sample_N_sample", type=int, default=2)
    p.add_argument("--sample.N_step", dest="sample_N_step", type=int, default=20)
    p.add_argument("--model.N_cycle", dest="model_N_cycle", type=int, default=10)
    p.add_argument("--use_deepspeed_evo_attention", default="true")
    p.add_argument("--use_cuequivariance_attention", default="true")
    p.add_argument("--use_cuequivariance_multiplicative_update", default="true")
    p.add_argument("--use_cuequivariance_attention_pair_bias", default="false")
    p.add_argument("--use_opm_tilelang", default="false")
    p.add_argument("--num_dl_workers", type=int, default=1)
    p.add_argument("--crop.sample_index", dest="crop_sample_index", type=int, default=0)

    # Registration method selection (mutually exclusive)
    reg_group = p.add_mutually_exclusive_group()
    reg_group.add_argument(
        "--svd_only",
        action="store_true",
        help="Use SVD registration only, skip comparison",
    )
    reg_group.add_argument(
        "--teaser_only",
        action="store_true",
        help="Use TEASER++ registration only, skip comparison",
    )
    reg_group.add_argument(
        "--vesper_only",
        action="store_true",
        help="Use VESPER registration only, skip comparison",
    )

    # VESPER-specific arguments (required when using --vesper_only or auto mode with VESPER)
    p.add_argument(
        "--vesper_map_path",
        type=str,
        default=None,
        help="Path to original MRC map for VESPER alignment (required for VESPER)",
    )
    p.add_argument(
        "--vesper_contour_level",
        type=float,
        default=None,
        help="Contour level for VESPER alignment (required for VESPER)",
    )
    p.add_argument(
        "--vesper_resolution",
        type=float,
        default=3.0,
        help="Resolution for simulating map from structure in VESPER (default: 3.0)",
    )
    p.add_argument(
        "--vesper_angle_spacing",
        type=float,
        default=5.0,
        help="Angular sampling interval in degrees for VESPER (default: 5.0)",
    )
    p.add_argument(
        "--vesper_num_threads",
        type=int,
        default=8,
        help="Number of CPU threads for VESPER (default: 8)",
    )

    return p.parse_args()


def main() -> None:
    LOG_FORMAT = (
        "%(asctime)s,%(msecs)-3d %(levelname)-8s "
        "[%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    )
    logging.basicConfig(
        format=LOG_FORMAT,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )

    args = parse_args()
    base_json = Path(args.base_json).expanduser().resolve()
    dump_root = Path(args.dump_root).expanduser().resolve()

    # Determine registration method from args
    vesper_configs: dict | None = None
    if args.svd_only:
        registration_method = "svd"
        logger.info("Registration method: SVD only (--svd_only)")
    elif args.teaser_only:
        registration_method = "teaser"
        logger.info("Registration method: TEASER++ only (--teaser_only)")
    elif args.vesper_only:
        registration_method = "vesper"
        logger.info("Registration method: VESPER only (--vesper_only)")
        # Validate VESPER arguments
        if args.vesper_map_path is None:
            raise ValueError("--vesper_map_path is required when using --vesper_only")
        if args.vesper_contour_level is None:
            raise ValueError("--vesper_contour_level is required when using --vesper_only")
        vesper_map_path = Path(args.vesper_map_path).expanduser().resolve()
        if not vesper_map_path.is_file():
            raise ValueError(f"VESPER map file not found: {vesper_map_path}")
        # Build VESPER configs dict to pass to inference
        vesper_configs = {
            "enabled": True,
            "map_path": str(vesper_map_path),
            "contour_level": args.vesper_contour_level,
            "resolution": args.vesper_resolution,
            "angle_spacing": args.vesper_angle_spacing,
            "num_threads": args.vesper_num_threads,
            "gpu_id": 0,
            "num_conformation": 10,
            "voxel_spacing": 2.0,
            "gaussian_bandwidth": 1.0,
        }
        logger.info(f"VESPER map path: {vesper_map_path}")
        logger.info(f"VESPER contour level: {args.vesper_contour_level}")
        logger.info(f"VESPER resolution: {args.vesper_resolution}")
    else:
        registration_method = "auto"
        logger.info("Registration method: Auto (compare TEASER++ and SVD, pick best)")

    # EM files are stored in dump_root/em/
    em_dir = dump_root / "em"
    em_dir.mkdir(parents=True, exist_ok=True)

    # AtomArray NPZ files are stored in dump_root/atom_arrays/ (flat structure)
    atom_arrays_dir = dump_root / "atom_arrays"
    atom_arrays_dir.mkdir(parents=True, exist_ok=True)

    # Temporary stage JSONs stored in dump_root/stage_jsons/
    stage_json_dir = dump_root / "stage_jsons"
    stage_json_dir.mkdir(parents=True, exist_ok=True)

    initial_em_pt = read_em_path_from_base_json(base_json)
    em_stem = initial_em_pt.stem  # e.g., 17127
    current_em_pt = initial_em_pt

    # Extract chains sorted by length (longest first), expanded by count
    chains = extract_chains_sorted_by_length(base_json)
    total_chains = len(chains)
    logger.info(
        f"Found {total_chains} chain copies (expanded by count), sorted by sequence length (descending)"
    )

    # Track stage names and NPZ paths for combine_stages
    stage_names: list[str] = []
    stage_npz_paths: list[Path | None] = []

    for stage_idx, (seq_idx, copy_idx, seq_entry) in enumerate(chains, start=1):
        seq_len = get_sequence_length(seq_entry)
        logger.info(
            f"Processing stage {stage_idx}/{total_chains}: seq_index={seq_idx}, copy={copy_idx}, length={seq_len}"
        )

        # Generate stage JSON
        temp_json, stage_name = generate_stage_json(
            base_json=base_json,
            seq_entry=seq_entry,
            stage_index=stage_idx,
            out_dir=stage_json_dir,
            em_pt=current_em_pt,
        )
        stage_names.append(stage_name)

        logger.info(f"[Stage {stage_name}] JSON: {temp_json}")
        logger.info(f"[Stage {stage_name}] Running inference...")

        # Run stage inference
        run_stage(
            stage_name=stage_name,
            temp_json=temp_json,
            dump_dir=dump_root,
            atom_arrays_dir=atom_arrays_dir,
            args=args,
            vesper_configs=vesper_configs,
        )
        logger.info(f"[Stage {stage_name}] Inference done")

        # For VESPER mode, use VESPER NPZ files directly without comparison
        if registration_method == "vesper":
            # VESPER registration is done inside the model, use _vesper NPZ files
            best_npz_path = None
            best_recall = 0.0
            em = torch.load(current_em_pt, weights_only=False, map_location="cpu")
            support_points_np = (
                em["main_atom_coords"].cpu().numpy()
                if isinstance(em["main_atom_coords"], torch.Tensor)
                else em["main_atom_coords"]
            )
            for sample_index in range(args.sample_N_sample):
                vesper_npz_path = get_stage_atomarray_npz(
                    atom_arrays_dir=atom_arrays_dir,
                    stage_name=stage_name,
                    seed=args.seeds,
                    sample_index=sample_index,
                    coord_type="vesper",
                )
                if vesper_npz_path.is_file():
                    arr = load_atom_array_npz(str(vesper_npz_path))
                    coords = extract_pick_mask_coords(arr)
                    if coords is not None and len(coords) > 0:
                        query_recall = calculate_query_recall(
                            support_points_np, coords, 3.0
                        )
                        if query_recall > best_recall:
                            best_recall = query_recall
                            best_npz_path = vesper_npz_path
            stage_npz_paths.append(best_npz_path)
            logger.info(
                f"[Stage {stage_name}] Using VESPER only (recall: {best_recall:.4f})"
            )
        else:
            # Compare TEASER++ and SVD registration methods
            em = torch.load(current_em_pt, weights_only=False, map_location="cpu")
            support_points_np = (
                em["main_atom_coords"].cpu().numpy()
                if isinstance(em["main_atom_coords"], torch.Tensor)
                else em["main_atom_coords"]
            )

            best_npz_path = None
            best_recall = 0.0
            for sample_index in range(args.sample_N_sample):
                _, npz_path, metrics = compare_registration_methods(
                    atom_arrays_dir=atom_arrays_dir,
                    stage_name=stage_name,
                    seed=args.seeds,
                    sample_index=sample_index,
                    support_points=support_points_np,
                    distance_threshold=3.0,
                    registration_method=registration_method,
                )
                if metrics['best_recall'] > best_recall:
                    best_recall = metrics['best_recall']
                    best_npz_path = npz_path

            stage_npz_paths.append(best_npz_path)
            if registration_method == "auto":
                logger.info(f"[Stage {stage_name}] Registration method comparison:")
                logger.info(
                    f"  TEASER++ query recall: {metrics.get('teaser_query_recall', 0.0):.4f}"
                )
                svd_recall = metrics.get('svd_query_recall')
                if svd_recall is None:
                    logger.info("  SVD query recall: N/A (SVD registration failed)")
                else:
                    logger.info(f"  SVD query recall: {svd_recall:.4f}")
                logger.info(
                    f"  >>> Selected method: {metrics['best_method'].upper()} (recall: {metrics['best_recall']:.4f})"
                )
            else:
                logger.info(
                    f"[Stage {stage_name}] Using {registration_method.upper()} only (recall: {metrics['best_recall']:.4f})"
                )

        # Crop EM points for next stage (skip for last stage)
        if stage_idx < total_chains:
            if best_npz_path is None:
                logger.error(f"[Stage {stage_name}] No valid NPZ path found, cannot crop EM points")
                raise ValueError(f"Registration failed for stage {stage_name}")
            next_em_pt = em_dir / f"{em_stem}_stage_{stage_idx}.pt"
            logger.info(f"[Stage {stage_name}] Cropping EM points -> {next_em_pt}")
            crop_em_points(
                prev_em_pt=current_em_pt,
                threshold=float(args.threshold),
                out_em_pt=next_em_pt,
                best_npz_path=best_npz_path,
            )
            current_em_pt = next_em_pt

    # Write stage names to a file for combine_stages.py
    stage_names_file = dump_root / "stage_names.txt"
    with open(stage_names_file, "w") as f:
        f.write("\n".join(stage_names))
    logger.info(f"Stage names written to: {stage_names_file}")

    # Write NPZ paths for combine_stages.py (records which registration method was used)
    npz_paths_file = dump_root / "stage_npz_paths.txt"
    with open(npz_paths_file, "w") as f:
        f.write("\n".join(str(p) for p in stage_npz_paths))
    logger.info(f"Stage NPZ paths written to: {npz_paths_file}")

    logger.info(f"All {total_chains} stages completed. Stage names: {stage_names}")


if __name__ == "__main__":
    main()
