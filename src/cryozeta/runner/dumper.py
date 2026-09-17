# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Modifications Copyright 2026 KiharaLab, Purdue University.
#
# This file is included in a GPLv3-licensed project. The original
# code remains under Apache-2.0; the combined work is distributed
# under GPLv3.

import os
from pathlib import Path

import numpy as np
import torch
from biotite.structure import AtomArray
from loguru import logger

from cryozeta.data.utils import save_structure_cif
from cryozeta.utils.file_io import save_json
from cryozeta.utils.torch_utils import round_values


def get_clean_full_confidence(full_confidence_dict: dict) -> dict:
    """
    Clean and format the full confidence dictionary by removing unnecessary keys and rounding values.

    Args:
        full_confidence_dict (dict): The dictionary containing full confidence data.

    Returns:
        dict: The cleaned and formatted dictionary.
    """
    # Remove atom_coordinate
    full_confidence_dict.pop("atom_coordinate")
    # Remove atom_is_polymer
    full_confidence_dict.pop("atom_is_polymer")
    # Keep two decimal places
    full_confidence_dict = round_values(full_confidence_dict)
    return full_confidence_dict


_COORD_TYPE_MAP = {
    "coordinate": None,
    "coordinate_teaser": "teaser",
    "coordinate_svd_0.8": "svd",
    "coordinate_vesper": "vesper",
}

# full_data keys indexed along the atom dimension
_ATOM_LEVEL_FULL_DATA_KEYS = ("atom_plddt", "atom_is_polymer", "atom_coordinate")
# full_data keys indexed along the token dimension (vector entries)
_TOKEN_LEVEL_FULL_DATA_KEYS = ("token_has_frame", "token_asym_id")
# full_data keys indexed along both token dimensions (matrix entries)
_TOKEN_PAIR_FULL_DATA_KEYS = ("token_pair_pde", "token_pair_pae", "contact_probs")


def filter_full_data_by_atoms(full_data: dict, atom_keep_mask: np.ndarray) -> dict:
    """Filter a per-sample full_data dict to the kept atoms.

    Atom-level entries are indexed by the keep mask.  Tokens that lose all of
    their atoms are dropped from the token-level entries and
    ``atom_to_token_idx`` is remapped to the new token indexing.  Tensors are
    moved to CPU.

    Args:
        full_data: Per-sample full_data dict (tensors without a batch dim).
        atom_keep_mask: Boolean numpy array of shape ``(N_atom,)``.

    Returns:
        A new dict with filtered entries.
    """
    full_data = {
        key: value.cpu() if hasattr(value, "cpu") else value
        for key, value in full_data.items()
    }
    keep_atom = torch.from_numpy(atom_keep_mask)
    atom_to_token_idx = full_data["atom_to_token_idx"]
    keep_token = torch.zeros(int(atom_to_token_idx.max().item()) + 1, dtype=torch.bool)
    keep_token[atom_to_token_idx[keep_atom].long()] = True
    token_remap = torch.cumsum(keep_token.long(), dim=0) - 1

    filtered = {}
    for key, value in full_data.items():
        if key in _ATOM_LEVEL_FULL_DATA_KEYS:
            filtered[key] = value[keep_atom]
        elif key == "atom_to_token_idx":
            filtered[key] = token_remap[value[keep_atom].long()]
        elif key in _TOKEN_LEVEL_FULL_DATA_KEYS:
            filtered[key] = value[keep_token]
        elif key in _TOKEN_PAIR_FULL_DATA_KEYS:
            filtered[key] = value[keep_token][:, keep_token]
        else:
            filtered[key] = value
    return filtered


class DataDumper:
    def __init__(
        self,
        base_dir: str,
        stage_name: str = "",
        need_atom_confidence: bool = False,
        atom_arrays_dir: str | None = None,
    ):
        self.base_dir = base_dir
        self.stage_name = stage_name
        self.need_atom_confidence = need_atom_confidence
        self.atom_arrays_dir = atom_arrays_dir
        if self.atom_arrays_dir:
            os.makedirs(self.atom_arrays_dir, exist_ok=True)

    def dump(
        self,
        pdb_id: str,
        seed: int,
        pred_dict: dict,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str],
        atom_masks: list[np.ndarray] | None = None,
    ):
        """
        Dump the predictions and related data to the specified directory.

        Output is organized as ``{base_dir}/{pdb_id}/{stage_name}/seed_{seed}/``.

        Args:
            pdb_id: Sample name.
            seed: Seed index.
            pred_dict: Model prediction dict.
            atom_array: Input AtomArray (one entry per atom).
            entity_poly_type: Entity-to-poly-type mapping.
            atom_masks: Optional list of per-sample boolean arrays of shape
                ``(N_atom,)``.  When given, atoms outside the density-support
                mask are excluded from the saved structures and atom-level
                confidence data.  Samples with an all-False mask are skipped.
        """
        dump_dir = self._get_dump_dir(pdb_id, seed)
        Path(dump_dir).mkdir(parents=True, exist_ok=True)

        self.dump_predictions(
            pred_dict=pred_dict,
            dump_dir=dump_dir,
            pdb_id=pdb_id,
            seed=seed,
            atom_array=atom_array,
            entity_poly_type=entity_poly_type,
            atom_masks=atom_masks,
        )

    def _get_dump_dir(self, sample_name: str, seed: int) -> str:
        """
        Generate the directory path for dumping data.

        Layout: ``{base_dir}/{sample_name}/{stage_name}/seed_{seed}``
        """
        dump_dir = os.path.join(
            self.base_dir, sample_name, self.stage_name, f"seed_{seed}"
        )
        return dump_dir

    def dump_predictions(
        self,
        pred_dict: dict,
        dump_dir: str,
        pdb_id: str,
        seed: int,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str],
        atom_masks: list[np.ndarray] | None = None,
    ):
        """
        Dump raw predictions from the model:
            structure: Save the predicted coordinates as CIF files.
            confidence: Save the confidence data as JSON files.
        """
        # Compute sorted indices by ranking_score (descending) so sample_0 = best
        N_sample = len(pred_dict["summary_confidence"])
        sorted_indices = sorted(
            range(N_sample),
            key=lambda i: pred_dict["summary_confidence"][i]["ranking_score"],
            reverse=True,
        )

        for result in [
            "coordinate",
            "coordinate_svd_0.8",
            "coordinate_svd_0.4",
            "coordinate_teaser",
            "coordinate_vesper",
            "coordinate_superimposed",
        ]:
            prediction_save_dir = os.path.join(
                dump_dir, result.replace("coordinate", "predictions")
            )
            if result in pred_dict and pred_dict[result] is not None:
                os.makedirs(prediction_save_dir, exist_ok=True)
                self._save_structure(
                    pred_coordinates=pred_dict[result],
                    prediction_save_dir=prediction_save_dir,
                    sample_name=pdb_id,
                    seed=seed,
                    atom_array=atom_array,
                    entity_poly_type=entity_poly_type,
                    sorted_indices=sorted_indices,
                    coord_type=_COORD_TYPE_MAP.get(result),
                    atom_masks=atom_masks,
                )
            if result == "coordinate":
                self._save_confidence(
                    data=pred_dict,
                    prediction_save_dir=prediction_save_dir,
                    sample_name=pdb_id,
                    sorted_indices=sorted_indices,
                    atom_masks=atom_masks,
                )

    def _save_structure(
        self,
        pred_coordinates: torch.Tensor,
        prediction_save_dir: str,
        sample_name: str,
        seed: int,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str],
        sorted_indices: list[int] | None = None,
        coord_type: str | None = None,
        atom_masks: list[np.ndarray] | None = None,
    ):
        assert atom_array is not None
        N_sample = pred_coordinates.shape[0]

        if self.atom_arrays_dir:
            npz_dir = self.atom_arrays_dir
            npz_prefix = f"{sample_name}_seed_{seed}"
        else:
            npz_dir = os.path.join(prediction_save_dir, "atom_arrays")
            os.makedirs(npz_dir, exist_ok=True)
            npz_prefix = sample_name

        if sorted_indices is None:
            sorted_indices = list(range(N_sample))

        suffix = f"_{coord_type}" if coord_type else ""

        for rank, idx in enumerate(sorted_indices):
            sample_coords = pred_coordinates[idx]
            sample_atom_array = atom_array
            if atom_masks is not None:
                keep_mask = atom_masks[idx]
                if not keep_mask.any():
                    logger.warning(
                        f"{sample_name} sample {idx}: all atoms fall outside the "
                        f"density-support mask, skipping structure dump "
                        f"(coord_type={coord_type or 'coordinate'})"
                    )
                    continue
                keep_tensor = torch.from_numpy(keep_mask).to(sample_coords.device)
                sample_coords = sample_coords[keep_tensor]
                sample_atom_array = atom_array[keep_mask]

            output_fpath = os.path.join(
                prediction_save_dir, f"{sample_name}_sample_{rank}.cif"
            )
            save_structure_cif(
                atom_array=sample_atom_array,
                pred_coordinate=sample_coords,
                output_fpath=output_fpath,
                entity_poly_type=entity_poly_type,
                pdb_id=sample_name,
                pred_atom_array_npz_path=os.path.join(
                    npz_dir, f"{npz_prefix}_sample_{rank}{suffix}.npz"
                ),
            )

    def _save_confidence(
        self,
        data: dict,
        prediction_save_dir: str,
        sample_name: str,
        sorted_indices: list[int] | None = None,
        atom_masks: list[np.ndarray] | None = None,
    ):
        N_sample = len(data["summary_confidence"])
        for idx in range(N_sample):
            if atom_masks is not None and not atom_masks[idx].any():
                continue
            if self.need_atom_confidence:
                full_data = data["full_data"][idx]
                if not full_data:
                    continue
                if atom_masks is not None:
                    full_data = filter_full_data_by_atoms(full_data, atom_masks[idx])
                data["full_data"][idx] = get_clean_full_confidence(full_data)

        if sorted_indices is None:
            sorted_indices = list(range(N_sample))

        for rank, idx in enumerate(sorted_indices):
            if atom_masks is not None and not atom_masks[idx].any():
                logger.warning(
                    f"{sample_name} sample {idx}: all atoms fall outside the "
                    "density-support mask, skipping confidence dump"
                )
                continue
            output_fpath = os.path.join(
                prediction_save_dir,
                f"{sample_name}_summary_confidence_sample_{rank}.json",
            )
            save_json(data["summary_confidence"][idx], output_fpath, indent=4)
            if self.need_atom_confidence:
                output_fpath = os.path.join(
                    prediction_save_dir,
                    f"{sample_name}_full_data_sample_{idx}.json",
                )
                save_json(data["full_data"][idx], output_fpath, indent=None)
