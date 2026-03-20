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

import torch
from biotite.structure import AtomArray

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
    ):
        """
        Dump the predictions and related data to the specified directory.

        Output is organized as ``{base_dir}/{pdb_id}/{stage_name}/seed_{seed}/``.
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
                )
            if result == "coordinate":
                self._save_confidence(
                    data=pred_dict,
                    prediction_save_dir=prediction_save_dir,
                    sample_name=pdb_id,
                    sorted_indices=sorted_indices,
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
            output_fpath = os.path.join(
                prediction_save_dir, f"{sample_name}_sample_{rank}.cif"
            )
            save_structure_cif(
                atom_array=atom_array,
                pred_coordinate=pred_coordinates[idx],
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
    ):
        N_sample = len(data["summary_confidence"])
        for idx in range(N_sample):
            if self.need_atom_confidence:
                data["full_data"][idx] = get_clean_full_confidence(
                    data["full_data"][idx]
                )

        if sorted_indices is None:
            sorted_indices = list(range(N_sample))

        for rank, idx in enumerate(sorted_indices):
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
