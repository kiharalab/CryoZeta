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

"""Data transforms for cryo-EM support point features.

Handles parsing of pre-processed EM map data (.pt files), support point
sampling/cropping, and construction of point-pair and point-residue pair
representations used by the model.
"""

import dataclasses
import os
from collections.abc import MutableMapping

import numpy as np
import torch
from loguru import logger

from cryozeta.configs.configs_base import configs as configs_base
from cryozeta.data.data_em import (
    PointPairRepresentationEmbedding,
    pointResiduePairRepresentationEmbedding,
)
from cryozeta.utils import em_utils

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
FeatureDict = MutableMapping[str, np.ndarray]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MAP_DIR = "/home/zhan1797/scratch/output_em"
DEFAULT_MAPPING_FILE = "/scratch/gautschi/zhan1797/full_dataset_2502.csv"

# Cropping thresholds for support-point selection around ground-truth CA atoms
CROP_DISTANCE_THRESHOLD = 5.0  # Angstroms
MIN_SUPPORT_POINTS = 17
MAX_SUPPORT_POINTS = 200

# Residue feature dimensions
NUM_PROTEIN_AA = 20
NUM_ALL_AA = 28  # 20 protein + 4 DNA + 4 RNA


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class MrcObject:
    """Container for pre-processed cryo-EM map data.

    Attributes:
        file_id: Unique identifier (e.g. PDB ID) for this entry.
        ca_data: Predicted main-atom coordinates, shape (N, 3).
        aa_data: Per-point residue-type features, shape (N, 20) or (N, 28).
        cluster: Cluster IDs for each point, shape (N,).
        ca_prob: Main-atom confidence scores, shape (N,) or (N, 3).
        interpolate_confidence: Optional pairwise interpolated confidence,
            shape (N, N, 5) or (N, N, 5, 3).
    """

    file_id: str
    ca_data: torch.Tensor
    aa_data: torch.Tensor
    cluster: torch.Tensor
    ca_prob: torch.Tensor
    interpolate_confidence: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _load_mapping(mapping_file: str) -> dict:
    """Load PDB-ID -> EMDB-ID mapping from a CSV file."""
    with open(mapping_file) as f:
        lines = f.read().splitlines()
    rows = [line.split(",")[:2] for line in lines[1:]]
    return {row[0]: row[1] for row in rows}


def parse_mrc(
    file_id: str,
    n_supportpoints: int,
    em_file: str,
) -> FeatureDict:
    """Load a pre-processed EM .pt file and return sampled EM features.

    Args:
        file_id: Identifier propagated into the MrcObject.
        n_supportpoints: Number of support points to sample.
        curr_pdb_id: PDB ID used to look up the corresponding .pt file.
        map_dir: Directory containing the .pt map files.
        mapping_file: Path to the CSV that maps PDB IDs to .pt file stems.

    Returns:
        A feature dictionary with keys: ``cluster``, ``em_support_points``,
        ``aa_feature``, ``ca_prob``, and optionally ``interpolate_confidence``.

    Raises:
        ValueError: If the .pt file for *curr_pdb_id* does not exist.
    """

    if not os.path.exists(em_file):
        em_file = em_file.replace("emd_", "")
    if not os.path.exists(em_file):
        raise ValueError(f"Corresponding map file not found: {em_file}")

    mrc_data = torch.load(em_file, weights_only=False)
    # Expected keys: emdb_id, main_atom_coords, main_atom_probs,
    #                res_features, cluster_ids, interpolate_confidence

    ca_label = torch.tensor(mrc_data["main_atom_coords"])  # (N, 3)

    has_interp = (
        "interpolate_confidence" in mrc_data
        and mrc_data["interpolate_confidence"] is not None
    )
    logger.info(f"has_interpolate_confidence: {has_interp}")

    # Use all features: 20 protein + 4 DNA + 4 RNA
    aa_label = mrc_data["res_features"]  # (N, 28)
    ca_prob = mrc_data["main_atom_probs"]  # (N, 3)
    inter_confidence = (
        mrc_data["interpolate_confidence"]  # (N, N, 5, 3)
        if has_interp
        else None
    )

    cluster_id = mrc_data["cluster_ids"]  # (N,)

    mrc_object = MrcObject(
        file_id=file_id,
        ca_data=ca_label,
        aa_data=aa_label,
        cluster=cluster_id,
        interpolate_confidence=inter_confidence,
        ca_prob=ca_prob,
    )

    return make_em_features(mrc_object, n_supportpoints)


# ---------------------------------------------------------------------------
# Feature construction helpers
# ---------------------------------------------------------------------------
def make_seq_mask(protein: FeatureDict) -> FeatureDict:
    """Add ``seq_mask`` and ``pair_mask`` entries to *protein*."""
    seq_mask = torch.ones(protein["restype"].shape[0], dtype=torch.float32)
    pair_mask = (seq_mask[..., None] * seq_mask[..., None, :]).to(torch.bfloat16)

    protein["seq_mask"] = seq_mask
    protein["pair_mask"] = pair_mask
    return protein


def make_em_features(
    mrc: MrcObject,
    n_supportpoints: int,
) -> FeatureDict:
    """Sample support points from *mrc* and build a feature dictionary."""

    if n_supportpoints == -1:
        support_points = mrc.ca_data
        mask = torch.ones(mrc.ca_data.size(0), dtype=torch.bool)
    else:
        support_points, mask = em_utils.sample_support_points_v2(
            N=n_supportpoints, indices_set=mrc.ca_data
        )

    features: FeatureDict = {
        "cluster": mrc.cluster[mask],
        "em_support_points": support_points,
        "all_support_points": mrc.ca_data,
        "aa_feature": mrc.aa_data[mask],
        "ca_prob": mrc.ca_prob[mask],
    }
    if mrc.interpolate_confidence is not None:
        features["interpolate_confidence"] = mrc.interpolate_confidence[mask, ...][
            :, mask, ...
        ]
    return features


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------
def _apply_index_selection(
    indices: torch.Tensor,
    support_points: torch.Tensor,
    m_conf: torch.Tensor | None,
    cluster: torch.Tensor,
    ca_prob: torch.Tensor,
    aa_feature: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select a subset of support points (and associated tensors) by index."""
    sel_mconf = m_conf[indices][:, indices] if m_conf is not None else None
    return (
        support_points[indices],
        sel_mconf,
        cluster[indices],
        ca_prob[indices],
        aa_feature[indices],
    )


def crop_cryoem_map_v2(
    support_points: torch.Tensor,
    ca: torch.Tensor,
    m_conf: torch.Tensor | None,
    cluster: torch.Tensor,
    ca_prob: torch.Tensor,
    aa_feature: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Crop support points to the neighbourhood of ground-truth CA atoms.

    Points within ``CROP_DISTANCE_THRESHOLD`` Å of any CA atom are kept.
    If fewer than ``MIN_SUPPORT_POINTS`` remain, the first
    ``MIN_SUPPORT_POINTS`` uncropped points are returned instead.  If more
    than ``MAX_SUPPORT_POINTS`` remain, a random subset is selected.

    Returns:
        Tuple of (support_points, m_conf, cluster, ca_prob, aa_feature) after
        cropping.
    """
    logger.debug(
        f"support_points: {support_points.size()} | ca: {ca.size()} | preselect: {support_points.size(0)}",
    )

    cdist = torch.cdist(support_points, ca, p=2)
    has_nearby_ca = (cdist < CROP_DISTANCE_THRESHOLD).any(dim=-1)  # (P,)

    cropped_sp = support_points[has_nearby_ca]
    cropped_mconf = (
        m_conf[has_nearby_ca][:, has_nearby_ca] if m_conf is not None else None
    )
    cropped_cluster = cluster[has_nearby_ca]
    cropped_ca_prob = ca_prob[has_nearby_ca]
    cropped_aa = aa_feature[has_nearby_ca]

    n_selected = cropped_sp.size(0)
    logger.debug(f"postselect: {n_selected}")

    # Too few points — fall back to the first MIN_SUPPORT_POINTS uncropped
    if n_selected < MIN_SUPPORT_POINTS:
        logger.warning(
            f"Only {n_selected} support points after cropping; falling back to first {MIN_SUPPORT_POINTS}.",
            n_selected,
            MIN_SUPPORT_POINTS,
        )
        n = MIN_SUPPORT_POINTS
        fallback_mconf = m_conf[:n, :n, ...] if m_conf is not None else None
        return (
            support_points[:n],
            fallback_mconf,
            cluster[:n],
            ca_prob[:n],
            aa_feature[:n],
        )

    # Too many points — randomly subsample
    if n_selected > MAX_SUPPORT_POINTS:
        logger.debug(
            f"Too many support points ({n_selected}); subsampling to {MAX_SUPPORT_POINTS}.",
        )
        idx = torch.randperm(n_selected)[:MAX_SUPPORT_POINTS]
        return _apply_index_selection(
            idx,
            cropped_sp,
            cropped_mconf,
            cropped_cluster,
            cropped_ca_prob,
            cropped_aa,
        )

    return cropped_sp, cropped_mconf, cropped_cluster, cropped_ca_prob, cropped_aa


# ---------------------------------------------------------------------------
# Point-pair representation
# ---------------------------------------------------------------------------
def get_pointpair(
    protein: FeatureDict,
    token_array,
    atom_array,
) -> FeatureDict:
    """Build the point-pair representation and attach it to *protein*.

    If ground-truth atom positions are available (``template_all_atom_positions``
    key), the cryo-EM map is cropped to the neighbourhood of the centre atoms
    before computing embeddings.
    """
    ca = None
    if "template_all_atom_positions" in protein:
        centre_atom_indices = token_array.get_annotation("centre_atom_index")
        ca = torch.tensor(atom_array.coord[centre_atom_indices])

    support_points = protein["em_support_points"]
    m_conf = protein.get("interpolate_confidence", None)
    cluster = protein["cluster"]
    ca_prob = protein["ca_prob"]
    aa_feature = protein["aa_feature"]

    # Crop the cryo-EM map around ground-truth CA positions
    if ca is not None:
        support_points, m_conf, cluster, ca_prob, aa_feature = crop_cryoem_map_v2(
            support_points,
            ca,
            m_conf,
            cluster,
            ca_prob,
            aa_feature,
        )

    _unique_coords, inverse_indices = torch.unique(cluster, dim=0, return_inverse=True)

    p = support_points.size(0)

    w_point = PointPairRepresentationEmbedding(
        support_points,
        m_conf,
        p,
        inverse_indices,
        _unique_coords.size(0),
        ca_prob,
        use_interpolate_confidence=configs_base["use_interpolation"],
    )

    protein["p"] = w_point
    protein["em_support_points"] = support_points
    protein["em_nsup"] = torch.tensor([p])
    protein["cropped_aa"] = aa_feature

    # Clean up intermediate keys no longer needed
    for key in ("interpolate_confidence", "cluster", "ca_prob", "aa_feature"):
        protein.pop(key, None)

    return protein


# ---------------------------------------------------------------------------
# Point-residue pair representation
# ---------------------------------------------------------------------------
def get_pointresiduepair(protein: FeatureDict) -> FeatureDict:
    """Build the point-residue pair representation and attach it to *protein*.

    Constructs an amino-acid confidence matrix ``A_conf`` (protein + DNA + RNA,
    28 channels) from the cropped AA features and delegates to
    :func:`pointResiduePairRepresentationEmbedding`.
    """
    Q = protein["em_support_points"]
    p = int(protein["em_nsup"][0])
    resi = protein["seq_mask"].size(0)
    tmp_A_conf = protein["cropped_aa"]

    # 20 protein + 4 DNA + 4 RNA features
    A_conf = torch.zeros(p, NUM_ALL_AA)
    A_conf[:, :NUM_ALL_AA] = tmp_A_conf[:, :NUM_ALL_AA]

    asym_ids = protein["asym_id"]

    x = pointResiduePairRepresentationEmbedding(
        protein["restype"],
        asym_ids,
        Q,
        A_conf,
        p,
        resi,
    )
    protein["pz"] = x

    # Clean up intermediate keys
    del protein["cropped_aa"]
    del protein["em_nsup"]

    return protein
