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

"""Density-support mask generation and prediction filtering.

Builds a "soft" macromolecule mask from the experimental map following the
author-recommended contouring strategy (independently implemented; inspired by
the masking approach used by EMProt/EMProtenix):

1. Seed support: voxels with density at or above the author-recommended
   contour level.  When no contour level is available, positive density
   (``> 0``) is used instead.
2. Gaussian blur: the binary seed support is smoothed with a 3-D Gaussian
   filter (sigma in voxel units, zero padding).
3. Re-binarize: Li's minimum cross-entropy threshold is recomputed on the
   blurred support values and applied back to the volume.
4. Cleanup: connected components smaller than a minimum size are removed.

Predicted atoms are then filtered by sampling the mask at their predicted
coordinates: atoms outside the mask are excluded from the output structures.
Filtering is residue-grained: a polymer residue is kept when its centre atom
(CA, C4', P, ...) falls inside the mask, and a non-polymer residue (ligand or
ion) is kept when any of its atoms falls inside the mask.
"""

import numpy as np
import scipy.ndimage as ndi
from biotite.structure import AtomArray
from loguru import logger

from .map import MapObject, parse_mrc

_LI_TOLERANCE = 1e-5
_LI_MAX_ITERATIONS = 100000


def compute_li_threshold(
    values: np.ndarray,
    tolerance: float = _LI_TOLERANCE,
    max_iterations: int = _LI_MAX_ITERATIONS,
) -> float:
    """Compute Li's minimum cross-entropy threshold for 1-D values.

    Iteratively solves ``t = (m_background - m_foreground) /
    (log(m_background) - log(m_foreground))`` where the means are taken over
    the voxels below/above the current threshold (Li & Lee, 1993).

    Args:
        values: 1-D array of (positive) support values.
        tolerance: Stopping tolerance on the threshold update.
        max_iterations: Maximum number of iterations.

    Returns:
        The threshold in the same intensity scale as ``values``.  Returns 0.0
        for empty input and the constant value itself for constant input.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    if np.all(values == values[0]):
        return float(values[0])

    values_min = values.min()
    values = values - values_min

    t_next = float(values.mean())
    t_curr = t_next - 2.0 * tolerance
    for _ in range(max_iterations):
        if abs(t_next - t_curr) <= tolerance:
            break
        t_curr = t_next
        foreground = values > t_curr
        background = ~foreground
        if not foreground.any() or not background.any():
            break
        mean_fore = values[foreground].mean()
        mean_back = values[background].mean()
        if mean_back <= 0.0 or mean_fore <= 0.0:
            break
        log_diff = np.log(mean_back) - np.log(mean_fore)
        if abs(log_diff) < 1e-15:
            break
        t_next = (mean_back - mean_fore) / log_diff

    return t_next + values_min


def gaussian_blur_binary_mask(
    mask: np.ndarray,
    sigma: float,
    truncate: float = 4.0,
) -> np.ndarray:
    """Blur a binary 3-D mask with a separable Gaussian filter.

    Zero padding is used so the support does not bleed in from outside the
    volume.

    Args:
        mask: 3-D binary (boolean) mask.
        sigma: Gaussian sigma in voxel units.
        truncate: Filter half-width in units of ``sigma``.

    Returns:
        Float32 blurred volume with the same shape as ``mask``.

    Raises:
        ValueError: If ``mask`` is not 3-D or ``sigma`` is not positive.
    """
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3D, got shape {tuple(mask.shape)}")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return ndi.gaussian_filter(
        mask.astype(np.float32, copy=False),
        sigma=float(sigma),
        truncate=float(truncate),
        mode="constant",
        cval=0.0,
    )


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected components smaller than ``min_size`` voxels.

    Uses 3-D connectivity 1 (face-adjacent voxels only).

    Args:
        mask: 3-D boolean mask.
        min_size: Minimum component size to keep.  ``<= 0`` disables removal.

    Returns:
        Boolean mask with small components removed.
    """
    if min_size <= 0:
        return mask.astype(dtype=bool, copy=False)
    structure = ndi.generate_binary_structure(rank=3, connectivity=1)
    labeled, _ = ndi.label(mask, structure=structure)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # background is not a component
    return sizes[labeled] >= min_size


def build_density_support_mask(
    grid_data: np.ndarray,
    contour_level: float | None = None,
    sigma: float = 5.0,
    min_component_size: int = 1024,
    truncate: float = 4.0,
) -> np.ndarray:
    """Build the density-support mask from map grid data.

    See module docstring for the algorithm (contour seed → Gaussian blur →
    Li re-threshold → small-component removal).  If the denoised mask ends up
    empty, the raw contour support is returned as a fallback.

    Args:
        grid_data: 3-D density array as parsed by
            :func:`cryozeta.em.map.parse_mrc`.
        contour_level: Author-recommended contour level in the map's native
            intensity scale.  ``None`` or non-positive falls back to positive
            density as the seed support.
        sigma: Gaussian sigma in voxel units for the blur step.
        min_component_size: Minimum connected-component size to keep.  ``0``
            disables component filtering.
        truncate: Filter half-width in units of ``sigma``.

    Returns:
        Boolean mask with the same shape as ``grid_data``.
    """
    density = np.asarray(grid_data, dtype=np.float32)
    if contour_level is not None and contour_level > 0:
        seed = density >= np.float32(contour_level)
    else:
        seed = density > 0

    if not seed.any():
        return np.zeros(density.shape, dtype=bool)

    blurred = gaussian_blur_binary_mask(seed, sigma=sigma, truncate=truncate)
    threshold = compute_li_threshold(blurred[blurred > 0])
    mask = blurred >= np.float32(threshold)
    if not mask.any():
        logger.warning(
            "Gaussian-blurred support re-thresholded to an empty mask; "
            "falling back to the raw contour support."
        )
        mask = seed
    return remove_small_components(mask, min_component_size)


def build_mask_from_map_file(
    map_path: str,
    contour_level: float | None = None,
    sigma: float = 5.0,
    min_component_size: int = 1024,
) -> tuple[np.ndarray, MapObject]:
    """Build the density-support mask from a map file.

    The map is parsed at its native grid (voxel size and origin are preserved
    so that predicted coordinates in the map's physical frame can be sampled
    directly).  Gzipped MRC files are supported transparently.

    Args:
        map_path: Path to the MRC map file.
        contour_level: Author-recommended contour level (native intensity
            scale).  ``None`` falls back to positive density.
        sigma: Gaussian sigma in voxel units.
        min_component_size: Minimum connected-component size to keep.

    Returns:
        Tuple of (boolean mask grid, parsed MapObject).

    Raises:
        ValueError: If the map is not orthogonal (raised by ``parse_mrc``).
    """
    map_obj = parse_mrc(map_path)
    mask = build_density_support_mask(
        map_obj.grid_data,
        contour_level=contour_level,
        sigma=sigma,
        min_component_size=min_component_size,
    )
    return mask, map_obj


def sample_mask_at_coords(
    mask: np.ndarray,
    map_obj: MapObject,
    coords: np.ndarray,
) -> np.ndarray:
    """Sample the mask at physical (Angstrom) coordinates.

    ``parse_mrc`` returns ``grid_data`` indexed as ``[z, y, x]`` while
    ``voxel_size`` and ``global_origin`` are in ``(x, y, z)`` order, matching
    the convention used throughout the detection pipeline
    (``physical = index[::-1] * voxel_size + origin``).

    Args:
        mask: 3-D boolean mask grid (indexed ``[z, y, x]``).
        map_obj: Map object providing ``global_origin`` and ``voxel_size``.
        coords: ``(N, 3)`` array of xyz coordinates in Angstroms.

    Returns:
        Boolean array of length ``N``; coordinates outside the grid bound
        sample as ``False``.
    """
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    origin = np.asarray(map_obj.global_origin, dtype=np.float64)
    voxel_size = np.asarray(map_obj.voxel_size, dtype=np.float64)

    index_xyz = np.rint((coords - origin) / voxel_size).astype(np.int64)
    x_idx, y_idx, z_idx = index_xyz[:, 0], index_xyz[:, 1], index_xyz[:, 2]
    nz, ny, nx = mask.shape

    in_bounds = (
        (x_idx >= 0)
        & (x_idx < nx)
        & (y_idx >= 0)
        & (y_idx < ny)
        & (z_idx >= 0)
        & (z_idx < nz)
    )

    inside = np.zeros(coords.shape[0], dtype=bool)
    inside[in_bounds] = mask[z_idx[in_bounds], y_idx[in_bounds], x_idx[in_bounds]]
    return inside


def _build_residue_groups(
    atom_array: AtomArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group atoms by residue and classify the groups.

    Args:
        atom_array: Biotite AtomArray of the predicted structure input.

    Returns:
        Tuple of:
            - ``group_id``: ``(N_atom,)`` integer residue-group id per atom.
            - ``is_polymer_group``: ``(n_groups,)`` whether the group is a
              polymer residue (protein/DNA/RNA) as opposed to a ligand or ion.
            - ``is_centre_atom``: ``(N_atom,)`` whether the atom is its
              residue's centre atom (0 when the annotation is unavailable).
    """
    asym = np.asarray(atom_array.label_asym_id).astype(str)
    res_id = np.asarray(atom_array.res_id).astype(str)
    pair = np.stack([asym, res_id], axis=1)
    _, group_id = np.unique(pair, axis=0, return_inverse=True)
    group_id = group_id.ravel()
    n_groups = int(group_id.max()) + 1

    mol_type = np.asarray(atom_array.mol_type)
    is_polymer_atom = np.isin(mol_type, ["protein", "dna", "rna"])
    is_polymer_group = (
        np.bincount(
            group_id, weights=is_polymer_atom.astype(np.float64), minlength=n_groups
        )
        > 0
    )

    centre_mask = getattr(atom_array, "centre_atom_mask", None)
    if centre_mask is not None:
        is_centre_atom = np.asarray(centre_mask).astype(bool)
    else:
        is_centre_atom = np.zeros(atom_array.array_length(), dtype=bool)

    return group_id, is_polymer_group, is_centre_atom


def _expand_keep_to_residue_level(
    atom_in_mask: np.ndarray,
    atom_array: AtomArray,
) -> np.ndarray:
    """Lift atom-level mask membership to residue-level keep decisions.

    Polymer residues are kept when their centre atom is inside the mask
    (falling back to "any atom inside" for residues without a centre atom);
    ligands and ions are kept when any of their atoms is inside.

    Args:
        atom_in_mask: ``(N_atom,)`` boolean array of per-atom mask membership.
        atom_array: Biotite AtomArray providing residue grouping annotations.

    Returns:
        ``(N_atom,)`` boolean array of atoms to keep.
    """
    group_id, is_polymer_group, is_centre_atom = _build_residue_groups(atom_array)
    n_groups = int(group_id.max()) + 1

    any_atom_in = (
        np.bincount(
            group_id, weights=atom_in_mask.astype(np.float64), minlength=n_groups
        )
        > 0
    )
    centre_atom_in = (
        np.bincount(
            group_id,
            weights=(atom_in_mask & is_centre_atom).astype(np.float64),
            minlength=n_groups,
        )
        > 0
    )
    has_centre_atom = (
        np.bincount(
            group_id, weights=is_centre_atom.astype(np.float64), minlength=n_groups
        )
        > 0
    )
    # Polymer residues keep by their centre atom; residues without any centre
    # atom (or ligands/ions) keep when any atom is inside the mask.
    polymer_keep = np.where(has_centre_atom, centre_atom_in, any_atom_in)
    keep_group = np.where(is_polymer_group, polymer_keep, any_atom_in)
    return keep_group[group_id]


def compute_keep_masks_for_predictions(
    pred_coordinates,
    atom_array: AtomArray,
    mask: np.ndarray,
    map_obj: MapObject,
) -> list[np.ndarray]:
    """Compute per-sample atom keep masks from the density-support mask.

    Args:
        pred_coordinates: ``(N_sample, N_atom, 3)`` predicted coordinates in
            the map's physical (Angstrom) frame.  Torch tensors and numpy
            arrays are both accepted.
        atom_array: Biotite AtomArray matching the atom dimension.
        mask: 3-D boolean density-support mask.
        map_obj: Map object the mask was built from.

    Returns:
        List of ``N_sample`` boolean arrays of shape ``(N_atom,)``.
    """
    if hasattr(pred_coordinates, "detach"):
        coords = pred_coordinates.detach().cpu().numpy()
    else:
        coords = np.asarray(pred_coordinates)

    keep_masks = []
    for sample_idx in range(coords.shape[0]):
        atom_in_mask = sample_mask_at_coords(mask, map_obj, coords[sample_idx])
        keep = _expand_keep_to_residue_level(atom_in_mask, atom_array)
        keep_masks.append(keep)
        n_kept = int(keep.sum())
        n_total = keep.shape[0]
        if n_kept < n_total:
            logger.info(
                f"Mask filtering: sample {sample_idx} keeps {n_kept}/{n_total} atoms "
                f"({n_total - n_kept} outside density support removed)"
            )
    return keep_masks
