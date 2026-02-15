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

"""
Structure blurring and CCC/OVR calculation for cryo-EM map-model validation.

Provides `calculate_ccc_ovr` which computes cross-correlation coefficient (CCC)
and overlap ratio (OVR) between an experimental cryo-EM map and a blurred
atomic model.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from numba import njit, prange

from cryozeta.em.map import MapObject, parse_mrc

# =============================================================================
# Constants – atomic masses
# =============================================================================

_atom_masses = {
    "H": 1.007941,
    "HE": 4.002602,
    "LI": 6.94,
    "BE": 9.0121831,
    "B": 10.811,
    "C": 12.01074,
    "N": 14.006703,
    "O": 15.999405,
    "F": 18.998403163,
    "NE": 20.1797,
    "NA": 22.98976928,
    "MG": 24.3051,
    "AL": 26.9815385,
    "SI": 28.0855,
    "P": 30.973761998,
    "S": 32.0648,
    "CL": 35.4529,
    "AR": 39.948,
    "K": 39.0983,
    "CA": 40.078,
    "SC": 44.955908,
    "TI": 47.867,
    "V": 50.9415,
    "CR": 51.9961,
    "MN": 54.938044,
    "FE": 55.845,
    "CO": 58.933194,
    "NI": 58.6934,
    "CU": 63.546,
    "ZN": 65.38,
    "GA": 69.723,
    "GE": 72.63,
    "AS": 74.921595,
    "SE": 78.971,
    "BR": 79.9035,
    "KR": 83.798,
    "RB": 85.4678,
    "SR": 87.62,
    "Y": 88.90584,
    "ZR": 91.224,
    "NB": 92.90637,
    "MO": 95.95,
    "TC": 97.9072,
    "RU": 101.07,
    "RH": 102.9055,
    "PD": 106.42,
    "AG": 107.8682,
    "CD": 112.414,
    "IN": 114.818,
    "SN": 118.71,
    "SB": 121.76,
    "TE": 127.6,
    "I": 126.90447,
    "XE": 131.293,
    "CS": 132.90545196,
    "BA": 137.327,
    "LA": 138.90547,
    "CE": 140.116,
    "PR": 140.90766,
    "ND": 144.242,
    "PM": 144.9128,
    "SM": 150.36,
    "EU": 151.964,
    "GD": 157.25,
    "TB": 158.92535,
    "DY": 162.5,
    "HO": 164.93033,
    "ER": 167.259,
    "TM": 168.93422,
    "YB": 173.054,
    "LU": 174.9668,
    "HF": 178.49,
    "TA": 180.94788,
    "W": 183.84,
    "RE": 186.207,
    "OS": 190.23,
    "IR": 192.217,
    "PT": 195.084,
    "AU": 196.966569,
    "HG": 200.592,
    "TL": 204.3834,
    "PB": 207.2,
    "BI": 208.9804,
    "PO": 208.9824,
    "AT": 209.9871,
    "RN": 222.0176,
    "FR": 223.0197,
    "RA": 226.0254,
    "AC": 227.0278,
    "TH": 232.0377,
    "PA": 231.03588,
    "U": 238.02891,
    "NP": 237.0482,
    "PU": 244.0642,
    "AM": 243.0614,
    "CM": 247.0704,
    "BK": 247.0703,
    "CF": 251.0796,
    "ES": 252.083,
    "FM": 257.0951,
    "MD": 258.0984,
    "NO": 259.101,
    "LR": 262.1096,
    "RF": 267.1218,
    "DB": 268.1257,
    "SG": 271.1339,
    "BH": 272.1383,
    "HS": 270.1343,
    "MT": 276.1516,
}


def get_atom_masses() -> dict[str, float]:
    """Return a dictionary mapping element symbols to atomic masses."""
    return _atom_masses


# =============================================================================
# Gaussian blur helpers (numba-accelerated)
# =============================================================================


def get_offsets_from_radius(label_radius: float, voxel_size: np.ndarray):
    """Compute integer grid offsets within a given physical radius."""
    int_range_x = int(np.ceil(label_radius / voxel_size[0]))
    int_range_y = int(np.ceil(label_radius / voxel_size[1]))
    int_range_z = int(np.ceil(label_radius / voxel_size[2]))
    offsets = np.array(
        [
            (dz, dy, dx)
            for dx in range(-int_range_x, int_range_x + 1)
            for dy in range(-int_range_y, int_range_y + 1)
            for dz in range(-int_range_z, int_range_z + 1)
        ],
        dtype=np.float32,
    )
    offsets = np.array(
        offsets[
            np.linalg.norm(offsets * voxel_size[::-1], axis=1)
            <= label_radius + np.sqrt(3)
        ]
    )
    return offsets


@njit
def add_gaussians(
    grid: np.ndarray,
    grid_indices: np.ndarray,
    atom_masses: np.ndarray,
    sigma: float,
    cutoff_dist: float,
    offsets: np.ndarray,
    voxel_size: np.ndarray,
) -> None:
    """Add Gaussian distributions to *grid* for each atom (numba-accelerated).

    Args:
        grid: 3-D density grid to accumulate into (modified in-place).
        grid_indices: Atom positions in grid-index space.
        atom_masses: Per-atom masses.
        sigma: Gaussian standard deviation.
        cutoff_dist: Distance cutoff for the Gaussian kernel.
        offsets: Pre-computed grid offsets to visit around each atom.
        voxel_size: Physical voxel dimensions.
    """
    grid_shape = np.array(grid.shape)
    exp_denom = -1.0 / (2 * sigma * sigma)
    pdf_coeff = 1.0 / (sigma**3 * np.power(2.0 * np.pi, 1.5))
    cutoff_sq = np.power(cutoff_dist, 2.0)

    for i in prange(len(grid_indices)):
        position = grid_indices[i]
        atom_mass = atom_masses[i]

        for offset in offsets:
            center = np.floor(position + offset).astype(np.int32)

            if (
                0 <= center[0] < grid_shape[0]
                and 0 <= center[1] < grid_shape[1]
                and 0 <= center[2] < grid_shape[2]
            ):
                shift = (center - position) * voxel_size[::-1]
                dist_sq = np.sum(shift**2)
                if dist_sq <= cutoff_sq:
                    height_at_point = np.exp(dist_sq * exp_denom)
                    v = atom_mass * pdf_coeff * height_at_point
                    grid[center[0], center[1], center[2]] += v


@njit
def _extend_mask(
    grid: np.ndarray,
    grid_coords: np.ndarray,
    offsets: np.ndarray,
    voxel_size: np.ndarray,
    mask_radius: float,
    mask_value: float = 1.0,
) -> None:
    """Fill *grid* with *mask_value* around each atom coordinate (numba-accelerated)."""
    for i in range(grid_coords.shape[0]):
        atom_coord = grid_coords[i][::-1]
        for offset in offsets:
            coord = np.floor(atom_coord + offset).astype(np.int32)
            curr_dis = np.linalg.norm((coord - atom_coord) * voxel_size[::-1])
            if (
                curr_dis <= mask_radius
                and 0 <= coord[0] < grid.shape[0]
                and 0 <= coord[1] < grid.shape[1]
                and 0 <= coord[2] < grid.shape[2]
            ):
                grid[coord[0], coord[1], coord[2]] = mask_value


# =============================================================================
# Blur / mask construction
# =============================================================================


def gaussian_blur_real_space_vc(
    atom_coords,
    elements,
    map_obj: MapObject,
    sigma_coeff: float = 0.356,
    resolution: float = 1.0,
    cutoff: float = 4.0,
    protein_only: bool = False,
    custom_logger=None,
) -> MapObject:
    """Apply Gaussian blur to atom positions in real space.

    Args:
        atom_coords: (N, 3) array of atom xyz coordinates.
        elements: Sequence of element symbols parallel to *atom_coords*.
        map_obj: Reference map providing grid geometry.
        sigma_coeff: Coefficient for sigma = sigma_coeff * resolution.
        resolution: Map resolution in Angstroms.
        cutoff: Truncation radius in units of sigma.
        protein_only: If *True*, keep only C/H/N/O/S atoms.
        custom_logger: Optional logger.

    Returns:
        A new :class:`MapObject` containing the blurred density.
    """
    global_origin = map_obj.global_origin
    voxel_size = map_obj.voxel_size

    masses = get_atom_masses()
    if protein_only:
        masses = {k: v for k, v in masses.items() if k in ["C", "H", "N", "O", "S"]}

    atom_list = []
    for i, coord in enumerate(atom_coords):
        atom_list.append({"mass": masses.get(elements[i], 1.0), "coord": coord})

    coords_arr = np.array([a["coord"] for a in atom_list], dtype=np.float32)
    masses_arr = np.array([a["mass"] for a in atom_list], dtype=np.float32)

    blurred_grid = np.zeros_like(map_obj.grid_data, dtype=np.float32)

    if len(coords_arr) == 0:
        return MapObject(
            grid_data=blurred_grid, voxel_size=voxel_size, global_origin=global_origin
        )

    grid_indices = (coords_arr - global_origin) / voxel_size
    grid_indices = grid_indices[:, ::-1]  # xyz -> ijk

    sigma = sigma_coeff * resolution
    cutoff_dist = cutoff * sigma
    if custom_logger:
        custom_logger.info(
            f"Blurring the grid with Gaussians... with sigma {sigma:.3f} and cutoff dist {cutoff_dist:3f}"
        )
    offsets = get_offsets_from_radius(cutoff_dist, voxel_size)
    add_gaussians(
        blurred_grid, grid_indices, masses_arr, sigma, cutoff_dist, offsets, voxel_size
    )

    return MapObject(
        grid_data=blurred_grid, voxel_size=voxel_size, global_origin=global_origin
    )


def extended_mask_real_space(
    atom_coords,
    mrc_obj: MapObject,
    mask_radius: float = 4.0,
    mask_value: float = 1.0,
    custom_logger=None,
) -> MapObject:
    """Create a binary mask around atom positions in real space.

    Args:
        atom_coords: (N, 3) array of atom xyz coordinates.
        mrc_obj: Reference map providing grid geometry.
        mask_radius: Radius (Angstroms) around each atom to set to *mask_value*.
        mask_value: Value written into the mask grid.
        custom_logger: Optional logger.

    Returns:
        A new :class:`MapObject` containing the mask.
    """
    global_origin = mrc_obj.global_origin
    voxel_size = mrc_obj.voxel_size

    grid = np.zeros_like(mrc_obj.grid_data, dtype=np.float32)

    if len(atom_coords) == 0:
        return MapObject(
            grid_data=grid, voxel_size=voxel_size, global_origin=global_origin
        )

    grid_coords = (atom_coords - global_origin) / voxel_size

    offsets = get_offsets_from_radius(mask_radius, voxel_size)
    _extend_mask(grid, grid_coords, offsets, voxel_size, mask_radius, mask_value)

    if custom_logger:
        nonzero_cnt = np.count_nonzero(grid)
        custom_logger.info(f"Extended mask done. {nonzero_cnt} grid points are masked.")

    return MapObject(grid_data=grid, voxel_size=voxel_size, global_origin=global_origin)


# =============================================================================
# Map–map comparison metrics
# =============================================================================


def calculate_ovr(target_map: MapObject, simu_map: MapObject) -> float:
    """Calculate the overlap ratio between two maps.

    Args:
        target_map: Experimental (or reference) map.
        simu_map: Simulated (blurred model) map.

    Returns:
        Overlap ratio in [0, 1], or -1.0 when the denominator is zero.
    """
    target_sum = min(np.sum(target_map.grid_data > 0), np.sum(simu_map.grid_data > 0))
    intersection = np.sum((target_map.grid_data * simu_map.grid_data) > 0)
    if np.isclose(target_sum, 0):
        return -1.0
    return intersection / target_sum


def calculate_ccc(target_map: MapObject, simu_map: MapObject) -> float:
    """Calculate the cross-correlation coefficient (CCC) between two maps.

    Args:
        target_map: Experimental (or reference) map.
        simu_map: Simulated (blurred model) map.

    Returns:
        CCC value, or -1.0 when the overlap mask is empty.
    """
    target_bin = target_map.grid_data > 0
    simu_bin = simu_map.grid_data > 0
    mask_array = (target_bin * simu_bin) > 0
    if np.allclose(mask_array, 0):
        return -1.0
    target_mask = target_bin * target_map.grid_data
    simu_mask = simu_bin * simu_map.grid_data
    return np.sum(target_mask * simu_mask) / np.sqrt(
        np.sum(np.square(target_mask)) * np.sum(np.square(simu_mask))
    )


# =============================================================================
# Public API CCC / OVR calculation
# =============================================================================


def calculate_ccc_ovr(
    coordinates,
    elements,
    map_path: str,
    resolution: float,
    contour_level: float,
):
    """Compute CCC and OVR between an experimental map and an atomic model.

    Args:
        coordinates: (N, 3) array of atom xyz coordinates.
        elements: Sequence of element symbols parallel to *coordinates*.
        map_path: Path to the MRC/MAP file.
        resolution: Map resolution in Angstroms.
        contour_level: Density threshold.

    Returns:
        Tuple of ``(ccc_mask, ovr_mask, ccc_box, ovr_box)``.

    Raises:
        ValueError: If resolution or contour level is not provided.
    """
    if resolution is None:
        raise ValueError("resolution must be provided")
    if contour_level is None:
        raise ValueError("contour_level must be provided")

    mrc_obj = parse_mrc(map_path)

    # Create mask around the full structure
    mask_obj = extended_mask_real_space(
        atom_coords=coordinates, mrc_obj=mrc_obj, mask_radius=2.0
    )

    # Apply contour threshold
    mrc_obj.grid_data[mrc_obj.grid_data <= contour_level] = 0

    # Masked map (for cc_mask)
    masked_obj = MapObject(
        grid_data=mrc_obj.grid_data * mask_obj.grid_data,
        voxel_size=mrc_obj.voxel_size,
        global_origin=mrc_obj.global_origin,
    )

    # Blur with all atom masses
    blurred_all = gaussian_blur_real_space_vc(
        atom_coords=coordinates,
        elements=elements,
        map_obj=masked_obj,
        resolution=resolution,
        protein_only=False,
    )

    # CC_mask and OVR_mask (with mask)
    ccc_mask = calculate_ccc(masked_obj, blurred_all)
    ovr_mask = calculate_ovr(masked_obj, blurred_all)
    logger.info(f"All Atoms CC_mask: {ccc_mask:.3f}, OVR_mask: {ovr_mask:.3f}")

    # CC_box and OVR_box (without mask)
    ccc_box = calculate_ccc(mrc_obj, blurred_all)
    ovr_box = calculate_ovr(mrc_obj, blurred_all)
    logger.info(f"All Atoms CC_box: {ccc_box:.3f}, OVR_box: {ovr_box:.3f}")
    return ccc_mask, ovr_mask, ccc_box, ovr_box
