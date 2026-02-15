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

import os
from dataclasses import dataclass
from typing import Any

import mrcfile
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Num
from loguru import _Logger


@dataclass
class MapObject:
    """Cryo-EM map object."""

    grid_data: Num[np.ndarray, "d h w"]
    voxel_size: Float[np.ndarray, "3"]
    global_origin: Float[np.ndarray, "3"]
    emdb_id: str | None = None

    def __repr__(self) -> str:
        """Get string representation of the MapObject.

        Returns:
            String representation of the object.
        """
        return f"Map(grid_data of shape {self.grid_data.shape}: voxel_size={self.voxel_size}, global_origin={self.global_origin})"

    @property
    def grid_size(self) -> tuple[int, int, int]:
        """Get the shape of the grid data.

        Returns:
            Tuple of (height, width, depth) dimensions.
        """
        shape = self.grid_data.shape
        if len(shape) != 3:
            raise ValueError("Grid data must be 3-dimensional")
        return (int(shape[0]), int(shape[1]), int(shape[2]))

    @property
    def spatial_size(self) -> tuple[float, float, float]:
        """Get the spatial size of the measured grid data in physical units.

        Returns:
            Tuple of (height, width, depth) in physical units.
        """
        nonzero_indices = np.nonzero(self.grid_data)
        grid_size = np.array(
            [max(dim) - min(dim) for dim in nonzero_indices],
            dtype=np.int32,
        )
        return tuple(grid_size * self.voxel_size)

    @property
    def is_empty(self) -> bool:
        """Check if the voxel data is effectively empty (all values close to zero)."""
        return bool(np.allclose(self.grid_data, 0, atol=1e-12))

    def __mul__(self, other: "MapObject") -> "MapObject":
        """Multiply two maps element-wise."""
        # assert voxel size and global origin are the same
        assert np.allclose(self.voxel_size, other.voxel_size)
        assert np.allclose(self.global_origin, other.global_origin)
        return MapObject(
            grid_data=self.grid_data * other.grid_data,
            voxel_size=self.voxel_size,
            global_origin=self.global_origin,
        )


def parse_header_from_mrc(
    mrc_path: str, logger: _Logger | None = None
) -> dict[str, Any]:
    """
    Parse the header from an MRC file.

    Args:
        mrc_path: Path to the MRC file.
        logger: Optional logger for output messages.

    Returns:
        Dictionary containing header information.
    """
    with mrcfile.mmap(mrc_path, mode="r") as mrc:
        if logger:
            print_header(logger, mrc)
        return mrc.header


def print_header(logger: _Logger, mrc: mrcfile.mrcfile.MrcFile) -> None:
    """
    Print the header information of an MRC file.

    Args:
        logger: Logger for output messages.
        mrc: MRC file object.
    """
    for item in mrc.header.dtype.names:
        logger.info(f"{item:15s} : {mrc.header[item]}")


def parse_mrc(
    mrc_path: os.PathLike | str, custom_logger: _Logger | None = None
) -> MapObject:
    """
    Parse an MRC file and return an MapObject.

    Args:
        mrc_path: Path to the MRC file.
        custom_logger: Optional logger for output messages.

    Returns:
        MapObject containing the parsed data.

    Raises:
        ValueError: If the map is not orthogonal.
    """
    with mrcfile.open(mrc_path, permissive=True, mode="r") as mrc:
        if custom_logger:
            print_header(custom_logger, mrc)
        grid_data = np.array(mrc.data.copy(), dtype=np.float32)
        header = mrc.header
        origin = np.array(header.origin.tolist(), dtype=np.float32)
        voxel_size = np.array(mrc.voxel_size.tolist(), dtype=np.float32)
        n_crs_start = np.array(
            [header.nxstart, header.nystart, header.nzstart], dtype=np.float32
        )
        angle = np.asarray(
            [header.cellb.alpha, header.cellb.beta, header.cellb.gamma],
            dtype=np.float32,
        )

        # Check orthogonal
        if not np.allclose(angle, 90.0):
            raise ValueError("Map is not orthogonal")

        # Reorder
        map_crs = np.subtract([header.mapc, header.mapr, header.maps], 1)
        sort = np.array([0, 1, 2], dtype=np.int64)
        for i in range(3):
            sort[map_crs[i]] = i

        n_xyz_start = np.array([n_crs_start[i] for i in sort])
        grid_data = np.transpose(grid_data, axes=2 - sort[::-1])

        # MRC2000 compatibility
        if np.isclose(origin, 0.0).all():
            origin += np.multiply(n_xyz_start, voxel_size)
            if custom_logger:
                custom_logger.info(
                    f"Origin is zero. Calculating origin from n_xyz_start and voxel size. New origin is {origin}"
                )

    return MapObject(grid_data=grid_data, voxel_size=voxel_size, global_origin=origin)


def determine_optimal_dtype(grid_data: np.ndarray) -> np.dtype:
    """
    Determine the optimal dtype for the voxel data.

    Args:
        grid_data: Input voxel data.

    Returns:
        Optimal numpy dtype for the data.
    """
    if np.issubdtype(grid_data.dtype, np.floating):
        # max_val = np.max(np.abs(grid_data))
        # if max_val < np.finfo(np.float16).max:
        #     return np.float16
        # just return float32 for compatibility with mol*
        return np.dtype(np.float32)

    # Check if dataset is integer
    if np.issubdtype(grid_data.dtype, np.integer):
        max_val = np.max(grid_data)
        min_val = np.min(grid_data)

        if min_val >= -(2**7) and max_val < 2**7:
            return np.dtype(np.int8)
        elif min_val >= -(2**15) and max_val < 2**15:
            return np.dtype(np.int16)
        else:
            return np.dtype(np.float32)

    return grid_data.dtype  # Default to existing dtype if no change is needed


def save_mrc(
    mrc_obj: MapObject,
    mrc_path: os.PathLike | str,
    custom_logger: _Logger | None = None,
) -> None:
    """
    Save an MapObject to a file.

    Args:
        mrc_path: Path to save the MRC file.
        mrc_obj: MapObject to save.
        custom_logger: Optional logger for output messages.
    """
    # Determine optimal dtype based on dataset range
    optimal_dtype = determine_optimal_dtype(mrc_obj.grid_data)

    if optimal_dtype != mrc_obj.grid_data.dtype and custom_logger:
        custom_logger.info(
            f"Converting data to optimal dtype: {optimal_dtype} and save to {mrc_path}"
        )

    # Convert dataset to optimal dtype
    grid_data = mrc_obj.grid_data.astype(optimal_dtype)

    with mrcfile.new(mrc_path, data=grid_data, overwrite=True) as mrc:
        mrc.header.origin = tuple(mrc_obj.global_origin)
        mrc.voxel_size = tuple(mrc_obj.voxel_size)
        mrc.update_header_from_data()

        if custom_logger:
            print_header(custom_logger, mrc)


def resample_mrc(
    mrc_object: MapObject,
    apix: float,
    use_gpu: bool = False,
    logger: _Logger | None = None,
) -> MapObject:
    """
    Resample an MapObject to a new voxel size.

    Args:
        mrc_object: Input MapObject.
        apix: Target voxel size.
        use_gpu: Whether to use GPU for resampling.
        logger: Optional logger for output messages.

    Returns:
        Resampled MapObject.
    """
    original_voxel_size = mrc_object.voxel_size
    original_data_shape = mrc_object.grid_data.shape
    target_voxel_size = np.array([apix, apix, apix], dtype=np.float32)
    target_grid_size = np.floor(
        mrc_object.grid_data.shape * original_voxel_size[::-1] / target_voxel_size[::-1]
    ).astype(np.int32)
    if logger:
        logger.info(f"Original voxel size: {original_voxel_size}")
        logger.info(f"Original data shape: {original_data_shape}")
        logger.info(f"Target voxel size: {target_voxel_size}")
        logger.info(f"Target grid size: {target_grid_size}")

    with torch.no_grad() and torch.amp.autocast("cuda", enabled=use_gpu):
        z = (
            torch.arange(0, target_grid_size[0], dtype=torch.float32)
            / original_voxel_size[2]
            / (original_data_shape[0] - 1)
            * 2
            - 1
        )
        y = (
            torch.arange(0, target_grid_size[1], dtype=torch.float32)
            / original_voxel_size[1]
            / (original_data_shape[1] - 1)
            * 2
            - 1
        )
        x = (
            torch.arange(0, target_grid_size[2], dtype=torch.float32)
            / original_voxel_size[0]
            / (original_data_shape[2] - 1)
            * 2
            - 1
        )

        new_grid = torch.stack(
            torch.meshgrid(x, y, z, indexing="ij"),
            dim=-1,
        ).unsqueeze(0)

        if logger:
            logger.info(f"New grid shape: {new_grid.shape}")

        original_data = (
            torch.from_numpy(mrc_object.grid_data).unsqueeze(0).unsqueeze(0)
        )  # volumetric input
        if use_gpu:
            # Check if GPU is available
            if torch.cuda.is_available():
                if logger:
                    logger.info("CUDA is available. Using GPU for resampling.")
                device = torch.device("cuda")
                original_data = original_data.to(device)
            # Check if MPS is available
            # elif torch.backends.mps.is_available():
            #     logger.info("MPS is available. Using MPS for resampling.")
            #     output_device = torch.output_device('mps')
            #     original_data = original_data.to(output_device)
            else:
                if logger:
                    logger.warning("GPU is not available. Using CPU for resampling.")

        target_data = (
            F.grid_sample(original_data, new_grid, mode="bilinear", align_corners=True)
            .cpu()
            .numpy()[0, 0]
            .transpose(2, 1, 0)
        )

        if logger:
            logger.info(f"Resampled data shape: {target_data.shape}")

    return MapObject(
        grid_data=target_data,
        voxel_size=target_voxel_size,
        global_origin=mrc_object.global_origin,
    )


def normalize_mrc(
    mrc_object: MapObject,
    contour_level: float = 0.0,
    quantile_fraction: float = 0.98,
    logger: _Logger | None = None,
) -> MapObject:
    """
    Normalize an MapObject.

    Args:
        mrc_object: Input MapObject.
        contour_level: Contour level for thresholding.
        quantile_fraction: Quantile fraction for upper bound.
        logger: Optional logger for output messages.

    Returns:
        Normalized MapObject.
    """
    grid = mrc_object.grid_data
    grid[grid < 0] = 0

    # Thresholding using contour level
    grid[grid < contour_level] = 0

    quantile_val = np.quantile(grid[grid > 0], quantile_fraction)
    grid[grid > quantile_val] = quantile_val

    # Min-max normalization
    grid = (grid - grid.min()) / (grid.max() - grid.min())

    if logger:
        logger.info(f"Quantile-{quantile_fraction} value: {quantile_val}")

    return MapObject(
        grid_data=grid,
        voxel_size=mrc_object.voxel_size,
        global_origin=mrc_object.global_origin,
    )


def crop_mrc(
    mrc_object: MapObject,
    extended_val: int = 16,
    min_spatial_size: int = 64,
    logger: _Logger | None = None,
) -> MapObject:
    """
    Crop an MapObject.

    Args:
        mrc_object: Input MapObject.
        extended_val: Number of voxels to extend the bounding box.
        min_spatial_size: Minimum size for each dimension after cropping.
        logger: Optional logger for output messages.

    Returns:
        Cropped MapObject.
    """
    grid = mrc_object.grid_data
    indices = np.nonzero(grid)
    bounding_box_ranges = [
        (
            max(0, np.min(dim) - extended_val),
            min(grid.shape[i], np.max(dim) + extended_val),
        )
        for i, dim in enumerate(indices)
    ]
    bbox = tuple(slice(start, end) for start, end in bounding_box_ranges)
    bbox_start = np.array([b.start for b in bbox], dtype=np.float32)
    bbox_xyz_start = bbox_start[::-1]

    if logger:
        logger.info(f"Bounding box ranges: {bounding_box_ranges}")
        logger.info(f"Bounding box start: {bbox_start}")
        bbox_end = np.array([b.stop for b in bbox], dtype=np.float32)
        logger.info(f"Bounding box end: {bbox_end}")

    origin = mrc_object.global_origin + np.multiply(
        bbox_xyz_start, mrc_object.voxel_size
    )

    cropped_grid = grid[bbox]

    pad_width = []
    for i in range(3):
        if cropped_grid.shape[i] < min_spatial_size:
            pad_width.append((0, min_spatial_size - cropped_grid.shape[i]))
        else:
            pad_width.append((0, 0))

    if any(pw[1] > 0 for pw in pad_width):
        if logger:
            logger.info(
                f"The MRC object with shape {cropped_grid.shape} need padding {pad_width} to meet the min_spatial_size."
            )
        cropped_grid = np.pad(
            cropped_grid, pad_width, mode="constant", constant_values=0
        )

        if logger:
            logger.info(f"Padded cropped grid to shape: {cropped_grid.shape}")

    return MapObject(
        grid_data=cropped_grid, voxel_size=mrc_object.voxel_size, global_origin=origin
    )
