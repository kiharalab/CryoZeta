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

from typing import Literal

import numpy as np
import numpy.typing as npt
from loguru import logger
from shot_fpfh.core import grid_subsampling
from shot_fpfh.descriptors import ShotMultiprocessor, compute_fpfh_descriptor
from shot_fpfh.descriptors.shot import compute_single_shot_descriptor, get_local_rf
from sklearn.neighbors import KDTree
from tqdm import tqdm


def _compute_descriptors_sequential(
    point_cloud: npt.NDArray[np.float64],
    normals: npt.NDArray[np.float64],
    keypoints: npt.NDArray[np.float64],
    neighborhoods: npt.NDArray[np.object_],
    local_rfs: npt.NDArray[np.float64],
    support: npt.NDArray[np.float64],
    radius: float,
    normalize: bool,
    min_neighborhood_size: int,
    disable_progress_bar: bool,
) -> npt.NDArray[np.float64]:
    """Sequential version of compute_descriptor (for n_procs=1 to avoid pool creation)."""
    descriptors = []
    for i, keypoint in enumerate(
        tqdm(
            keypoints,
            desc=f"SHOT desc with radius {radius}",
            disable=disable_progress_bar,
        )
    ):
        desc = compute_single_shot_descriptor(
            (
                keypoint,
                support[neighborhoods[i]],
                normals[neighborhoods[i]],
                radius,
                local_rfs[i],
                normalize,
                min_neighborhood_size,
            )
        )
        descriptors.append(desc)
    return np.array(descriptors)


def _compute_local_rfs_sequential(
    keypoints: npt.NDArray[np.float64],
    neighborhoods: npt.NDArray[np.object_],
    support: npt.NDArray[np.float64],
    radius: float,
    disable_progress_bar: bool,
) -> npt.NDArray[np.float64]:
    """Sequential version of compute_local_rf (for n_procs=1 to avoid pool creation)."""
    local_rfs = []
    for i, keypoint in enumerate(
        tqdm(
            keypoints,
            desc=f"Local RFs with radius {radius}",
            disable=disable_progress_bar,
        )
    ):
        rf = get_local_rf(
            (
                keypoint,
                support[neighborhoods[i]],
                radius,
            )
        )
        local_rfs.append(rf)
    return np.array(local_rfs)


def compute_descriptors(
    point_cloud: npt.NDArray[np.float64],
    normals: npt.NDArray[np.float64],
    radius: float,
    descriptor_choice: Literal[
        "fpfh", "shot_single_scale", "shot_bi_scale", "shot_multiscale"
    ] = "shot_single_scale",
    *,
    # Optional parameters with defaults
    fpfh_n_bins: int = 5,
    phi: float = 3.0,
    rho: float = 10.0,
    n_scales: int = 2,
    subsample_support: bool = True,
    normalize: bool = True,
    share_local_rfs: bool = True,
    min_neighborhood_size: int = 100,
    n_procs: int = 8,
    disable_progress_bars: bool = False,
    verbose: bool = True,
) -> npt.NDArray[np.float64]:
    """
    Compute descriptors for every point in the point cloud.

    Args:
        point_cloud: Input point cloud (N, 3)
        normals: Normal vectors for the point cloud (N, 3)
        radius: Base radius for descriptor computation
        descriptor_choice: Type of descriptor to compute
        fpfh_n_bins: Number of bins for FPFH descriptor
        phi: Scale multiplier between levels
        rho: Divisor for subsampling voxel size
        n_scales: Number of scales for multiscale descriptor
        subsample_support: Whether to subsample support region
        normalize: Whether to normalize descriptors
        share_local_rfs: Whether to share local reference frames
        min_neighborhood_size: Minimum neighborhood size
        n_procs: Number of processes for parallel computation
        disable_progress_bars: Whether to disable progress bars
        verbose: Whether to print verbose output

    Returns:
        Computed descriptors as numpy array
    """
    # When n_procs=1, use sequential computation to avoid creating a multiprocessing.Pool
    # This prevents nested pool issues when used with ProcessPoolExecutor
    if n_procs == 1:
        return _compute_descriptors_sequential_impl(
            point_cloud=point_cloud,
            normals=normals,
            keypoints=point_cloud,
            descriptor_choice=descriptor_choice,
            radius=radius,
            phi=phi,
            rho=rho,
            n_scales=n_scales,
            subsample_support=subsample_support,
            normalize=normalize,
            share_local_rfs=share_local_rfs,
            min_neighborhood_size=min_neighborhood_size,
            disable_progress_bars=disable_progress_bars,
            verbose=verbose,
            fpfh_n_bins=fpfh_n_bins,
        )

    # Use ShotMultiprocessor for n_procs > 1
    with ShotMultiprocessor(
        normalize=normalize,
        share_local_rfs=share_local_rfs,
        min_neighborhood_size=min_neighborhood_size,
        n_procs=n_procs,
        disable_progress_bar=disable_progress_bars,
        verbose=verbose,
    ) as shot_multiprocessor:
        if descriptor_choice == "shot_single_scale":
            return shot_multiprocessor.compute_descriptor_single_scale(
                point_cloud=point_cloud,
                keypoints=point_cloud,
                normals=normals,
                radius=radius,
                subsampling_voxel_size=radius / rho if subsample_support else None,
            )

        elif descriptor_choice == "shot_bi_scale":
            return shot_multiprocessor.compute_descriptor_bi_scale(
                point_cloud=point_cloud,
                keypoints=point_cloud,
                normals=normals,
                local_rf_radius=radius,
                shot_radius=radius * phi,
                subsampling_voxel_size=radius / rho if subsample_support else None,
            )

        elif descriptor_choice == "shot_multiscale":
            return shot_multiprocessor.compute_descriptor_multiscale(
                point_cloud=point_cloud,
                keypoints=point_cloud,
                normals=normals,
                radii=radius * phi ** np.arange(n_scales),
                voxel_sizes=radius * phi ** np.arange(n_scales) / rho
                if subsample_support
                else None,
            )

        elif descriptor_choice == "fpfh":
            return compute_fpfh_descriptor(
                np.arange(len(point_cloud)),
                point_cloud,
                normals,
                radius=radius,
                n_bins=fpfh_n_bins,
                disable_progress_bars=disable_progress_bars,
                verbose=verbose,
            )

        else:
            raise ValueError("Incorrect descriptor choice")


def _compute_descriptors_sequential_impl(
    point_cloud: npt.NDArray[np.float64],
    normals: npt.NDArray[np.float64],
    keypoints: npt.NDArray[np.float64],
    descriptor_choice: Literal[
        "fpfh", "shot_single_scale", "shot_bi_scale", "shot_multiscale"
    ],
    radius: float,
    phi: float,
    rho: float,
    n_scales: int,
    subsample_support: bool,
    normalize: bool,
    share_local_rfs: bool,
    min_neighborhood_size: int,
    disable_progress_bars: bool,
    verbose: bool,
    fpfh_n_bins: int,
) -> npt.NDArray[np.float64]:
    """Sequential implementation to avoid pool creation when n_procs=1."""

    if descriptor_choice == "shot_single_scale":
        subsampling_voxel_size = radius / rho if subsample_support else None
        support = (
            grid_subsampling(point_cloud, subsampling_voxel_size)
            if subsampling_voxel_size is not None
            else None
        )
        if verbose and support is not None:
            logger.info(
                f"Keeping a support of {support.shape[0]} points out of {point_cloud.shape[0]} "
                f"(voxel size: {subsampling_voxel_size:.2f})"
            )
        neighborhoods = KDTree(
            point_cloud[support] if support is not None else point_cloud
        ).query_radius(keypoints, radius)
        local_rfs = _compute_local_rfs_sequential(
            keypoints=keypoints,
            neighborhoods=neighborhoods,
            support=point_cloud[support] if support is not None else point_cloud,
            radius=radius,
            disable_progress_bar=disable_progress_bars,
        )
        return _compute_descriptors_sequential(
            point_cloud=point_cloud,
            normals=normals[support] if support is not None else normals,
            keypoints=keypoints,
            neighborhoods=neighborhoods,
            local_rfs=local_rfs,
            support=point_cloud[support] if support is not None else point_cloud,
            radius=radius,
            normalize=normalize,
            min_neighborhood_size=min_neighborhood_size,
            disable_progress_bar=disable_progress_bars,
        )

    elif descriptor_choice == "shot_bi_scale":
        subsampling_voxel_size = radius / rho if subsample_support else None
        support = (
            grid_subsampling(point_cloud, subsampling_voxel_size)
            if subsampling_voxel_size is not None
            else None
        )
        if verbose and support is not None:
            logger.info(
                f"Keeping a support of {support.shape[0]} points out of {point_cloud.shape[0]} "
                f"(voxel size: {subsampling_voxel_size})"
            )
        local_rf_radius = radius
        shot_radius = radius * phi

        neighborhoods = KDTree(
            point_cloud[support] if support is not None else point_cloud
        ).query_radius(keypoints, local_rf_radius)
        local_rfs = _compute_local_rfs_sequential(
            keypoints=keypoints,
            neighborhoods=neighborhoods,
            support=point_cloud[support] if support is not None else point_cloud,
            radius=local_rf_radius,
            disable_progress_bar=disable_progress_bars,
        )
        neighborhoods = KDTree(
            point_cloud[support] if support is not None else point_cloud
        ).query_radius(keypoints, shot_radius)
        return _compute_descriptors_sequential(
            point_cloud=point_cloud,
            normals=normals[support] if support is not None else normals,
            keypoints=keypoints,
            neighborhoods=neighborhoods,
            local_rfs=local_rfs,
            support=point_cloud[support] if support is not None else point_cloud,
            radius=shot_radius,
            normalize=normalize,
            min_neighborhood_size=min_neighborhood_size,
            disable_progress_bar=disable_progress_bars,
        )

    elif descriptor_choice == "shot_multiscale":
        radii = radius * phi ** np.arange(n_scales)
        voxel_sizes = (
            radius * phi ** np.arange(n_scales) / rho if subsample_support else None
        )

        all_descriptors = np.zeros((len(radii), keypoints.shape[0], 352))
        local_rfs = None

        for scale, r in enumerate(radii):
            support = (
                grid_subsampling(point_cloud, voxel_sizes[scale])
                if voxel_sizes is not None
                else None
            )
            if verbose and support is not None:
                logger.info(
                    f"Keeping a support of {support.shape[0]} points out of {point_cloud.shape[0]} "
                    f"(voxel size: {voxel_sizes[scale]})"
                )
            neighborhoods = KDTree(
                point_cloud[support] if support is not None else point_cloud
            ).query_radius(keypoints, r)

            if local_rfs is None or not share_local_rfs:
                local_rfs = _compute_local_rfs_sequential(
                    keypoints=keypoints,
                    neighborhoods=neighborhoods,
                    support=point_cloud[support]
                    if support is not None
                    else point_cloud,
                    radius=r,
                    disable_progress_bar=disable_progress_bars,
                )

            all_descriptors[scale, :, :] = _compute_descriptors_sequential(
                point_cloud=point_cloud,
                normals=normals[support] if support is not None else normals,
                keypoints=keypoints,
                neighborhoods=neighborhoods,
                local_rfs=local_rfs,
                support=point_cloud[support] if support is not None else point_cloud,
                radius=r,
                normalize=normalize,
                min_neighborhood_size=min_neighborhood_size,
                disable_progress_bar=disable_progress_bars,
            )

        return all_descriptors.reshape(keypoints.shape[0], 352 * len(radii))

    elif descriptor_choice == "fpfh":
        return compute_fpfh_descriptor(
            np.arange(len(point_cloud)),
            point_cloud,
            normals,
            radius=radius,
            n_bins=fpfh_n_bins,
            disable_progress_bars=disable_progress_bars,
            verbose=verbose,
        )

    else:
        raise ValueError("Incorrect descriptor choice")
