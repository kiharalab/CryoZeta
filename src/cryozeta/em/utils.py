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

import numpy as np
import torch
from loguru import logger
from numba import njit
from tqdm import tqdm  # type: ignore


def write_coords_to_pdb(coords, output_path=None, atom_type="CA"):
    """
    Write coordinates to a PDB file.

    Args:
        coords (torch.Tensor or numpy.ndarray): Coordinates to write, shape (N, 3).
        output_path (str, optional): Path to save the PDB file. If None, saves in the current directory.
        atom_type (str): Atom type to write. "CA" for protein CA atoms (ALA residue),
            "C1P" for nucleic acid C1' atoms (DA residue).

    Returns:
        str: Path of the written PDB file.
    """
    # Atom name (cols 13-16) and residue name (cols 18-20) per atom type
    atom_formats = {
        "CA": (" CA ", "ALA"),  # protein C-alpha
        "C1P": (" C1'", " DA"),  # nucleic acid C1'
    }
    if atom_type not in atom_formats:
        raise ValueError(
            f"Unsupported atom_type: {atom_type!r}. Choose from {list(atom_formats)}"
        )
    atom_name, res_name = atom_formats[atom_type]

    with open(output_path, "w") as f:
        f.write("MODEL        1\n")  # Start of model
        for i, coord in enumerate(coords):
            # PDB format: https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
            f.write(
                f"ATOM  {i + 1:5d} {atom_name} {res_name} A{i + 1:4d}    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00           C\n"
            )
        f.write("ENDMDL\n")  # End of model
        f.write("END\n")  # End of file

    return output_path


def meanshiftpp_torch(
    X: torch.Tensor, bandwidth: float, n_steps: int, tol: float = 1e-3, base: int = 3
) -> torch.Tensor:
    """
    Mean shift++ clustering with PyTorch batch operations.

    Parameters
    ----------
    X : torch.Tensor
        (n, d) array of points.
    bandwidth : float
        Radius for binning points.
    n_steps : int
        Number of iterations.
    tol : float
        Tolerance for convergence.
    base : int
        Base for offsets. Default is 3.

    Returns
    -------
    X_shifted : torch.Tensor
        (n, d) array of new points after mean shift clustering.
    """

    n, d = X.shape

    # Generate offsets
    ranges = [torch.arange(-1, base - 1) for _ in range(d)]
    mesh = torch.meshgrid(ranges, indexing="ij")
    offsets = torch.stack(mesh, dim=-1).reshape(-1, d)
    offsets = offsets.to(X.device)

    X_shifted = X.clone().detach()

    for i in range(n_steps):
        bins = (X_shifted / bandwidth).int()  # Shape: [n, d]

        # Create a large tensor for all shifted bins
        all_shifted_bins = bins.unsqueeze(1) + offsets
        all_shifted_bins_flat = all_shifted_bins.reshape(-1, d)

        # Unique bins and their inverse indices
        unique_bins, inverse_indices = torch.unique(
            all_shifted_bins_flat, dim=0, return_inverse=True
        )

        # Compute sum and count for each unique bin
        sum_per_bin = torch.zeros_like(unique_bins, dtype=torch.float, device=X.device)

        sum_per_bin = sum_per_bin.scatter_add_(
            0,
            inverse_indices.unsqueeze(1).expand(-1, d),
            X_shifted.repeat(1, base**d).reshape(-1, d),
        )

        count_per_bin = torch.zeros(
            len(unique_bins), dtype=torch.float, device=X.device
        )
        count_per_bin = count_per_bin.scatter_add_(
            0,
            inverse_indices,
            torch.ones(n * base**d, dtype=torch.float, device=X.device),
        )

        # Map sums and counts back to the original bins
        sum_mapped = sum_per_bin[inverse_indices]  # Shape: [n * base ** d, d]
        count_mapped = count_per_bin[inverse_indices]  # Shape: [n * base ** d]

        # Compute new positions
        X_shifted_new = sum_mapped.reshape(n, -1, d).sum(dim=1) / count_mapped.reshape(
            n, -1
        ).sum(dim=1).unsqueeze(1)

        # Check for convergence
        if torch.max(torch.norm(X_shifted_new - X_shifted, dim=1)) <= tol:
            print(f"MeanShift++ Converged at {i + 1} steps.")
            break

        X_shifted = X_shifted_new.clone()

    return X_shifted


@njit
def localshift_numba(point_cd, reference, fmaxd, fsiv, n_steps, tol):
    cnt = point_cd.shape[0]
    ref_shape = reference.shape
    fsiv_neg = -1.5 * fsiv

    for i in range(cnt):
        for _step in range(n_steps):
            pos = point_cd[i]
            stp = np.array(
                [
                    max(int(pos[0] - fmaxd), 0),
                    max(int(pos[1] - fmaxd), 0),
                    max(int(pos[2] - fmaxd), 0),
                ],
                dtype=np.int32,
            )
            endp = np.array(
                [
                    min(int(pos[0] + fmaxd + 1), ref_shape[0] - 1),
                    min(int(pos[1] + fmaxd + 1), ref_shape[1] - 1),
                    min(int(pos[2] + fmaxd + 1), ref_shape[2] - 1),
                ],
                dtype=np.int32,
            )

            pos2 = np.zeros(3, dtype=np.float32)
            dtotal = 0.0

            for xp in range(stp[0], endp[0]):
                for yp in range(stp[1], endp[1]):
                    for zp in range(stp[2], endp[2]):
                        offset = np.array([xp, yp, zp], dtype=np.float32)
                        d2 = (
                            (offset[0] - pos[0]) ** 2
                            + (offset[1] - pos[1]) ** 2
                            + (offset[2] - pos[2]) ** 2
                        )
                        kernel_weight = np.exp(fsiv_neg * d2) * reference[xp, yp, zp]
                        if kernel_weight > 0:
                            dtotal += kernel_weight
                            pos2[0] += kernel_weight * offset[0]
                            pos2[1] += kernel_weight * offset[1]
                            pos2[2] += kernel_weight * offset[2]

            if dtotal > 0:
                pos2 /= dtotal
                shift_dis_square = (
                    (pos[0] - pos2[0]) ** 2
                    + (pos[1] - pos2[1]) ** 2
                    + (pos[2] - pos2[2]) ** 2
                )
                if shift_dis_square < tol:
                    break
                point_cd[i, 0] = pos2[0]
                point_cd[i, 1] = pos2[1]
                point_cd[i, 2] = pos2[2]
            else:
                break

    return point_cd


def get_shifted_indices(
    point_cd: torch.Tensor,
    reference_np: np.ndarray,
    fmaxd: float,
    fsiv: float,
    n_steps: int,
    tol: float,
):
    point_cd_np = point_cd.cpu().numpy()
    point_cd_shifted_np = localshift_numba(
        point_cd_np, reference_np, fmaxd, fsiv, n_steps, tol
    )
    point_cd_shifted = torch.from_numpy(point_cd_shifted_np)
    point_cd_shifted = point_cd_shifted.round_(decimals=3).unique(dim=0)
    point_cd_shifted = meanshiftpp_torch(
        point_cd_shifted, bandwidth=0.5, n_steps=100, tol=1e-5
    )
    point_cd_shifted = point_cd_shifted.round_(decimals=3).unique(dim=0)

    return point_cd_shifted


def generate_gaussian_importance_map(
    roi_size, sigma_scale=0.125, device=torch.device("cpu")
):
    """Generate a Gaussian importance map."""
    center = torch.tensor(roi_size, device=device) // 2
    sigma = torch.tensor(roi_size, device=device) * sigma_scale
    grid = torch.stack(
        torch.meshgrid(
            *[torch.arange(s, device=device) for s in roi_size], indexing="ij"
        )
    )
    dist = torch.sum(
        ((grid - center.view(-1, 1, 1, 1)) / sigma.view(-1, 1, 1, 1)) ** 2, dim=0
    )
    importance_map = torch.exp(-dist / 2)

    # handle non-positive weights
    min_non_zero = max(torch.min(importance_map).item(), 1e-3)
    importance_map = torch.clamp_(importance_map.to(torch.float), min=min_non_zero)
    return importance_map


@torch.no_grad()
def sliding_window_inference(
    input_map: torch.Tensor,
    output_num_channels: int,
    roi_size: int,
    batch_size: int,
    model,
    device=torch.device("cuda"),
    output_device=torch.device("cpu"),
    overlap_ratio=0.5,
    gaussian=False,
    sigma_scale=0.125,
):

    model.eval()
    depth, height, width = input_map.shape[2:]
    logger.info(f"Input tensor shape: {depth}/{height}/{width}")
    if depth < roi_size or height < roi_size or width < roi_size:
        raise ValueError(
            "Input tensor spatial dimensions are smaller than the ROI size."
        )

    # Generate importance map if gaussian is True
    importance_map = (
        generate_gaussian_importance_map(
            (roi_size, roi_size, roi_size), sigma_scale, device=output_device
        )
        if gaussian
        else None
    )

    # Initialize output and count maps for each type of output
    output_map = torch.zeros(
        (1, output_num_channels, depth, height, width),
        device=output_device,
        dtype=torch.float,
    )
    count_map = torch.zeros(
        (1, output_num_channels, depth, height, width),
        device=output_device,
        dtype=torch.float,
    )
    stride = int(roi_size * (1 - overlap_ratio))  # 50% overlap

    # Collect all possible coordinates
    coords = []
    starts_set = set()
    num_voxels = 0
    for z in range(0, depth, stride):
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                z_end = min(z + roi_size, depth)
                y_end = min(y + roi_size, height)
                x_end = min(x + roi_size, width)
                z_start = z_end - roi_size
                y_start = y_end - roi_size
                x_start = x_end - roi_size
                if (z_start, y_start, x_start) in starts_set:
                    continue
                starts_set.add((z_start, y_start, x_start))
                num_voxels += 1
                sub_volume = input_map[
                    :, :, z_start:z_end, y_start:y_end, x_start:x_end
                ]
                if torch.all(sub_volume == 0):
                    continue
                coords.append((z_start, y_start, x_start, z_end, y_end, x_end))

    logger.info(
        f"Number of non-empty ROIs / All possible ROIs: {len(coords)} / {num_voxels}"
    )

    # Process coordinates in batches with tqdm progress bar
    for i in tqdm(range(0, len(coords), batch_size), desc="Processing inference"):
        batch_coords = coords[i : i + batch_size]
        batch = [
            input_map[:, :, z_start:z_end, y_start:y_end, x_start:x_end]
            for (z_start, y_start, x_start, z_end, y_end, x_end) in batch_coords
        ]
        batch_tensor = torch.cat(batch, dim=0).to(device)
        batch_diff = batch_size - batch_tensor.shape[0]
        if batch_diff > 0:
            batch_tensor = torch.cat(
                [
                    batch_tensor,
                    torch.zeros(batch_diff, *batch_tensor.shape[1:], device=device),
                ],
                dim=0,
            )
        model_output = model(batch_tensor)  # Expecting tuple of tensors as output

        if isinstance(model_output, tuple):
            model_output = model_output[0]

        if batch_diff > 0:
            model_output = model_output[:batch_size]

        model_output = model_output.to(output_device)

        for idx, (z_start, y_start, x_start, z_end, y_end, x_end) in enumerate(
            batch_coords
        ):
            if gaussian:
                assert importance_map is not None, (
                    "Importance map is required for gaussian inference"
                )
                weighted_output = model_output[idx] * importance_map
                output_map[:, :, z_start:z_end, y_start:y_end, x_start:x_end] += (
                    weighted_output
                )
                count_map[:, :, z_start:z_end, y_start:y_end, x_start:x_end] += (
                    importance_map
                )
            else:
                output_map[:, :, z_start:z_end, y_start:y_end, x_start:x_end] += (
                    model_output[idx]
                )
                count_map[:, :, z_start:z_end, y_start:y_end, x_start:x_end] += 1

        del model_output

    # Normalize each output map by its count map
    output_map[count_map > 0] /= count_map[count_map > 0]

    return output_map  # Return a tuple of all normalized output maps
