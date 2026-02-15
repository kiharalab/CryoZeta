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

import torch
from torch import nn


def generate_offsets(d: int, base: int = 3):
    """
    Parameters
    ----------
    d : int
        Dimensionality of the data.
    base : int
        Base for offsets. Default is 3.

    Returns
    -------
    offsets : torch.Tensor
        (3 ** d, d) array of offsets.
    """

    ranges = [torch.arange(-1, base - 1) for _ in range(d)]
    mesh = torch.meshgrid(ranges, indexing="ij")
    offsets = torch.stack(mesh, dim=-1).reshape(-1, d)

    return offsets


class MeanShiftPP(nn.Module):
    """
    Mean shift clustering with PyTorch batch operations.

    Parameters
    ----------
    bandwidth : float
        Radius for binning points.
    n_steps : int
        Number of iterations.
    tol : float
        Tolerance for convergence.
    base : int
        Base for offsets. Default is 3.
    """

    def __init__(
        self, bandwidth: float, n_steps: int, tol: float = 1e-3, base: int = 3
    ):
        super().__init__()
        self.bandwidth = bandwidth
        self.n_steps = n_steps
        self.tol = tol
        self.base = base

    def forward(self, X: torch.Tensor):
        """
        Parameters
        ----------
        X : torch.Tensor
            (n, d) array of points.

        Returns
        -------
        X_shifted : torch.Tensor
            (n, d) array of new points after one iteration of shift.
        """

        _n, d = X.shape
        offsets = generate_offsets(d, self.base)
        offsets = offsets.to(X.device)

        X_shifted = X.clone()

        for _i in range(self.n_steps):
            X_shifted = self.step(X_shifted, offsets)

            if torch.max(torch.norm(X_shifted - X, dim=1)) <= self.tol:
                # clean verbose output
                # print(f"Meanshift++ Converged at {i + 1} steps.")
                break

            X = X_shifted.clone()

        return X_shifted

    def step(self, X: torch.Tensor, offsets: torch.Tensor):
        """
        Parameters
        ----------
        X : torch.Tensor
            (n, d) array of points.

        Returns
        -------
        X_shifted : torch.Tensor
            (n, d) array of new points after one iteration of shift.
        """

        n, d = X.shape

        bins = (X / self.bandwidth).int()  # Shape: [n, d]

        # Create a large tensor for all shifted bins
        all_shifted_bins = bins.unsqueeze(1) + offsets
        all_shifted_bins_flat = all_shifted_bins.reshape(-1, d)

        # Unique bins and their inverse indices
        unique_bins, inverse_indices = torch.unique(
            all_shifted_bins_flat, dim=0, return_inverse=True
        )

        # Compute sum and count for each unique bin
        sum_per_bin = torch.zeros_like(
            unique_bins, dtype=torch.float, device=X.device
        ).scatter_add_(
            0,
            inverse_indices.unsqueeze(1).expand(-1, d),
            X.repeat(1, self.base**d).reshape(-1, d),
        )

        count_per_bin = torch.zeros(
            len(unique_bins), dtype=torch.float, device=X.device
        ).scatter_add_(
            0,
            inverse_indices,
            torch.ones(n * self.base**d, dtype=torch.float, device=X.device),
        )

        # Map sums and counts back to the original bins
        sum_mapped = sum_per_bin[inverse_indices]  # Shape: [n * base ** d, d]
        count_mapped = count_per_bin[inverse_indices]  # Shape: [n * base ** d]

        # Compute new positions
        X_shifted = sum_mapped.reshape(n, -1, d).sum(dim=1) / count_mapped.reshape(
            n, -1
        ).sum(dim=1).unsqueeze(1)

        return X_shifted


def mean_shift_pp_torch(
    X: torch.Tensor, bandwidth: float, n_steps: int, tol: float = 1e-3, base: int = 3
) -> torch.Tensor:
    """
    Mean shift clustering with PyTorch batch operations.

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

        # print(sum_per_bin.device)
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


def sample_support_points(MConf: torch.Tensor, N: int = 320, indices_set=None):
    """
    Sample N support points from the input tensor MConf.

    Args:
        MConf (torch.Tensor): A PyTorch tensor with dimensions [W, H, L] containing values between 0 and 1.
        N (int, optional): The number of support points to sample. Defaults to 320.

    Returns:
        torch.Tensor: A tensor containing N support points.

    Raises:
        ValueError: If there are not enough points where MConf > 0.5 to sample 3N points.
    """
    assert MConf.dim() == 3, "MConf must be a 3D tensor"

    # Step 1: Filter points where MConf > 0.5 and flatten the indices
    indices = (MConf > 0.5).nonzero(as_tuple=False)
    # Step 3: Randomly select 3N points from the filtered indices
    if indices.size(0) < 3 * N:
        raise ValueError("Not enough points to sample 3N points where MConf > 0.5")
    indices_3N = indices[torch.randperm(indices.size(0))[: 3 * N]].float()
    if indices_set is not None:
        indices_3N = indices_set
    # Step 4: Compute the Euclidean distance between each pair of points in S2
    # Using torch.cdist for efficient pairwise distance computation
    D = torch.cdist(indices_3N, indices_3N, p=2)

    # Step 5: Support Points Reduction
    S3 = indices_3N.clone()
    mask = torch.ones(S3.size(0), dtype=torch.bool)
    D.fill_diagonal_(float("inf"))
    _temp, sorted_indecies = torch.sort(torch.flatten(D), stable=True)

    cur_index = 0

    while torch.sum(mask) > N:
        # print(S3.size())
        # Set the diagonal to infinity to avoid selecting the same point

        # Find the indices of the minimum value in D
        # as our sorted_indecies is sorted, we can just take the top values,
        # of which the x or y was not previously choosen.
        while True:
            temp_i, temp_j = (
                torch.div(sorted_indecies[cur_index], D.size(0), rounding_mode="trunc"),
                sorted_indecies[cur_index] % D.size(1),
            )
            if not mask[temp_i] or not mask[temp_j]:
                cur_index = cur_index + 1
                continue
            else:
                # cur_index = cur_index + 1
                break

        i1, j1 = (
            torch.div(sorted_indecies[cur_index], D.size(0), rounding_mode="trunc"),
            sorted_indecies[cur_index] % D.size(1),
        )
        cur_index = cur_index + 1

        # Randomly select an index k from the pair [i, j]
        k1 = torch.randint(0, 2, (1,)).item()

        k1 = i1 if k1 == 0 else j1
        # visited[k1] = False

        mask[k1] = False

    # Step 6: Return the set S3 containing N support points
    return S3[mask], indices_3N


def sample_support_points_v2(N, indices_set):
    """
    Sample N support points from the input tensor MConf.

    Args:
        MConf (torch.Tensor): A PyTorch tensor with dimensions [W, H, L] containing values between 0 and 1.
        N (int, optional): The number of support points to sample. Defaults to 320.

    Returns:
        torch.Tensor: A tensor containing N support points.

    Raises:
        ValueError: If there are not enough points where MConf > 0.5 to sample 3N points.
    """
    indices_3N = indices_set
    # Step 4: Compute the Euclidean distance between each pair of points in S2
    # Using torch.cdist for efficient pairwise distance computation
    D = torch.cdist(indices_3N, indices_3N, p=2)

    # Step 5: Support Points Reduction
    S3 = indices_3N.clone()
    mask = torch.ones(S3.size(0), dtype=torch.bool)
    D.fill_diagonal_(float("inf"))
    _temp, sorted_indecies = torch.sort(torch.flatten(D), stable=True)

    cur_index = 0

    while torch.sum(mask) > N:  # 2 * N
        # print(S3.size())
        # Set the diagonal to infinity to avoid selecting the same point

        # Find the indices of the minimum value in D
        # as our sorted_indecies is sorted, we can just take the top values,
        # of which the x or y was not previously choosen.
        while True:
            temp_i, temp_j = (
                torch.div(sorted_indecies[cur_index], D.size(0), rounding_mode="trunc"),
                sorted_indecies[cur_index] % D.size(1),
            )
            if not mask[temp_i] or not mask[temp_j]:
                cur_index = cur_index + 1
                continue
            else:
                # cur_index = cur_index + 1
                break

        i1, j1 = (
            torch.div(sorted_indecies[cur_index], D.size(0), rounding_mode="trunc"),
            sorted_indecies[cur_index] % D.size(1),
        )
        cur_index = cur_index + 1

        # Randomly select an index k from the pair [i, j]
        k1 = torch.randint(0, 2, (1,)).item()

        k1 = i1 if k1 == 0 else j1
        # visited[k1] = False

        mask[k1] = False

    # Step 6: Return the set S3 containing N support points
    return S3[mask], mask


def get_conf(conf, support_points):
    """
    conf: 3D tensor, w x h x l
    support_points: N x 3
    return: dict, {(x,y,z): conf}
    """
    m_conf = {}
    for i in range(support_points.size(0)):
        x, y, z = (
            int(support_points[i][0].item()),
            int(support_points[i][1].item()),
            int(support_points[i][2].item()),
        )
        m_conf.update({(x, y, z): conf[x, y, z]})
    return m_conf


@torch.jit.script
def localshift_torch(
    point_cd: torch.Tensor,
    reference: torch.Tensor,
    fmaxd: float,
    fsiv: float,
    n_steps: int,
    tol: float = 1e-3,
):
    device = point_cd.device
    cnt, _ = point_cd.shape
    ref_shape_tensor = torch.tensor(reference.shape, device=device)
    pos2 = torch.zeros(3, device=device)
    fsiv_neg = -1.5 * fsiv

    for i in range(cnt):
        pos = point_cd[i]
        for _step in range(n_steps):
            # Define the bounding box for local shifts
            stp = torch.clamp(pos - fmaxd, min=0).to(torch.int)
            endp = torch.clamp(pos + fmaxd + 1, max=ref_shape_tensor).to(torch.int)

            # Generate all grid points within the box (vectorized over all points)
            x = torch.arange(stp[0], endp[0], device=device)
            y = torch.arange(stp[1], endp[1], device=device)
            z = torch.arange(stp[2], endp[2], device=device)
            grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")

            # Flatten the grid points and compute the squared distances to 'pos'
            offsets = torch.stack(
                [grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=-1
            ).float()
            dist_sq = ((offsets - pos.unsqueeze(0)) ** 2).sum(dim=1)

            # Compute kernel weights for all points at once
            kernel_weights = (
                torch.exp(fsiv_neg * dist_sq)
                * reference[grid_x, grid_y, grid_z].flatten()
            )

            # Filter out the points with zero weights and compute the weighted sum of positions
            valid = kernel_weights > 0
            weighted_positions = (
                offsets[valid] * kernel_weights[valid].unsqueeze(1)
            ).sum(dim=0)
            total_weight = kernel_weights[valid].sum()

            if total_weight == 0:
                break
            else:
                pos2 = weighted_positions / total_weight
                shift_dis_square = ((pos - pos2) ** 2).sum()
                if shift_dis_square < tol:
                    break

                pos.copy_(pos2)

    return point_cd


def pad_em_features(tensor_list):
    max_dim = max([a.shape[0] for a in tensor_list])
    pad_tensor_list = []
    print(tensor_list[0].shape)
    if tensor_list[0].shape[1] == 3 and len(tensor_list[0].shape) == 3:
        for tensor in tensor_list:
            try:
                point, _, recyle_dim = tensor.shape
            except Exception:
                print("problem")
                print(tensor)
                print(tensor.shape)

            pad_tensor = torch.zeros(
                (max_dim, 3, recyle_dim),
                dtype=tensor.dtype,
                layout=tensor.layout,
                device=tensor.device,
            )
            pad_tensor[:point, :, :] = tensor
            pad_tensor_list.append(pad_tensor)
    elif len(tensor_list[0].shape) == 2:
        for tensor in tensor_list:
            point, recyle_dim = tensor.shape
            pad_tensor = torch.zeros(
                (max_dim, recyle_dim),
                dtype=tensor.dtype,
                layout=tensor.layout,
                device=tensor.device,
            )
            pad_tensor[:point, :] = tensor
            pad_tensor_list.append(pad_tensor)
    elif tensor_list[0].shape[2] == 100:
        for tensor in tensor_list:
            point, _, feat_dim, recyle_dim = tensor.shape
            pad_tensor = torch.zeros(
                (max_dim, max_dim, feat_dim, recyle_dim),
                dtype=tensor.dtype,
                layout=tensor.layout,
                device=tensor.device,
            )
            pad_tensor[:point, :point, :, :] = tensor
            pad_tensor_list.append(pad_tensor)
    else:
        for tensor in tensor_list:
            point, res, feat_dim, recyle_dim = tensor.shape
            pad_tensor = torch.zeros(
                (max_dim, res, feat_dim, recyle_dim),
                dtype=tensor.dtype,
                layout=tensor.layout,
                device=tensor.device,
            )
            pad_tensor[:point, :, :, :] = tensor
            pad_tensor_list.append(pad_tensor)

    return pad_tensor_list
