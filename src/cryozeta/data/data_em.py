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
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def make_one_hot(x, num_classes):
    """Convert integer tensor to one-hot encoding.

    Args:
        x: Integer tensor
        num_classes: Number of classes

    Returns:
        One-hot encoded tensor
    """
    x_one_hot = torch.zeros(*x.shape, num_classes, device=x.device)
    x_one_hot.scatter_(-1, x.unsqueeze(-1), 1)
    return x_one_hot


def manifold_connection(assign_conf, num_clust=None):
    """Create binary connectivity matrix for points in same cluster.

    Args:
        assign_conf: Tensor of shape (p,) with cluster assignments
        num_clust: Number of clusters

    Returns:
        Tensor of shape (p, p) with binary connectivity
    """
    assign_onehot = make_one_hot(assign_conf, num_clust)
    manifold_connection = torch.matmul(assign_onehot, assign_onehot.T)
    return manifold_connection


def manifold_distance(
    assign_conf,
    distance,
    high_res=True,
    eps=1e-9,
    min_bin=2.3125,
    max_bin=21.6875,
    no_bins=50,
):
    """Compute shortest-path distances within clusters and bin them.

    Args:
        assign_conf: Tensor of shape (p,) with cluster assignments
        distance: Pairwise distance matrix of shape (p, p)
        eps: Small epsilon value
        min_bin: Minimum bin boundary
        max_bin: Maximum bin boundary
        no_bins: Number of distance bins

    Returns:
        One-hot binned distance features of shape (p, p, no_bins)
    """
    clusters = torch.unique(assign_conf)
    max_dist = torch.max(distance)
    ret_manifold = torch.zeros_like(distance).double()

    for cluster in clusters:
        mask = torch.where(assign_conf == cluster, 1, 0)
        mask = mask[..., None] * mask[None, ...]
        dist = torch.where(mask == 1, distance, max_dist + 1)
        graph = csr_matrix(dist)
        dist_matrix = shortest_path(csgraph=graph, directed=False)
        dist_matrix = torch.tensor(dist_matrix)
        ret_manifold += dist_matrix * mask

    # Bin distances
    if not high_res:
        distance = ret_manifold
    boundaries = torch.linspace(min_bin, max_bin, no_bins - 1)
    distance_bins = torch.sum(distance[..., None] > boundaries, dim=-1)
    distance_feature = make_one_hot(distance_bins, no_bins)

    return distance_feature


def take_confidence_endpoints(M_conf):
    """Extract confidence values at point pair endpoints.

    Args:
        M_conf: Confidence tensor of shape (p, 3)

    Returns:
        Tensor of shape (p, p, 6) with confidence at endpoints
    """
    replication_factor = len(M_conf)
    tiled_A = torch.tile(M_conf, (replication_factor, 1))
    pairwise_matrix = tiled_A.view(replication_factor, replication_factor, 3)
    pairwise_matrix_t = pairwise_matrix.transpose(1, 0)
    conf_endpoints = torch.cat([pairwise_matrix, pairwise_matrix_t], dim=-1)
    return conf_endpoints


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Generate 1D sinusoidal positional embeddings.

    Reference: https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L49C1-L49C1

    Args:
        embed_dim: Output dimension for each position
        pos: Positions to be encoded, shape (M,)

    Returns:
        Sinusoidal embeddings of shape (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)

    return emb


def PointPairRepresentationEmbedding(
    Q,
    M_conf,
    p,
    assign_conf,
    num_clust,
    ca_prob,
    eps=1e-9,
    min_bin=2.3125,
    max_bin=21.6875,
    no_bins=39,
    use_interpolate_confidence=True,
):
    """Create point pair representation embedding.

    Combines multiple geometric and structural features into a unified
    representation for each pair of points.

    Args:
        Q: Point positions, shape (p, 3)
        M_conf: Point confidences, shape (p, 3)
        p: Number of points
        assign_conf: Cluster assignments, shape (p,)
        num_clust: Number of clusters
        ca_prob: Confidence scores, shape (p, 3)
        eps: Small epsilon for numerical stability
        min_bin: Minimum distance bin boundary
        max_bin: Maximum distance bin boundary
        no_bins: Number of distance bins
        use_interpolate_confidence: Whether to include interpolated confidence

    Returns:
        Point pair features of shape:
        - (p, p, 114) if use_interpolate_confidence=True
        - (p, p, 99) if use_interpolate_confidence=False
    """
    if use_interpolate_confidence:
        w_point = torch.zeros(p, p, 114)  # 3 + 39 + 21 + 50 + 1
    else:
        w_point = torch.zeros(p, p, 99)  # 3 + 39 + 6 + 50 + 1

    # Orientation feature: normalized pairwise differences (3D)
    orientation = Q[:, None, :] - Q[None, :, :]
    norm = torch.norm(torch.abs(orientation) + eps, dim=-1, keepdim=True)
    w_point[:, :, 0:3] = orientation / norm

    # Distance feature: binned pairwise distances (39D)
    boundaries = torch.linspace(min_bin, max_bin, no_bins - 1)
    distance = torch.cdist(Q, Q)
    distance_bins = torch.sum(distance[..., None] > boundaries, dim=-1)
    distance_feature = make_one_hot(distance_bins, no_bins)
    w_point[:, :, 3:42] = distance_feature

    # Confidence feature: endpoint confidence values (6D)
    point_confidence = take_confidence_endpoints(ca_prob)

    if use_interpolate_confidence:
        # Interpolated confidence (15D)
        interpolate_confidence = M_conf
        interpolate_confidence = interpolate_confidence.view(p, p, -1)
        confidence_feature = torch.cat(
            [point_confidence, interpolate_confidence], dim=-1
        )
        w_point[:, :, 42:63] = confidence_feature

        # Manifold distance feature (50D)
        manifold_distance_feature = manifold_distance(assign_conf, distance)
        w_point[:, :, 63:113] = manifold_distance_feature

        # Manifold connection feature (1D)
        manifold_connection_feature = manifold_connection(assign_conf, num_clust)
        w_point[:, :, 113] = manifold_connection_feature
    else:
        # Confidence feature (6D)
        w_point[:, :, 42:48] = point_confidence

        # Manifold distance feature (50D)
        manifold_distance_feature = manifold_distance(assign_conf, distance)
        w_point[:, :, 48:98] = manifold_distance_feature

        # Manifold connection feature (1D)
        manifold_connection_feature = manifold_connection(assign_conf, num_clust)
        w_point[:, :, 98] = manifold_connection_feature

    return w_point


def pointResiduePairRepresentationEmbedding(aatype, asym_ids, Q, A_conf, p, r):
    """Create point-to-residue pair representation embedding.

    Combines amino acid type, confidence, and positional information
    into a unified representation for point-residue pairs.

    Args:
        aatype: Amino acid types, shape (r, 32)
        asym_ids: Chain identifiers, shape (r,)
        Q: Point positions, shape (p, 3)
        A_conf: Residue confidence, shape (p, 28)
        p: Number of points
        r: Number of residues

    Returns:
        Point-residue features of shape (p, r, 125)
        Components: 32 (aa type) + 28 (confidence) + 64 (pos encoding) + 1
    """
    x = torch.zeros(p, r, 32 + 28 + 64 + 1)

    # Amino acid features (32D)
    aatype_feature = torch.tile(aatype, (p, 1, 1))
    x[:, :, 0:32] = aatype_feature

    # Confidence features (28D)
    amino_confidence = torch.tile(A_conf.unsqueeze(1), (1, r, 1))
    x[:, :, 32:60] = amino_confidence

    # Sinusoidal positional encoding (64D)
    pos_emd = get_1d_sincos_pos_embed_from_grid(64, asym_ids)
    x[:, :, 60:124] = torch.tensor(pos_emd)

    # Affinity feature placeholder (1D) - set to 1 after second recycle
    # x[:, :, 124] remains 0 (initialized)

    return x
