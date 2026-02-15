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

import random
from collections import defaultdict

import numpy as np
import open3d as o3d
import small_gicp
import teaserpp_python
import torch
import torch.nn as nn
from Bio.SVDSuperimposer import SVDSuperimposer
from loguru import logger
from scipy.spatial import KDTree

from cryozeta.em.strucblur import calculate_ccc_ovr
from cryozeta.model.modules.calc_descriptors import compute_descriptors

# =============================================================================
# Recall / F1 Metrics
# =============================================================================


def calculate_query_recall(
    reference_points, query_points, distance_threshold: float = 3.0
) -> float:
    """
    Percentage of query points within distance_threshold of reference points.

    Args:
        reference_points: Reference point cloud (N, 3)
        query_points: Query point cloud (M, 3)
        distance_threshold: Distance threshold in Angstroms

    Returns:
        Recall value between 0 and 1
    """
    tree = KDTree(reference_points)
    dist, _ = tree.query(query_points, k=1, workers=-1)
    return float((dist < distance_threshold).sum()) / float(len(query_points))


def calculate_reference_recall(
    reference_points, query_points, distance_threshold: float = 3.0
) -> float:
    """
    Percentage of reference points within distance_threshold of query points.

    Args:
        reference_points: Reference point cloud (N, 3)
        query_points: Query point cloud (M, 3)
        distance_threshold: Distance threshold in Angstroms

    Returns:
        Recall value between 0 and 1
    """
    tree = KDTree(query_points)
    dist, _ = tree.query(reference_points, k=1, workers=-1)
    return float((dist < distance_threshold).sum()) / float(len(reference_points))


def calculate_f1(query_recall: float, reference_recall: float) -> float:
    """
    Harmonic mean of query recall and reference recall.

    Args:
        query_recall: Query recall value
        reference_recall: Reference recall value

    Returns:
        F1 score between 0 and 1
    """
    if query_recall + reference_recall == 0:
        return 0.0
    return 2.0 * (query_recall * reference_recall) / (query_recall + reference_recall)


# =============================================================================
# TEASER++ Helper Functions
# =============================================================================


def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
    """Find k-nearest neighbors using KDTree."""
    feat1tree = KDTree(feat1)
    dists, nn_inds = feat1tree.query(feat0, k=knn, workers=-1)
    if return_distance:
        return nn_inds, dists
    else:
        return nn_inds


def find_correspondences(feats0, feats1, mutual_filter=True):
    """Find correspondences between two descriptor sets using mutual nearest neighbors."""
    nns01 = find_knn_cpu(feats0, feats1, knn=1, return_distance=False)
    corres01_idx0 = np.arange(len(nns01))
    corres01_idx1 = nns01

    if not mutual_filter:
        return corres01_idx0, corres01_idx1

    nns10 = find_knn_cpu(feats1, feats0, knn=1, return_distance=False)

    if max(corres01_idx1) == feats1.shape[0]:
        corres10_idx0 = np.append(nns10, np.nan)
    else:
        corres10_idx0 = nns10

    mutual_filter = corres10_idx0[corres01_idx1] == corres01_idx0
    corres_idx0 = corres01_idx0[mutual_filter]
    corres_idx1 = corres01_idx1[mutual_filter]

    return corres_idx0, corres_idx1


def get_teaser_solver(noise_bound):
    """Create and configure TEASER++ solver."""
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1.0
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = False
    solver_params.inlier_selection_mode = (
        teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
    )
    solver_params.rotation_tim_graph = (
        teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN
    )
    solver_params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 10000
    solver_params.rotation_cost_threshold = 1e-16
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    return solver


def FitModelPointsTeaser(
    ca_coordinate: torch.Tensor,
    all_atom: torch.Tensor,
    elements: list[str],
    support_points: torch.Tensor,
    noise_bound: float = 1.0,
    desc_type: str = "shot_bi_scale",
    desc_r: float = 10,
    desc_scales: int = 2,
    nn_for_normal: int = 100,
    r_for_normal: float = 10,
    gicp_downsampling: float = 1,
    min_neighborhood_size: int = 100,
    phi: float = 3.0,
    rho: float = 10.0,
    map_path: str | None = None,
    resolution: float | None = None,
    contour_level: float | None = None,
) -> torch.Tensor | None:
    """
    Fit model points to support points using TEASER++ followed by GICP refinement.

    This function performs two-stage registration:
    1. TEASER++ for robust initial alignment using descriptor-based correspondences
    2. GICP for fine-grained refinement

    Args:
        ca_coordinate: CA/token coordinates [N_sample, N_res, 3]
        all_atom: All atom coordinates [N_sample, N_atom, 3]
        support_points: Reference EM support points [N_point, 3]
        noise_bound: Noise bound for TEASER++ (default: 1.0)
        desc_type: Descriptor type (default: "shot_bi_scale")
        desc_r: Descriptor radius (default: 10)
        desc_scales: Number of descriptor scales (default: 2)
        nn_for_normal: Max neighbors for normal estimation (default: 100)
        r_for_normal: Radius for normal estimation (default: 10)
        gicp_downsampling: Downsampling resolution for GICP (default: 1)
        min_neighborhood_size: Min neighborhood size for descriptor computation (default: 100)
        phi: Scale multiplier between descriptor levels (default: 3.0)
        rho: Divisor for subsampling voxel size (default: 10.0)
        map_path: Optional path to EM map file (default: None)
        resolution: Optional resolution of EM map (default: None)
        contour_level: Optional contour level for EM map (default: None)
    Returns:
        Transformed all_atom coordinates, or None if fitting failed
    """
    n_sample = all_atom.shape[0]
    recall_score_list = [0] * n_sample
    ccc_mask_list = [0] * n_sample
    ccc_box_list = [0] * n_sample
    # Convert to numpy
    device = ca_coordinate.device
    dtype = ca_coordinate.dtype
    ca_coordinate_np = ca_coordinate.cpu().numpy()
    all_atom_np = all_atom.cpu().numpy()
    ref_points = support_points.cpu().numpy()

    logger.info(f"Reference points (original): {len(ref_points)}")

    all_atom_fitted = all_atom_np.copy()

    for sample_idx in range(n_sample):
        # Get query points (predicted CA coordinates for this sample)
        query_points = ca_coordinate_np[sample_idx]
        logger.info(f"Sample {sample_idx}: Query points: {len(query_points)}")

        # Create point clouds and estimate normals
        logger.info(f"Sample {sample_idx}: Estimating normals for reference...")
        ref_pcd = o3d.geometry.PointCloud()
        ref_pcd.points = o3d.utility.Vector3dVector(ref_points)
        ref_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=r_for_normal, max_nn=nn_for_normal
            )
        )

        logger.info(f"Sample {sample_idx}: Estimating normals for query...")
        query_pcd = o3d.geometry.PointCloud()
        query_pcd.points = o3d.utility.Vector3dVector(query_points)
        query_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=r_for_normal, max_nn=nn_for_normal
            )
        )

        # Stage 1: Compute descriptors and run TEASER++
        logger.info(
            f"Sample {sample_idx}: Computing {desc_type} descriptors for reference..."
        )
        ref_descriptors = compute_descriptors(
            np.asarray(ref_pcd.points),
            np.asarray(ref_pcd.normals),
            radius=desc_r,
            n_scales=desc_scales,
            descriptor_choice=desc_type,
            min_neighborhood_size=min_neighborhood_size,
            phi=phi,
            rho=rho,
            n_procs=32,
        )
        logger.info(f"Sample {sample_idx}:   Descriptor shape: {ref_descriptors.shape}")

        logger.info(
            f"Sample {sample_idx}: Computing {desc_type} descriptors for query..."
        )
        query_descriptors = compute_descriptors(
            np.asarray(query_pcd.points),
            np.asarray(query_pcd.normals),
            radius=desc_r,
            n_scales=desc_scales,
            descriptor_choice=desc_type,
            min_neighborhood_size=min_neighborhood_size,
            phi=phi,
            rho=rho,
            n_procs=32,
        )
        logger.info(
            f"Sample {sample_idx}:   Descriptor shape: {query_descriptors.shape}"
        )

        # Find correspondences
        logger.info(f"Sample {sample_idx}: Finding correspondences...")
        corrs_ref, corrs_query = find_correspondences(
            ref_descriptors, query_descriptors, mutual_filter=True
        )
        logger.info(f"Sample {sample_idx}:   Found {len(corrs_ref)} correspondences")

        if len(corrs_ref) < 10:
            logger.warning(
                f"Sample {sample_idx}: Too few correspondences ({len(corrs_ref)}) for TEASER++"
            )
            continue

        # Extract corresponding points
        A_corr = ref_points[corrs_ref, :].T
        B_corr = query_points[corrs_query, :].T

        # Run TEASER++
        logger.info(f"Sample {sample_idx}: Running TEASER++...")
        teaser_solver = get_teaser_solver(noise_bound=noise_bound)
        teaser_solver.solve(B_corr, A_corr)
        solution = teaser_solver.getSolution()

        if np.allclose(solution.rotation, 0.0) and np.allclose(
            solution.translation, 0.0
        ):
            logger.warning(f"Sample {sample_idx}: TEASER++ returned invalid solution")
            continue

        # Create transformation matrix
        teaser_transform = np.eye(4)
        teaser_transform[:3, :3] = solution.rotation
        teaser_transform[:3, 3] = solution.translation

        # Transform query points with TEASER++ result
        teaser_transformed_points = (
            query_points @ teaser_transform[:3, :3].T + teaser_transform[:3, 3]
        )

        # Metrics for TEASER++-only alignment (before GICP)
        teaser_query_recall = calculate_query_recall(
            ref_points, teaser_transformed_points, distance_threshold=3.0
        )
        teaser_reference_recall = calculate_reference_recall(
            ref_points, teaser_transformed_points, distance_threshold=3.0
        )
        teaser_f1 = calculate_f1(teaser_query_recall, teaser_reference_recall)
        logger.info(
            f"Sample {sample_idx}: TEASER++ query recall: {teaser_query_recall:.4f}"
        )
        logger.info(
            f"Sample {sample_idx}: TEASER++ reference recall: {teaser_reference_recall:.4f}"
        )
        logger.info(f"Sample {sample_idx}: TEASER++ F1: {teaser_f1:.4f}")

        # Stage 2: GICP refinement
        logger.info(f"Sample {sample_idx}: Running GICP refinement...")
        gicp_result = small_gicp.align(
            ref_points,
            teaser_transformed_points,
            downsampling_resolution=gicp_downsampling,
        )

        # Compute combined transformation (GICP * TEASER)
        combined_transform = np.eye(4)
        combined_transform[:3, :3] = (
            gicp_result.T_target_source[:3, :3] @ teaser_transform[:3, :3]
        )
        combined_transform[:3, 3] = (
            gicp_result.T_target_source[:3, :3] @ teaser_transform[:3, 3]
            + gicp_result.T_target_source[:3, 3]
        )

        # Metrics for TEASER++ + GICP alignment
        final_points = (
            teaser_transformed_points @ gicp_result.T_target_source[:3, :3].T
            + gicp_result.T_target_source[:3, 3]
        )
        final_query_recall = calculate_query_recall(
            ref_points, final_points, distance_threshold=3.0
        )
        recall_score_list[sample_idx] = final_query_recall

        final_reference_recall = calculate_reference_recall(
            ref_points, final_points, distance_threshold=3.0
        )
        final_f1 = calculate_f1(final_query_recall, final_reference_recall)
        logger.info(
            f"Sample {sample_idx}: Final query recall (after GICP): {final_query_recall:.4f}"
        )
        logger.info(
            f"Sample {sample_idx}: Final reference recall (after GICP): {final_reference_recall:.4f}"
        )
        logger.info(f"Sample {sample_idx}: Final F1 (after GICP): {final_f1:.4f}")

        # Compare methods
        logger.info(f"Sample {sample_idx}: ========== Comparison ==========")
        logger.info(
            f"Sample {sample_idx}: TEASER++ - Query: {teaser_query_recall:.4f}, Ref: {teaser_reference_recall:.4f}, F1: {teaser_f1:.4f}"
        )
        logger.info(
            f"Sample {sample_idx}: TEASER++ + GICP - Query: {final_query_recall:.4f}, Ref: {final_reference_recall:.4f}, F1: {final_f1:.4f}"
        )
        if teaser_f1 > final_f1:
            logger.info(
                f"Sample {sample_idx}: Best method: TEASER++ (F1 diff: {teaser_f1 - final_f1:.4f})"
            )
        else:
            logger.info(
                f"Sample {sample_idx}: Best method: TEASER++ + GICP (F1 diff: {final_f1 - teaser_f1:.4f})"
            )
        logger.info(f"Sample {sample_idx}: ================================")

        # Apply to all atoms (always uses combined TEASER++ + GICP transform)
        all_atom_fitted[sample_idx] = (
            all_atom_np[sample_idx] @ combined_transform[:3, :3].T
            + combined_transform[:3, 3]
        )
        ccc_mask, _, ccc_box, _ = calculate_ccc_ovr(
            all_atom_fitted[sample_idx],
            elements,
            map_path=map_path,
            resolution=resolution,
            contour_level=contour_level,
        )
        ccc_mask_list[sample_idx] = ccc_mask
        ccc_box_list[sample_idx] = ccc_box

    if sum(recall_score_list) == 0:
        return None, recall_score_list, ccc_mask_list, ccc_box_list

    return (
        torch.from_numpy(all_atom_fitted).to(device=device, dtype=dtype),
        recall_score_list,
        ccc_mask_list,
        ccc_box_list,
    )


def PointResidueMatching(P, threshold=0.4):
    """
    Match points and residues based on a predicted distance distribution (P)
    and threshold requirements (distance <= 5 A with high probability >= 0.8).

    :param P: torch.Tensor of shape (p, r, 10)
        Point-residue distance probability distribution (after softmax).
        The 10 distance bins correspond to approximate distance boundaries
        [0.5, 1, 2, 3, 5, 7.5, 10, 12.5, 15, and >15].
    :return:
        res2pnts: dictionary mapping {res_idx: set_of_point_indices}
        dist_bin: torch.Tensor of shape (p, r)
                  argmax index of the predicted distance bin for each (point, residue)
    """
    P = nn.functional.softmax(P, dim=-1)
    # 1) Find the predicted distance bin (argmax over axis=2)
    #    -> shape (p, r), values in [0..9]
    dist_bin = torch.argmax(P, dim=2)  # shape (p, r)

    # 2) Create a mask for point-residue pairs where:
    #    - predicted distance bin <= 4 (i.e., within 5 A, because bins [0..4] roughly
    #      correspond to distances up to 5 A), AND
    #    - sum of probabilities from bins [0..4] > 0.8 (i.e., high confidence
    #      of being <= 5 A)
    mask = (dist_bin <= 4) & (P[:, :, :5].sum(dim=2) > threshold)

    # 3) Extract the point indices (pnt_idx) and residue indices (res_idx)
    #    for which the mask is True.
    pnt_idx, res_idx = torch.where(mask)

    # 4) Build a dictionary that maps each residue index to the set of
    #    point indices that satisfy the criteria.
    res2pnts = defaultdict(set)
    for i in range(len(pnt_idx)):
        # Note that res_idx[i] is the residue index, pnt_idx[i] is the point index
        r_idx = res_idx[i].item()
        p_idx = pnt_idx[i].item()
        res2pnts[r_idx].add(p_idx)

    return dict(res2pnts), dist_bin


def get_common_points(points_list: list[set[int]]) -> set[int]:
    """
    Given a list of sets of point indices, return the intersection
    (common points) among all sets.
    """
    if not points_list:
        return set()
    common_set = points_list[0].copy()
    for s in points_list[1:]:
        common_set &= s
    return common_set


def filter_close_points(
    point_set: set[int], Q: torch.Tensor, distance_threshold: float = 8.0
) -> set[int]:
    """
    Given a set of point indices 'point_set' and their coordinates Q (shape: (p, 3)),
    remove points that lie within 'distance_threshold' of one another.

    Returns a subset of point_set such that no two points are within the threshold.
    This naive implementation is O(N^2); for large sets, consider more efficient methods.

    Args:
        point_set (Set[int]): A set of point indices.
        Q (torch.Tensor): Coordinates of shape (p, 3), where p = number of points.
        distance_threshold (float): Distance cutoff. Points closer than this
                                    are not kept simultaneously.

    Returns:
        Set[int]: A subset of point_set where no two points are within distance_threshold.
    """
    point_list = list(point_set)
    keep = []

    for p_idx in point_list:
        p_coord = Q[p_idx]  # shape: (3,)
        # Check distance to any point already in 'keep'
        is_far_enough = True
        for kept_idx in keep:
            dist = torch.norm(p_coord - Q[kept_idx])
            if dist < distance_threshold:
                is_far_enough = False
                break
        if is_far_enough:
            keep.append(p_idx)

    return set(keep)


def SetPntResAffinity(
    res2pnts: dict[int, set[int]],
    asym_ids,
    entity_ids,
    sym_ids,
    Q,
    distance_threshold: float = 8.0,
) -> torch.Tensor:
    """
    Set Point-Residue Affinity during Inference, using PyTorch.

    Args:
        res2pnts (dict): A mapping residue_index -> set(point_indices).
                         E.g. {res_i: {pnt1, pnt2, ...}, ...}
        asym_ids (array-like of shape (r,)):
            Identifies chain number for each residue (AlphaFold 'asym_id').
        entity_ids (array-like of shape (r,)):
            Identifies each set of identical chains (AlphaFold 'entity_id').
        sym_ids (array-like of shape (r,)):
            Identifies symmetry index within a set of identical chains (AlphaFold 'sym_id').
        Q (array-like or torch.Tensor of shape (p,3)):
            Coordinates of p points.
        distance_threshold (float): Minimum distance allowed between chosen points
                                    for symmetrical residues.

    Returns:
        Affinity (torch.Tensor of shape (p, r)):
            A binary matrix indicating chosen point-residue affinity (1 if chosen, 0 otherwise).
    """
    p = Q.size(0)  # number of points
    r = asym_ids.size(0)  # number of residues

    # Initialize an empty (p x r) affinity matrix
    Affinity = torch.zeros((p, r), dtype=torch.bfloat16)

    # Loop over each unique entity
    for entity_id in torch.unique(entity_ids):
        # Step 1: select all residues belonging to the current entity
        mask1 = entity_ids == entity_id

        # The maximum sym_id for these residues
        max_sym_in_this_entity = sym_ids[mask1].max()

        # If there is no actual symmetry (max_sym == 0), skip
        if max_sym_in_this_entity.item() == 0:
            continue

        # Step 2: For each chain (unique asym_id) within this entity, collect residue indices
        entity_asym_ids = torch.unique(asym_ids[mask1])
        asym_list = []
        for cur_asym_id in entity_asym_ids:
            mask2 = mask1 & (asym_ids == cur_asym_id)
            # gather the residue indices for this chain
            chain_res_indices = torch.where(mask2)[0]
            asym_list.append(chain_res_indices)

        # 'num_sym' is the number of symmetrical chains
        num_sym = len(asym_list)
        if num_sym < 2:
            # Means the entity was labeled symmetrical but we only found 1 chain in it
            continue

        # We'll assume each asym_list[i] has the same number of residues (same positions).
        # That is a typical assumption in symmetrical chains.
        num_res_of_sym = len(asym_list[0])
        # If any chain does not have the same residue count, skip
        if any(len(lst) != num_res_of_sym for lst in asym_list):
            continue

        # We'll gather potential (res_list, pnt_list) pairs in a list
        pairs = []

        # Step 3: For each residue position 'idx' along one chain
        for idx in range(num_res_of_sym):
            # sym_res_list: the corresponding residue indices across all symmetrical chains
            sym_res_list = [chain[idx].item() for chain in asym_list]
            # E.g. [res_of_chain1, res_of_chain2, ...]

            # Check if each of those residues has at least 'num_sym' points
            # in res2pnts (meaning we can possibly pick a point for each chain).
            has_enough_points = True
            for r_ in sym_res_list:
                if r_ not in res2pnts or len(res2pnts[r_]) < num_sym:
                    has_enough_points = False
                    break
            if not has_enough_points:
                continue

            # Get the sets of points for these symmetrical residues
            points_sets = [res2pnts[r_] for r_ in sym_res_list]
            # Intersection of all these sets
            common_pnt_list = get_common_points(points_sets)

            # Must have at least 'num_sym' points in common
            if len(common_pnt_list) < num_sym:
                continue

            # Filter out points that are too close to each other (< distance_threshold)
            common_pnt_list = filter_close_points(
                common_pnt_list, Q, distance_threshold
            )

            # Check again if we still have enough points
            if len(common_pnt_list) >= num_sym:
                # Randomly pick exactly 'num_sym' points
                chosen_points = random.sample(list(common_pnt_list), num_sym)
                # Store this candidate pair
                pairs.append((sym_res_list, chosen_points))

        # Step 4: If we have any candidate pairs, pick one at random
        if len(pairs) > 0:
            sym_res_list, chosen_points = random.choice(pairs)
            # Mark them in the Affinity matrix
            for i in range(len(sym_res_list)):
                Affinity[chosen_points[i], sym_res_list[i]] = 1.0

    return Affinity


def FitModelPoints(
    point_residue_logits,
    ca_coordinate,
    all_atom,
    elements,
    support_points,
    all_support_points,
    confidence=0.8,
    dump_dir=None,
    pdb_id=None,
    map_path: str | None = None,
    resolution: float | None = None,
    contour_level: float | None = None,
):
    n_sample = all_atom.shape[0]
    recall_score_list = [0] * n_sample
    ccc_mask_list = [0] * n_sample
    ccc_box_list = [0] * n_sample
    res2pnts, _ = PointResidueMatching(
        point_residue_logits.squeeze(0), threshold=confidence
    )
    if len(res2pnts) < 3:
        print(len(res2pnts), "< 3 pairs")
        return None, recall_score_list, ccc_mask_list, ccc_box_list

    device = ca_coordinate.device
    dtype = ca_coordinate.dtype
    ca_coordinate = ca_coordinate.cpu().numpy()
    all_atom = all_atom.cpu().numpy()
    support_points = support_points.cpu().numpy()
    all_support_points = all_support_points.cpu().numpy()

    all_atom_fitted = all_atom.copy()
    ca_coordinate_fitted = ca_coordinate.copy()

    for i in range(n_sample):
        res_idx = list(res2pnts.keys())
        pnt_idx = [
            random.choice(tuple(res2pnts[i])) for i in res_idx
        ]  # random choice per pair

        pairing_pnt_coors = support_points[pnt_idx]  # support-point coordinates
        pairing_res_coors = ca_coordinate[i][res_idx]  # corresponding atom coordinates

        # Optimal (least-squares) rigid-body transform = Kabsch/SVD
        sup = SVDSuperimposer()
        sup.set(pairing_pnt_coors, pairing_res_coors)
        sup.run()  # SVD super-imposer
        rot, trans = sup.get_rotran()[:2]

        # Apply to **all** atoms
        all_atom_fitted[i] = (all_atom[i] @ rot) + trans
        ca_coordinate_fitted[i] = (ca_coordinate[i] @ rot) + trans

        recall = calculate_query_recall(
            all_support_points, ca_coordinate_fitted[i], distance_threshold=3.0
        )
        ccc_mask, _, ccc_box, _ = calculate_ccc_ovr(
            all_atom_fitted[i],
            elements,
            map_path=map_path,
            resolution=resolution,
            contour_level=contour_level,
        )
        recall_score_list[i] = recall
        ccc_mask_list[i] = ccc_mask
        ccc_box_list[i] = ccc_box

        # saved_data_dict = {
        #     'sample_idx': i,
        #     'recall_score': recall,
        #     'res_idx': res_idx,
        #     'pnt_idx': pnt_idx,
        #     'pairing_pnt_coors': pairing_pnt_coors,
        #     'pairing_res_coors': pairing_res_coors,
        #     'rmsd': sup.get_rms()
        # }

    # if dump_dir is not None:
    #     with open(f"{dump_dir}/saved_data/svd_data_{confidence}_{pdb_id}.pkl", "wb") as f:
    #         pickle.dump(saved_data_dict, f)
    # Convert back to PyTorch tensor with the same device and dtype as all_atom
    return (
        torch.from_numpy(all_atom_fitted).to(device=device, dtype=dtype),
        recall_score_list,
        ccc_mask_list,
        ccc_box_list,
    )


def FitModelVESPER(
    all_atom: torch.Tensor,
    atom_array,
    entity_poly_type: dict[str, str],
    pdb_id: str,
    contour_level: float,
    resolution: float = 3.0,
    num_conformation: int = 10,
    gpu_id: int = 0,
    angle_spacing: float = 5.0,
    voxel_spacing: float = 2.0,
    gaussian_bandwidth: float = 1.0,
    num_threads: int = 8,
    output_dir: str | None = None,
    support_points: torch.Tensor = None,
    map_path: str | None = None,
    resolution_map: float | None = None,
    contour_level_map: float | None = None,
) -> torch.Tensor | None:
    """
    Fit model points to EM map using VESPER alignment.

    Unlike TEASER++ and SVD which use extracted support points, VESPER directly
    aligns the structure to the original EM map by simulating a density from the
    predicted structure and performing map-to-map alignment.

    This function:
    1. Creates a CIF file from the predicted structure (saved as *_vanilla.cif)
    2. Runs VESPER to get the optimal rotation/translation
    3. Applies the transformation to all atom coordinates

    Args:
        all_atom: All atom coordinates [N_sample, N_atom, 3]
        atom_array: Biotite AtomArray object with structure metadata
        entity_poly_type: Entity poly type information for CIF writing
        pdb_id: PDB ID for the structure
        map_path: Path to reference MRC map (supports .mrc, .map, and .gz compressed)
        contour_level: Density threshold for reference map
        resolution: Resolution for simulating map from structure (default: 3.0)
        num_conformation: Number of top conformations to evaluate (default: 10)
        gpu_id: GPU device ID (default: 0)
        angle_spacing: Angular sampling interval in degrees (default: 5.0)
        voxel_spacing: Voxel spacing for sampling (default: 2.0)
        gaussian_bandwidth: Gaussian filter bandwidth (default: 1.0)
        num_threads: Number of CPU threads (default: 8)
        output_dir: Directory to save vanilla CIF files (optional)

    Returns:
        Transformed all_atom coordinates [N_sample, N_atom, 3], or None if fitting failed
    """
    import copy
    import gzip
    import os
    import shutil
    import tempfile

    from vesper.api import run_vesper_fit

    from cryozeta.data.utils import save_atoms_to_cif

    n_sample = all_atom.shape[0]
    recall_score_list = [0] * n_sample
    ccc_mask_list = [0] * n_sample
    ccc_box_list = [0] * n_sample

    device = all_atom.device
    dtype = all_atom.dtype
    all_atom_np = all_atom.cpu().numpy()
    all_atom_fitted = all_atom_np.copy()

    # Handle gzip-compressed map files
    map_path_resolved = map_path
    temp_map_file = None
    if map_path.endswith(".gz"):
        logger.info(f"Decompressing gzip map file: {map_path}")
        # Decompress to a temporary file
        temp_map_fd, temp_map_path = tempfile.mkstemp(suffix=".map")
        os.close(temp_map_fd)
        try:
            with gzip.open(map_path, "rb") as f_in:
                with open(temp_map_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            map_path_resolved = temp_map_path
            temp_map_file = temp_map_path
            logger.info(f"Decompressed map saved to: {temp_map_path}")
        except Exception as e:
            logger.error(f"Failed to decompress map file: {e}")
            if os.path.exists(temp_map_path):
                os.remove(temp_map_path)
            return None

    # Create output directory for vanilla CIF files if specified
    vanilla_cif_dir = None
    if output_dir:
        vanilla_cif_dir = os.path.join(output_dir, "predictions_vanilla")
        os.makedirs(vanilla_cif_dir, exist_ok=True)
        logger.info(f"Vanilla CIF files will be saved to: {vanilla_cif_dir}")

    try:
        for sample_idx in range(n_sample):
            logger.info(f"Sample {sample_idx}: ========== VESPER Fitting ==========")

            # Create a CIF file from the predicted structure
            pred_atom_array = copy.deepcopy(atom_array)
            pred_atom_array.coord = all_atom_np[sample_idx]

            # Determine CIF output path
            if vanilla_cif_dir:
                # Save to output directory with _vanilla.cif suffix
                cif_path = os.path.join(
                    vanilla_cif_dir, f"{pdb_id}_sample_{sample_idx}_vanilla.cif"
                )
            else:
                # Use a temporary directory
                tmpdir = tempfile.mkdtemp()
                cif_path = os.path.join(tmpdir, f"{pdb_id}_sample_{sample_idx}.cif")

            save_atoms_to_cif(
                output_cif_file=cif_path,
                atom_array=pred_atom_array,
                entity_poly_type=entity_poly_type,
                pdb_id=pdb_id,
            )
            logger.info(f"Sample {sample_idx}: Saved vanilla CIF to {cif_path}")
            logger.info(
                f"Sample {sample_idx}: CIF file size: {os.path.getsize(cif_path)} bytes"
            )

            # Log VESPER parameters
            logger.info(f"Sample {sample_idx}: VESPER parameters:")
            logger.info(f"  map_path: {map_path_resolved}")
            logger.info(f"  contour_level: {contour_level}")
            logger.info(f"  resolution: {resolution}")
            logger.info(f"  num_conformation: {num_conformation}")
            logger.info(f"  gpu_id: {gpu_id}")
            logger.info(f"  angle_spacing: {angle_spacing}")
            logger.info(f"  voxel_spacing: {voxel_spacing}")
            logger.info(f"  gaussian_bandwidth: {gaussian_bandwidth}")
            logger.info(f"  num_threads: {num_threads}")

            # Run VESPER alignment
            try:
                logger.info(f"Sample {sample_idx}: Running VESPER alignment...")
                rotation, translation = run_vesper_fit(
                    map_path=map_path_resolved,
                    input_pdb=cif_path,
                    contour_level=contour_level,
                    resolution=resolution,
                    num_conformation=num_conformation,
                    gpu_id=gpu_id,
                    angle_spacing=angle_spacing,
                    voxel_spacing=voxel_spacing,
                    gaussian_bandwidth=gaussian_bandwidth,
                    num_threads=num_threads,
                    only_best=True,
                )
                logger.info(f"Sample {sample_idx}: VESPER fitting succeeded")

                # Log transformation details
                logger.info(f"Sample {sample_idx}: Rotation matrix:")
                logger.info(
                    f"  [{rotation[0, 0]:.6f}, {rotation[0, 1]:.6f}, {rotation[0, 2]:.6f}]"
                )
                logger.info(
                    f"  [{rotation[1, 0]:.6f}, {rotation[1, 1]:.6f}, {rotation[1, 2]:.6f}]"
                )
                logger.info(
                    f"  [{rotation[2, 0]:.6f}, {rotation[2, 1]:.6f}, {rotation[2, 2]:.6f}]"
                )
                logger.info(
                    f"Sample {sample_idx}: Translation: [{translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}]"
                )

                # Apply transformation to all atoms
                all_atom_fitted[sample_idx] = (
                    all_atom_np[sample_idx] @ rotation.T + translation
                )

                ref_points = support_points.cpu().numpy()
                query_points = all_atom_fitted[sample_idx]
                recall_score_list[sample_idx] = calculate_query_recall(
                    ref_points, query_points, distance_threshold=3.0
                )
                ccc_mask, _, ccc_box, _ = calculate_ccc_ovr(
                    all_atom_fitted[sample_idx],
                    atom_array.element,
                    map_path=map_path,
                    resolution=resolution_map,
                    contour_level=contour_level_map,
                )
                ccc_mask_list[sample_idx] = ccc_mask
                ccc_box_list[sample_idx] = ccc_box

                # Log coordinate statistics before and after transformation
                orig_coords = all_atom_np[sample_idx]
                fitted_coords = all_atom_fitted[sample_idx]
                logger.info(f"Sample {sample_idx}: Coordinate statistics:")
                logger.info(
                    f"  Original - min: [{orig_coords.min(axis=0)[0]:.2f}, {orig_coords.min(axis=0)[1]:.2f}, {orig_coords.min(axis=0)[2]:.2f}], max: [{orig_coords.max(axis=0)[0]:.2f}, {orig_coords.max(axis=0)[1]:.2f}, {orig_coords.max(axis=0)[2]:.2f}]"
                )
                logger.info(
                    f"  Fitted   - min: [{fitted_coords.min(axis=0)[0]:.2f}, {fitted_coords.min(axis=0)[1]:.2f}, {fitted_coords.min(axis=0)[2]:.2f}], max: [{fitted_coords.max(axis=0)[0]:.2f}, {fitted_coords.max(axis=0)[1]:.2f}, {fitted_coords.max(axis=0)[2]:.2f}]"
                )

                logger.info(
                    f"Sample {sample_idx}: ====================================="
                )

            except Exception as e:
                import traceback

                logger.warning(f"Sample {sample_idx}: VESPER fitting failed: {e}")
                logger.warning(
                    f"Sample {sample_idx}: Traceback:\n{traceback.format_exc()}"
                )
                # Keep original coordinates on failure
                continue
    finally:
        # Clean up temporary decompressed map file
        if temp_map_file and os.path.exists(temp_map_file):
            os.remove(temp_map_file)
            logger.info(f"Cleaned up temporary map file: {temp_map_file}")

    return (
        torch.from_numpy(all_atom_fitted).to(device=device, dtype=dtype),
        recall_score_list,
        ccc_mask_list,
        ccc_box_list,
    )
