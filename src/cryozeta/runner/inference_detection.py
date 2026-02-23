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

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import typer
from loguru import logger

from cryozeta.em import (
    MapObject,
    crop_mrc,
    get_detection_model,
    get_shifted_indices,
    normalize_mrc,
    parse_mrc,
    resample_mrc,
    save_mrc,
    sliding_window_inference,
    write_coords_to_pdb,
)
from cryozeta.em.utils import _meanshiftpp_gpu_fallback


@dataclass
class CryoEMInferenceConfig:
    """Configuration for cryo-EM inference"""

    roi_size: int = 64
    batch_size: int = 4
    compile: bool = False
    device: str = "cuda"
    resolution: float | None = None
    contour_level: float | None = None
    resolution_cutoff: float = 4.5
    ca_wgt: float = 1.0
    bb_wgt: float = 1.0
    c1p_wgt: float = 3.0
    sg_wgt: float = 1.0
    ca_threshold_highres: float = 0.25
    c1p_threshold_highres: float = 0.20
    ca_threshold_lowres: float = 0.20
    c1p_threshold_lowres: float = 0.15
    save_map_predictions: bool = False
    contour_level_scale: float = 0.5
    no_protein: bool = False
    no_dna_rna: bool = False
    interp_threshold: int = 3000


def interpolate_and_sample(sampled_indices, atom_label, num_steps=5, num_points=None):
    """Interpolate between pairs of points and sample values from a multi-channel label map.

    Args:
        sampled_indices: Tensor of shape [N, 3] containing coordinates to interpolate between
        atom_label: Tensor of shape [D, H, W, C] containing label values
        num_steps: Number of interpolation steps between each pair of points
        num_points: Optional number of points to randomly sample for interpolation

    Returns:
        Tensor of interpolated values with shape [N, M, num_steps, C] where M is min(N, num_points)
    """
    # Randomly sample points if specified
    if num_points is not None and len(sampled_indices) > num_points:
        torch.manual_seed(42)  # For reproducibility
        perm = torch.randperm(len(sampled_indices))[:num_points]
        sampled_for_interp = sampled_indices[perm]
    else:
        sampled_for_interp = sampled_indices

    steps = torch.arange(
        1, num_steps + 1, dtype=torch.float32, device=sampled_indices.device
    ) / (num_steps + 1)

    indices_i = sampled_indices.unsqueeze(1).unsqueeze(2)  # [N, 1, 1, 3]
    indices_j = sampled_for_interp.unsqueeze(0).unsqueeze(2)  # [1, M, 1, 3]
    steps = steps.view(1, 1, -1, 1)  # [1, 1, num_steps, 1]

    interpolated_points = indices_i + (indices_j - indices_i) * steps

    interpolated_indices = torch.round(interpolated_points).to(torch.int64)
    max_dims = (
        torch.tensor(atom_label.shape[:3], device=interpolated_indices.device) - 1
    )
    min_dims = torch.zeros(3, device=interpolated_indices.device, dtype=torch.int64)

    interpolated_indices = torch.clamp(
        interpolated_indices, min=min_dims, max=max_dims
    ).cpu()

    interpolated_values = atom_label[
        interpolated_indices[..., 0],
        interpolated_indices[..., 1],
        interpolated_indices[..., 2],
        :,
    ]

    return interpolated_values


def post_process(
    maps_dict,
    voxel_size,
    global_origin,
    output_dir: Path,
    emdb_id: str,
    num_points: int | None = None,
    resolution: float = 1.0,
    disable_interpolation: bool = False,
    config: CryoEMInferenceConfig = None,
):
    """Post-process inference results for a single entry"""
    logger.info(f"Post-processing {emdb_id}...")

    fmaxd = 2.0
    fsiv = 0.5
    n_steps = 100
    tol = 1e-5

    prot_ca_map_np = maps_dict["prot_ca_map"].cpu().numpy()
    rdna_c1p_map_np = maps_dict["rdna_c1p_map"].cpu().numpy()

    if resolution <= config.resolution_cutoff:
        prot_ca_indices = (
            (maps_dict["prot_ca_map"] >= config.ca_threshold_highres).nonzero().float()
        )
        rdna_c1p_indices = (
            (maps_dict["rdna_c1p_map"] >= config.c1p_threshold_highres)
            .nonzero()
            .float()
        )
    else:
        prot_ca_indices = (
            (
                (
                    (
                        config.ca_wgt * maps_dict["prot_ca_map"]
                        + config.bb_wgt * maps_dict["prot_bb_map"]
                    )
                    / (config.ca_wgt + config.bb_wgt)
                )
                >= config.ca_threshold_lowres
            )
            .nonzero()
            .float()
        )
        rdna_c1p_indices = (
            (
                (
                    (
                        config.c1p_wgt * maps_dict["rdna_c1p_map"]
                        + config.sg_wgt * maps_dict["rdna_sugar_map"]
                    )
                    / (config.c1p_wgt + config.sg_wgt)
                )
                >= config.c1p_threshold_lowres
            )
            .nonzero()
            .float()
        )

    # Initialize cluster ids
    prot_ca_cluster_ids = None
    rdna_c1p_cluster_ids = None
    main_atom_indices = None

    if len(prot_ca_indices) > 0:
        prot_ca_shifted_indices = get_shifted_indices(
            point_cd=prot_ca_indices,
            reference_np=prot_ca_map_np,
            fmaxd=fmaxd,
            fsiv=fsiv,
            n_steps=n_steps,
            tol=tol,
        )
        logger.info(f"[{emdb_id}] Number of CA indices: {len(prot_ca_shifted_indices)}")
        prot_ca_clustered_indices = _meanshiftpp_gpu_fallback(
            prot_ca_shifted_indices, bandwidth=5.0, n_steps=100, tol=1e-5
        )
        _, prot_ca_cluster_ids = torch.unique(
            prot_ca_clustered_indices, dim=0, return_inverse=True
        )
        main_atom_indices = prot_ca_shifted_indices.clone()
        num_ca_atoms = len(prot_ca_shifted_indices)
    else:
        logger.warning(f"No protein CA indices found for {emdb_id}")
        # Initialize with empty tensor
        main_atom_indices = torch.empty(0, 3)
        prot_ca_cluster_ids = torch.empty(0, dtype=torch.long)
        num_ca_atoms = 0

    if len(rdna_c1p_indices) > 0:
        rdna_c1p_shifted_indices = get_shifted_indices(
            point_cd=rdna_c1p_indices,
            reference_np=rdna_c1p_map_np,
            fmaxd=fmaxd,
            fsiv=fsiv,
            n_steps=n_steps,
            tol=tol,
        )
        logger.info(
            f"[{emdb_id}] Number of C1P(R/DNA) indices: {len(rdna_c1p_shifted_indices)}"
        )
        rdna_c1p_clustered_indices = _meanshiftpp_gpu_fallback(
            rdna_c1p_shifted_indices, bandwidth=5.0, n_steps=100, tol=1e-5
        )
        _, rdna_c1p_cluster_ids = torch.unique(
            rdna_c1p_clustered_indices, dim=0, return_inverse=True
        )
        if main_atom_indices.numel() > 0:
            main_atom_indices = torch.cat(
                [main_atom_indices, rdna_c1p_shifted_indices], dim=0
            )
        else:
            main_atom_indices = rdna_c1p_shifted_indices.clone()

    # Handle case where no atoms are found
    if main_atom_indices is None or main_atom_indices.numel() == 0:
        logger.warning(f"No main atom indices found for {emdb_id}")
        return

    num_prot_ca_clusters = (
        len(torch.unique(prot_ca_cluster_ids)) if prot_ca_cluster_ids.numel() > 0 else 0
    )
    cluster_ids = (
        prot_ca_cluster_ids.clone()
        if prot_ca_cluster_ids.numel() > 0
        else torch.empty(0, dtype=torch.long)
    )
    logger.info(f"[{emdb_id}] Number of protein CA clusters: {num_prot_ca_clusters}")

    if rdna_c1p_cluster_ids is not None and rdna_c1p_cluster_ids.numel() > 0:
        num_rdna_c1p_clusters = len(torch.unique(rdna_c1p_cluster_ids))
        logger.info(
            f"[{emdb_id}] Number of R/DNA C1P clusters: {num_rdna_c1p_clusters}"
        )
        rdna_c1p_cluster_ids += num_prot_ca_clusters
        cluster_ids = torch.cat([cluster_ids, rdna_c1p_cluster_ids], dim=0)
    else:
        num_rdna_c1p_clusters = 0
        logger.warning(f"No R/DNA C1P indices found for {emdb_id}")

    logger.info(f"[{emdb_id}] Number of clusters: {len(torch.unique(cluster_ids))}")

    main_atom_prob = torch.stack(
        [
            maps_dict["prot_ca_map"],
            maps_dict["rdna_c1p_map"],
            maps_dict["rdna_c1p_map"],
        ],
        dim=0,
    )
    logger.info(f"[{emdb_id}] Main atom prob shape: {main_atom_prob.shape}")
    # permute to (D, H, W, C)
    main_atom_prob = main_atom_prob.permute(1, 2, 3, 0)

    logger.info(f"[{emdb_id}] voxel_size: {voxel_size}, global_origin: {global_origin}")
    main_atom_coords = (
        main_atom_indices.cpu().numpy()[:, ::-1] * voxel_size + global_origin
    )
    logger.info(f"[{emdb_id}] main_atom_coords shape: {main_atom_coords.shape}")
    ca_coords = main_atom_coords[:num_ca_atoms]
    c1p_coords = main_atom_coords[num_ca_atoms:]
    if len(ca_coords) > 0:
        write_coords_to_pdb(ca_coords, output_dir / f"{emdb_id}_CA.pdb", atom_type="CA")
    if len(c1p_coords) > 0:
        write_coords_to_pdb(
            c1p_coords, output_dir / f"{emdb_id}_C1P.pdb", atom_type="C1P"
        )
    main_atom_indices_int = main_atom_indices.to(torch.int64)
    main_atom_probs = main_atom_prob[
        main_atom_indices_int[:, 0],
        main_atom_indices_int[:, 1],
        main_atom_indices_int[:, 2],
        :,
    ]

    res_features = maps_dict["residue_features"].permute(1, 2, 3, 0)  # (D, H, W, C)
    res_features = res_features[
        main_atom_indices_int[:, 0],
        main_atom_indices_int[:, 1],
        main_atom_indices_int[:, 2],
        :,
    ]

    # Calculate interpolated confidence with optional random sampling
    num_total_points = len(main_atom_indices)
    if disable_interpolation:
        logger.info(f"[{emdb_id}] Disabling interpolation")
        interpolated_confidence = None
    elif config is not None and num_total_points > config.interp_threshold:
        logger.info(
            f"[{emdb_id}] Number of points ({num_total_points}) exceeds "
            f"interp_threshold ({config.interp_threshold}), skipping interpolation"
        )
        interpolated_confidence = None
    else:
        if num_points is not None:
            logger.info(
                f"[{emdb_id}] Using random sampling with {num_points} points for interpolation"
            )
        interpolated_confidence = interpolate_and_sample(
            main_atom_indices, main_atom_prob, num_points=num_points
        )

    pt_dict = {
        "emdb_id": emdb_id,
        "main_atom_coords": main_atom_coords,
        "main_atom_probs": main_atom_probs,
        "res_features": res_features,
        "cluster_ids": cluster_ids.cpu(),
        "interpolate_confidence": interpolated_confidence.cpu()
        if interpolated_confidence is not None
        else None,
    }

    output_pt_path = output_dir / f"{emdb_id}.pt"
    torch.save(pt_dict, output_pt_path)
    logger.info(f"[{emdb_id}] Saved output to {output_pt_path}")


class CryoEMInference:
    def __init__(self, config: CryoEMInferenceConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using device: {self.device}")
        logger.info(
            f"PyTorch version: {torch.__version__}, "
            f"CUDA compiled version: {torch.version.cuda}, "
            f"cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}"
        )

        # Load models
        logger.info("Loading models...")
        self.detection_model = get_detection_model(
            load_pretrained=True, compile=config.compile
        )
        self.detection_model.to(self.device).to(torch.bfloat16)

        logger.info("Models loaded successfully")

    def preprocess_map(self, mrc_path: Path, emdb_id: str):
        """Download and preprocess map for given MRC file"""
        logger.info(f"Preprocessing map from {mrc_path}")

        if not mrc_path.exists():
            raise FileNotFoundError(f"MRC file not found: {mrc_path}")

        original_mrc_obj = parse_mrc(mrc_path)

        contour_level = self.config.contour_level
        if contour_level is None:
            raise ValueError(
                f"contour_level must be provided in the config for {emdb_id}"
            )
        logger.info(f"Using contour level for {emdb_id}: {contour_level}")

        # For inference, we assume the map is already processed
        # But we can still do basic normalization and cropping
        resampled_mrc_obj = resample_mrc(original_mrc_obj, 1.0)
        normalized_mrc_obj = normalize_mrc(
            resampled_mrc_obj,
            contour_level=contour_level * self.config.contour_level_scale,
        )
        unified_mrc_obj = crop_mrc(
            normalized_mrc_obj, extended_val=8, min_spatial_size=self.config.roi_size
        )

        emap = (
            torch.from_numpy(unified_mrc_obj.grid_data)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
        )
        logger.info(f"Finished preprocessing, shape: {emap.shape}")

        return emap, unified_mrc_obj.voxel_size, unified_mrc_obj.global_origin

    @torch.no_grad()
    def run_inference(
        self,
        mrc_path: Path,
        output_dir: Path,
        emdb_id: str | None = None,
        num_points: int | None = None,
        disable_interpolation: bool = False,
        overwrite: bool = False,
    ):
        """Run inference for a single MRC file"""
        if emdb_id is None:
            emdb_id = mrc_path.stem.split(".")[0]
            if "_" in emdb_id:
                emdb_id = emdb_id.split("_")[1]
            elif "-" in emdb_id:
                emdb_id = emdb_id.split("-")[1]

        # Skip if output already exists
        output_pt_path = output_dir / f"{emdb_id}.pt"
        if not overwrite and output_pt_path.exists():
            logger.info(
                f"[{emdb_id}] Output already exists at {output_pt_path}, skipping. "
                "Use --overwrite to force."
            )
            return emdb_id

        logger.info(f"Running inference for {emdb_id}")

        resolution = self.config.resolution
        if resolution is None:
            raise ValueError(f"resolution must be provided in the config for {emdb_id}")
        logger.info(f"Using resolution for {emdb_id}: {resolution}")

        t0 = time.time()
        # Preprocess map
        emap, voxel_size, global_origin = self.preprocess_map(mrc_path, emdb_id)

        # Check map size
        spatial_size = emap.shape[2:]
        if any(dim > 600 for dim in spatial_size):
            logger.warning(
                f"Large spatial size {spatial_size} for {emdb_id}, this may take a while"
            )

        t1 = time.time()
        logger.info(f"Start inference for {emdb_id}")

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(self.device.type == "cuda"),
        ):
            # Move input to device
            emap = emap.to(self.device, dtype=torch.bfloat16)  # (1, 1, D, H, W)
            spatial_mask = (emap > 0.0).squeeze(0).squeeze(0).cpu()  # (D, H, W)

            detection_output = sliding_window_inference(
                input_map=emap,
                output_num_channels=9 + 25,
                roi_size=self.config.roi_size,
                batch_size=self.config.batch_size,
                model=self.detection_model,
                device=self.device,
                gaussian=True,
            )

            t2 = time.time()

            atom_prob = detection_output[:, :9, ...]  # (1, 9, D, H, W)
            residue_prob = detection_output[:, 9:, ...]  # (1, 25, D, H, W)

            # Residue rearrangement: 20 AAs + 4 DNA + 4 RNA (duplicate DRNA logits)
            aa_probs = residue_prob[0, 1:21, ...]
            drna_probs = residue_prob[0, 21:25, ...]
            residue_features = torch.cat(
                [aa_probs, drna_probs, drna_probs], dim=0
            )  # (28, D, H, W)

            prot_ca_proba = (
                atom_prob[0, 2, ...]
                * spatial_mask
                * (0.0 if self.config.no_protein else 1.0)
            )
            prot_bb_proba = (
                atom_prob[0, 1, ...]
                * spatial_mask
                * (0.0 if self.config.no_protein else 1.0)
            )
            rdna_c1p_proba = (
                atom_prob[0, 5, ...]
                * spatial_mask
                * (0.0 if self.config.no_dna_rna else 1.0)
            )
            rdna_sugar_proba = (
                atom_prob[0, 6, ...]
                * spatial_mask
                * (0.0 if self.config.no_dna_rna else 1.0)
            )
            residue_prob = residue_prob[0, ...] * spatial_mask.unsqueeze(0)

            # Save MRC files
            if self.config.save_map_predictions:
                ca_prb_mrc_obj = MapObject(
                    grid_data=prot_ca_proba.cpu().numpy(),
                    voxel_size=voxel_size,
                    global_origin=global_origin,
                )
                save_mrc(ca_prb_mrc_obj, output_dir / f"{emdb_id}_prot_ca_prb.mrc")
                logger.info(f"Saved {emdb_id}_prot_ca_prb.mrc")
                rdna_c1p_prb_mrc_obj = MapObject(
                    grid_data=rdna_c1p_proba.cpu().numpy(),
                    voxel_size=voxel_size,
                    global_origin=global_origin,
                )
                save_mrc(
                    rdna_c1p_prb_mrc_obj, output_dir / f"{emdb_id}_rdna_c1p_prb.mrc"
                )
                logger.info(f"Saved {emdb_id}_rdna_c1p_prb.mrc")
                prot_bb_prb_mrc_obj = MapObject(
                    grid_data=prot_bb_proba.cpu().numpy(),
                    voxel_size=voxel_size,
                    global_origin=global_origin,
                )
                save_mrc(prot_bb_prb_mrc_obj, output_dir / f"{emdb_id}_prot_bb_prb.mrc")
                logger.info(f"Saved {emdb_id}_prot_bb_prb.mrc")
                prot_sg_prb_mrc_obj = MapObject(
                    grid_data=rdna_sugar_proba.cpu().numpy(),
                    voxel_size=voxel_size,
                    global_origin=global_origin,
                )
                save_mrc(prot_sg_prb_mrc_obj, output_dir / f"{emdb_id}_rdna_sg_prb.mrc")
                logger.info(f"Saved {emdb_id}_rdna_sg_prb.mrc")

                np.save(
                    output_dir / f"{emdb_id}_residue.npy", residue_prob.cpu().numpy()
                )

            maps_dict = {
                "prot_ca_map": prot_ca_proba.cpu(),
                "rdna_c1p_map": rdna_c1p_proba.cpu(),
                "prot_bb_map": prot_bb_proba.cpu(),
                "rdna_sugar_map": rdna_sugar_proba.cpu(),
                "detection_map": residue_prob.cpu(),
                "residue_features": residue_features.cpu(),
            }

            t3 = time.time()

            # Run post-processing
            post_process(
                maps_dict,
                voxel_size,
                global_origin,
                output_dir,
                emdb_id,
                num_points,
                resolution,
                disable_interpolation=disable_interpolation,
                config=self.config,
            )

            logger.info(f"Completed inference for {emdb_id}")

            t4 = time.time()

            with open(output_dir / f"{emdb_id}_timing.txt", "w") as f:
                f.write(f"Map dimension: {detection_output.shape[2:]}\n")
                f.write(f"Preprocessing time: {t1 - t0:.2f} seconds\n")
                f.write(f"UNet Inference time: {t2 - t1:.2f} seconds\n")
                f.write(f"Saving outputs time: {t3 - t2:.2f} seconds\n")
                f.write(f"Post-processing time: {t4 - t3:.2f} seconds\n")
                f.write(f"Total time: {t4 - t0:.2f} seconds\n")

            return emdb_id


app = typer.Typer(help="Cryo-EM atom detection inference")


@app.command("run")
def run(
    input_file: str = typer.Argument(..., help="Input MRC file path"),
    output_dir: str = typer.Argument(..., help="Output directory"),
    resolution: float | None = typer.Option(
        None,
        help="Resolution of the map (optional, will be fetched by EMDB ID if not provided)",
    ),
    contour_level: float | None = typer.Option(
        None,
        help="Contour level of the map (optional, will be fetched by EMDB ID if not provided)",
    ),
    emdb_id: str | None = typer.Option(
        None, help="EMDB ID (defaults to filename stem)"
    ),
    batch_size: int = typer.Option(4, help="Batch size for inference"),
    device: str = typer.Option("cuda", help="Device to use (cuda/cpu)"),
    compile_models: bool = typer.Option(
        False, "--compile", help="Compile models for faster inference"
    ),
    num_points: int | None = typer.Option(
        None, help="Number of points to randomly sample for interpolation"
    ),
    disable_interpolation: bool = typer.Option(
        False,
        "--disable-interpolation",
        help="Disable interpolation; set interpolate_confidence=None in outputs",
    ),
    resolution_cutoff: float | None = typer.Option(
        4.5,
        help="The resolution cutoff to change post-processing behavior. Resolution <= cutoff will use only CA and C1P for post-processing. Resolution > cutoff will use (CA,BB), (C1P,SG) for post-processing.",
    ),
    ca_wgt: float | None = typer.Option(
        1.0, "--ca-wgt", help="Weight for CA when resolution > cutoff"
    ),
    bb_wgt: float | None = typer.Option(
        1.0, "--bb-wgt", help="Weight for BB when resolution > cutoff"
    ),
    c1p_wgt: float | None = typer.Option(
        3.0, "--c1p-wgt", help="Weight for C1P when resolution > cutoff"
    ),
    sg_wgt: float | None = typer.Option(
        1.0, "--sg-wgt", help="Weight for SG when resolution > cutoff"
    ),
    ca_threshold_highres: float | None = typer.Option(
        0.25,
        "--ca-threshold-highres",
        help="Threshold for CA when resolution <= cutoff",
    ),
    c1p_threshold_highres: float | None = typer.Option(
        0.20,
        "--c1p-threshold-highres",
        help="Threshold for C1P when resolution <= cutoff",
    ),
    ca_threshold_lowres: float | None = typer.Option(
        0.20,
        "--ca-threshold-lowres",
        help="Threshold for (CA,BB) when resolution > cutoff",
    ),
    c1p_threshold_lowres: float | None = typer.Option(
        0.15,
        "--c1p-threshold-lowres",
        help="Threshold for (C1P,SG) when resolution > cutoff",
    ),
    save_map_predictions: bool | None = typer.Option(
        False,
        "--save-map-predictions",
        help="Save predicted probability maps as MRC files",
    ),
    contour_level_scale: float = typer.Option(
        0.5,
        "--contour-level-scale",
        help="Scale factor for contour levels in post-processing",
    ),
    no_protein: bool = typer.Option(
        False, "--no-protein", help="Exclude protein predictions"
    ),
    no_dna_rna: bool = typer.Option(
        False, "--no-dna-rna", help="Exclude DNA/RNA predictions"
    ),
    interp_threshold: int = typer.Option(
        3000,
        "--interp-threshold",
        help="Skip interpolation feature calculation when the number of detected points exceeds this threshold",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite/--no-overwrite", help="Overwrite existing output files"
    ),
):
    """Run cryo-EM inference on a single MRC file."""

    config = CryoEMInferenceConfig(
        roi_size=64,  # Fixed to 64
        batch_size=batch_size,
        compile=compile_models,
        device=device,
        resolution=resolution,
        contour_level=contour_level,
        resolution_cutoff=resolution_cutoff,
        ca_wgt=ca_wgt,
        bb_wgt=bb_wgt,
        c1p_wgt=c1p_wgt,
        sg_wgt=sg_wgt,
        ca_threshold_highres=ca_threshold_highres,
        c1p_threshold_highres=c1p_threshold_highres,
        ca_threshold_lowres=ca_threshold_lowres,
        c1p_threshold_lowres=c1p_threshold_lowres,
        save_map_predictions=save_map_predictions,
        contour_level_scale=contour_level_scale,
        no_protein=no_protein,
        no_dna_rna=no_dna_rna,
        interp_threshold=interp_threshold,
    )

    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    inference = CryoEMInference(config)
    inference.run_inference(
        input_path,
        output_path,
        emdb_id,
        num_points,
        disable_interpolation=disable_interpolation,
        overwrite=overwrite,
    )

    typer.echo(f"Inference completed successfully! Results saved to {output_path}")


@app.command("json-run")
def json_run(
    input_json: str = typer.Argument(
        ..., help="Path to input JSON file (list of entries with 'map_path')"
    ),
    output_dir: str = typer.Argument(
        ..., help="Output directory for atom detection results"
    ),
    device: str = typer.Option("cuda", help="Device to use (cuda/cpu)"),
    batch_size: int = typer.Option(4, help="Batch size for sliding-window inference"),
    compile_models: bool = typer.Option(
        False, "--compile", help="Compile models for faster inference"
    ),
    interp_threshold: int = typer.Option(
        3000,
        "--interp-threshold",
        help="Skip interpolation feature calculation when the number of detected points exceeds this threshold",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite/--no-overwrite", help="Overwrite existing output files"
    ),
):
    """Run cryo-EM inference on all entries in a JSON file.

    The JSON should be a list of objects, each with at least ``"name"``,
    ``"map_path"``, ``"resolution"``, and ``"contour_level"`` keys.  The model
    is loaded once and reused for every entry.

    Output is organized per entry as ``{output_dir}/{name}/CryoZeta-Detection/``.
    """
    with open(input_json) as f:
        data = json.load(f)

    base_output_path = Path(output_dir)

    config = CryoEMInferenceConfig(
        batch_size=batch_size,
        compile=compile_models,
        device=device,
        interp_threshold=interp_threshold,
    )
    inference = CryoEMInference(config)

    for i, item in enumerate(data):
        em_map_path = item["map_path"]
        name = item.get("name")
        if name is None:
            logger.error(
                f"[batch {i + 1}/{len(data)}] 'name' missing for {em_map_path}, skipping"
            )
            continue
        resolution = item.get("resolution")
        contour_level = item.get("contour_level")
        if resolution is None:
            logger.error(
                f"[batch {i + 1}/{len(data)}] 'resolution' missing for {em_map_path}, skipping"
            )
            continue
        if contour_level is None:
            logger.error(
                f"[batch {i + 1}/{len(data)}] 'contour_level' missing for {em_map_path}, skipping"
            )
            continue
        # Per-entry output directory: {output_dir}/{name}/CryoZeta-Detection/
        entry_output_path = base_output_path / name / "CryoZeta-Detection"
        entry_output_path.mkdir(parents=True, exist_ok=True)
        # Update per-entry config values
        inference.config.resolution = float(resolution)
        inference.config.contour_level = float(contour_level)
        logger.info(f"[batch {i + 1}/{len(data)}] Processing {em_map_path}")
        try:
            inference.run_inference(
                mrc_path=Path(em_map_path),
                output_dir=entry_output_path,
                emdb_id=name,
                overwrite=overwrite,
            )
        except Exception as e:
            logger.error(f"[batch {i + 1}/{len(data)}] Failed on {em_map_path}: {e}")

    typer.echo(f"Batch inference completed. Results saved to {base_output_path}")


if __name__ == "__main__":
    app()
