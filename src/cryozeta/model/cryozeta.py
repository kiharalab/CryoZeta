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

import functools
import operator
import os
import time
from typing import Any

import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

from cryozeta.model import sample_confidence
from cryozeta.model.generator import (
    InferenceNoiseScheduler,
    sample_diffusion,
)
from cryozeta.model.utils import simple_merge_dict_list
from cryozeta.openfold_local.model.primitives import LayerNorm
from cryozeta.utils.torch_utils import autocasting_disable_decorator

from .modules.confidence import ConfidenceHead
from .modules.diffusion import DiffusionModule
from .modules.embedders import InputFeatureEmbedder, RelativePositionEncoding
from .modules.fitting import (
    FitModelPoints,
    FitModelPointsTeaser,
    FitModelVESPER,
    PointResidueMatching,
    SetPntResAffinity,
)
from .modules.head import DistogramHead, PointNoiseHead, PointResidueClassHead
from .modules.empairformer import EMPairformerStack, MSAModule, TemplateEmbedder
from .modules.primitives import LinearNoBias


class CryoZeta(nn.Module):
    """
    Implements Algorithm 1 [Main Inference Loop] in AF3
    """

    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs

        # Some constants
        self.N_cycle = self.configs.model.N_cycle
        self.N_model_seed = self.configs.model.N_model_seed

        # Diffusion scheduler
        self.inference_noise_scheduler = InferenceNoiseScheduler(
            **configs.inference_noise_scheduler
        )

        # Model
        self.input_embedder = InputFeatureEmbedder(**configs.model.input_embedder)
        self.relative_position_encoding = RelativePositionEncoding(
            **configs.model.relative_position_encoding
        )
        self.template_embedder = TemplateEmbedder(**configs.model.template_embedder)
        self.msa_module = MSAModule(
            **configs.model.msa_module,
            msa_configs=configs.data.get("msa", {}),
        )
        self.empairformer_stack = EMPairformerStack(**configs.model.empairformer)
        self.diffusion_module = DiffusionModule(**configs.model.diffusion_module)
        self.distogram_head = DistogramHead(**configs.model.distogram_head)
        self.point_residue_head = PointResidueClassHead(
            **configs.model.point_residue_class
        )
        self.point_noise_head = PointNoiseHead(**configs.model.point_noise)
        self.confidence_head = ConfidenceHead(**configs.model.confidence_head)

        self.c_s, self.c_z, self.c_s_inputs, self.c_pz, self.c_p = (
            configs.c_s,
            configs.c_z,
            configs.c_s_inputs,
            configs.c_pz,
            configs.c_p,
        )
        self.linear_no_bias_sinit = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_s
        )
        self.linear_no_bias_zinit1 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_zinit2 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_pzinit = LinearNoBias(
            in_features=self.c_pz, out_features=self.c_pz
        )
        self.linear_no_bias_pinit = LinearNoBias(
            in_features=self.c_p, out_features=self.c_p
        )
        self.linear_no_bias_token_bond = LinearNoBias(
            in_features=1, out_features=self.c_z
        )
        self.linear_no_bias_z_cycle = LinearNoBias(
            in_features=self.c_z, out_features=self.c_z
        )
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s, out_features=self.c_s
        )
        self.linear_no_bias_pz = LinearNoBias(
            in_features=self.c_pz, out_features=self.c_pz
        )
        self.linear_no_bias_p = LinearNoBias(
            in_features=self.c_p, out_features=self.c_p
        )
        self.layernorm_z_cycle = LayerNorm(self.c_z)
        self.layernorm_s = LayerNorm(self.c_s)
        self.layernorm_pz = LayerNorm(self.c_pz)
        self.layernorm_p = LayerNorm(self.c_p)

        # Zero init the recycling layer
        nn.init.zeros_(self.linear_no_bias_z_cycle.weight)
        nn.init.zeros_(self.linear_no_bias_s.weight)
        nn.init.zeros_(self.linear_no_bias_pz.weight)
        nn.init.zeros_(self.linear_no_bias_p.weight)

    def get_empairformer_output(
        self,
        input_feature_dict: dict[str, Any],
        N_cycle: int,
        inplace_safe: bool = False,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """
        The forward pass from the input to empairformer output

        Args:
            input_feature_dict (dict[str, Any]): input features
            N_cycle (int): number of cycles
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            Tuple[torch.Tensor, ...]: s_inputs, s, z
        """
        N_token = input_feature_dict["residue_index"].shape[-1]
        if N_token <= 16:
            # Deepspeed_evo_attention do not support token <= 16
            deepspeed_evo_attention_condition_satisfy = False
        else:
            deepspeed_evo_attention_condition_satisfy = True

        # Line 1-5
        s_inputs, pz_inputs, p_inputs = self.input_embedder(
            input_feature_dict, inplace_safe=False, chunk_size=chunk_size
        )  # [..., N_token, 449]
        s_init = self.linear_no_bias_sinit(s_inputs)  #  [..., N_token, c_s]
        z_init = (
            self.linear_no_bias_zinit1(s_init)[..., None, :]
            + self.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )  #  [..., N_token, N_token, c_z]
        pz_init = self.linear_no_bias_pzinit(pz_inputs)
        p_init = self.linear_no_bias_pinit(p_inputs)

        if inplace_safe:
            z_init += self.relative_position_encoding(input_feature_dict)
            z_init += self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
        else:
            z_init = z_init + self.relative_position_encoding(input_feature_dict)
            z_init = z_init + self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
        # Line 6
        z = torch.zeros_like(z_init)
        s = torch.zeros_like(s_init)
        pz = torch.zeros_like(pz_init)
        p = torch.zeros_like(p_init)

        # Line 7-13 recycling
        for _cycle_no in tqdm(
            range(N_cycle),
            total=N_cycle,
            desc="EMPairformer cycles",
            leave=False,
            dynamic_ncols=True,
        ):
            with torch.no_grad():
                z = z_init + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z))
                if inplace_safe:
                    if self.template_embedder.n_blocks > 0:
                        z += self.template_embedder(
                            input_feature_dict,
                            z,
                            use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
                            use_deepspeed_evo_attention=self.configs.use_deepspeed_evo_attention
                            and deepspeed_evo_attention_condition_satisfy,
                            use_lma=self.configs.use_lma,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_feature_dict,
                        z,
                        s_inputs,
                        pair_mask=None,
                        use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
                        use_deepspeed_evo_attention=self.configs.use_deepspeed_evo_attention
                        and deepspeed_evo_attention_condition_satisfy,
                        use_lma=self.configs.use_lma,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                else:
                    if self.template_embedder.n_blocks > 0:
                        z = z + self.template_embedder(
                            input_feature_dict,
                            z,
                            use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
                            use_deepspeed_evo_attention=self.configs.use_deepspeed_evo_attention
                            and deepspeed_evo_attention_condition_satisfy,
                            use_lma=self.configs.use_lma,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_feature_dict,
                        z,
                        s_inputs,
                        pair_mask=None,
                        use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
                        use_deepspeed_evo_attention=self.configs.use_deepspeed_evo_attention
                        and deepspeed_evo_attention_condition_satisfy,
                        use_lma=self.configs.use_lma,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                s = s_init + self.linear_no_bias_s(self.layernorm_s(s))
                pz = pz_init + self.linear_no_bias_pz(self.layernorm_pz(pz))
                p = p_init + self.linear_no_bias_p(self.layernorm_p(p))

                s, z, pz, p = self.empairformer_stack(
                    s,
                    z,
                    pz,
                    p,
                    pair_mask=input_feature_dict["pair_mask"],
                    use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
                    use_deepspeed_evo_attention=self.configs.use_deepspeed_evo_attention
                    and deepspeed_evo_attention_condition_satisfy,
                    use_lma=self.configs.use_lma,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )

                if self.configs.use_affinity:
                    with torch.no_grad():
                        distogram = self.point_residue_head(pz.unsqueeze(0))
                    res2pnts, _ = PointResidueMatching(distogram.squeeze(0))
                    if res2pnts:
                        affinity = SetPntResAffinity(
                            res2pnts,
                            input_feature_dict["asym_id"],
                            input_feature_dict["entity_id"],
                            input_feature_dict["sym_id"],
                            input_feature_dict["em_support_points"],
                        )

                        pz = pz.clone()
                        pz[..., -1] = affinity

        return s_inputs, s, z, pz, p

    def sample_diffusion(self, **kwargs) -> torch.Tensor:
        """
        Samples diffusion process based on the provided configurations.

        Returns:
            torch.Tensor: The result of the diffusion sampling process.
        """
        _configs = {
            key: self.configs.sample_diffusion.get(key)
            for key in [
                "gamma0",
                "gamma_min",
                "noise_scale_lambda",
                "step_scale_eta",
            ]
        }
        _configs.update(
            {
                "attn_chunk_size": self.configs.infer_setting.chunk_size,
                "diffusion_chunk_size": self.configs.infer_setting.sample_diffusion_chunk_size,
            }
        )
        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
            sample_diffusion
        )(**_configs, **kwargs)

    def run_confidence_head(self, *args, **kwargs):
        """
        Runs the confidence head with optional automatic mixed precision (AMP) disabled.

        Returns:
            Any: The output of the confidence head.
        """
        return autocasting_disable_decorator(self.configs.skip_amp.confidence_head)(
            self.confidence_head
        )(*args, **kwargs)

    def main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        N_cycle: int,
        inplace_safe: bool = True,
        chunk_size: int | None = 4,
        N_model_seed: int = 1,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop (multiple model seeds) for the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            N_cycle (int): Number of cycles.
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to True.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to 4.
            N_model_seed (int): Number of model seeds. Defaults to 1.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        pred_dicts = []
        log_dicts = []
        time_trackers = []
        for _ in range(N_model_seed):
            pred_dict, log_dict, time_tracker = self._main_inference_loop(
                input_feature_dict=input_feature_dict,
                N_cycle=N_cycle,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            pred_dicts.append(pred_dict)
            log_dicts.append(log_dict)
            time_trackers.append(time_tracker)

        # Combine outputs of multiple models
        def _cat(dict_list, key):
            return torch.cat([x[key] for x in dict_list], dim=0)

        def _list_join(dict_list, key):
            return functools.reduce(operator.iadd, [x[key] for x in dict_list], [])

        all_pred_dict = {
            "coordinate": _cat(pred_dicts, "coordinate"),
            "summary_confidence": _list_join(pred_dicts, "summary_confidence"),
            "full_data": _list_join(pred_dicts, "full_data"),
            "plddt": _cat(pred_dicts, "plddt"),
            "pae": _cat(pred_dicts, "pae"),
            "pde": _cat(pred_dicts, "pde"),
            "resolved": _cat(pred_dicts, "resolved"),
            "point_residue_logits": _cat(pred_dicts, "point_residue_logits"),
            "point_noise_logits": _cat(pred_dicts, "point_noise_logits"),
        }

        if all(pred_dict["coordinate_svd_0.8"] is not None for pred_dict in pred_dicts):
            all_pred_dict["coordinate_svd_0.8"] = _cat(pred_dicts, "coordinate_svd_0.8")
        if all(pred_dict["coordinate_svd_0.4"] is not None for pred_dict in pred_dicts):
            all_pred_dict["coordinate_svd_0.4"] = _cat(pred_dicts, "coordinate_svd_0.4")
        if all(pred_dict["coordinate_teaser"] is not None for pred_dict in pred_dicts):
            all_pred_dict["coordinate_teaser"] = _cat(pred_dicts, "coordinate_teaser")
        if all(pred_dict["coordinate_vesper"] is not None for pred_dict in pred_dicts):
            all_pred_dict["coordinate_vesper"] = _cat(pred_dicts, "coordinate_vesper")
        if all(
            pred_dict["coordinate_superimposed"] is not None for pred_dict in pred_dicts
        ):
            all_pred_dict["coordinate_superimposed"] = _cat(
                pred_dicts, "coordinate_superimposed"
            )

        all_log_dict = simple_merge_dict_list(log_dicts)
        all_time_dict = simple_merge_dict_list(time_trackers)
        return all_pred_dict, all_log_dict, all_time_dict

    def _main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        N_cycle: int,
        inplace_safe: bool = True,
        chunk_size: int | None = 4,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop (single model seed) for the Alphafold3 model.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        step_st = time.time()
        N_token = input_feature_dict["residue_index"].shape[-1]
        if N_token <= 16:
            deepspeed_evo_attention_condition_satisfy = False
        else:
            deepspeed_evo_attention_condition_satisfy = True

        log_dict = {}
        pred_dict = {}
        time_tracker = {}

        s_inputs, s, z, pz, _p = self.get_empairformer_output(
            input_feature_dict=input_feature_dict,
            N_cycle=N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        keys_to_delete = []
        for key in input_feature_dict.keys():
            if "template_" in key or key in [
                "msa",
                "has_deletion",
                "deletion_value",
                "profile",
                "deletion_mean",
                "token_bonds",
            ]:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del input_feature_dict[key]
        torch.cuda.empty_cache()

        step_trunk = time.time()
        time_tracker.update({"empairformer": step_trunk - step_st})
        logger.info(f"EMPairformer done in {step_trunk - step_st:.1f}s. Starting diffusion sampling...")
        # Sample diffusion
        # [..., N_sample, N_atom, 3]
        N_sample = self.configs.sample_diffusion["N_sample"]
        N_step = self.configs.sample_diffusion["N_step"]

        noise_schedule = self.inference_noise_scheduler(
            N_step=N_step, device=s_inputs.device, dtype=s_inputs.dtype
        )
        pred_dict["coordinate"] = self.sample_diffusion(
            denoise_net=self.diffusion_module,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            N_sample=N_sample,
            noise_schedule=noise_schedule,
            inplace_safe=inplace_safe,
        )

        step_diffusion = time.time()
        time_tracker.update({"diffusion": step_diffusion - step_trunk})
        logger.info(f"Diffusion sampling done in {step_diffusion - step_trunk:.1f}s. Starting model fitting...")
        if N_token > 2000:
            torch.cuda.empty_cache()

        # Distogram logits: log contact_probs only, to reduce the dimension
        pred_dict["contact_probs"] = sample_confidence.compute_contact_prob(
            distogram_logits=self.distogram_head(z),
            **sample_confidence.get_bin_params(self.configs.loss.distogram),
        )  # [N_token, N_token]
        pred_dict["point_residue_logits"] = self.point_residue_head(pz.unsqueeze(0))
        pred_dict["point_noise_logits"] = self.point_noise_head(pz.unsqueeze(0))

        ca_mask = input_feature_dict["atom_array"].centre_atom_mask
        ca_mask = ca_mask.astype(bool)
        ca_coordinate = pred_dict["coordinate"][..., ca_mask, :]

        pdb_id = input_feature_dict["pdb_id"]
        elements = input_feature_dict["atom_array"].element  # element symbols

        sequences_indices = []
        sequence_type = []
        current_index = 0

        for seq in input_feature_dict["sequences"]:
            seq_type = next(iter(seq.keys()))
            length = len(seq[seq_type]["sequence"])
            count = seq[seq_type]["count"]
            for _i in range(count):
                sequences_indices.append([current_index, current_index + length])
                current_index += length
                sequence_type.append(seq_type)

        num_big_distances = []
        for n in range(N_sample):
            big_distance_count = 0
            for i in range(len(sequences_indices)):
                indices = sequences_indices[i]
                chain = ca_coordinate[n, indices[0] : indices[1], :]
                chain_a = chain[:-1, :]
                chain_b = chain[1:, :]
                distances = torch.norm(chain_b - chain_a, dim=1)
                if sequence_type[i] != "proteinChain":
                    continue
                for distance in distances:
                    if distance > 4:
                        big_distance_count += 1
            num_big_distances.append(big_distance_count)

        coordinate_svd_08, recall_score_svd_08, ccc_mask_svd_08, ccc_box_svd_08 = (
            FitModelPoints(
                pred_dict["point_residue_logits"],
                ca_coordinate,
                pred_dict["coordinate"],
                elements,
                input_feature_dict["em_support_points"],
                input_feature_dict["all_support_points"],
                0.8,
                input_feature_dict["dump_dir"],
                pdb_id,
                map_path=input_feature_dict.get("map_path", None),
                resolution=input_feature_dict.get("resolution", None),
                contour_level=input_feature_dict.get("contour_level", None),
            )
        )
        coordinate_svd_04, recall_score_svd_04, ccc_mask_svd_04, ccc_box_svd_04 = (
            FitModelPoints(
                pred_dict["point_residue_logits"],
                ca_coordinate,
                pred_dict["coordinate"],
                elements,
                input_feature_dict["em_support_points"],
                input_feature_dict["all_support_points"],
                0.4,
                input_feature_dict["dump_dir"],
                pdb_id,
                map_path=input_feature_dict.get("map_path", None),
                resolution=input_feature_dict.get("resolution", None),
                contour_level=input_feature_dict.get("contour_level", None),
            )
        )
        coordinate_teaser, recall_score_teaser, ccc_mask_teaser, ccc_box_teaser = (
            FitModelPointsTeaser(
                ca_coordinate,
                pred_dict["coordinate"],
                elements,
                input_feature_dict["all_support_points"],
                map_path=input_feature_dict.get("map_path", None),
                resolution=input_feature_dict.get("resolution", None),
                contour_level=input_feature_dict.get("contour_level", None),
            )
        )
        pred_dict["coordinate_svd_0.8"] = coordinate_svd_08
        pred_dict["coordinate_svd_0.4"] = coordinate_svd_04
        pred_dict["coordinate_teaser"] = coordinate_teaser
        pred_dict["coordinate_vesper"] = None
        pred_dict["coordinate_superimposed"] = None

        total_score_svd_08 = [
            recall_score_svd_08[i] + ccc_mask_svd_08[i] - num_big_distances[i]
            for i in range(N_sample)
        ]
        total_score_svd_04 = [
            recall_score_svd_04[i] + ccc_mask_svd_04[i] - num_big_distances[i]
            for i in range(N_sample)
        ]
        total_score_teaser = [
            recall_score_teaser[i] + ccc_mask_teaser[i] - num_big_distances[i]
            for i in range(N_sample)
        ]

        coordinate_superimposed = pred_dict["coordinate"].clone()
        if (
            sum(recall_score_svd_08)
            + sum(recall_score_svd_04)
            + sum(recall_score_teaser)
            > 0
        ):
            # sort the total_score by descending order and make the coordinates the same order
            combined_scores = (
                total_score_svd_08 + total_score_svd_04 + total_score_teaser
            )
            sorted_indices = sorted(
                range(len(combined_scores)),
                key=lambda idx: combined_scores[idx],
                reverse=True,
            )
            for i in range(N_sample):
                if sorted_indices[i] < N_sample:
                    coordinate_superimposed[i] = coordinate_svd_08[sorted_indices[i]]
                elif sorted_indices[i] < N_sample * 2:
                    coordinate_superimposed[i] = coordinate_svd_04[
                        sorted_indices[i] - N_sample
                    ]
                elif sorted_indices[i] < N_sample * 3:
                    coordinate_superimposed[i] = coordinate_teaser[
                        sorted_indices[i] - N_sample * 2
                    ]
        else:
            try:
                (
                    coordinate_vesper,
                    recall_score_vesper,
                    ccc_mask_vesper,
                    ccc_box_vesper,
                ) = FitModelVESPER(
                    all_atom=pred_dict["coordinate"],
                    atom_array=input_feature_dict["atom_array"],
                    entity_poly_type=input_feature_dict["entity_poly_type"],
                    pdb_id=pdb_id,
                    contour_level=3,
                    support_points=input_feature_dict["all_support_points"],
                    map_path=input_feature_dict.get("map_path", None),
                    resolution_map=input_feature_dict.get("resolution", None),
                    contour_level_map=input_feature_dict.get("contour_level", None),
                )
                pred_dict["coordinate_vesper"] = coordinate_vesper
                total_score_vesper = [
                    recall_score_vesper[i] + ccc_mask_vesper[i] - num_big_distances[i]
                    for i in range(N_sample)
                ]
                sorted_indices = sorted(
                    range(len(total_score_vesper)),
                    key=lambda idx: total_score_vesper[idx],
                    reverse=True,
                )
                for i in range(N_sample):
                    coordinate_superimposed[i] = coordinate_vesper[sorted_indices[i]]

            except Exception as e:
                logger.error(f"Error in FitModelVESPER: {e}")

        pred_dict["coordinate_superimposed"] = coordinate_superimposed

        output_dir = f"{input_feature_dict['dump_dir']}/saved_data"
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(f"{output_dir}/scores.csv"):
            with open(f"{output_dir}/scores.csv", "w") as f:
                f.write(
                    "pdb_id,sample_idx,method,recall,ccc_mask,ccc_box,num_dist_over_4,recall_ccmask_ca\n"
                )
        if sum(recall_score_svd_08) > 0:
            for n in range(N_sample):
                with open(f"{output_dir}/scores.csv", "a") as f:
                    f.write(
                        f"{pdb_id},{n},svd_0.8,{recall_score_svd_08[n]},{ccc_mask_svd_08[n]},{ccc_box_svd_08[n]},{num_big_distances[n]},{total_score_svd_08[n]}\n"
                    )
        if sum(recall_score_svd_04) > 0:
            for n in range(N_sample):
                with open(f"{output_dir}/scores.csv", "a") as f:
                    f.write(
                        f"{pdb_id},{n},svd_0.4,{recall_score_svd_04[n]},{ccc_mask_svd_04[n]},{ccc_box_svd_04[n]},{num_big_distances[n]},{total_score_svd_04[n]}\n"
                    )
        if sum(recall_score_teaser) > 0:
            for n in range(N_sample):
                with open(f"{output_dir}/scores.csv", "a") as f:
                    f.write(
                        f"{pdb_id},{n},teaser,{recall_score_teaser[n]},{ccc_mask_teaser[n]},{ccc_box_teaser[n]},{num_big_distances[n]},{total_score_teaser[n]}\n"
                    )
        if (
            sum(recall_score_svd_08)
            + sum(recall_score_svd_04)
            + sum(recall_score_teaser)
            == 0
        ):
            for n in range(N_sample):
                with open(f"{output_dir}/scores.csv", "a") as f:
                    f.write(
                        f"{pdb_id},{n},vesper,{recall_score_vesper[n]},{ccc_mask_vesper[n]},{ccc_box_vesper[n]},{num_big_distances[n]},{total_score_vesper[n]}\n"
                    )

        output_dict = {}
        output_dict["ca_mask"] = ca_mask
        output_dict["pred_all_coordinate"] = pred_dict["coordinate"]
        output_dict["em_support_points"] = input_feature_dict["em_support_points"]
        output_dict["all_support_points"] = input_feature_dict["all_support_points"]
        output_dict["point_residue_logits"] = pred_dict["point_residue_logits"]
        torch.save(output_dict, f"{output_dir}/output_dict_{pdb_id}.pt")

        # Confidence logits
        step_fitting = time.time()
        logger.info(f"Model fitting done in {step_fitting - step_diffusion:.1f}s. Running confidence head...")
        (
            pred_dict["plddt"],
            pred_dict["pae"],
            pred_dict["pde"],
            pred_dict["resolved"],
        ) = self.run_confidence_head(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            pair_mask=None,
            x_pred_coords=pred_dict["coordinate"],
            use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
            use_deepspeed_evo_attention=self.configs.use_deepspeed_evo_attention
            and deepspeed_evo_attention_condition_satisfy,
            use_lma=self.configs.use_lma,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        step_confidence = time.time()
        time_tracker.update({"confidence": step_confidence - step_fitting})
        time_tracker.update({"model_forward": time.time() - step_st})
        logger.info(
            f"Confidence head done in {step_confidence - step_fitting:.1f}s. "
            f"Total model forward: {time.time() - step_st:.1f}s "
            f"(empairformer={time_tracker['empairformer']:.1f}s, "
            f"diffusion={time_tracker['diffusion']:.1f}s, "
            f"fitting={step_fitting - step_diffusion:.1f}s, "
            f"confidence={time_tracker['confidence']:.1f}s)"
        )

        # Summary Confidence & Full Data
        pred_dict["summary_confidence"], pred_dict["full_data"] = (
            sample_confidence.compute_full_data_and_summary(
                configs=self.configs,
                pae_logits=pred_dict["pae"],
                plddt_logits=pred_dict["plddt"],
                pde_logits=pred_dict["pde"],
                contact_probs=pred_dict.get(
                    "per_sample_contact_probs", pred_dict["contact_probs"]
                ),
                token_asym_id=input_feature_dict["asym_id"],
                token_has_frame=input_feature_dict["has_frame"],
                atom_coordinate=pred_dict["coordinate"],
                atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
                atom_is_polymer=1 - input_feature_dict["is_ligand"],
                N_recycle=N_cycle,
                interested_atom_mask=None,
                return_full_data=True,
                mol_id=None,
                elements_one_hot=None,
            )
        )

        return pred_dict, log_dict, time_tracker

    def forward(
        self,
        input_feature_dict: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Forward pass of the CryoZeta model (inference only).

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, log, and time dictionaries.
        """
        inplace_safe = not (self.training or torch.is_grad_enabled())
        chunk_size = self.configs.infer_setting.chunk_size if inplace_safe else None

        pred_dict, log_dict, time_tracker = self.main_inference_loop(
            input_feature_dict=input_feature_dict,
            N_cycle=self.N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            N_model_seed=self.N_model_seed,
        )
        log_dict.update({"time": time_tracker})

        return pred_dict, log_dict, time_tracker
