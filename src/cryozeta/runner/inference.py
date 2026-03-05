# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Modifications Copyright 2026 KiharaLab, Purdue University.
#
# This file is included in a GPLv3-licensed project. The original
# code remains under Apache-2.0; the combined work is distributed
# under GPLv3.

import json
import os
import traceback
from collections.abc import Mapping
from contextlib import nullcontext
from os.path import exists as opexists, join as opjoin
from typing import Any

import torch
import torch.distributed as dist
from loguru import logger
from safetensors.torch import load_file

from cryozeta.configs import parse_configs, parse_sys_args
from cryozeta.configs.configs_base import configs as configs_base
from cryozeta.configs.configs_data import data_configs
from cryozeta.configs.configs_inference import inference_configs
from cryozeta.data.infer_data_pipeline import get_inference_dataloader
from cryozeta.model.cryozeta import CryoZeta
from cryozeta.runner.dumper import DataDumper
from cryozeta.utils.distributed import DIST_WRAPPER
from cryozeta.utils.seed import seed_everything
from cryozeta.utils.torch_utils import to_device


class InferenceRunner:
    def __init__(self, configs: Any) -> None:
        self.configs = configs
        self.init_env()
        self.init_basics()
        self.init_model()
        self.load_checkpoint()
        self.init_dumper(need_atom_confidence=configs.need_atom_confidence)

    def init_env(self) -> None:
        logger.info(
            f"PyTorch version: {torch.__version__}, "
            f"CUDA compiled version: {torch.version.cuda}, "
            f"cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}"
        )
        self.print(
            f"Distributed environment: world size: {DIST_WRAPPER.world_size}, "
            + f"global rank: {DIST_WRAPPER.rank}, local rank: {DIST_WRAPPER.local_rank}"
        )
        self.use_cuda = torch.cuda.device_count() > 0
        if self.use_cuda:
            self.device = torch.device(f"cuda:{DIST_WRAPPER.local_rank}")
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
            devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
            logger.info(
                f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
            )
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        if DIST_WRAPPER.world_size > 1:
            dist.init_process_group(backend="nccl")
        if self.configs.use_deepspeed_evo_attention:
            env = os.getenv("CUTLASS_PATH", None)
            self.print(f"env: {env}")
            assert env is not None, (
                "if use ds4sci, set env as https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/"
            )
            if env is not None:
                logger.info(
                    "The kernels will be compiled when DS4Sci_EvoformerAttention is called for the first time."
                )
        use_fastlayernorm = os.getenv("LAYERNORM_TYPE", None)
        if use_fastlayernorm == "fast_layernorm":
            logger.info(
                "The kernels will be compiled when fast_layernorm is called for the first time."
            )

        logger.info("Finished init ENV.")

    def init_basics(self) -> None:
        self.dump_dir = self.configs.dump_dir
        self.stage_name = (
            "CryoZeta-Interpolate" if self.configs.use_interpolation else "CryoZeta"
        )
        os.makedirs(self.dump_dir, exist_ok=True)

    def init_model(self) -> None:
        self.model = CryoZeta(self.configs).to(self.device)

    def load_checkpoint(self) -> None:
        checkpoint_path = self.configs.load_checkpoint_path
        if not os.path.exists(checkpoint_path):
            raise Exception(f"Given checkpoint path not exist [{checkpoint_path}]")
        self.print(f"Loading from {checkpoint_path}")

        assert checkpoint_path.endswith(".safetensors"), (
            "Checkpoint must be a safetensors file"
        )
        state_dict = load_file(checkpoint_path, device=str(self.device))

        self.model.load_state_dict(
            state_dict=state_dict,
            strict=True,
        )
        self.model.eval()
        self.print("Finish loading checkpoint.")

    def init_dumper(self, need_atom_confidence: bool = False):
        self.dumper = DataDumper(
            base_dir=self.dump_dir,
            stage_name=self.stage_name,
            need_atom_confidence=need_atom_confidence,
        )

    @torch.no_grad()
    def predict(self, data: Mapping[str, Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]

        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if torch.cuda.is_available()
            else nullcontext()
        )

        data = to_device(data, self.device)
        # dump_dir is set per-entry in main() before calling predict()
        with enable_amp:
            prediction, _, _ = self.model(
                input_feature_dict=data["input_feature_dict"],
            )

        return prediction

    def print(self, msg: str):
        if DIST_WRAPPER.rank == 0:
            logger.info(msg)


def _has_interpolation_features(em_file_dir: str, sample_name: str) -> bool:
    """Check whether the detection .pt file contains interpolation features."""
    pt_path = opjoin(
        em_file_dir, sample_name, "CryoZeta-Detection", f"{sample_name}.pt"
    )
    if not opexists(pt_path):
        return False
    pt_data = torch.load(pt_path, weights_only=False)
    return (
        "interpolate_confidence" in pt_data
        and pt_data["interpolate_confidence"] is not None
    )


def main(configs: Any) -> None:
    # When use_interpolation is requested, pre-check which entries actually
    # have interpolation features in their detection output.  Entries that
    # lack interpolation features are collected into a skip-set so we can
    # avoid featurisation errors and skip them gracefully.
    skip_interp_names: set[str] = set()
    if configs.use_interpolation:
        with open(configs.input_json_path) as f:
            input_entries = json.load(f)
        for entry in input_entries:
            name = entry.get("name")
            if name and not _has_interpolation_features(configs.em_file_dir, name):
                skip_interp_names.add(name)
                logger.info(
                    f"[pre-check] {name}: no interpolation features in detection output, "
                    "will skip CryoZeta-Interpolate inference for this entry"
                )

    # Runner
    runner = InferenceRunner(configs)

    # Data
    logger.info(f"Loading data from\n{configs.input_json_path}")
    dataloader = get_inference_dataloader(configs=configs, skip_names=skip_interp_names)

    num_data = len(dataloader.dataset)
    for seed in configs.seeds:
        seed_everything(seed=seed, deterministic=True)
        for batch in dataloader:
            try:
                data, atom_array, data_error_message = batch[0]
                
                sample_name = data.get("sample_name")
                
                # Skip if featurization was skipped (no input features)
                if "input_feature_dict" not in data:
                    logger.info(
                        f"[Rank {DIST_WRAPPER.rank}] {sample_name}: skipping "
                        "CryoZeta-Interpolate (no interpolation features)"
                    )
                    continue
                
                data["input_feature_dict"]["atom_array"] = atom_array
                data["input_feature_dict"]["entity_poly_type"] = data[
                    "entity_poly_type"
                ]

                sample_name = data["sample_name"]

                # Per-entry stage directory: {dump_dir}/{name}/{stage_name}
                entry_stage_dir = opjoin(
                    runner.dump_dir, sample_name, runner.stage_name
                )
                error_dir = opjoin(entry_stage_dir, "ERR")

                if len(data_error_message) > 0:
                    logger.info(data_error_message)
                    os.makedirs(error_dir, exist_ok=True)
                    with open(
                        opjoin(error_dir, f"{sample_name}.txt"),
                        "w",
                    ) as f:
                        f.write(data_error_message)
                    continue

                logger.info(
                    f"[Rank {DIST_WRAPPER.rank} ({data['sample_index'] + 1}/{num_data})] {sample_name}: "
                    f"N_asym {data['N_asym'].item()}, N_token {data['N_token'].item()}, "
                    f"N_atom {data['N_atom'].item()}, N_msa {data['N_msa'].item()}"
                )

                # Skip if output already exists
                predictions_dir = opjoin(entry_stage_dir, f"seed_{seed}", "predictions")
                if not configs.overwrite and opexists(predictions_dir):
                    logger.info(
                        f"[Rank {DIST_WRAPPER.rank}] {sample_name} seed_{seed} "
                        "already exists, skipping. Use --overwrite true to force."
                    )
                    continue

                # Pass per-entry stage dir to the model for saved_data output
                data["input_feature_dict"]["dump_dir"] = entry_stage_dir

                prediction = runner.predict(data)
                runner.dumper.dump(
                    pdb_id=sample_name,
                    seed=seed,
                    pred_dict=prediction,
                    atom_array=atom_array,
                    entity_poly_type=data["entity_poly_type"],
                )

                logger.info(
                    f"[Rank {DIST_WRAPPER.rank}] {data['sample_name']} succeeded.\nResults saved to {entry_stage_dir}"
                )

            except Exception as e:
                error_message = f"[Rank {DIST_WRAPPER.rank}]{data['sample_name']} {e}:\n{traceback.format_exc()}"
                logger.info(error_message)
                # Save error info
                sample_name = data["sample_name"]
                error_dir = opjoin(
                    runner.dump_dir, sample_name, runner.stage_name, "ERR"
                )
                os.makedirs(error_dir, exist_ok=True)
                if opexists(error_path := opjoin(error_dir, f"{sample_name}.txt")):
                    os.remove(error_path)
                with open(error_path, "w") as f:
                    f.write(error_message)
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()


def cli():
    """Entry point for the cryozeta-inference console script."""
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(
        configs=configs,
        arg_str=parse_sys_args(),
        fill_required_with_null=True,
    )
    configs_base["use_interpolation"] = configs.use_interpolation
    if configs.use_interpolation:
        configs.model.input_embedder.p_dim = 114
    else:
        configs.model.input_embedder.p_dim = 99
    main(configs)


if __name__ == "__main__":
    cli()
