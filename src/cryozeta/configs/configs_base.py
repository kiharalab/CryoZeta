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

# pylint: disable=C0114,C0301
from cryozeta.configs.extend_types import (
    GlobalConfigValue,
    ValueMaybeNone,
)

basic_configs = {
    "load_checkpoint_path": "",
    "em_file_dir": "",
    "load_strict": False,
    "load_params_only": True,
    "seed": 42,
    "deterministic": False,
    "use_affinity": True,
    "use_interpolation": True,
}
model_configs = {
    # Model
    "c_s": 384,
    "c_z": 128,
    "c_pz": 256,
    "c_p": 128,
    "c_s_inputs": 449,  # c_s_inputs == c_token + 32 + 32 + 1
    "c_atom": 128,
    "c_atompair": 16,
    "c_token": 384,
    "n_blocks": 48,
    "max_atoms_per_token": 24,  # DNA G max_atoms = 23
    "no_bins": 64,
    "sigma_data": 16.0,
    "blocks_per_ckpt": ValueMaybeNone(
        1
    ),  # NOTE: Number of blocks in each activation checkpoint, if None, no checkpointing is performed.
    # switch of kernels
    "use_memory_efficient_kernel": False,
    "use_deepspeed_evo_attention": True,
    "use_flash": False,
    "use_lma": False,
    "use_xformer": False,
    "use_cuequivariance_attention": True,
    "use_cuequivariance_multiplicative_update": True,
    "use_cuequivariance_attention_pair_bias": False,
    "find_unused_parameters": False,
    "dtype": "bf16",
    "loss_metrics_sparse_enable": True,
    "skip_amp": {
        "sample_diffusion": True,
        "confidence_head": True,
    },
    "infer_setting": {
        "chunk_size": ValueMaybeNone(256),
        "sample_diffusion_chunk_size": ValueMaybeNone(1),
        "lddt_metrics_sparse_enable": GlobalConfigValue("loss_metrics_sparse_enable"),
        "lddt_metrics_chunk_size": ValueMaybeNone(1),
    },
    "inference_noise_scheduler": {
        "s_max": 160.0,
        "s_min": 4e-4,
        "rho": 7,
        "sigma_data": 16.0,  # NOTE: in EDM, this is 1.0
    },
    "sample_diffusion": {
        "gamma0": 0.8,
        "gamma_min": 1.0,
        "noise_scale_lambda": 1.003,
        "step_scale_eta": 1.5,
        "N_step": 200,
        "N_sample": 5,
    },
    "model": {
        "N_model_seed": 1,  # for inference
        "N_cycle": 4,
        "input_embedder": {
            "c_atom": GlobalConfigValue("c_atom"),
            "c_atompair": GlobalConfigValue("c_atompair"),
            "c_token": GlobalConfigValue("c_token"),
            "pz_dim": 125,  # 119 for prot only, 125 for rna/dna
            "p_dim": 114,  # 100 for prot only, 114 for rna/dna
            "c_pz": GlobalConfigValue("c_pz"),
            "c_p": GlobalConfigValue("c_p"),
        },
        "relative_position_encoding": {
            "r_max": 32,
            "s_max": 2,
            "c_z": GlobalConfigValue("c_z"),
        },
        "template_embedder": {
            "c": 64,
            "c_z": GlobalConfigValue("c_z"),
            "n_blocks": 0,
            "dropout": 0.25,
            "blocks_per_ckpt": GlobalConfigValue("blocks_per_ckpt"),
        },
        "msa_module": {
            "c_m": 64,
            "c_z": GlobalConfigValue("c_z"),
            "c_s_inputs": GlobalConfigValue("c_s_inputs"),
            "n_blocks": 4,
            "msa_dropout": 0.15,
            "pair_dropout": 0.25,
            "blocks_per_ckpt": GlobalConfigValue("blocks_per_ckpt"),
        },
        "pairformer": {
            "n_blocks": GlobalConfigValue("n_blocks"),
            "c_z": GlobalConfigValue("c_z"),
            "c_s": GlobalConfigValue("c_s"),
            "n_heads": 16,
            "dropout": 0.25,
            "blocks_per_ckpt": GlobalConfigValue("blocks_per_ckpt"),
        },
        "empairformer": {
            "n_blocks": GlobalConfigValue("n_blocks"),
            "c_z": GlobalConfigValue("c_z"),
            "c_s": GlobalConfigValue("c_s"),
            "c_pz": GlobalConfigValue("c_pz"),
            "c_p": GlobalConfigValue("c_p"),
            "n_heads": 16,
            "dropout": 0.25,
            "blocks_per_ckpt": GlobalConfigValue("blocks_per_ckpt"),
            "c_hidden_msa_att": 32,
            "c_hidden_opm": 32,
            "c_hidden_mul": 128,
            "c_hidden_pair_att": 32,
            "no_heads_msa": 8,
            "no_heads_pair": 4,
            "transition_n": 4,
            "pair_dropout": 0.25,
            "no_column_attention": False,
            "opm_first": False,
            "fuse_projection_weights": False,
            "clear_cache_between_blocks": False,
            "tune_chunk_size": True,
            "inf": 1e9,
            "eps": 1e-8,
        },
        "diffusion_module": {
            "use_fine_grained_checkpoint": True,
            "sigma_data": GlobalConfigValue("sigma_data"),
            "c_token": 768,
            "c_atom": GlobalConfigValue("c_atom"),
            "c_atompair": GlobalConfigValue("c_atompair"),
            "c_z": GlobalConfigValue("c_z"),
            "c_s": GlobalConfigValue("c_s"),
            "c_s_inputs": GlobalConfigValue("c_s_inputs"),
            "initialization": {
                "zero_init_condition_transition": False,
                "zero_init_atom_encoder_residual_linear": False,
                "he_normal_init_atom_encoder_small_mlp": False,
                "he_normal_init_atom_encoder_output": False,
                "glorot_init_self_attention": False,
                "zero_init_adaln": True,
                "zero_init_residual_condition_transition": False,
                "zero_init_dit_output": True,
                "zero_init_atom_decoder_linear": False,
            },
            "atom_encoder": {
                "n_blocks": 3,
                "n_heads": 4,
            },
            "transformer": {
                "n_blocks": 24,
                "n_heads": 16,
            },
            "atom_decoder": {
                "n_blocks": 3,
                "n_heads": 4,
            },
            "blocks_per_ckpt": GlobalConfigValue("blocks_per_ckpt"),
        },
        "confidence_head": {
            "c_z": GlobalConfigValue("c_z"),
            "c_s": GlobalConfigValue("c_s"),
            "c_s_inputs": GlobalConfigValue("c_s_inputs"),
            "n_blocks": 4,
            "max_atoms_per_token": GlobalConfigValue("max_atoms_per_token"),
            "pairformer_dropout": 0.0,
            "blocks_per_ckpt": GlobalConfigValue("blocks_per_ckpt"),
            "distance_bin_start": 3.375,
            "distance_bin_end": 21.375,
            "distance_bin_step": 1.25,
            "stop_gradient": True,
        },
        "distogram_head": {
            "c_z": GlobalConfigValue("c_z"),
            "no_bins": GlobalConfigValue("no_bins"),
        },
        "point_residue_class": {
            "c_p": 256,
            "c_out": 10,
        },
        "point_noise": {
            "c_p": 256,
            "c_out": 2,
        },
    },
}
inference_loss_and_metrics_configs = {
    "loss": {
        "plddt": {
            "min_bin": 0,
            "max_bin": 100,
            "no_bins": 50,
            "eps": 1e-6,
        },
        "pde": {
            "min_bin": 0,
            "max_bin": 32,
            "no_bins": 64,
            "eps": 1e-6,
        },
        "pae": {
            "min_bin": 0,
            "max_bin": 32,
            "no_bins": 64,
            "eps": 1e-6,
        },
        "distogram": {
            "min_bin": 2.3125,
            "max_bin": 21.6875,
            "no_bins": 64,
            "eps": 1e-6,
        },
    },
    "metrics": {
        "clash": {"af3_clash_threshold": 1.1, "vdw_clash_threshold": 0.75},
    },
}

configs = {
    **basic_configs,
    **model_configs,
    **inference_loss_and_metrics_configs,
}
