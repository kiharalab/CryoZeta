#!/usr/bin/env bash
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

export LAYERNORM_TYPE=fast_layernorm
export CUTLASS_PATH=$(pwd)/externals/cutlass

N_sample=5
N_step=20
N_cycle=10
seed=101
use_deepspeed_evo_attention=true
mode="combined" # cryozeta, cryozeta-interpolate, or combined
overwrite=false # set to true to re-run even if output already exists
checkpoint_path="assets/cryozeta-v0.0.1.safetensors"
checkpoint_interpolation_path="assets/cryozeta-interpolate-v0.0.1.safetensors"
input_json_path="examples/example.json"
dump_dir="output/example"

# ── GPU configuration ──────────────────────────────────────────────────────────
# Specify which single GPU to use (e.g. "0", "1", "2").
gpu_ids="0"
# ───────────────────────────────────────────────────────────────────────────────

echo "==> GPU configuration: gpu_ids=${gpu_ids}"

# Build --overwrite flag for typer-based CLIs (cryozeta-detection)
em_overwrite_flag=""
[ "$overwrite" = "true" ] && em_overwrite_flag="--overwrite"

# ── Step 1: Atom detection (single GPU) ───────────────────────────────────────
# Outputs to: ${dump_dir}/<name>/CryoZeta-Detection/
CUDA_VISIBLE_DEVICES=${gpu_ids} uv run cryozeta-detection json-run \
    ${input_json_path} ${dump_dir} --device cuda ${em_overwrite_flag}

# ── Step 2: CryoZeta inference (single GPU) ──────────────────────────────────
# Outputs to: ${dump_dir}/<name>/CryoZeta/seed_<seed>/
if [ "$mode" = "combined" ] || [ "$mode" = "cryozeta" ]; then
    CUDA_VISIBLE_DEVICES=${gpu_ids} uv run cryozeta-inference \
    --seeds ${seed} \
    --load_checkpoint_path ${checkpoint_path} \
    --em_file_dir ${dump_dir} \
    --dump_dir ${dump_dir} \
    --input_json_path ${input_json_path} \
    --use_deepspeed_evo_attention ${use_deepspeed_evo_attention} \
    --model.N_cycle ${N_cycle} \
    --sample_diffusion.N_sample ${N_sample} \
    --sample_diffusion.N_step ${N_step} \
    --data.num_dl_workers 1 \
    --use_interpolation false \
    --overwrite ${overwrite}
fi

# Outputs to: ${dump_dir}/<name>/CryoZeta-Interpolate/seed_<seed>/
if [ "$mode" = "combined" ] || [ "$mode" = "cryozeta-interpolate" ]; then
    CUDA_VISIBLE_DEVICES=${gpu_ids} uv run cryozeta-inference \
    --seeds ${seed} \
    --load_checkpoint_path ${checkpoint_interpolation_path} \
    --dump_dir ${dump_dir} \
    --em_file_dir ${dump_dir} \
    --input_json_path ${input_json_path} \
    --use_deepspeed_evo_attention ${use_deepspeed_evo_attention} \
    --model.N_cycle ${N_cycle} \
    --sample_diffusion.N_sample ${N_sample} \
    --sample_diffusion.N_step ${N_step} \
    --data.num_dl_workers 1 \
    --use_interpolation true \
    --overwrite ${overwrite}
fi

# ── Step 3: Combine best results (CPU only) ───────────────────────────────────
# Outputs to: ${dump_dir}/<name>/CryoZeta-Final/
if [ "$mode" = "combined" ]; then
    uv run cryozeta-combine \
    --dump-dir ${dump_dir} \
    --input-json-path ${input_json_path} \
    --seeds ${seed} \
    --num-select ${N_sample}
fi
