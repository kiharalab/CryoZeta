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
# code remains under Apache 2.0; the combined work is distributed
# under GPLv3.
set -e

# ── Usage ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<'USAGE'
Usage: inference_demo.sh [OPTIONS]

Run CryoZeta inference pipeline (detection → structure prediction → combine).

Options:
  -e, --env ENV       Pixi environment name (cu11, cu13, default, …)
                      or CUDA major version (11, 12, 13).
                      Overrides auto-detection and PIXI_ENV / CRYOZETA_CUDA.
  -g, --gpu IDS       Comma-separated GPU device IDs (e.g. "0", "0,1").
                      Default: 0
  -h, --help          Show this help message and exit.

Environment variables (lower priority than flags):
  PIXI_ENV            Pixi environment name (set by env_setup.sh activation).
  CRYOZETA_CUDA       CUDA major version shorthand (11, 12, 13).

Examples:
  sh inference_demo.sh                        # auto-detect everything
  sh inference_demo.sh -e cu11 -g 0           # CUDA 11, GPU 0
  sh inference_demo.sh --env 13 --gpu 2       # CUDA 13, GPU 2
  CRYOZETA_CUDA=11 sh inference_demo.sh -g 1  # CUDA 11 via env var, GPU 1
USAGE
}

# ── Auto-detect pixi environment based on GPU and driver ──────────────────────
# Two constraints determine the best CUDA environment:
#   1. Compute capability (GPU architecture support)
#   2. Driver-supported CUDA version (hard upper bound from nvidia-smi)
#
# Architecture preference:  cc >= 10 → cu13,  cc >= 8 → cu12 (default),  else → cu11
# Driver constraint:        driver must support the target CUDA major version.
detect_pixi_env() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "default"; return
    fi

    # Max CUDA version the installed driver supports (major only)
    local driver_cuda
    driver_cuda=$(nvidia-smi 2>/dev/null \
        | sed -n 's/.*CUDA Version: *\([0-9]*\).*/\1/p')

    # Highest compute capability among all GPUs
    local compute_cap
    compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
        | sort -t. -k1,1nr -k2,2nr | head -1 | tr -d '[:space:]')

    if [ -z "$driver_cuda" ] || [ -z "$compute_cap" ]; then
        echo "default"; return
    fi

    local major="${compute_cap%%.*}"

    if [ "$major" -ge 10 ] 2>/dev/null && [ "$driver_cuda" -ge 13 ] 2>/dev/null; then
        echo "cu13"
    elif [ "$major" -ge 8 ] 2>/dev/null && [ "$driver_cuda" -ge 12 ] 2>/dev/null; then
        echo "default"
    elif [ "$driver_cuda" -ge 11 ] 2>/dev/null; then
        echo "cu11"
    else
        echo "default"
    fi
}

# ── Map CUDA version shorthand to pixi env name ──────────────────────────────
cuda_version_to_env() {
    case "$1" in
        11)       echo "cu11"    ;;
        12)       echo "default" ;;
        13)       echo "cu13"    ;;
        cu11|cu13|default|dev|dev-cu11|dev-cu13) echo "$1" ;;
        *)
            echo "ERROR: unrecognised environment/CUDA version '$1'" >&2
            echo "       Use 11, 12, 13 or a pixi env name (cu11, cu13, default, …)" >&2
            exit 1
            ;;
    esac
}

# ── Parse command-line arguments ──────────────────────────────────────────────
cli_env=""
cli_gpu=""

while [ $# -gt 0 ]; do
    case "$1" in
        -e|--env)
            [ -z "${2:-}" ] && { echo "ERROR: $1 requires an argument" >&2; usage; exit 1; }
            cli_env="$2"; shift 2 ;;
        -g|--gpu)
            [ -z "${2:-}" ] && { echo "ERROR: $1 requires an argument" >&2; usage; exit 1; }
            cli_gpu="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "ERROR: unknown option '$1'" >&2; usage; exit 1 ;;
    esac
done

# ── Resolve pixi environment ─────────────────────────────────────────────────
# Priority: CLI flag > PIXI_ENV (from activation) > CRYOZETA_CUDA > auto-detect
if [ -n "$cli_env" ]; then
    PIXI_ENV="$(cuda_version_to_env "$cli_env")"
elif [ -n "${PIXI_ENV:-}" ]; then
    : # already set by env_setup.sh activation
elif [ -n "${CRYOZETA_CUDA:-}" ]; then
    PIXI_ENV="$(cuda_version_to_env "$CRYOZETA_CUDA")"
else
    PIXI_ENV="$(detect_pixi_env)"
fi

export LAYERNORM_TYPE=fast_layernorm
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export USE_OPM_CHUNKED=1  # Set to 0 to use original einsum OPM

# Triton needs ptxas on PATH; Blackwell (sm_100+) requires CUDA 13+ ptxas
CONDA_PREFIX="$(pwd)/.pixi/envs/${PIXI_ENV}"
if [ -x "${CONDA_PREFIX}/bin/ptxas" ]; then
    export TRITON_PTXAS_PATH="${CONDA_PREFIX}/bin/ptxas"
    export TRITON_PTXAS_BLACKWELL_PATH="${CONDA_PREFIX}/bin/ptxas"
fi

# Point CUDA_HOME to the pixi environment so that PyTorch's cpp_extension
# finds the pixi-managed nvcc (and matching host-compiler compatibility)
# instead of a system-installed CUDA toolkit.
PIXI_PREFIX="$(pixi run --frozen -e "${PIXI_ENV}" bash -c 'echo $CONDA_PREFIX')"
export CUDA_HOME="${PIXI_PREFIX}"

# ── Inference parameters ──────────────────────────────────────────────────────
N_sample=5
N_step=20
N_cycle=10
seed=101
use_deepspeed_evo_attention=true
use_cuequivariance_attention=true
use_cuequivariance_multiplicative_update=true
use_cuequivariance_attention_pair_bias=true
use_opm_tilelang=false  # Set to true to use TileLang OPM kernel (overrides USE_OPM_CHUNKED)
mode="combined"  # cryozeta, cryozeta-interpolate, or combined
overwrite=false   # set to true to re-run even if output already exists
checkpoint_path="assets/cryozeta-v0.0.1.safetensors"
checkpoint_interpolation_path="assets/cryozeta-interpolate-v0.0.1.safetensors"
input_json_path="examples/example.json"
dump_dir="output/example"

# ── GPU configuration ────────────────────────────────────────────────────────
# CLI flag overrides the default; use -g/--gpu to set from command line.
gpu_ids="${cli_gpu:-0}"
# ─────────────────────────────────────────────────────────────────────────────

echo "==> Using pixi environment: ${PIXI_ENV}"
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${gpu_ids}" 2>/dev/null | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' || echo "unknown")
gpu_cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "${gpu_ids}" 2>/dev/null | tr -d '[:space:]' || echo "unknown")
cuda_driver_ver=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' || echo "unknown")
echo "==> GPU ${gpu_ids}: ${gpu_name} (compute capability ${gpu_cc})"
echo "==> CUDA driver version: ${cuda_driver_ver}"

# Build --overwrite flag for typer-based CLIs (cryozeta-detection)
em_overwrite_flag=""
[ "$overwrite" = "true" ] && em_overwrite_flag="--overwrite"

# ── Step 1: Atom detection (single GPU) ──────────────────────────────────────
# Outputs to: ${dump_dir}/<name>/CryoZeta-Detection/
CUDA_VISIBLE_DEVICES=${gpu_ids} pixi run --frozen -e "${PIXI_ENV}" cryozeta-detection json-run \
    ${input_json_path} ${dump_dir} --device cuda ${em_overwrite_flag}

# ── Step 2: CryoZeta inference (single GPU) ──────────────────────────────────
# Outputs to: ${dump_dir}/<name>/CryoZeta/seed_<seed>/
if [ "$mode" = "combined" ] || [ "$mode" = "cryozeta" ]; then
    CUDA_VISIBLE_DEVICES=${gpu_ids} pixi run --frozen -e "${PIXI_ENV}" cryozeta-inference \
    --seeds ${seed} \
    --load_checkpoint_path ${checkpoint_path} \
    --em_file_dir ${dump_dir} \
    --dump_dir ${dump_dir} \
    --input_json_path ${input_json_path} \
    --use_deepspeed_evo_attention ${use_deepspeed_evo_attention} \
    --use_cuequivariance_attention ${use_cuequivariance_attention} \
    --use_cuequivariance_multiplicative_update ${use_cuequivariance_multiplicative_update} \
    --use_cuequivariance_attention_pair_bias ${use_cuequivariance_attention_pair_bias} \
    --use_opm_tilelang ${use_opm_tilelang} \
    --model.N_cycle ${N_cycle} \
    --sample_diffusion.N_sample ${N_sample} \
    --sample_diffusion.N_step ${N_step} \
    --data.num_dl_workers 1 \
    --use_interpolation false \
    --overwrite ${overwrite}
fi

# Outputs to: ${dump_dir}/<name>/CryoZeta-Interpolate/seed_<seed>/
if [ "$mode" = "combined" ] || [ "$mode" = "cryozeta-interpolate" ]; then
    CUDA_VISIBLE_DEVICES=${gpu_ids} pixi run --frozen -e "${PIXI_ENV}" cryozeta-inference \
    --seeds ${seed} \
    --load_checkpoint_path ${checkpoint_interpolation_path} \
    --dump_dir ${dump_dir} \
    --em_file_dir ${dump_dir} \
    --input_json_path ${input_json_path} \
    --use_deepspeed_evo_attention ${use_deepspeed_evo_attention} \
    --use_cuequivariance_attention ${use_cuequivariance_attention} \
    --use_cuequivariance_multiplicative_update ${use_cuequivariance_multiplicative_update} \
    --use_cuequivariance_attention_pair_bias ${use_cuequivariance_attention_pair_bias} \
    --use_opm_tilelang ${use_opm_tilelang} \
    --model.N_cycle ${N_cycle} \
    --sample_diffusion.N_sample ${N_sample} \
    --sample_diffusion.N_step ${N_step} \
    --data.num_dl_workers 1 \
    --use_interpolation true \
    --overwrite ${overwrite}
fi

# ── Step 3: Combine best results (CPU only) ──────────────────────────────────
# Outputs to: ${dump_dir}/<name>/CryoZeta-Final/
if [ "$mode" = "combined" ]; then
    pixi run --frozen -e "${PIXI_ENV}" cryozeta-combine \
    --dump-dir ${dump_dir} \
    --input-json-path ${input_json_path} \
    --seeds ${seed} \
    --num-select ${N_sample}
fi
