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
Usage: large_inference_demo.sh [OPTIONS]

Run CryoZeta large complex inference pipeline (cycle prediction with stage-wise EM filtering).

Options:
  -e, --env ENV       Pixi environment name (cu11, cu13, default, …)
                      or CUDA major version (11, 12, 13).
                      Overrides auto-detection and PIXI_ENV / CRYOZETA_CUDA.
  -g, --gpu IDS       Comma-separated GPU device IDs (e.g. "0", "0,1").
                      Default: 1
  -x, --example SEL   Select one entry from assets/examples/large_examples.json.
                      SEL can be a 0-based index (e.g. 0) or entry name (e.g. 9nb5).
                      Default: 0
  -r, --registration  Registration method: auto (default), teaser, svd, vesper.
  -h, --help          Show this help message and exit.

Environment variables (lower priority than flags):
  PIXI_ENV            Pixi environment name (set by env_setup.sh activation).
  CRYOZETA_CUDA       CUDA major version shorthand (11, 12, 13).

Examples:
  sh large_inference_demo.sh                        # auto-detect everything
  sh large_inference_demo.sh -e cu11 -g 1           # CUDA 11, GPU 1
  sh large_inference_demo.sh --env 13 --gpu 2       # CUDA 13, GPU 2
  sh large_inference_demo.sh --example 2            # run entry at index 2
  sh large_inference_demo.sh --example 9ey0         # run entry named 9ey0
  sh large_inference_demo.sh -r teaser              # TEASER++ registration only
  CRYOZETA_CUDA=11 sh large_inference_demo.sh -g 1  # CUDA 11 via env var, GPU 1
USAGE
}

# ── Auto-detect pixi environment based on GPU and driver ──────────────────────
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
cli_example="0"
cli_registration="auto"

while [ $# -gt 0 ]; do
    case "$1" in
        -e|--env)
            [ -z "${2:-}" ] && { echo "ERROR: $1 requires an argument" >&2; usage; exit 1; }
            cli_env="$2"; shift 2 ;;
        -g|--gpu)
            [ -z "${2:-}" ] && { echo "ERROR: $1 requires an argument" >&2; usage; exit 1; }
            cli_gpu="$2"; shift 2 ;;
        -x|--example)
            [ -z "${2:-}" ] && { echo "ERROR: $1 requires an argument" >&2; usage; exit 1; }
            cli_example="$2"; shift 2 ;;
        -r|--registration)
            [ -z "${2:-}" ] && { echo "ERROR: $1 requires an argument" >&2; usage; exit 1; }
            cli_registration="$2"; shift 2 ;;
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
use_cuequivariance=true  # Master toggle for all cuEquivariance modules
use_cuequivariance_attention=${use_cuequivariance}
use_cuequivariance_multiplicative_update=${use_cuequivariance}
use_cuequivariance_attention_pair_bias=${use_cuequivariance}
use_opm_tilelang=false  # Set to true to use TileLang OPM kernel (overrides USE_OPM_CHUNKED)
checkpoint_path="assets/cryozeta-v0.0.1.safetensors"
detection_checkpoint_path="assets/cryozeta-detection-v0.0.1.safetensors"

# ── Large inference / cycle prediction parameters ──────────────────────────────
# Registration method: auto (default), teaser, svd, vesper
registration_method="${cli_registration}"
# EM point cropping threshold (Å)
em_threshold=5.0
# Input and output paths
input_json_path="assets/examples/large_examples.json"
dump_dir="output/large_examples"
selected_entry="${cli_example}"
selected_input_json="${dump_dir}/selected_input_entry.json"
cycle_input_json="${dump_dir}/selected_cycle_input_entry.json"

# ── GPU configuration ────────────────────────────────────────────────────────
# CLI flag overrides the default; use -g/--gpu to set from command line.
gpu_ids="${cli_gpu:-1}"
# ─────────────────────────────────────────────────────────────────────────────

echo "==> Using pixi environment: ${PIXI_ENV}"
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${gpu_ids}" 2>/dev/null | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' || echo "unknown")
gpu_cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "${gpu_ids}" 2>/dev/null | tr -d '[:space:]' || echo "unknown")
cuda_driver_ver=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' || echo "unknown")
echo "==> GPU ${gpu_ids}: ${gpu_name} (compute capability ${gpu_cc})"
echo "==> CUDA driver version: ${cuda_driver_ver}"

# ── Run cycle prediction ──────────────────────────────────────────────────────

# Build registration flag
registration_flag=""
case "$registration_method" in
    auto)
        echo "==> Registration method: Auto (comparing TEASER++ and SVD)"
        ;;
    teaser)
        echo "==> Registration method: TEASER++ only"
        registration_flag="--teaser_only"
        ;;
    svd)
        echo "==> Registration method: SVD only"
        registration_flag="--svd_only"
        ;;
    vesper)
        echo "==> Registration method: VESPER only"
        registration_flag="--vesper_only"
        # Note: Additional VESPER parameters would be added here if needed
        ;;
    *)
        echo "ERROR: Unknown registration method '$registration_method'" >&2
        exit 1
        ;;
esac

echo "==> Starting large complex cycle prediction..."
echo "    Input JSON: ${input_json_path}"
echo "    Dump directory: ${dump_dir}"
echo "    EM threshold: ${em_threshold}"
echo "    Selected entry: ${selected_entry}"

if [ ! -f "${input_json_path}" ]; then
    echo "ERROR: Input JSON not found: ${input_json_path}" >&2
    exit 1
fi

if [ ! -f "${detection_checkpoint_path}" ]; then
    echo "ERROR: Detection checkpoint not found: ${detection_checkpoint_path}" >&2
    echo "       Please place cryozeta-detection-v0.0.1.safetensors under ./assets" >&2
    exit 1
fi

if [ ! -f "${checkpoint_path}" ]; then
    echo "ERROR: Inference checkpoint not found: ${checkpoint_path}" >&2
    echo "       Please place cryozeta-v0.0.1.safetensors under ./assets" >&2
    exit 1
fi

mkdir -p "${dump_dir}"

echo "==> Preparing single-entry input from ${input_json_path}"
python3 - "${input_json_path}" "${selected_input_json}" "${cycle_input_json}" "${dump_dir}" "${selected_entry}" <<'PY'
import json
import sys
from pathlib import Path

input_json = Path(sys.argv[1])
selected_input_json = Path(sys.argv[2])
cycle_input_json = Path(sys.argv[3])
dump_dir = Path(sys.argv[4])
selection = sys.argv[5]

data = json.loads(input_json.read_text())
if not isinstance(data, list) or len(data) == 0:
    raise ValueError(f"Input JSON must be a non-empty list: {input_json}")

if selection.isdigit():
    entry_index = int(selection)
    if not (0 <= entry_index < len(data)):
        raise IndexError(
            f"example index {entry_index} is out of range for {len(data)} entries"
        )
else:
    matches = [i for i, item in enumerate(data) if item.get("name") == selection]
    if not matches:
        names = ", ".join(item.get("name", "<missing>") for item in data)
        raise ValueError(
            f"example '{selection}' not found in {input_json}. Available names: {names}"
        )
    entry_index = matches[0]

entry = dict(data[entry_index])
name = entry.get("name")
if not name:
    raise ValueError("Selected entry must contain 'name'")

selected_input_json.parent.mkdir(parents=True, exist_ok=True)
selected_input_json.write_text(json.dumps([entry], indent=2))

entry["em_file"] = str(
    (dump_dir / name / "CryoZeta-Detection" / f"{name}.pt").resolve()
)
cycle_input_json.parent.mkdir(parents=True, exist_ok=True)
cycle_input_json.write_text(json.dumps([entry], indent=2))

print(f"Prepared sample: {name} (index {entry_index})")
print(f"  detection input: {selected_input_json}")
print(f"  cycle input:     {cycle_input_json}")
PY

echo "==> Running detection to generate EM .pt for selected sample..."
CUDA_VISIBLE_DEVICES=${gpu_ids} pixi run --frozen -e "${PIXI_ENV}" cryozeta-detection json-run \
    "${selected_input_json}" "${dump_dir}" --device cuda --overwrite

CUDA_VISIBLE_DEVICES=${gpu_ids} pixi run --frozen -e "${PIXI_ENV}" cryozeta-cycle-predict \
    --base_json "${cycle_input_json}" \
    --load_checkpoint_path "${checkpoint_path}" \
    --dump_root "${dump_dir}" \
    --threshold "${em_threshold}" \
    --seeds "${seed}" \
    --sample.N_sample "${N_sample}" \
    --sample.N_step "${N_step}" \
    --model.N_cycle "${N_cycle}" \
    --use_deepspeed_evo_attention "${use_deepspeed_evo_attention}" \
    --use_cuequivariance_attention "${use_cuequivariance_attention}" \
    --use_cuequivariance_multiplicative_update "${use_cuequivariance_multiplicative_update}" \
    --use_cuequivariance_attention_pair_bias "${use_cuequivariance_attention_pair_bias}" \
    --use_opm_tilelang "${use_opm_tilelang}" \
    --num_dl_workers 1 \
    ${registration_flag}

echo "==> Cycle prediction completed!"
echo "    Output directory: ${dump_dir}"

# ── Combine stages ───────────────────────────────────────────────────────────

echo "==> Combining stages into final structure..."

pixi run --frozen -e "${PIXI_ENV}" cryozeta-combine-stages \
    --dump_root "${dump_dir}" \
    --output "${dump_dir}/combined.cif"

echo "==> Done! Combined structure saved to: ${dump_dir}/combined.cif"
