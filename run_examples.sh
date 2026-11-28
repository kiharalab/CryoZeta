#!/usr/bin/env bash
set -e

# Usage: bash run_examples.sh [example_name] [gpu_id]
# Example: bash run_examples.sh 9dci 1
#          bash run_examples.sh all 0

EXAMPLE_NAME=${1:-all}
GPU_ID=${2:-0}

export CUDA_DEVICE_ORDER=PCI_BUS_ID

. "$(dirname -- "${BASH_SOURCE[0]}")/scripts/detect_pixi_env.sh"

PIXI_ENV="${PIXI_ENV:-$(detect_pixi_env "$GPU_ID")}"
echo "==> Using pixi environment: ${PIXI_ENV}"

if [ -d "$(pwd)/.pixi/envs/${PIXI_ENV}/lib" ]; then
    export LD_LIBRARY_PATH="$(pwd)/.pixi/envs/${PIXI_ENV}/lib:${LD_LIBRARY_PATH:-}"
elif [ -d "$(pwd)/.pixi/envs/dev/lib" ]; then
    export LD_LIBRARY_PATH="$(pwd)/.pixi/envs/dev/lib:${LD_LIBRARY_PATH:-}"
fi

# ── Inference parameters ──────────────────────────────────────────────────────
N_sample=5
N_step=20
N_cycle=10
seed=101
use_cuequivariance_attention=true
use_cuequivariance_multiplicative_update=true
use_cuequivariance_attention_pair_bias=false
overwrite=false
checkpoint_path="assets/cryozeta-v0.0.1.safetensors"
checkpoint_interpolation_path="assets/cryozeta-interpolate-v0.0.1.safetensors"

run_example() {
    local name=$1
    echo "================================================================================"
    echo "Running example: ${name}"
    echo "================================================================================"

    local input_json_path="examples/${name}.json"
    if [ ! -f "${input_json_path}" ]; then
        echo "Error: ${input_json_path} not found!"
        return 1
    fi

    local dump_dir="output/example_${name}"

    # ── Step 1: Atom detection ────────────────────────────────────────────────
    echo "[${name}] Step 1/4: Detection..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} pixi run --frozen -e "${PIXI_ENV}" cryozeta-detection json-run \
        ${input_json_path} ${dump_dir} --device cuda

    # ── Step 2: CryoZeta inference ────────────────────────────────────────────
    echo "[${name}] Step 2/4: CryoZeta inference..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} pixi run --frozen -e "${PIXI_ENV}" cryozeta-inference \
        --seeds ${seed} \
        --load_checkpoint_path ${checkpoint_path} \
        --em_file_dir ${dump_dir} \
        --dump_dir ${dump_dir} \
        --input_json_path ${input_json_path} \
        --use_cuequivariance_attention ${use_cuequivariance_attention} \
        --use_cuequivariance_multiplicative_update ${use_cuequivariance_multiplicative_update} \
        --use_cuequivariance_attention_pair_bias ${use_cuequivariance_attention_pair_bias} \
        --model.N_cycle ${N_cycle} \
        --sample_diffusion.N_sample ${N_sample} \
        --sample_diffusion.N_step ${N_step} \
        --data.num_dl_workers 1 \
        --use_interpolation false \
        --overwrite ${overwrite}

    # ── Step 3: CryoZeta-Interpolate inference ────────────────────────────────
    echo "[${name}] Step 3/4: CryoZeta-Interpolate inference..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} pixi run --frozen -e "${PIXI_ENV}" cryozeta-inference \
        --seeds ${seed} \
        --load_checkpoint_path ${checkpoint_interpolation_path} \
        --em_file_dir ${dump_dir} \
        --dump_dir ${dump_dir} \
        --input_json_path ${input_json_path} \
        --use_cuequivariance_attention ${use_cuequivariance_attention} \
        --use_cuequivariance_multiplicative_update ${use_cuequivariance_multiplicative_update} \
        --use_cuequivariance_attention_pair_bias ${use_cuequivariance_attention_pair_bias} \
        --model.N_cycle ${N_cycle} \
        --sample_diffusion.N_sample ${N_sample} \
        --sample_diffusion.N_step ${N_step} \
        --data.num_dl_workers 1 \
        --use_interpolation true \
        --overwrite ${overwrite}

    # ── Step 4: Combine best results ──────────────────────────────────────────
    echo "[${name}] Step 4/4: Combining results..."
    pixi run --frozen -e "${PIXI_ENV}" cryozeta-combine \
        --dump-dir ${dump_dir} \
        --input-json-path ${input_json_path} \
        --seeds ${seed} \
        --num-select ${N_sample}

    echo "[${name}] Done. Output: ${dump_dir}"
    echo ""
}

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPU_ID}" 2>/dev/null | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' || echo "unknown")
echo "==> GPU ${GPU_ID}: ${gpu_name}"

if [ "${EXAMPLE_NAME}" = "all" ]; then
    run_example "9bh4"
    run_example "9dci"
    run_example "9j8v"
else
    run_example "${EXAMPLE_NAME}"
fi

echo "All done."
