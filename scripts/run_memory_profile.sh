#!/usr/bin/env bash
set -e

export LAYERNORM_TYPE=fast_layernorm
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1  # <-- adjust based on nvidia-smi

# Pixi environment setup
detect_pixi_env() {
    local compute_cap
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "default"; return
    fi
    compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -t. -k1,1nr -k2,2nr | head -1 | tr -d '[:space:]')
    if [ -z "$compute_cap" ]; then
        echo "default"; return
    fi
    local major="${compute_cap%%.*}"
    if [ "$major" -ge 10 ] 2>/dev/null; then
        echo "cu13"
    else
        echo "default"
    fi
}

PIXI_ENV=$(detect_pixi_env)
echo "==> Using pixi environment: ${PIXI_ENV}"
export CUTLASS_PATH="$(pwd)/.pixi/envs/${PIXI_ENV}"

if [ -d "$(pwd)/.pixi/envs/${PIXI_ENV}/lib" ]; then
    export LD_LIBRARY_PATH="$(pwd)/.pixi/envs/${PIXI_ENV}/lib:$LD_LIBRARY_PATH"
elif [ -d "$(pwd)/.pixi/envs/dev/lib" ]; then
    export LD_LIBRARY_PATH="$(pwd)/.pixi/envs/dev/lib:$LD_LIBRARY_PATH"
fi

# Cleanup
OUTPUT_DIR=output/memory_profile_run
echo "Cleaning up previous results in ${OUTPUT_DIR}..."
rm -rf "${OUTPUT_DIR}"

echo "Starting memory profiling..."
pixi run --frozen -e "${PIXI_ENV}" python scripts/profile_memory.py \
    --seeds 101 \
    --load_checkpoint_path assets/cryozeta-v0.0.1.safetensors \
    --em_file_dir output/example \
    --dump_dir "${OUTPUT_DIR}" \
    --input_json_path examples/example.json \
    --use_deepspeed_evo_attention true \
    --model.N_cycle 1 \
    --sample_diffusion.N_sample 1 \
    --sample_diffusion.N_step 2 \
    --data.num_dl_workers 1 \
    --use_interpolation false \
    --overwrite true

echo ""
echo "Memory profiling complete. Outputs:"
echo "  ${OUTPUT_DIR}/memory_profile/memory_timeline.html  — interactive memory timeline"
echo "  ${OUTPUT_DIR}/memory_profile/memory_snapshot.pickle — drag to https://pytorch.org/memory_viz"
echo "  ${OUTPUT_DIR}/memory_profile/memory_trace.json      — open at https://ui.perfetto.dev"
echo "  ${OUTPUT_DIR}/memory_profile/memory_stacks.txt      — for flamegraph generation"
