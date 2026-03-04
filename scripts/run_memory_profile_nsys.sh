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

OUTPUT_DIR=output/nsys_profile
INPUT_JSON=examples/example.json
REPORT_NAME="${OUTPUT_DIR}/cryozeta_memory"
mkdir -p "${OUTPUT_DIR}"

# ── Step 1: Run detection if needed ───────────────────────────────────────
DETECTION_PT="${OUTPUT_DIR}/9b0l/CryoZeta-Detection/9b0l.pt"
if [ ! -f "${DETECTION_PT}" ]; then
    echo "Detection output not found at ${DETECTION_PT}"
    echo "Running detection step first..."
    pixi run --frozen -e "${PIXI_ENV}" cryozeta-detection json-run \
        "${INPUT_JSON}" \
        "${OUTPUT_DIR}" \
        --overwrite
    echo "Detection complete."
else
    echo "Detection output already exists, skipping."
fi

# ── Step 2: nsys memory profiling ─────────────────────────────────────────
echo "Starting nsys memory profiling..."

# Find the pixi python
PIXI_PYTHON="$(pixi run --frozen -e "${PIXI_ENV}" which python)"

nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=true \
    --stats=true \
    -o "${REPORT_NAME}" \
    "${PIXI_PYTHON}" scripts/profile_memory_nsys.py \
        --seeds 101 \
        --load_checkpoint_path assets/cryozeta-v0.0.1.safetensors \
        --em_file_dir "${OUTPUT_DIR}" \
        --dump_dir "${OUTPUT_DIR}" \
        --input_json_path "${INPUT_JSON}" \
        --use_deepspeed_evo_attention true \
        --model.N_cycle 1 \
        --sample_diffusion.N_sample 1 \
        --sample_diffusion.N_step 2 \
        --data.num_dl_workers 1 \
        --use_interpolation false \
        --overwrite true

echo ""
echo "nsys profiling complete. Outputs:"
echo "  ${REPORT_NAME}.nsys-rep  — open in Nsight Systems GUI"
echo "  ${REPORT_NAME}.sqlite    — queryable with nsys stats / sqlite3"
echo ""
echo "Quick stats from CLI:"
echo "  nsys stats ${REPORT_NAME}.nsys-rep"
echo ""
echo "To view memory timeline + NVTX ranges:"
echo "  Open ${REPORT_NAME}.nsys-rep in Nsight Systems GUI"
echo "  -> Enable 'CUDA Memory Operations' and 'NVTX' rows"
