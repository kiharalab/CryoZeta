#!/usr/bin/env bash
set -e

export LAYERNORM_TYPE=fast_layernorm
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1  # <-- adjust based on nvidia-smi

# ── Auto-detect pixi environment based on GPU ────────────────────────────────
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

CONDA_PREFIX="$(pwd)/.pixi/envs/${PIXI_ENV}"
export CUTLASS_PATH="${CONDA_PREFIX}"

if [ -x "${CONDA_PREFIX}/bin/ptxas" ]; then
    export TRITON_PTXAS_PATH="${CONDA_PREFIX}/bin/ptxas"
    export TRITON_PTXAS_BLACKWELL_PATH="${CONDA_PREFIX}/bin/ptxas"
fi

PIXI_PREFIX="$(pixi run --frozen -e "${PIXI_ENV}" bash -c 'echo $CONDA_PREFIX')"
export CUDA_HOME="${PIXI_PREFIX}"

if [ -d "${CONDA_PREFIX}/lib" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH"
elif [ -d "$(pwd)/.pixi/envs/dev/lib" ]; then
    export LD_LIBRARY_PATH="$(pwd)/.pixi/envs/dev/lib:$LD_LIBRARY_PATH"
fi

# ── Inference parameters (same as inference_demo.sh) ─────────────────────────
N_sample=5
N_step=20
N_cycle=10
seed=101
use_deepspeed_evo_attention=true
use_cuequivariance=true
use_cuequivariance_attention=${use_cuequivariance}
use_cuequivariance_multiplicative_update=${use_cuequivariance}
use_cuequivariance_attention_pair_bias=${use_cuequivariance}
use_opm_tilelang=false
checkpoint_path="assets/cryozeta-v0.0.1.safetensors"
input_json_path="assets/examples/example.json"

OUTPUT_DIR=output/memory_profile_run

# ── Step 1: Run detection if needed ──────────────────────────────────────────
DETECTION_PT="${OUTPUT_DIR}/9b0l/CryoZeta-Detection/9b0l.pt"
if [ ! -f "${DETECTION_PT}" ]; then
    echo "Detection output not found at ${DETECTION_PT}"
    echo "Running detection step first..."
    pixi run --frozen -e "${PIXI_ENV}" cryozeta-detection json-run \
        "${input_json_path}" \
        "${OUTPUT_DIR}" \
        --device cuda \
        --overwrite
    echo "Detection complete."
else
    echo "Detection output already exists, skipping."
fi

# ── Step 2: Memory profiling ─────────────────────────────────────────────────
echo "Starting memory profiling..."
pixi run --frozen -e "${PIXI_ENV}" python scripts/profile_memory.py \
    --seeds ${seed} \
    --load_checkpoint_path ${checkpoint_path} \
    --em_file_dir "${OUTPUT_DIR}" \
    --dump_dir "${OUTPUT_DIR}" \
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
    --overwrite true

echo ""
echo "Memory profiling complete. Outputs:"
echo "  ${OUTPUT_DIR}/memory_profile/memory_timeline.html  — interactive memory timeline"
echo "  ${OUTPUT_DIR}/memory_profile/memory_snapshot.pickle — drag to https://pytorch.org/memory_viz"
echo "  ${OUTPUT_DIR}/memory_profile/memory_trace.json      — open at https://ui.perfetto.dev"
echo "  ${OUTPUT_DIR}/memory_profile/memory_stacks.txt      — for flamegraph generation"
