#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────
SIF="/net/kihara/home/zhu773/CryoZeta_build/CryoZeta/CryoZeta.sif"
GPU=""
# ─────────────────────────────────────────────────────────────────────

usage() {
    cat <<EOF
Usage: $(basename "$0") --gpu ID [OPTIONS] <input_json> <output_dir>
       $(basename "$0") prepare-ui [--sif PATH]

Run the CryoZeta pipeline (detection → inference → combine) via Apptainer.

Subcommands:
  prepare-ui          Launch the web UI for preparing input JSON files

Arguments:
  input_json    Path to input JSON file (e.g. examples/example.json)
  output_dir    Directory for output results

Required:
  --gpu ID            CUDA device ID (e.g. 0, 1, 2)

Options:
  --sif PATH          Path to CryoZeta.sif  (default: $SIF)
  --interpolation     Use interpolation model in Stage 2
  --seeds SEEDS       Random seeds           (default: 101)
  --n-sample N        Number of diffusion samples (default: 5)
  --n-step N          Number of diffusion steps   (default: 20)
  --n-cycle N         Number of model cycles      (default: 10)
  --num-select N      Number to select in combine (default: 5)
  --skip-detection    Skip Stage 1
  --skip-inference    Skip Stage 2
  --skip-combine      Skip Stage 3
  -h, --help          Show this help
EOF
    exit 0
}

# ── Defaults ─────────────────────────────────────────────────────────
USE_INTERPOLATION=false
SEEDS=101
N_SAMPLE=5
N_STEP=20
N_CYCLE=10
NUM_SELECT=5
SKIP_DETECTION=false
SKIP_INFERENCE=false
SKIP_COMBINE=false

# ── Handle prepare-ui subcommand ─────────────────────────────────────
if [[ "${1:-}" == "prepare-ui" ]]; then
    shift
    while [[ $# -gt 0 ]]; do
        case $1 in
            --sif) SIF="$2"; shift 2 ;;
            *)     echo "Unknown option: $1" >&2; exit 1 ;;
        esac
    done
    [[ -f "$SIF" ]] || { echo "Error: SIF not found: $SIF" >&2; exit 1; }
    apptainer exec --bind "$(pwd):$(pwd)" "$SIF" bash -c \
        "cd /app/CryoZeta && source /opt/conda/etc/profile.d/conda.sh && conda activate cryozeta && cryozeta-prepare ui"
    exit 0
fi

# ── Parse arguments ──────────────────────────────────────────────────
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --sif)            SIF="$2";            shift 2 ;;
        --gpu)            GPU="$2";            shift 2 ;;
        --interpolation)  USE_INTERPOLATION=true; shift ;;
        --seeds)          SEEDS="$2";          shift 2 ;;
        --n-sample)       N_SAMPLE="$2";       shift 2 ;;
        --n-step)         N_STEP="$2";         shift 2 ;;
        --n-cycle)        N_CYCLE="$2";        shift 2 ;;
        --num-select)     NUM_SELECT="$2";     shift 2 ;;
        --skip-detection) SKIP_DETECTION=true; shift ;;
        --skip-inference) SKIP_INFERENCE=true; shift ;;
        --skip-combine)   SKIP_COMBINE=true;   shift ;;
        -h|--help)        usage ;;
        -*)               echo "Unknown option: $1" >&2; exit 1 ;;
        *)                POSITIONAL+=("$1");  shift ;;
    esac
done

if [[ ${#POSITIONAL[@]} -ne 2 ]]; then
    echo "Error: expected 2 positional arguments, got ${#POSITIONAL[@]}" >&2
    usage
fi

INPUT_JSON="$(realpath "${POSITIONAL[0]}")"
OUTPUT_DIR="$(realpath -m "${POSITIONAL[1]}")"

# ── Validate ─────────────────────────────────────────────────────────
[[ -n "$GPU" ]]        || { echo "Error: --gpu is required" >&2; usage; }
[[ -f "$SIF" ]]        || { echo "Error: SIF not found: $SIF" >&2; exit 1; }
[[ -f "$INPUT_JSON" ]] || { echo "Error: input JSON not found: $INPUT_JSON" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"

# ── Extract paths from JSON that need bind-mounting ──────────────────
# Parses map_path, precomputed_msa_dir, and FILE_ ligand paths, then
# collects their parent directories as unique bind-mount targets.
BIND_PATHS=()
while IFS= read -r p; do
    [[ -n "$p" ]] && BIND_PATHS+=("$p")
done < <(python3 -c "
import json, os, sys

with open('$INPUT_JSON') as f:
    entries = json.load(f)

dirs = set()
json_dir = os.path.dirname(os.path.abspath('$INPUT_JSON'))

def resolve(p):
    return os.path.abspath(os.path.join(json_dir, p)) if not os.path.isabs(p) else os.path.abspath(p)

for entry in entries:
    # map_path
    mp = entry.get('map_path', '')
    if mp:
        dirs.add(os.path.dirname(resolve(mp)))

    # sequences: msa dirs and FILE_ ligands
    for seq in entry.get('sequences', []):
        for kind in ('proteinChain', 'rnaSequence'):
            chain = seq.get(kind, {})
            msa_dir = chain.get('msa', {}).get('precomputed_msa_dir', '')
            if msa_dir:
                dirs.add(resolve(msa_dir))
        lig = seq.get('ligand', {}).get('ligand', '')
        if lig.startswith('FILE_'):
            dirs.add(os.path.dirname(resolve(lig[5:])))

for d in sorted(dirs):
    print(d)
")

# Build --bind flags, skip paths that don't exist
BIND_FLAGS=("--bind" "$INPUT_JSON:$INPUT_JSON" "--bind" "$OUTPUT_DIR:$OUTPUT_DIR")
for bp in "${BIND_PATHS[@]}"; do
    if [[ -e "$bp" ]]; then
        BIND_FLAGS+=("--bind" "$bp:$bp")
    else
        echo "Warning: skipping bind for non-existent path: $bp" >&2
    fi
done

echo "Bind mounts: ${BIND_FLAGS[*]}"

# Helper: run a command inside the container
run_in_container() {
    local need_gpu="$1"; shift
    local cmd="cd /app/CryoZeta && source /opt/conda/etc/profile.d/conda.sh && conda activate cryozeta && $*"

    if [[ "$need_gpu" == "gpu" ]]; then
        CUDA_VISIBLE_DEVICES="$GPU" apptainer exec --nv \
            "${BIND_FLAGS[@]}" \
            "$SIF" bash -c "$cmd"
    else
        apptainer exec \
            "${BIND_FLAGS[@]}" \
            "$SIF" bash -c "$cmd"
    fi
}

# Pick checkpoint based on interpolation flag
if [[ "$USE_INTERPOLATION" == true ]]; then
    CHECKPOINT="/app/CryoZeta/assets/cryozeta-interpolate-v0.0.1.safetensors"
else
    CHECKPOINT="/app/CryoZeta/assets/cryozeta-v0.0.1.safetensors"
fi

# ── Stage 1: Atom Detection ─────────────────────────────────────────
if [[ "$SKIP_DETECTION" == false ]]; then
    echo "=== Stage 1: Atom Detection ==="
    run_in_container gpu \
        cryozeta-detection json-run "$INPUT_JSON" "$OUTPUT_DIR" --device cuda
fi

# ── Stage 2: Structure Prediction ────────────────────────────────────
if [[ "$SKIP_INFERENCE" == false ]]; then
    echo "=== Stage 2: Structure Prediction (interpolation=$USE_INTERPOLATION) ==="
    run_in_container gpu \
        "export LAYERNORM_TYPE=fast_layernorm && cryozeta-inference \
            --seeds $SEEDS \
            --load_checkpoint_path $CHECKPOINT \
            --em_file_dir $OUTPUT_DIR \
            --dump_dir $OUTPUT_DIR \
            --input_json_path $INPUT_JSON \
            --use_deepspeed_evo_attention true \
            --model.N_cycle $N_CYCLE \
            --sample_diffusion.N_sample $N_SAMPLE \
            --sample_diffusion.N_step $N_STEP \
            --data.num_dl_workers 1 \
            --use_interpolation $USE_INTERPOLATION \
            --overwrite false"
fi

# ── Stage 3: Combine Results ─────────────────────────────────────────
if [[ "$SKIP_COMBINE" == false ]]; then
    echo "=== Stage 3: Combine Results ==="
    run_in_container cpu \
        cryozeta-combine \
            --dump-dir "$OUTPUT_DIR" \
            --input-json-path "$INPUT_JSON" \
            --seeds "$SEEDS" \
            --num-select "$NUM_SELECT"
fi

echo "=== Done. Results in: $OUTPUT_DIR ==="
