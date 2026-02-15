#!/bin/sh
# Pixi activation script for CryoZeta.
# Sets environment variables used at runtime.
# This runs automatically on every `pixi run` / `pixi shell`.
#
# NOTE: Do NOT put heavy operations (installs, builds, downloads) here.
#       Use `pixi run setup` for one-time project initialization.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Auto-detect pixi environment based on GPU compute capability ──────────────
# Matches the logic in inference_demo.sh: compute_cap major >= 10 → cu13, else default (cu12).
detect_pixi_env() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "default"; return
    fi
    compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
        | sort -t. -k1,1nr -k2,2nr | head -1 | tr -d '[:space:]')
    if [ -z "$compute_cap" ]; then
        echo "default"; return
    fi
    major="${compute_cap%%.*}"
    if [ "$major" -ge 10 ] 2>/dev/null; then
        echo "cu13"
    else
        echo "default"
    fi
}

export PIXI_ENV=$(detect_pixi_env)

# TEASER-plusplus location
export TEASERPP_DIR="${SCRIPT_DIR}/externals/TEASER-plusplus"

# CUTLASS headers (provided by the pixi conda package)
export CUTLASS_PATH="${CONDA_PREFIX}"
