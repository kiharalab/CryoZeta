#!/bin/sh
# Pixi activation script for CryoZeta.
# Sets environment variables used at runtime.
# This runs automatically on every `pixi run` / `pixi shell`.
#
# NOTE: Do NOT put heavy operations (installs, builds, downloads) here.
#       Use `pixi run setup` for one-time project initialization.
#
# ── Customising the CUDA environment ─────────────────────────────────────────
# Priority (highest → lowest):
#   1. PIXI_ENVIRONMENT_NAME  – set automatically by `pixi shell -e <env>`
#   2. CRYOZETA_CUDA           – user override, e.g. CRYOZETA_CUDA=11
#      Accepted values: 11 → cu11, 12 → default (cu12), 13 → cu13,
#                       or a pixi env name directly (cu11, cu13, default, …)
#   3. Auto-detection from GPU compute capability and driver version

PROJECT_ROOT="${PIXI_PROJECT_ROOT:-$(cd "$(dirname "$0")" && pwd)}"

# ── Auto-detect pixi environment based on GPU and driver ──────────────────────
# Two constraints determine the best CUDA environment:
#   1. Compute capability (GPU architecture support)
#   2. Driver-supported CUDA version (hard upper bound from nvidia-smi)
#
# Architecture preference:  cc >= 10 → cu13,  cc >= 8 → cu12 (default),  else → cu11
# Driver constraint:        driver must support the target CUDA major version.
#
# If running inside a pixi environment (PIXI_ENVIRONMENT_NAME is set), the
# user's explicit choice is respected and auto-detection is skipped.
detect_pixi_env() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "default"; return
    fi

    # Max CUDA version the installed driver supports (major only)
    driver_cuda=$(nvidia-smi 2>/dev/null \
        | sed -n 's/.*CUDA Version: *\([0-9]*\).*/\1/p')

    # Highest compute capability among all GPUs
    compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
        | sort -t. -k1,1nr -k2,2nr | head -1 | tr -d '[:space:]')

    if [ -z "$driver_cuda" ] || [ -z "$compute_cap" ]; then
        echo "default"; return
    fi

    major="${compute_cap%%.*}"

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

# ── Map CRYOZETA_CUDA shorthand to pixi env name ─────────────────────────────
cuda_version_to_env() {
    case "$1" in
        11)       echo "cu11"    ;;
        12)       echo "default" ;;
        13)       echo "cu13"    ;;
        cu11|cu13|default|dev|dev-cu11|dev-cu13) echo "$1" ;;
        *)
            echo "WARNING: unrecognised CRYOZETA_CUDA='$1', falling back to auto-detect" >&2
            detect_pixi_env
            ;;
    esac
}

# Determine PIXI_ENV with the documented priority order.
if [ -n "$PIXI_ENVIRONMENT_NAME" ]; then
    # 1. Pixi told us (e.g. `pixi shell -e cu11`)
    export PIXI_ENV="$PIXI_ENVIRONMENT_NAME"
elif [ -n "$CRYOZETA_CUDA" ]; then
    # 2. User override via CRYOZETA_CUDA
    export PIXI_ENV="$(cuda_version_to_env "$CRYOZETA_CUDA")"
else
    # 3. Auto-detect from hardware
    export PIXI_ENV="$(detect_pixi_env)"
fi

# ── TEASER-plusplus ───────────────────────────────────────────────────────────
export TEASERPP_DIR="${PROJECT_ROOT}/externals/TEASER-plusplus"

# Make teaserpp_python importable from the cmake build tree.
# The build produces the .so in build/python/teaserpp_python/ but does not
# copy __init__.py from the source tree, so we do it here (idempotent).
_TEASER_BUILD_PY="${TEASERPP_DIR}/build/python"
if [ -d "${_TEASER_BUILD_PY}/teaserpp_python" ]; then
    if [ ! -f "${_TEASER_BUILD_PY}/teaserpp_python/__init__.py" ] && \
       [ -f "${TEASERPP_DIR}/python/teaserpp_python/__init__.py" ]; then
        cp "${TEASERPP_DIR}/python/teaserpp_python/__init__.py" \
           "${_TEASER_BUILD_PY}/teaserpp_python/__init__.py" 2>/dev/null || true
    fi
    export PYTHONPATH="${_TEASER_BUILD_PY}${PYTHONPATH:+:$PYTHONPATH}"
fi
unset _TEASER_BUILD_PY

# CUTLASS headers (provided by the pixi conda package)
export CUTLASS_PATH="${CONDA_PREFIX}"
