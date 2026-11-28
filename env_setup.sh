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

. "$PROJECT_ROOT/scripts/detect_pixi_env.sh"

# ── Map CRYOZETA_CUDA shorthand to pixi env name ─────────────────────────────
cuda_version_to_env() {
    case "$1" in
        11)       echo "cu11"    ;;
        12)       echo "default" ;;
        13)       echo "cu13"    ;;
        cu11|cu13|default|dev|dev-cu11|dev-cu13) echo "$1" ;;
        *)
            echo "WARNING: unrecognised CRYOZETA_CUDA='$1', falling back to auto-detect" >&2
            detect_pixi_env "${CUDA_VISIBLE_DEVICES:-0}"
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
    PIXI_ENV=$(detect_pixi_env "${CUDA_VISIBLE_DEVICES:-0}") || return 1
    export PIXI_ENV
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


# Add Conda lib to LD_LIBRARY_PATH for TVM/CUDA to find libnvrtc
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

# Triton needs ptxas on PATH and Blackwell-capable ptxas for sm_100+ GPUs
if [ -x "${CONDA_PREFIX}/bin/ptxas" ]; then
    export TRITON_PTXAS_PATH="${CONDA_PREFIX}/bin/ptxas"
    # CUDA 13+ ptxas supports Blackwell; set both paths
    export TRITON_PTXAS_BLACKWELL_PATH="${CONDA_PREFIX}/bin/ptxas"
fi
