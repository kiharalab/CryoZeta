#!/bin/bash
# Initialize script for CryoZeta dependencies
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
EXTERNALS_DIR="${SCRIPT_DIR}/externals"
TEASERPP_DIR="${EXTERNALS_DIR}/TEASER-plusplus"
DEPS_DIR="${SCRIPT_DIR}/.deps"

# ─── Helper functions ────────────────────────────────────────────────────────

# Find Eigen3 CMake config directory from system/conda installations
find_eigen3_dir() {
    # 1. Check common system locations directly (fast path)
    for dir in \
        "/usr/share/eigen3/cmake" \
        "/usr/lib/cmake/eigen3" \
        "/usr/local/share/eigen3/cmake" \
        "/usr/local/lib/cmake/eigen3"; do
        if [ -f "$dir/Eigen3Config.cmake" ] || [ -f "$dir/eigen3-config.cmake" ]; then
            echo "$dir"
            return 0
        fi
    done

    # 2. Broader search under /usr and /usr/local
    local config_file
    config_file=$(find /usr /usr/local \( -name "Eigen3Config.cmake" -o -name "eigen3-config.cmake" \) 2>/dev/null | head -1)
    if [ -n "$config_file" ]; then
        echo "$(dirname "$config_file")"
        return 0
    fi

    # 3. Check conda if available
    if command -v conda &>/dev/null; then
        local conda_base
        conda_base=$(conda info --base 2>/dev/null)
        for dir in "${conda_base}/share/eigen3/cmake" "${conda_base}/lib/cmake/eigen3"; do
            if [ -f "$dir/Eigen3Config.cmake" ] || [ -f "$dir/eigen3-config.cmake" ]; then
                echo "$dir"
                return 0
            fi
        done
    fi

    # 4. Check local build from a previous run of this script
    local local_cmake="${DEPS_DIR}/eigen-install/share/eigen3/cmake"
    if [ -f "${local_cmake}/Eigen3Config.cmake" ]; then
        echo "$local_cmake"
        return 0
    fi

    return 1
}

# Find Boost CMake config directory from system/conda installations
find_boost_dir() {
    # 1. Check common system locations
    for base in "/usr/lib/cmake" "/usr/local/lib/cmake"; do
        if [ -d "$base" ]; then
            local found
            found=$(find "$base" -maxdepth 1 -name "Boost-*" -type d 2>/dev/null | head -1)
            if [ -n "$found" ]; then
                echo "$found"
                return 0
            fi
        fi
    done

    # 2. Check conda if available
    if command -v conda &>/dev/null; then
        local conda_base
        conda_base=$(conda info --base 2>/dev/null)
        if [ -d "${conda_base}/lib/cmake" ]; then
            local found
            found=$(find "${conda_base}/lib/cmake" -maxdepth 1 -name "Boost-*" -type d 2>/dev/null | head -1)
            if [ -n "$found" ]; then
                echo "$found"
                return 0
            fi
        fi
    fi

    return 1
}

# Find pybind11 CMake config directory from the venv
find_pybind11_dir() {
    # 1. Search venv for CMake config files
    local config_file
    config_file=$(find "${VENV_DIR}" -name "pybind11Config.cmake" -o -name "pybind11-config.cmake" 2>/dev/null | head -1)
    if [ -n "$config_file" ]; then
        echo "$(dirname "$config_file")"
        return 0
    fi

    # 2. Ask Python directly
    local py_path
    py_path=$(python3 -c "import pybind11, os; print(os.path.join(os.path.dirname(pybind11.__file__), 'share', 'cmake', 'pybind11'))" 2>/dev/null)
    if [ -n "$py_path" ] && [ -d "$py_path" ]; then
        echo "$py_path"
        return 0
    fi

    return 1
}

# ─── 1. Python environment ──────────────────────────────────────────────────

echo "==> Setting up Python environment..."
uv sync
source "${VENV_DIR}/bin/activate"

# Override any stale paths injected by the venv activate script
TEASERPP_DIR="${EXTERNALS_DIR}/TEASER-plusplus"

# ─── 2. CUTLASS (headers only, for DeepSpeed DS4Sci_EvoformerAttention) ──────

export CUTLASS_PATH="${EXTERNALS_DIR}/cutlass"

if [ ! -d "${EXTERNALS_DIR}/cutlass" ]; then
    echo "==> Cloning CUTLASS (headers only via sparse checkout)..."
    mkdir -p "${EXTERNALS_DIR}"
    git clone --depth 1 --filter=blob:none --sparse \
        -b v3.5.1 https://github.com/NVIDIA/cutlass.git "${EXTERNALS_DIR}/cutlass"
    cd "${EXTERNALS_DIR}/cutlass"
    git sparse-checkout set include
    cd "${SCRIPT_DIR}"
else
    echo "==> CUTLASS directory already exists, skipping."
fi

# ─── 3. Download CryoZeta model weights ─────────────────────────────────────

echo "==> Downloading CryoZeta model weights..."
hf download "KiharaLab/CryoZeta" \
    --repo-type model \
    --local-dir "assets"

# ─── 4. TEASER-plusplus ──────────────────────────────────────────────────────

if [ ! -d "${TEASERPP_DIR}" ]; then
    echo "==> Cloning TEASER-plusplus..."
    mkdir -p "${EXTERNALS_DIR}"
    git clone --depth 1 https://github.com/MIT-SPARK/TEASER-plusplus.git "${TEASERPP_DIR}"
else
    echo "==> TEASER-plusplus directory already exists, skipping clone."
fi

# Build only if the shared library doesn't already exist
if [ ! -f "${TEASERPP_DIR}/build/libteaser.so" ] && \
   [ ! -f "${TEASERPP_DIR}/build/libteaser.dylib" ] && \
   [ ! -f "${TEASERPP_DIR}/build/teaser.lib" ]; then

    echo "==> Building TEASER-plusplus..."

    # ── Locate dependencies ──

    echo "Looking for Eigen3 and Boost..."
    EIGEN3_DIR=$(find_eigen3_dir) || true
    BOOST_DIR=$(find_boost_dir) || true

    # Build Eigen3 from source if not found anywhere
    if [ -z "$EIGEN3_DIR" ]; then
        EIGEN3_SRC="${DEPS_DIR}/eigen"
        EIGEN3_BUILD="${DEPS_DIR}/eigen-build"
        EIGEN3_INSTALL="${DEPS_DIR}/eigen-install"

        if [ ! -f "${EIGEN3_INSTALL}/share/eigen3/cmake/Eigen3Config.cmake" ]; then
            echo "Building Eigen3 from source..."
            mkdir -p "${DEPS_DIR}"
            if [ ! -d "${EIGEN3_SRC}" ]; then
                git clone --branch 3.4 --depth 1 \
                    https://gitlab.com/libeigen/eigen.git "${EIGEN3_SRC}" || {
                    echo "git clone failed, trying wget fallback..."
                    wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz \
                        -O "${DEPS_DIR}/eigen.tar.gz"
                    tar -xzf "${DEPS_DIR}/eigen.tar.gz" -C "${DEPS_DIR}"
                    mv "${DEPS_DIR}/eigen-3.4.0" "${EIGEN3_SRC}" || {
                        echo "Error: Failed to download Eigen3. Please install manually:"
                        echo "  Ubuntu/Debian: sudo apt-get install libeigen3-dev"
                        echo "  RHEL/CentOS:   sudo yum install eigen3-devel"
                        echo "  Conda:         conda install -c conda-forge eigen"
                        exit 1
                    }
                }
            fi

            mkdir -p "${EIGEN3_BUILD}"
            cmake -S "${EIGEN3_SRC}" -B "${EIGEN3_BUILD}" \
                -DCMAKE_INSTALL_PREFIX="${EIGEN3_INSTALL}" -DCMAKE_BUILD_TYPE=Release
            cmake --build "${EIGEN3_BUILD}" --target install -j"$(nproc 2>/dev/null || echo 4)"
        fi

        EIGEN3_DIR="${EIGEN3_INSTALL}/share/eigen3/cmake"
        if [ -f "${EIGEN3_DIR}/Eigen3Config.cmake" ]; then
            echo "Eigen3 built and installed to: ${EIGEN3_INSTALL}"
        else
            echo "Error: Eigen3 build failed."
            exit 1
        fi
    fi

    if [ -z "$BOOST_DIR" ]; then
        echo "Warning: Boost not found. TEASER-plusplus may still work without it."
        echo "  To install: sudo apt-get install libboost-all-dev  (or conda install -c conda-forge boost)"
    fi

    # ── Install pybind11 if needed ──

    echo "Ensuring pybind11 is installed..."
    uv pip list 2>/dev/null | grep -q pybind11 || uv pip install pybind11

    # ── CMake configure & build ──

    echo "Configuring TEASER-plusplus with CMake..."
    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON"

    # Eigen3 (required)
    echo "  Eigen3_DIR: $EIGEN3_DIR"
    CMAKE_ARGS="$CMAKE_ARGS -DEigen3_DIR=$EIGEN3_DIR"
    export Eigen3_DIR="$EIGEN3_DIR"

    # Boost (optional)
    if [ -n "$BOOST_DIR" ]; then
        echo "  Boost_DIR:  $BOOST_DIR"
        CMAKE_ARGS="$CMAKE_ARGS -DBoost_DIR=$BOOST_DIR"
        export Boost_DIR="$BOOST_DIR"
    fi

    # pybind11
    PYBIND11_DIR=$(find_pybind11_dir) || true
    if [ -n "$PYBIND11_DIR" ]; then
        echo "  pybind11_DIR: $PYBIND11_DIR"
        CMAKE_ARGS="$CMAKE_ARGS -Dpybind11_DIR=$PYBIND11_DIR"
        export pybind11_DIR="$PYBIND11_DIR"
    fi

    mkdir -p "${TEASERPP_DIR}/build"
    cmake -S "${TEASERPP_DIR}" -B "${TEASERPP_DIR}/build" $CMAKE_ARGS
    cmake --build "${TEASERPP_DIR}/build" -j"$(nproc 2>/dev/null || echo 4)"

    # ── Install Python bindings ──

    echo "Installing TEASER-plusplus Python bindings..."
    cd "${TEASERPP_DIR}"
    uv pip install -e . || {
        echo "Warning: uv pip install failed, trying pip directly..."
        pip install -e . || {
            echo "Error: Failed to install Python bindings."
            echo "  Try manually: cd ${TEASERPP_DIR} && pip install -e ."
        }
    }
    cd "${SCRIPT_DIR}"

    echo ""
    echo "TEASER-plusplus build complete!"
    echo ""
else
    echo "==> TEASER-plusplus already built, skipping."
fi

# ─── 5. Export environment variables for current session ─────────────────────

# Re-detect if variables weren't set (e.g., build was skipped on re-run)
[ -z "$EIGEN3_DIR" ] && EIGEN3_DIR=$(find_eigen3_dir) || true
[ -z "$BOOST_DIR" ]  && BOOST_DIR=$(find_boost_dir)   || true

export TEASERPP_DIR="${TEASERPP_DIR}"
[ -n "$EIGEN3_DIR" ] && export Eigen3_DIR="$EIGEN3_DIR"
[ -n "$BOOST_DIR" ]  && export Boost_DIR="$BOOST_DIR"

echo ""
echo "==> Initialization complete!"
echo "    CUTLASS_PATH:  ${CUTLASS_PATH}"
echo "    TEASERPP_DIR:  ${TEASERPP_DIR}"
[ -n "$EIGEN3_DIR" ] && echo "    Eigen3_DIR:    ${EIGEN3_DIR}"
[ -n "$BOOST_DIR" ]  && echo "    Boost_DIR:     ${BOOST_DIR}"
echo ""
