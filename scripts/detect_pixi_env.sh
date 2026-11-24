# Select a locked toolkit for the requested nvidia-smi indices or UUIDs.
detect_pixi_env() (
    command -v nvidia-smi >/dev/null 2>&1 || { echo default; return; }
    driver=$(nvidia-smi 2>/dev/null | sed -nE 's/.*CUDA (UMD )?Version: *([0-9]+)\.([0-9]+).*/\2 \3/p' | head -1)
    [ -n "$driver" ] || { echo default; return; }
    caps=$(nvidia-smi -i "$1" --query-gpu=compute_cap --format=csv,noheader) || return 1
    set -- $driver
    version=$(($1 * 100 + $2))
    min=$(printf '%s\n' "$caps" | sort -n | head -1 | cut -d. -f1)
    max=$(printf '%s\n' "$caps" | sort -n | tail -1 | cut -d. -f1)
    if [ "$min" -ge 10 ] && [ "$version" -ge 1302 ]; then
        echo cu13
    elif [ "$min" -ge 8 ] && [ "$version" -ge 1208 ]; then
        echo default
    elif [ "$max" -lt 10 ] && [ "$version" -ge 1108 ]; then
        echo cu11
    else
        echo "No compatible locked CUDA toolkit for compute capability $caps and driver CUDA $1.$2" >&2
        return 1
    fi
)
