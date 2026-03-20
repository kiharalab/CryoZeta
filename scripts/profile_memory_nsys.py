#!/usr/bin/env python
"""
Profile CryoZeta inference memory with NVIDIA Nsight Systems (nsys).

This script is designed to be launched via `nsys profile`. It uses NVTX
annotations to label per-module regions and emits CUDA memory markers so
that nsys captures a full GPU memory timeline alongside kernel activity.

The resulting .nsys-rep / .sqlite can be opened in Nsight Systems GUI for
interactive memory + kernel + NVTX analysis.

Usage (wrapped by scripts/run_memory_profile_nsys.sh):
  nsys profile \
      --trace=cuda,nvtx,osrt \
      --gpu-metrics-device=all \
      --cuda-memory-usage=true \
      --force-overwrite=true \
      -o output/nsys_profile/report \
      python scripts/profile_memory_nsys.py \
          --load_checkpoint_path assets/cryozeta-v0.0.1.safetensors \
          --input_json_path assets/examples/example.json \
          --em_file_dir output/example \
          --dump_dir output/nsys_profile \
          ...
"""

import copy
import gc
import os
import sys
import traceback
from contextlib import nullcontext
from typing import Any

import torch
import torch.cuda.nvtx as nvtx
from loguru import logger

from cryozeta.configs import parse_configs, parse_sys_args
from cryozeta.configs.configs_base import configs as configs_base
from cryozeta.configs.configs_data import data_configs
from cryozeta.configs.configs_inference import inference_configs
from cryozeta.data.infer_data_pipeline import get_inference_dataloader
from cryozeta.model.cryozeta import CryoZeta
from cryozeta.utils.seed import seed_everything
from cryozeta.utils.torch_utils import to_device

from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# NVTX per-module hooks — shows up as named ranges in Nsight Systems
# ---------------------------------------------------------------------------

class _NvtxHook:
    """Push/pop an NVTX range around a module's forward pass."""

    def __init__(self, name: str):
        self.name = name

    def pre_hook(self, module, args):
        nvtx.range_push(self.name)

    def post_hook(self, module, args, output):
        nvtx.range_pop()


def attach_nvtx_hooks(model: torch.nn.Module, max_depth: int = 4):
    """Attach NVTX push/pop hooks to submodules up to max_depth.

    Returns list of hook handles for cleanup.
    """
    handles = []
    for name, module in model.named_modules():
        depth = name.count(".") + 1 if name else 0
        if depth > max_depth or not name:
            continue
        hook = _NvtxHook(name)
        h1 = module.register_forward_pre_hook(hook.pre_hook)
        h2 = module.register_forward_hook(hook.post_hook)
        handles.extend([h1, h2])
    logger.info(f"Attached NVTX hooks to {len(handles) // 2} modules (max_depth={max_depth})")
    return handles


# ---------------------------------------------------------------------------
# Config / model setup (same as profile_memory.py)
# ---------------------------------------------------------------------------

def build_configs():
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(
        configs=configs,
        arg_str=parse_sys_args(),
        fill_required_with_null=True,
    )
    configs_base["use_interpolation"] = configs.use_interpolation
    if configs.use_interpolation:
        configs.model.input_embedder.p_dim = 114
    else:
        configs.model.input_embedder.p_dim = 99
    return configs


def setup_model(configs, device):
    model = CryoZeta(configs).to(device)
    checkpoint_path = configs.load_checkpoint_path
    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    assert checkpoint_path.endswith(".safetensors"), "Checkpoint must be a .safetensors file"
    state_dict = load_file(checkpoint_path, device=str(device))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    logger.info("Model loaded and set to eval mode.")
    return model


def print_memory_stats(device, label=""):
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
    print(f"\n{'=' * 80}")
    print(f"CUDA MEMORY — {label}")
    print(f"{'=' * 80}")
    print(f"  Allocated     : {allocated:.3f} GB")
    print(f"  Reserved      : {reserved:.3f} GB")
    print(f"  Peak allocated: {peak_allocated:.3f} GB")
    print(f"  Peak reserved : {peak_reserved:.3f} GB")
    print(f"{'=' * 80}\n")


# ---------------------------------------------------------------------------
# Main profiled inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_nsys_inference(model, data, configs, device):
    eval_precision = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[configs.dtype]

    enable_amp = (
        torch.autocast(device_type="cuda", dtype=eval_precision)
        if torch.cuda.is_available()
        else nullcontext()
    )

    data = to_device(data, device)

    # ── Warmup (outside nsys capture if using --capture-range) ────────────
    logger.info("Running warmup pass...")
    nvtx.range_push("warmup")
    warmup_data = copy.deepcopy(data)
    with enable_amp:
        _ = model(input_feature_dict=warmup_data["input_feature_dict"])
    del warmup_data
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    nvtx.range_pop()  # warmup

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.reset_accumulated_memory_stats(device)
    print_memory_stats(device, "After warmup + cache clear")

    # ── Attach NVTX hooks ─────────────────────────────────────────────────
    hook_handles = attach_nvtx_hooks(model, max_depth=4)

    # ── Profiled inference pass ───────────────────────────────────────────
    logger.info("Starting nsys-profiled inference pass...")

    # Signal to nsys --capture-range=cudaProfilerApi
    torch.cuda.cudart().cudaProfilerStart()
    nvtx.range_push("inference")

    with enable_amp:
        prediction, _, _ = model(
            input_feature_dict=data["input_feature_dict"],
        )

    torch.cuda.synchronize()
    nvtx.range_pop()  # inference
    torch.cuda.cudart().cudaProfilerStop()

    # Cleanup hooks
    for h in hook_handles:
        h.remove()

    print_memory_stats(device, "After profiled inference")

    return prediction


def main():
    configs = build_configs()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    model = setup_model(configs, device)
    print_memory_stats(device, "After model load")

    seed = configs.seeds[0] if configs.seeds else 101
    seed_everything(seed=seed, deterministic=True)
    dataloader = get_inference_dataloader(configs=configs)

    for batch in dataloader:
        try:
            data, atom_array, data_error_message = batch[0]
            if len(data_error_message) > 0:
                logger.warning(f"Data error: {data_error_message}")
                continue

            data["input_feature_dict"]["atom_array"] = atom_array
            data["input_feature_dict"]["entity_poly_type"] = data["entity_poly_type"]

            sample_name = data["sample_name"]
            logger.info(
                f"Profiling sample: {sample_name}  "
                f"N_token={data['N_token'].item()}, "
                f"N_atom={data['N_atom'].item()}, "
                f"N_msa={data['N_msa'].item()}"
            )

            entry_stage_dir = os.path.join(
                configs.dump_dir, sample_name, "CryoZeta",
            )
            os.makedirs(entry_stage_dir, exist_ok=True)
            data["input_feature_dict"]["dump_dir"] = entry_stage_dir

            run_nsys_inference(model, data, configs, device)
            logger.info("nsys profiling pass complete.")

        except Exception as e:
            logger.error(f"Error during profiling: {e}\n{traceback.format_exc()}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        break  # Only profile first sample


if __name__ == "__main__":
    main()
