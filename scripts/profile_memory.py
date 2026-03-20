#!/usr/bin/env python
"""
Profile CUDA memory usage of CryoZeta inference using torch.profiler.

Produces:
  1. Memory timeline HTML (interactive plot of allocations over time)
  2. Memory snapshot pickle (for torch.cuda._memory_viz tools)
  3. Chrome trace JSON with memory events (viewable at https://ui.perfetto.dev)
  4. Printed tables: top memory-consuming ops, allocation summary

Usage:
  python scripts/profile_memory.py \
      --load_checkpoint_path assets/cryozeta-v0.0.1.safetensors \
      --input_json_path assets/examples/example.json \
      --em_file_dir output/example \
      --dump_dir output/memory_profile \
      --model.N_cycle 1 \
      --sample_diffusion.N_sample 1 \
      --sample_diffusion.N_step 2

The script accepts the same CLI flags as cryozeta-inference.
"""

import argparse
import copy
import gc
import os
import pickle
import sys
import traceback
from contextlib import nullcontext
from typing import Any

import torch
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
# Lightweight per-module memory tracking via record_function hooks
# ---------------------------------------------------------------------------

class _RecordFunctionHook:
    """Wraps a module's forward in a torch.profiler.record_function context.

    This lets the profiler attribute memory to module names (e.g.
    "CryoZeta.evoformer.blocks.0.tri_mul_out") without with_stack overhead.
    """

    def __init__(self, name: str):
        self.name = name
        self.rf = None

    def pre_hook(self, module, args):
        self.rf = torch.profiler.record_function(self.name)
        self.rf.__enter__()

    def post_hook(self, module, args, output):
        if self.rf is not None:
            self.rf.__exit__(None, None, None)
            self.rf = None


def attach_module_memory_hooks(model: torch.nn.Module, max_depth: int = 4):
    """Attach record_function hooks to submodules up to max_depth.

    Returns a list of hook handles (call .remove() to detach).
    Only hooks modules that have direct parameters or are at a "leaf-ish"
    level to avoid excessive nesting in the trace.
    """
    handles = []
    for name, module in model.named_modules():
        depth = name.count(".") + 1 if name else 0
        if depth > max_depth:
            continue
        # Skip the root module (empty name) — it would wrap the entire forward
        if not name:
            continue
        hook = _RecordFunctionHook(name)
        h1 = module.register_forward_pre_hook(hook.pre_hook)
        h2 = module.register_forward_hook(hook.post_hook)
        handles.extend([h1, h2])
    logger.info(f"Attached record_function hooks to {len(handles) // 2} modules (max_depth={max_depth})")
    return handles


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
    """Print current CUDA memory statistics."""
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


@torch.no_grad()
def run_memory_profile(model, data, configs, device, output_dir, record_history=False):
    """Run inference with detailed memory profiling.

    Args:
        record_history: If True, enable torch.cuda.memory._record_memory_history()
            for detailed per-allocation tracking (snapshot pickle). This is VERY slow
            (~10-50x overhead). Default False — the profiler tables, timeline HTML,
            and chrome trace are still produced without it.
    """
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
    os.makedirs(output_dir, exist_ok=True)

    # ── Warmup pass ───────────────────────────────────────────────────────
    logger.info("Running warmup pass (not profiled)...")
    warmup_data = copy.deepcopy(data)
    with enable_amp:
        _ = model(input_feature_dict=warmup_data["input_feature_dict"])
    del warmup_data
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    # Reset memory tracking for the profiled run
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.reset_accumulated_memory_stats(device)

    print_memory_stats(device, "After warmup + cache clear")

    # ── Optionally enable memory history for snapshot ───────────────────
    if record_history:
        logger.info("Enabling CUDA memory history recording (SLOW — use for detailed snapshots)...")
        torch.cuda.memory._record_memory_history(max_entries=100_000)

    # ── Attach per-module record_function hooks ─────────────────────────
    hook_handles = attach_module_memory_hooks(model, max_depth=4)

    # ── Profiled pass ─────────────────────────────────────────────────────
    logger.info("Starting memory-profiled inference pass...")

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        profile_memory=True,
        record_shapes=True,
        with_stack=record_history,  # stacks add overhead; only enable with --record_history
    ) as prof:
        with enable_amp:
            prediction, _, _ = model(
                input_feature_dict=data["input_feature_dict"],
            )

    torch.cuda.synchronize()

    # Remove hooks
    for h in hook_handles:
        h.remove()

    # ── Capture memory snapshot (only if history was recorded) ──────────
    if record_history:
        snapshot = torch.cuda.memory._snapshot()
        torch.cuda.memory._record_memory_history(enabled=None)
        snapshot_path = os.path.join(output_dir, "memory_snapshot.pickle")
        with open(snapshot_path, "wb") as f:
            pickle.dump(snapshot, f)
        logger.info(f"Memory snapshot saved to {snapshot_path}")
        logger.info("  -> Visualize at https://pytorch.org/memory_viz  (drag & drop the pickle)")

    # ── Export memory timeline HTML ───────────────────────────────────────
    try:
        mem_timeline_path = os.path.join(output_dir, "memory_timeline.html")
        prof.export_memory_timeline(mem_timeline_path, device=str(device))
        logger.info(f"Memory timeline HTML saved to {mem_timeline_path}")
    except Exception as e:
        logger.warning(f"Could not export memory timeline HTML: {e}")

    # ── Export Chrome trace ───────────────────────────────────────────────
    trace_path = os.path.join(output_dir, "memory_trace.json")
    prof.export_chrome_trace(trace_path)
    logger.info(f"Chrome trace saved to {trace_path}")
    logger.info("  -> Open at https://ui.perfetto.dev")

    # ── Export stacks (for flamegraph, requires with_stack) ─────────────
    if record_history:
        stacks_path = os.path.join(output_dir, "memory_stacks.txt")
        prof.export_stacks(stacks_path, "self_device_memory_usage")
        logger.info(f"Memory stacks saved to {stacks_path}")
        logger.info("  -> Use with flamegraph.pl to generate SVG flamegraph")

    # ── Print summary tables ──────────────────────────────────────────────
    sep = "=" * 120

    # Per-module memory summary (from record_function hooks)
    print(f"\n{sep}")
    print("PER-MODULE MEMORY USAGE (from record_function hooks)")
    print(sep)
    events = prof.key_averages()
    module_events = []
    for evt in events:
        # record_function hooks produce events with module path names (contain '.')
        if "." in evt.key and not evt.key.startswith("aten::"):
            module_events.append(evt)
    # Sort by self CUDA memory descending
    module_events.sort(key=lambda e: e.self_device_memory_usage, reverse=True)
    if module_events:
        print(f"  {'Module':<70s} {'Self CUDA Mem':>14s} {'CUDA Mem':>14s} {'CUDA Time':>14s}")
        print(f"  {'-'*70} {'-'*14} {'-'*14} {'-'*14}")
        for evt in module_events[:40]:
            self_mem = evt.self_device_memory_usage / (1024**2)
            total_mem = evt.device_memory_usage / (1024**2)
            cuda_time = evt.self_cuda_time_total / 1000  # ms
            print(f"  {evt.key:<70s} {self_mem:>11.1f} MB {total_mem:>11.1f} MB {cuda_time:>11.1f} ms")
    else:
        print("  (no module-level events found)")

    # Top ops by CUDA memory usage
    print(f"\n{sep}")
    print("TOP 30 OPERATORS BY CUDA MEMORY (self_device_memory_usage)")
    print(sep)
    print(
        prof.key_averages().table(
            sort_by="self_device_memory_usage",
            row_limit=30,
        )
    )

    # Top ops by total CUDA memory
    print(f"\n{sep}")
    print("TOP 30 OPERATORS BY TOTAL CUDA MEMORY (device_memory_usage)")
    print(sep)
    print(
        prof.key_averages().table(
            sort_by="device_memory_usage",
            row_limit=30,
        )
    )

    # Group by stack to find memory hotspots (only useful with --record_history)
    if record_history:
        print(f"\n{sep}")
        print("TOP 20 MEMORY HOTSPOTS BY STACK TRACE")
        print(sep)
        print(
            prof.key_averages(group_by_stack_n=5).table(
                sort_by="self_device_memory_usage",
                row_limit=20,
            )
        )

    # ── Detailed memory stats from CUDA allocator ─────────────────────────
    print_memory_stats(device, "After profiled inference")

    if torch.cuda.is_available():
        print(f"\n{sep}")
        print("CUDA ALLOCATOR STATISTICS")
        print(sep)
        stats = torch.cuda.memory_stats(device)
        keys_of_interest = [
            "allocation.all.current",
            "allocation.all.peak",
            "allocation.all.allocated",
            "allocation.all.freed",
            "allocation.large_pool.current",
            "allocation.large_pool.peak",
            "allocation.small_pool.current",
            "allocation.small_pool.peak",
            "allocated_bytes.all.current",
            "allocated_bytes.all.peak",
            "allocated_bytes.large_pool.current",
            "allocated_bytes.large_pool.peak",
            "reserved_bytes.all.current",
            "reserved_bytes.all.peak",
            "active_bytes.all.current",
            "active_bytes.all.peak",
            "num_alloc_retries",
            "num_ooms",
        ]
        for key in keys_of_interest:
            val = stats.get(key, "N/A")
            if isinstance(val, int) and "bytes" in key:
                print(f"  {key:50s}: {val / (1024**3):.3f} GB  ({val:,} bytes)")
            else:
                print(f"  {key:50s}: {val}")
        print(sep)

    return prediction


def main():
    # Parse --record_history before cryozeta config parsing sees it
    record_history = "--record_history" in sys.argv
    if record_history:
        sys.argv.remove("--record_history")

    configs = build_configs()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    output_dir = os.path.join(configs.dump_dir, "memory_profile")

    # Model
    model = setup_model(configs, device)
    print_memory_stats(device, "After model load")

    # Data
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

            run_memory_profile(model, data, configs, device, output_dir, record_history=record_history)
            logger.info(f"Memory profiling complete. Results in {output_dir}/")

        except Exception as e:
            logger.error(f"Error during profiling: {e}\n{traceback.format_exc()}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        break  # Only profile first sample


if __name__ == "__main__":
    main()
