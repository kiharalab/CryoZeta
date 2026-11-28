# Kernel bench (long-token inference plan)

This directory holds the harnesses, routing controls and candidate adapters for the approved plan
"Kernel plan for long token inference", audited against baseline `c544c6bbd0366951d8b4e0bc4c0da547836f7b8f`.
Status of each phase: [LEDGER.md](LEDGER.md). Results: [RESULTS.md](RESULTS.md).

| Path | What |
|---|---|
| `src/cryozeta/model/kernel_routing.py` | Seam for op/call-site routing: variant loading, strict/observe mode, counters, M-CLEAR policy, stage timer hooks. It is inert unless a `CRYOZETA_KERNEL_*` variable is set. |
| `src/cryozeta/model/kernel_providers.py` | FPF K2B / TriMul-v4 adapters and the C-CAT derived-weight cache (non-persistent buffers) |
| `src/cryozeta/third_party/fpf/` | Vendored FlashPairformer subset ([VENDORED.md](../../src/cryozeta/third_party/fpf/VENDORED.md)) |
| `src/cryozeta/kernel_bench/` | `op_bench`, `capture_activations`, `run_trial` (+ `mode_b`), `synth_inputs`, `compare_outputs`, `provenance`, `v0_regression` |
| `docs/kernel-bench/variants/*.yaml` | V0, kernel candidates (K-*), caching (C-CAT), memory policies (M-*), existing-flag controls (L2-EX-*) |
| `docs/kernel-bench/tolerances.template.yaml` | Unsealed gate template (P0-5 fills and seals it) |
| `tests/kernel_bench/` | CPU checks: harness CLI and classification, P-LC stage args and preflight, routing/strict mode, M-CLEAR per stack, C-CAT through `TriMul.forward`, adapter contracts, `v0_regression` comparator |

## Recommended configuration

For long inputs, use `M-TCHUNK-mem-ES-TMV4A-OFF-NBM-X34` (X1–X4, LEDGER K24 and K28). Without it, routing is inert and CryoZeta runs as before.

What it changes:
- It runs FlashPairformer `fpf_trimul_v4` at the msa, empairformer and confidence trimul sites for stages with at least 101 tokens. Shorter stages, including single chains in cycle prediction, use the configured baseline TriMul. The counters record these calls as `below_min_tokens`.
- It adds memory-only policies.
- On the 24 GB card it moves the 9b0l synthetic limit from 1280 to 2304 tokens (RESULTS, "Capacity track").
- End-to-end speed is unchanged within ±5 %.

The variant's `env` block is applied only by `run_trial`. For a plain CLI run, export those variables yourself:

```bash
export CRYOZETA_KERNEL_VARIANT=$PWD/docs/kernel-bench/variants/M-TCHUNK-mem-ES-TMV4A-OFF-NBM-X34.yaml
export CRYOZETA_TRANSITION_CHUNK=memory PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CRYOZETA_CONF_OFFLOAD_MIN_N=0 \
       CRYOZETA_DROP_BOND_MASK=1 CRYOZETA_FREE_EMBED_INPUTS=1 CRYOZETA_HEADS_LATE=1
```

Values versus V0:
- X1, X3 and X4 leave values unchanged.
- X2 changes them at rounding level. It passes the sealed 9b0l gates, but on 10DN and 36PP it was not shown equivalent (LEDGER K27); the user accepted it anyway (K28).
- To keep V0's kernels and use only the memory policies, export the second line and leave out `CRYOZETA_KERNEL_VARIANT`. The model reads these variables directly.

## Setup

Use CryoZeta's own Pixi environment and lockfile. Do not upgrade anything for a kernel A/B. Run every command below from the repository root, inside the environment:

```bash
pixi run -e default --frozen build-teaser        # once; the model import needs teaserpp_python
KB=/path/to/kernel-bench       # assets/, entries/, det/, trials/, provenance/ (outside git)
P="pixi run -e default --frozen"
export PYTHONNOUSERSITE=1                        # see below
```

`PYTHONNOUSERSITE=1` is required. Python's user site is on, so `~/.local/lib/python3.12/site-packages` otherwise
shadows the frozen environment: on this account it imports numpy 2.3.3 instead of the lock's 1.26.4, plus pillow,
protobuf and typing_extensions. `run_trial` and `v0_regression` set it for their children, and `v0_regression`
fails an arm whose numpy does not come from the frozen prefix. The parent process still needs it set by hand.

Variant variables read at import:
- `CRYOZETA_KERNEL_VARIANT` (YAML)
- `CRYOZETA_KERNEL_COUNTERS` (JSON written at exit)
- `CRYOZETA_SYNC_STAGE_TIMER`, `CRYOZETA_STAGE_PEAKS`, `CRYOZETA_TRUNK_DUMP` (instrumentation, diagnosis only)

`run_trial` sets all of them. Use it rather than setting them by hand.

## Checks (CPU, no GPU needed)

```bash
CUDA_VISIBLE_DEVICES= $P python -I -m unittest discover -s tests/kernel_bench -p test_harness.py -v   # harness
CUDA_VISIBLE_DEVICES= CRYOZETA_ASSETS_DIR=$KB/assets $P python -m unittest discover -s tests/kernel_bench -p "test_*.py" -v  # all
```

Without `CRYOZETA_ASSETS_DIR` (the CCD assets), `test_points.RealFeaturizer` is skipped.

`test_harness` itself imports no torch, but its `run_trial` dry-runs run the preflight, which imports the model in a
child process (TEASER++ comes from the pixi activation, so use `$P`, not the bare interpreter; about 1 min each).

### V0 numerical regression

`v0_regression` runs this working tree's inference against a baseline revision (default `c544c6b…`, extracted
with `git archive`; the working tree is never touched) with identical weights, inputs, settings, seeds and
environment, and compares every dumped file:

```bash
CUDA_VISIBLE_DEVICES= $P python -m cryozeta.kernel_bench.v0_regression --out $KB/trials/v0-regression/<new> \
    --route -- --seeds 101 --load_checkpoint_path $KB/assets/cryozeta-v0.0.1.safetensors \
    --em_file_dir <det tree> --input_json_path <entry.json> <flags: the same for every arm>
```

On a GPU, use the V0 flags and the environment `run_trial` sets for V0, with task-scoped compile caches. From
the repo root, with one idle, authorized GPU (selected by UUID) and a new `G`. This is the command run on hostA
(run 5, [RESULTS.md](RESULTS.md)):

```bash
G=$KB/trials/v0-regression/gpu-$(hostname)-1; X=$PWD/.pixi/envs/default; U=<GPU UUID>
test ! -e $G && $P env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$U CUDA_DEVICE_ORDER=PCI_BUS_ID LAYERNORM_TYPE=fast_layernorm USE_OPM_CHUNKED=1 \
    CUDA_HOME=$X TRITON_PTXAS_PATH=$X/bin/ptxas TRITON_PTXAS_BLACKWELL_PATH=$X/bin/ptxas \
    TRITON_CACHE_DIR=$G.cache/triton TORCH_EXTENSIONS_DIR=$G.cache/torch_extensions TILELANG_CACHE_DIR=$G.cache/tilelang \
    CUEQ_TRITON_CACHE_DIR=$G.cache/cueq_triton CUDA_CACHE_PATH=$G.cache/cuda \
  python -m cryozeta.kernel_bench.v0_regression --out $G --route -- --seeds 101 \
    --load_checkpoint_path $KB/assets/cryozeta-v0.0.1.safetensors --em_file_dir $KB/det/v1 \
    --input_json_path $KB/entries/9b0l.json --use_deepspeed_evo_attention true \
    --use_cuequivariance_attention true --use_cuequivariance_multiplicative_update true \
    --use_cuequivariance_attention_pair_bias true --use_opm_tilelang false --model.N_cycle 2 \
    --sample_diffusion.N_sample 2 --sample_diffusion.N_step 5 --data.num_dl_workers 1 --use_interpolation false
```

The report must show `cpu_shim: false`. The input is the real 9b0l entry (805 tokens) with the frozen detection
`det/v1` (620 points), at reduced sampler settings (smoke, not a production setting). The 96-token `s96_real`
fixture is CPU-only. On CUDA, its 64 synthetic points recall nothing under SVD or TEASER. c544c6b then always runs
the VESPER rotation search (`cryozeta.py`, about 184 k rotations at about 29 it/s, roughly 1.75 h per sample), and
no flag turns that off (hostA run 2).

On CUDA the bitwise gate cannot pass, whatever the patch does. Custom kernels (DS4Sci, cuEq Triton,
fast_layernorm) are nondeterministic run to run even with the baseline's `seed_everything(deterministic=True)`.
In hostA run 3, `patched` vs `repeat` differed by about as much as `baseline` vs `patched` (RESULTS.md). A
CUDA V0 check needs either a deterministic kernel mode or sealed tolerances calibrated from `repeat`.

- Arms `baseline`, `patched` (routing inert), `v0trial` (V0.yaml + counters + sync stage timer, as `run_trial` sets
  them) and `repeat` (run-to-run noise floor). Each arm must reproduce every baseline file bitwise. For older dumps, the harness also reconstructs the new seed-local `scores.csv` from the baseline scores and coordinates and checks every row. The report distinguishes literal tree equality (`bitwise`) from this compatibility check (`outputs_match`).
- `--route` adds one strict arm per op the V0 counters saw (OPM einsum/chunked, row/column LMA/torch, TriMul
  C-CAT), replacing it at every site reached. Its counters must show the provider served at every replaced site.
  APB has no servable alternative in this model: the c544c6b gate `c_hidden % 32` rejects cueq at every site
  (pairformer 384/16 = 24, diffusion 768/16 = 48) and atom attention is local. A strict cueq request raises
  `KernelFallback`, so the APB arm is identity routing (strict torch at every site). An arm that keeps V0's served
  provider everywhere, or swaps cueq TriMul for C-CAT, must also be bitwise equal; the others report max |diff|.
  A site with more than one V0 served backend is never expected bitwise. `--route` fails (`route_missing`) unless
  the V0 counters reached all four ops.
- An arm passes only if it exits 0, imported `cryozeta` from its own tree and dumped a `.cif`. Inference catches a
  failed sample, writes `ERR/<name>.txt` and still exits 0; the arm's `why` quotes that file's last line.
- `.npz` files count as identical when every member's bytes are (zip headers carry timestamps). Object members
  (atom names) are compared as raw bytes, never unpickled.
- With `--use_cuequivariance_attention true` the baseline itself cannot run on CPU: `Attention` falls back from
  cueq to DS4Sci whenever `cueq_would_fall_back` holds (N_token at or below its threshold, or an unsupported
  hidden dim/dtype), and DS4Sci is CUDA-only. CPU runs keep cueq/DS4Sci attention off.
- Without CUDA, the child redirects `Module.to("cuda")` to CPU, identically in every arm. The baseline's OOM-recovery
  branch (`cryozeta.py:467-476`) otherwise moves a CPU-resident trunk to CUDA. The report records `cpu_shim`.
- On GPU, a kernel that autotunes by timing without persisting its choice can make later arms differ from the
  baseline. That is a false failure, never a false pass: if `repeat` also differs, the run is nondeterministic and
  says nothing about the patch.

## Phase 0

```bash
# P0-1 provenance: git diff vs the baseline, lock hashes vs the plan, pixi list, nvidia-smi, asset hashes
$P python -m cryozeta.kernel_bench.provenance snapshot --repo . --assets $KB/assets \
    --assets-sha $KB/provenance/assets.sha256 --out $KB/provenance/$(hostname)-$(date +%Y%m%d-%H%M)
# P0-2 detection once per entry (GPU), then freeze (manifest + read-only, records n_detected and N_point)
$P cryozeta-detection json-run $KB/entries/9b0l.json $KB/det/v1 --device cuda      # repeat per entry
$P python -m cryozeta.kernel_bench.provenance freeze-det --det $KB/det/v1 --entries $KB/entries
# P0-3/P0-6 one baseline trial (observe mode, counters on). Mode A = real CLI; Mode B = warmed in-process repeat
$P python -m cryozeta.kernel_bench.run_trial --protocol P-CZ --mode A --entry $KB/entries/9b0l.json \
    --frozen-det $KB/det/v1 --assets $KB/assets --variant docs/kernel-bench/variants/V0.yaml \
    --gpu 0 --timeout 7200 --cache cold --out $KB/trials/<host>/P-CZ/9b0l/V0/r1
```

Add `--dry-run` to any `run_trial` call to print the resolved steps, environment and effective timeout without running anything.

Things `run_trial` does:
- It refuses to start (exit 3) if the GPU is not idle.
- It refuses (exit 2) a non-P-LC trial whose `--frozen-det` has no `freeze-det` manifest giving N_token and N_point for every entry: the requested cell must be checkable (amendment 1). It also refuses a tree whose hash differs from the manifest's `tree_sha256`, and an entry that sets `em_file` (that would bypass `--em_file_dir`).
- It records the model input of every forward (`inputs`: N_token, P, valid points, `p`/`pz` shapes) and checks each run against the manifest (`cell`). A mismatch turns any outcome into INVALID_CELL; an unverified run blocks SUCCESS.
- Each inference process or Mode B iteration must record the requested provider at every declared eligible site. Missing evidence yields UNSUPPORTED and zero coverage. Below a declared token floor, the counters must show the baseline at that site.
- `outcome` is the runtime class only. Accuracy gates are never evaluated here: use `compare_outputs` with sealed tolerances.
- A trial is labelled `synthetic` when `--synthetic` is given, the entry name matches `*_synN_real|single`, or a `synthetic.json` sits in the parent of the entries directory (the `synth_inputs` layout).
- A variant cannot set harness-controlled flags (seeds, paths, N_cycle, N_sample, N_step, workers, `p_dim`, which `build_configs` overwrites) or env (GPU selection, compile caches, and the instrumentation variables `CRYOZETA_KERNEL_COUNTERS`, `CRYOZETA_KERNEL_VARIANT`, `CRYOZETA_SYNC_STAGE_TIMER`, `CRYOZETA_STAGE_PEAKS`, `CRYOZETA_TRUNK_DUMP`). `sample_diffusion_chunk_size` ≠ 1 is refused unless the variant is `sample_paired: false`.
  - Workers include `--num_workers`, the setting the DataLoader actually reads (`data.num_dl_workers` is never read). Workers > 0 draw the featurizer's random point subsample in another RNG stream, which breaks pairing.
- `cli_args` are tokenized the way the CryoZeta config parser does it (joined, then split on whitespace) and must be `--flag value` pairs. The `--flag=value` form, values containing spaces, and argparse prefix abbreviations of a guarded flag are refused.
- **P-LC variants.** The five `use_*` flags are `cryozeta-cycle-predict`'s own options. Every other variant flag (`--use_lma`, `--infer_setting.chunk_size`) goes after a bare `--`, and `cycle_predict` appends it to every stage's inference config. `cycle_predict` refuses, with exit 2 before any stage work:
  - malformed pairs, duplicates and the `=` form;
  - flags a stage sets itself (seeds, paths, sampler sizes, the `use_*` flags), `p_dim` and `--em_file_dir`, plus argparse prefixes of these;
  - values the config parser rejects;
  - more than one attention algorithm (`use_memory_efficient_kernel`, DS4Sci/cueq attention, `use_lma`: the rule in `primitives.Attention`), `use_flash`, which no config passes to the model, and `use_memory_efficient_kernel`, which `Attention` has no branch for (it runs torch attention). Both would be silent no-ops.
- P-LC trials have no cell check (no frozen-det manifest).
- **Preflight.** Before `--dry-run` output and before any launch, `run_trial` passes every model step's argv through the real parsers. For P-LC that is `cycle_predict.parse_args` plus every stage config; for the other protocols it is the inference config. The check runs in a CPU-only child with throwaway caches. It refuses (exit 2, `refusing: preflight <step>: <the parser's error line>`) any combination the model would reject or ignore, with the same attention rule as `cycle_predict` (`attention_conflict`). Nothing is created and no GPU work starts.
- It refuses an existing `--out` (exit 2).
- It uses task-scoped compile caches for `--cache cold`: Triton, torch extensions, TileLang, cueq Triton (`CUEQ_TRITON_CACHE_DIR`) and the CUDA JIT cache (`CUDA_CACHE_PATH`).
- `--cache warm` uses `--cache-root`. Never purge a shared cache to make a run cold. The first warm trial on an empty root fills it and is effectively cold: discard it.

Timing keys in `trial.json`:
- **Mode A** (real CLI): `T4_s` per step and `T5_s` total. `T5_scope` says what T5 covers: P-LC includes per-stage detection, while the other protocols exclude the frozen detection. `T3_first_s` is the first forward per process; it includes the FPF one-time warm probe and an `empty_cache`, so it is first-request cost. `T3_subsequent_s` covers the later forwards.
- **Mode B** (warmed in-process repeat): per-iteration forward times are `mode_b.iterations[].t3`; the warmed ones are collected in `T3_warm_s` (and `throughput_per_s`). `process_wall_s` is the wall time of each whole Mode B process (one per step, warmup included) and `process_wall_total_s` their sum; neither is a user-facing T4/T5. The warmup excludes first-request cost from `T3_warm_s`.

## L1: operators

```bash
# synthetic inputs (seeded init: contract only; pass --checkpoint for real weights)
$P python -m cryozeta.kernel_bench.op_bench --module empairformer_stack.blocks.0.tri_att_start \
    --checkpoint $KB/assets/cryozeta-v0.0.1.safetensors --n 1024 --baseline-backend cueq \
    --backend fpf_k2b --strict --out $KB/trials/<host>/L1/triatt.jsonl
# real activations: capture once, then replay
$P python -m cryozeta.kernel_bench.capture_activations --out $KB/acts/9b0l \
    --modules 'empairformer_stack.blocks.0.*' -- <cryozeta-inference args>
$P python -m cryozeta.kernel_bench.op_bench --module empairformer_stack.blocks.0.tri_mul_out \
    --inputs $KB/acts/9b0l --backend fpf_trimul_v4 --baseline-backend cueq --strict --out ...
```

Providers per op are listed in `kernel_routing.OPS`. EM ops (`opm`, `rowcol`) need `--n-point`.

`op_bench` appends one record per backend, plus a comparison record, to `--out`, and rewrites `<out>.counters.json` on every call. Use a fresh `--out` per sweep. G-op and `speed_ratio` are computed only when both arms are SUCCESS.

## L2 to L4: end to end

Run the same `run_trial` call with a candidate variant. Interleave arms on one GPU, never concurrently:

```bash
for v in V0 K-TA-K2B-empairformer; do
  $P python -m cryozeta.kernel_bench.run_trial --protocol P-CZ --mode B --entry $KB/entries/9b0l.json \
      --frozen-det $KB/det/v1 --assets $KB/assets --variant docs/kernel-bench/variants/$v.yaml \
      --gpu 0 --timeout 7200 --cache warm --cache-root $KB/cache/<host> --out $KB/trials/<host>/L2/9b0l/$v/r1
done
$P python -m cryozeta.kernel_bench.compare_outputs --ref <V0 dump>/9b0l/CryoZeta --cand <cand dump>/9b0l/CryoZeta \
    --name 9b0l --tolerances docs/kernel-bench/tolerances.yaml --variant docs/kernel-bench/variants/$v.yaml \
    --out cmp.json
# L3 capacity cells (amendment 1): P levels from the P0-2 manifest; ineligible cells are listed, never clamped
$P python -c "import json, statistics as st; from cryozeta.kernel_bench.synth_inputs import capacity_cells; \
P = [e['N_point'] for e in json.load(open('$KB/det/v1.manifest.json'))['entries'].values()]; \
print(json.dumps(capacity_cells(range(512, 8193, 512), {'min': min(P), 'med': int(st.median(P)), \
'max': max(P), '2xmax': 2 * max(P)}), indent=1))"
# one cell: synthetic N_token and exactly P detection points (labelled synthetic; capacity/latency evidence only)
$P python -m cryozeta.kernel_bench.synth_inputs --base-json $KB/entries/9b0l.json --entry 9b0l --root $KB/assets \
    --n-token 3000 --n-point 1200 --det $KB/det/v1/9b0l/CryoZeta-Detection/9b0l.pt --out $KB/synth/9b0l-3000-1200
$P python -m cryozeta.kernel_bench.provenance freeze-det --det $KB/synth/9b0l-3000-1200/det \
    --entries $KB/synth/9b0l-3000-1200/entries
$P python -m cryozeta.kernel_bench.run_trial --protocol P-CZ --mode A --synthetic \
    --entry $KB/synth/9b0l-3000-1200/entries/9b0l_syn3000_real.json --frozen-det $KB/synth/9b0l-3000-1200/det ...
```

`compare_outputs` evaluates the sample-paired gates only with `--variant`, and refuses a variant marked
`sample_paired: false`. `M-SDC-5` is such a variant: status `deferred`, excluded from the paired benchmark and
from winner selection ([AMENDMENT-1.md](AMENDMENT-1.md)). No seed-spread threshold is substituted.

With `--tolerances`, `compare_outputs` also requires the following:
- **Trial records.** `--ref` and `--cand` must lie under `run_trial` outputs, i.e. have a `trial.json` above them.
- **Matching variant.** `--variant` must be the variant the candidate trial ran: the same `variant_id`, and the same file sha256 when the trial recorded one.
- **Right reference.** `--ref` must be a trial of the candidate's `baseline_id` (V0 for every current variant).
- **SUCCESS only.** Both trials must have outcome SUCCESS. A failed run is never gated.
- **Paired trials.** Both trials must share protocol, entries, frozen-detection hash, and seed, N_sample, N_step and N_cycle. `--seed` and `--name` must be the trials' seed and one of their entries.
- **No synthetic trials.** Neither trial may be synthetic. Synthetic runs are capacity/latency evidence only, never accuracy evidence.
- **Complete sealed file.** A `sealed: true` file must fill every gate and list `calibration.trials`.

Further behavior:
- `G-trunk` is null (not a pass) when a trunk threshold is set but no trunk tensors exist.
- A NaN or infinite delta fails its gate (it is never dropped by `max`).
- `all_pass` is true only if every gate is true.
- An existing `--out` is refused.

A protein/RNA chain left without an MSA (a truncated chain, or `--msa single`) gets a query-only MSA under
`OUT/msa/<i>`. The featurizer needs one; without it inference skips the sample and still exits 0.

`synth_inputs --n-point P` writes exactly P detection points: a seeded subset (`--seed`, default 0) of a larger
detection, or translated copies of a smaller one. P must be at most min(N_token, 2000), the unchanged V0
support-point cap; a larger P exits 2 and is a separate policy change, not a baseline cell.

Outcome classes (`trial.json`):
- SUCCESS, SKIPPED (an interpolation skip is never a success)
- UNSUPPORTED, FALLBACK, GPU_OOM, HOST_OOM, TIMEOUT, NUMERICAL, CRASH
- INVALID_CELL: the model input did not match the requested (N_token, P), or could not be verified. The raw class is kept in `cell.outcome_before_check`.

Allowed V0 rescues (cueq→DS4Sci TriAtt, cueq→torch APB) count as SUCCESS. They are reported in `baseline_rescues`.

Provider coverage is the lowest per-run fraction of eligible calls served by the requested provider. It is N/A when all runs have empty scope or contain only verified below-floor calls.
