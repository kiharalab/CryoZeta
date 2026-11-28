# Kernel-bench results, 2026-11-06 (GPU section 2026-11-10)

**First GPU measurements: hostA, RTX A5000 (24 GB, sm_86), 2026-11-10.** They cover one GPU type and one real
entry (9b0l). See [GPU validation on hostA](#gpu-validation-on-hostA-2026-11-10). No H100, A100 or RTX 5090
number exists. The only capacity statement is the 3-success support bracket 1280 ≤ N_max < 1344. It holds for V0 and for
fpf_k2b, for one 9b0l-synthetic cell at P = 620 on the A5000, and is runtime evidence only (Step 5).

This file records:
- what was built;
- what the CPU checks show (contract and correctness only, never speed);
- what ran on CUDA, with its measured results and failures.

## GPU validation on hostA (2026-11-10)

### Setup and provenance

| Item | Value |
|---|---|
| Host | hostA, a lab machine. It reads the shared the network filesystem checkout, so no code was transferred. Driver 595.91.07; 247 GB RAM; 32 CPUs at load about 15 (other users). |
| GPUs used | GPU0 RTX A5000 `GPU-<uuid>` and GPU3 RTX A5000 `GPU-<uuid>`, selected by UUID (`CUDA_VISIBLE_DEVICES`), or by PCI index with `CUDA_DEVICE_ORDER=PCI_BUS_ID` for `run_trial --gpu`. Each was checked idle right before each launch. GPU2 (A6000, another user's job), GPU1 (T400) and hostG were not touched. |
| Source | Branch `kernel-bench`, HEAD = `main` = `c544c6b`. Working tree as run: `$KB/provenance/hostA-20261110-gpu2/snapshot.json` (taken on hostA after the last source edit at 09:14; `diff_sha256 e3c38ea1…`, per-file `untracked_sha256`, e.g. `v0_regression.py bf94786c…`, `run_trial.py 441f0c9e…`). No source file was edited while a GPU job ran. Later edits are docs only. |
| Lock | `pixi.lock`, `uv.lock`, `pyproject.toml` sha256 match the plan (snapshot `lock_hashes`). 36 assets, 0 mismatched. No dependency changed. |
| Detection | `$KB/det/v1`, frozen and read-only: tree sha256 `3bb43405…`. For 9b0l, n_detected = N_point = 620 and N_token = 805. The tree hash was identical before and after every trial. Detection ran on GPU3 (`$KB/logs/det-v1-hostA-20261110.launch.sh`, exit 0) under the *leaked env* (below); every trial uses its frozen output as data. |
| Caches | Task-scoped: every trial or run has its own `cache/` (Triton, torch extensions, cuEq Triton, TileLang, CUDA). No shared cache was used or purged. |

**Environment leak, found and fixed in this pass.** Python's user site was active, so
`~/.local/lib/python3.12/site-packages` shadowed the frozen env. It imported numpy 2.3.3 instead of the lock's
1.26.4, plus pillow 11.3, protobuf and typing_extensions. This applied to **every earlier run on this account**:
the CPU closure on workstation, hostA P-CZ V0 r1, the P-LC smoke, and v0_regression runs 3–4.
- Fix: `PYTHONNOUSERSITE=1`. `run_trial.build_env` and `v0_regression` now set it for their children. Each
  `v0_regression` arm also records `numpy_file` and fails unless numpy comes from the frozen prefix.
- Run 5, the L1 r2 sweep and the paired Mode B trials ran with it, and their numpy is 1.26.4 from
  `.pixi/envs/default`.
- Earlier results are labelled *leaked env* below. The synthetic ladder inputs `$KB/synth/9b0l-{1024,1536,2048}-620`
  were also generated without it; they are data files, and their manifests check out.

### Step 2: v0_regression CUDA smoke (V0 GPU settings)

Command: `$KB/trials/v0-regression/gpu-hostA-20261110-5.launch.sh` (it is the README GPU block). It uses:
- the real 9b0l entry with `det/v1`;
- the V0 flags DS4Sci + cuEq attention + cuEq TriMul + cuEq APB, `LAYERNORM_TYPE=fast_layernorm`, `USE_OPM_CHUNKED=1`;
- `--route --` before the inference args;
- reduced smoke settings: N_cycle 2, N_sample 2, N_step 5, seed 101.

**Deviation from the 96-token fixture.** On CUDA, `s96_real` (64 synthetic points) recalls nothing under SVD or
TEASER. c544c6b then always runs VESPER: about 184 k rotations at about 29 it/s, roughly 1.75 h per sample, with
no switch to turn it off (run 2, stopped). 9b0l + `det/v1` is the smallest real input that exercises all routed ops.

Runs:

| Run | What happened | Evidence? |
|---|---|---|
| gpu-hostA-20261110-1 | Aborted earlier | No |
| 2 (`s96_real`) | Entered VESPER; stopped | No |
| 3 | 4 core arms ran; the comparator then crashed on a bf16 tensor (`point_residue_logits`), because numpy has no bf16 | No: leaked env. Used for the noise analysis only |
| 4 | Stopped when the env leak was found | No |
| **5** | The run below | **Yes** |

Bf16 fix in `_max_diff`: an exact upcast to fp32. A unit test (`V0Regression.test_bf16_pt`) covers it.

**Run 5:**
- Output: `$KB/trials/v0-regression/gpu-hostA-20261110-5/report.json`. Exit 1, wall 40:40, max RSS 8.33 GB.
- Arms: baseline 777.0 s (includes the cold layernorm build and Triton JIT), patched 441.7 s (it rebuilds the
  layernorm extension for its own source path), v0trial 253.5 s, repeat 255.9 s, route_apb 255.8 s,
  route_trimul 248.1 s, route_opm and route_rowcol OOM.

| Acceptance item | Result |
|---|---|
| `cpu_shim` | **false** in all 8 arms |
| `route_missing` | **[]** (the V0 counters reached trimul, triatt, opm, apb and rowcol) |
| frozen numpy | all 8 arms import `.pixi/envs/default/.../numpy` |
| served backends (v0trial counters) | trimul cueq 416 calls (384 empairformer, 16 msa, 16 confidence); triatt cueq 416; rowcol ds4sci 192; opm chunked 200; APB torch everywhere (cueq rejected as `ineligible_shape` by c544c6b's `c_hidden % 32` gate: empairformer 96, diffusion 240, confidence 8; atom 63 torch) |
| route_trimul (C-CAT at all 3 sites) | ran; strict counters show `cueq_cached_cat` served at every replaced site; `problems: []` |
| route_apb (strict torch identity) | ran; `problems: []` |
| route_opm (einsum at empairformer + msa) | **GPU_OOM**: the unchunked einsum OPM asks for 6.29 GiB more at N = 805 on the 24 GB card. A resource limit of that provider, not a routing fault |
| route_rowcol (LMA at empairformer) | **GPU_OOM** (asks for 4.95 GiB more). Same class |
| bitwise `patched`/`v0trial`/`repeat` vs baseline | **false** (0/20 files identical) |
| **`v0_ok`, `ok`** | **false** |

**Why the bitwise check fails, and why the patch is not the cause.** The `repeat` arm runs the identical
patched code a second time. It differs from `baseline` and from `patched` as much as `patched` differs from
`baseline`. Raw model output: `output_dict_9b0l.pt`, 10283 atoms × 2 samples; per-atom displacement in Å
(`$KB/trials/v0-regression/gpu-hostA-20261110-5.pairwise.txt`, script `$KB/trials/v0-regression/pairwise.py`):

| Pair (same math in every arm) | coord median | coord RMS | coord max | logits mean \|Δ\| | logits max |
|---|---|---|---|---|---|
| baseline vs **repeat** (noise floor, rev tree vs patched tree) | 0.545 | 0.888 | 7.55 | 0.0418 | 0.77 |
| **patched vs repeat** (identical code, twice) | 0.527 | 0.922 | 7.95 | 0.0424 | 2.69 |
| baseline vs patched | 0.564 | 0.985 | 10.28 | 0.0424 | 2.97 |
| baseline vs v0trial | 0.598 | 1.012 | 20.41 | 0.0419 | 1.06 |
| baseline vs route_apb | 0.540 | 0.908 | 11.79 | 0.0420 | 1.25 |
| baseline vs route_trimul (C-CAT) | 0.596 | 1.020 | 13.07 | 0.0420 | 1.75 |

- `point_residue_logits` is a trunk output, so the trunk itself is nondeterministic across processes. This holds
  despite c544c6b's `seed_everything(seed, deterministic=True)` (`inference.py:205`), which sets
  `use_deterministic_algorithms` and `CUBLAS_WORKSPACE_CONFIG` but does not cover DS4Sci, cuEq Triton or
  fast_layernorm.
- Every pair lies inside the same band.
- Confidence JSON differences are 0.0625–0.125 in every pair.
- The registered and ranked files (`*_svd.npz`, ranked `.cif`) differ by up to about 400 Å only because sample
  order and the registration choice flip downstream.

Result:
- The CUDA bitwise V0 gate is **void**. It cannot pass for any patch.
- The data shows no patch-attributable difference. The patch-minus-noise gap is indistinguishable from zero at
  this sample size (one run each); that is not a tolerance-based pass.
- Run 3 (leaked env) showed the same pattern: patched vs repeat 7.80 Å max, baseline vs patched 7.96 Å.

**Decision needed (scientific setting):** either a deterministic CUDA mode for these kernels, or sealed
tolerances calibrated from `repeat` (P0-5, D_rep).

### Step 3: real P-CZ and P-LC trials (V0)

**Dry runs.** `run_trial --dry-run`:
- P-CZ: `$KB/logs/dryrun-hostA-PCZ-V0-r1.log`;
- P-LC: `dryrun-hostA-PLC-V0-smoke-svd-r1.log`;
- candidate K-TM-V4 Mode B: workstation CPU, rc 0.

All passed. They resolve the expected cell `9b0l: [805, 620]`.

**P-CZ V0 r1** (full V0 settings, *leaked env*):
- Command: `run_trial --protocol P-CZ --mode A --entry $KB/entries/9b0l.json --frozen-det $KB/det/v1 --variant
  V0.yaml --gpu 3 --timeout 7200 --cache cold`.
- Settings: N_cycle 10, N_sample 5, N_step 20. GPU3.
- Output: `$KB/trials/hostA/P-CZ/9b0l/V0/r1/trial.json`.

| Field | Value |
|---|---|
| outcome | SUCCESS, exit 0, 0 missing outputs, no ERR files |
| model input | N_token 805, N_atom 10283, N_msa 5038, N_point 620 (`em_support_points` [620,3], `pz` [620,805,125]); matches the manifest |
| T5 (cold process, total) | 1413.3 s; T0 model ready 263.7 s |
| T3 first forward (cold JIT) | 1118.6 s: EMPairformer 905.9 s (cycle 1 384.7 s, then about 58 s per cycle), diffusion 20.7 s, fitting 183.2 s, confidence 7.5 s |
| M3 NVML peak | 13130 MiB |
| M2 torch peak / M1 | 13.45 GB / 9.77 GB |
| host | cgroup peak 12.59 GB; max RSS 8.37 GB |
| finite | all output tensors and coords finite; recall about 0.75; plddt 43.5–46.3; ptm 0.386 |

**P-LC smoke-svd-r1** (*smoke settings*, *leaked env*, **INCOMPLETE**):
- Command: `$KB/logs/hostA-gpu3-chain-1.sh`, i.e. `--protocol P-LC --n-cycle 2 --n-sample 2 --n-step 5
  --registration svd_only`.
- Why `svd_only`: smoke-r1 with `auto` exited 1 in c544c6b's `compare_registration_methods`, because TEASER
  found one correspondence and wrote no npz.
- Results:
  - Stage 1: N_token 520, N_point 520, forward 438.1 s (cold).
  - Stage 2: N_token 247, P 247, forward 82.0 s; T3_subsequent 82.8 s.
  - Both stages had finite outputs.
  - Stage 3 (a 23-token DNA chain) recalled nothing (`MAP SUM=0.0 COUNT=0`) and entered the baseline VESPER
    search, which put M3 at 24110 MiB.
- I stopped it (`kill -TERM` of my own process group). The trial records CRASH, exit −15, with
  `stage_names.txt` and `combined.cif` missing. Wall time to the stop: 2060.6 s.
- Counters: trimul/triatt cueq; rowcol ds4sci; opm chunked; APB torch.
- A complete P-LC run needs a decision on the unconditional VESPER fallback, which takes hours per stage that
  recalls nothing.

### Step 4: L1 op_bench (triangle multiplication and triangle attention)

- Command: `$KB/trials/hostA/L1/r2/sweep.sh GPU-<uuid>` (GPU3), i.e. `op_bench --module
  empairformer_stack.blocks.0.<m> --checkpoint cryozeta-v0.0.1 --n {256,805} --baseline-backend v0 --backend <b>
  --strict`.
- Inputs: real weights, synthetic inputs, bf16 autocast.
- Records: `$KB/trials/hostA/L1/r2/<m>.n<N>.jsonl`; table: `python3 -I tools/summarize.py r2`.
- G-op compares rel. Frobenius error vs fp64 against 2× the baseline's, at N = 256 only. speed_ratio =
  candidate median / V0 median, below 1 means faster.
- All outcomes were deterministic within a process.

| Module | Candidate | N | Outcome | G-op | rel_fro vs fp64 (cand / V0) | rel_fro vs V0 | median ms (V0 → cand) | speed_ratio | peak MB (V0 → cand) |
|---|---|---|---|---|---|---|---|---|---|
| tri_mul_out | cueq_cached_cat | 256 | SUCCESS | pass | 0.00564 / 0.00564 | 0 | 1.31 → 1.39 | 1.06 | 201 → 193 |
| tri_mul_out | fpf_trimul_v4 | 256 | SUCCESS | pass | 0.00530 / 0.00564 | 0.0037 | 1.31 → 0.823 | 0.63 | 201 → 183 |
| tri_mul_out | torch | 256 | SUCCESS | pass | 0.00693 / 0.00564 | 0.0067 | 1.31 → 3.54 | 2.71 | 201 → 177 |
| tri_mul_in | cueq_cached_cat | 256 | SUCCESS | pass | 0.00581 / 0.00581 | 0 | 1.29 → 1.41 | 1.09 | 201 → 193 |
| tri_mul_in | fpf_trimul_v4 | 256 | SUCCESS | pass | 0.00549 / 0.00581 | 0.0037 | 1.29 → 0.817 | 0.63 | 201 → 183 |
| tri_mul_in | torch | 256 | SUCCESS | pass | 0.00714 / 0.00581 | 0.0070 | 1.46 → 3.71 | 2.55 | 201 → 177 |
| tri_att_start | fpf_k2b | 256 | SUCCESS | pass | 0.00675 / 0.00675 | 0.0013 | 1.63 → 1.54 | 0.94 | 189 → 155 |
| tri_att_start | lma | 256 | SUCCESS | pass | 0.0110 / 0.00675 | 0.0093 | 1.63 → 7.07 | 4.33 | 189 → 898 |
| tri_att_start | torch | 256 | SUCCESS | pass | 0.0110 / 0.00675 | 0.0093 | 1.63 → 3.76 | 2.31 | 189 → 369 |
| tri_att_start | ds4sci | 256 | **CRASH** | – | – | – | – | – | – |
| tri_att_end | fpf_k2b | 256 | SUCCESS | pass | 0.00783 / 0.00783 | 0.0012 | 1.63 → 1.53 | 0.94 | 189 → 155 |
| tri_att_end | lma / torch | 256 | SUCCESS | pass | 0.0132 / 0.00783 | 0.0112 | 1.64 → 7.07 / 3.74 | 4.31 / 2.27 | 189 → 898 / 369 |
| tri_att_end | ds4sci | 256 | **CRASH** | – | – | – | – | – | – |
| tri_mul_out | fpf_trimul_v4 | 805 | SUCCESS | n/a (N > 256) | – | 0.0036 | 15.2 → 8.90 | **0.59** | 1910 → 1440 |
| tri_mul_in | fpf_trimul_v4 | 805 | SUCCESS | n/a | – | 0.0037 | 14.2 → 8.91 | **0.63** | 1910 → 1440 |
| tri_mul_out / in | cueq_cached_cat | 805 | SUCCESS | n/a | – | 0 | 15.2 → 15.2 / 14.2 → 14.2 | 1.00 | 1910 → 1900 |
| tri_mul_out / in | torch | 805 | SUCCESS | n/a | – | 0.0064 / 0.0067 | about 46.5 | 3.06 / 3.29 | 1910 → 1480 |
| tri_att_start / end | fpf_k2b | 805 | SUCCESS | n/a | – | 0.0010 | 25.9 → 18.9 | **0.73** | 1800 → 1540 |
| tri_att_start / end | torch | 805 | SUCCESS | n/a | – | 0.010 / 0.012 | 25.9 → 108 | 4.16 | 1800 → 9080 |
| tri_att_start / end | lma | 805 | **GPU_OOM** | – | – | – | – | – | – |
| tri_att_start / end | ds4sci | 805 | **CRASH** | – | – | – | – | – | – |

- First-call (compile) cost, kept separate from the medians:
  - fpf_trimul_v4: 0.64–7.1 s;
  - fpf_k2b: 0.65–6.8 s;
  - cueq: 0.2–6.6 s, the larger values on the first module of a fresh cache.
- **ds4sci triangle attention is unsupported in this configuration.** `RuntimeError: expected scalar type
  BFloat16 but found Float`. c544c6b's `_deepspeed_evo_attn` casts the biases to bf16 only when q is fp32. Under
  bf16 autocast, q is bf16 but the triangle mask bias is fp32. That code is unchanged from the baseline. ds4sci is
  only V0's fallback for N at or below cuEq's threshold, not a replacement candidate. The shared function also
  serves V0's row attention. **2026-11-16: fixed on user request** (biases cast to q's dtype; no-op where c544c6b ran).
  See [AMENDMENT-2.md](AMENDMENT-2.md) A2.2. The rows above are pre-fix. GPU recheck queued (`$KB/trials/hostA/L1/r3`).
- **Candidates advanced:**
  - `K-TM-V4-empairformer` (fpf_trimul_v4): G-op pass; 0.59–0.63× op time and 25% less op memory at N = 805.
  - `K-TA-K2B-empairformer` (fpf_k2b): G-op pass; 0.73× and 14% less memory at N = 805.
  - These are single-op numbers, not end-to-end speedups.

### Step 4b: paired P-CZ Mode B vs V0 (9b0l real, N = 805, P = 620, full V0 settings)

- Command (`$KB/logs/hostA-pair.sh GPU_INDEX GPU_UUID TAG VARIANT...`, one fresh process per variant, cold
  task-scoped cache, busy guard before each variant):
  `run_trial --protocol P-CZ --mode B --warmup-forwards 1 --timed-forwards 1 --entry $KB/entries/9b0l.json
  --frozen-det $KB/det/v1 --assets $KB/assets --variant docs/kernel-bench/variants/<V>.yaml --gpu <i>
  --timeout 7200 --cache cold --out $KB/trials/hostA/P-CZ/9b0l/<V>/modeB-<tag>`.
- iter0 is the cold forward (compile events and new cache files counted). iter1 is the warmed, timed forward
  with 0 compile events. Stage times are the time to each stage's end mark. M2 is the caching allocator's
  `max_memory_reserved` per iteration and M1 is `max_memory_allocated` (live tensors), both reset before each
  iteration (`kernel_bench/mode_b.py`, `kernel_routing.snapshot`; corrected 2026-11-11, this line said
  `max_memory_allocated` for M2 before). M3 is the NVML peak over the process. Summary: `python3 -I $KB/trials/hostA/P-CZ/modeb_summary.py
  <trial dirs>`.
- Served backends: fpf_trimul_v4 served all 1920 `trimul@empairformer` calls per forward and fpf_k2b served
  all 1920 `triatt@empairformer` calls. Every other call site stayed on V0 (cueq trimul/triatt at msa and
  confidence, ds4sci rowcol 960, chunked OPM 1000, APB torch). No in-scope fallback.
- Every trial below: SUCCESS, exit 0, 0 missing outputs, cell match, all output tensors finite.

**Same GPU (hostA GPU0 `GPU-<uuid>`, A5000), both orders: r2 = K2B → TM-V4 → V0, r3 = V0 → TM-V4 → K2B.**
n = 1 per variant and order. Δ is against V0 in the same round.

| Variant (kernel source) | round | cold iter0 s (compile events, new cache files) | warm T3 s | EMPairformer s (Δ) | diffusion s | fitting s | confidence s | M2 cold / warm GiB | M3 MiB | process wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| V0 (NVIDIA cuEq trimul + triatt) | r2 | 1097.6 (9, 74) | 761.1 | 535.8 | 19.8 | 200.0 | 4.4 | 12.53 / 10.68 | 13130 | 2157.7 |
| V0 | r3 | 1074.2 (9, 74) | 743.5 | 535.2 | 19.8 | 183.1 | 4.4 | 12.53 / 10.68 | 13130 | 2116.0 |
| K-TM-V4 (FlashPairformer `fpf_trimul_v4`, Triton) | r2 | 1085.6 (12, 96) | 753.4 | 531.4 (−4.4) | 19.8 | 196.6 | 4.4 | 12.53 / 10.97 | 13130 | 2154.5 |
| K-TM-V4 | r3 | 1068.3 (12, 96) | 750.4 | 529.0 (−6.2) | 19.8 | 196.1 | 4.4 | 12.53 / 10.99 | 13130 | 2107.9 |
| K-TA-K2B (FlashPairformer `fpf_k2b`, Triton) | r2 | 1064.6 (14, 112) | 731.2 | 521.9 (−13.9) | 19.7 | 184.3 | 4.4 | 12.53 / 10.68 | 13132 | 2074.7 |
| K-TA-K2B | r3 | 1078.1 (14, 112) | 731.7 | 522.1 (−13.1) | 19.8 | 184.2 | 4.4 | 12.53 / 10.68 | 13132 | 2113.1 |

- **Warm EMPairformer, mean of the two orders:** V0 535.5 s; fpf_trimul_v4 530.2 s (−5.3 s, −1.0%);
  fpf_k2b 522.0 s (−13.5 s, −2.5%). V0's own spread across the two rounds is 0.6 s. Both candidates are
  faster in both orders, n = 2 per arm on one GPU, so this is a measured observation, not a general speedup
  claim.
- **Cold forward:** no difference beyond noise. The candidates add compile work (12 and 14 compile events vs 9;
  96 and 112 new cache files vs 74), and the cold EMPairformer stays at 863–877 s for every variant.
- In the warm forward, EMPairformer is about 70% of T3 and fitting (prediction heads, SVD 0.8 / 0.4 and
  TEASER point fitting, scoring) about 25%. Fitting varied by 17 s between these runs (183–200 s), more than
  either kernel's EMPairformer change. Compare EMPairformer, not T3.
- Op-level expectation from Step 4 (1920 calls × per-call saving at N = 805): about 12 s for fpf_trimul_v4 and
  about 13 s for fpf_k2b. Observed −5.3 s and −13.5 s. fpf_k2b matches its op-level estimate; fpf_trimul_v4
  realises less than half of its estimate in the full forward.
- Neither kernel lowers whole-forward peak memory. M3 is flat; warm M2 (reserved) rises 0.29 GiB with fpf_trimul_v4.
  M1 (allocated) is 9.10 cold / 9.04 warm GiB for V0 and K2B in every round, and 9.13 / 9.06 GiB for fpf_trimul_v4.
- Other forward stages are untouched by these kernels: compile (cold minus warm, about 320–340 s), fitting,
  and process start / model load / featurization / writing (about 295 s in the Mode A V0 cell).

**Not paired.**
- V0 r1 on GPU3 (`GPU-<uuid>`): warm T3 777.2 s, EMPairformer 574.0 s, fitting 176.8 s, M3 13130 MiB. That
  is 38 s more EMPairformer than V0 r2 on GPU0, more than either kernel effect, so cross-GPU deltas are not used.
- K-TM-V4 r1 on GPU3: void for timing (K7 in the ledger). Another user's training job shared the GPU from
  10:33:14, about 3 minutes after the trial started: cold 1605.0 s, warm 1411.2 s, M3 23474 MiB (device-wide).
  Its outputs are finite and are kept for correctness only. K-TA-K2B r1 did not start (busy guard).
- The reverse order ran on GPU0 as r3 (`$KB/logs/hostA-gpu0-chain-1.sh`, `$KB/logs/hostA-pair-gpu0-r3.log`).

**Output differences (descriptive; the bitwise gate is void on CUDA, no tolerances sealed).** Raw
`output_dict_9b0l.pt`, per-atom distance of `pred_all_coordinate`, `$KB/trials/hostA/P-CZ/pairwise_pt.py`,
GPU0 r2 (`$KB/trials/hostA/P-CZ/pairwise-gpu0-r2.txt`; r3 in `pairwise-gpu0-r3.txt`):

| Pair | coord median Å | coord max Å | logits mean \|Δ\| |
|---|---|---|---|
| V0 it0 vs V0 it1 (same process) | 0.579 | 11.85 | 0.0510 |
| V0 vs K-TM-V4 (4 iteration pairs) | 0.277–0.721 | 9.77–19.46 | 0.0509–0.0510 |
| V0 vs K-TA-K2B (4 iteration pairs) | 0.231–0.721 | 11.67–14.55 | 0.0509–0.0511 |
| K-TM-V4 it0 vs it1 / K-TA-K2B it0 vs it1 | 0.342 / 0.545 | 18.53 / 12.35 | 0.0509 / 0.0510 |

r3 has the same pattern: V0 vs V0 (same process, and r3 vs r2) has a median of 0.41–0.55 Å and a max of
11.4–13.4 Å. V0 vs candidates has a median of 0.23–0.59 Å and a max of 7.4–20.2 Å. Every dump checked is finite.
V0's own forward-to-forward difference has the same size as V0-vs-candidate. This is no accuracy claim:
a G-struct verdict needs sealed tolerances (P0-5).

### Step 5: fixed-P token ladder (9b0l synthetic, P = 620, hostA GPU0 and GPU3) — done 2026-11-11 15:08

A capacity claim needs 3 consecutive successes per level plus correctness checks. Only N = 1280 meets that rule, for
both V0 and K2B. Every other level is a single cell and a provisional resource observation. The refinement and repeats
(2026-11-11) are further down.

- Protocol: Mode A, fresh process, cold task-scoped cache per cell, full V0 settings.
- Inputs: `$KB/synth/9b0l-<N>-620` (synthetic; N_point = n_detected = 620).
- Timeout: min(10800, max(3 × 1413.3 × (N/805)³, 1800)), which is 8727 s at N = 1024.
- Order: V0 and K-TA-K2B-empairformer (FlashPairformer `fpf_k2b`, Triton, from Anthropic
  `uplifting-biomolecular-modeling` @ f4f62fa) interleaved per N, at N = 1024, then 1536, then 2048.
- GPU: GPU0 `GPU-<uuid>`.
- Scripts: `$KB/logs/hostA-ladder.sh` (IO guard added 15:48, K9; the pre-guard copy is `hostA-ladder.sh.v1`)
  and `$KB/logs/hostA-gpu0-chain-3.sh`. Log: `$KB/logs/hostA-ladder-gpu0-r1b.log`.
- Cells: `$KB/trials/hostA/L3/9b0l-P620/<V>/N<N>-r1`.

```
cd $REPO && PYTHONNOUSERSITE=1 pixi run -e default --frozen python -m cryozeta.kernel_bench.run_trial \
  --protocol P-CZ --mode A --synthetic --entry $KB/synth/9b0l-$N-620/entries/9b0l_syn${N}_real.json \
  --frozen-det $KB/synth/9b0l-$N-620/det --assets $KB/assets --variant docs/kernel-bench/variants/$V.yaml \
  --gpu 0 --timeout 10800 --smaller 805:1413.3 --n-token $N --cache cold --out $KB/trials/hostA/L3/9b0l-P620/$V/N$N-r1
```

| Variant | N | Outcome | Cell (N, P) | M3 NVML peak MiB | T5 s | Cold forward s (EMPairformer / diffusion / fitting / confidence) | Notes |
|---|---|---|---|---|---|---|---|
| V0 | 1024 | SUCCESS, exit 0, 0 missing | match (1024, 620) | 15338 | 1749.3 | 1304.9 (1067.6 / 28.9 / 197.3 / 11.1) | 5 samples, 12228 atoms, raw output finite; M4 RSS 8.0 GiB |
| K-TA-K2B | 1024 | INCOMPLETE (r1): killed by hostA's 16:32–16:59 reboots, no outcome in `trial.json` | match (1024, 620) | 18398 before contention (18810 device-wide, with another user's 444 MiB process) | — | EMPairformer 1047.3 (clean, before 14:36); diffusion 637.4 and fitting stalled by hostA IO (K9) | timing after 14:36 void; rerun as r2 |

**Driver break.** hostA came back from the reboots on kernel 7.0.0-43 and NVIDIA driver 595.99.02. Every
earlier hostA result above ran on 7.0.0-31 and 595.91.07. The ladder was relaunched at 17:30 with
`$KB/logs/hostA-gpu0-chain-4.sh` (log `hostA-ladder-gpu0-r2b.log`). N = 1024 is rerun as r2 for both variants,
so each same-N V0/K2B pair runs on one driver; then come N = 1536 and 2048 as r1. Rows below are on the new driver.

**M2 check, in parallel on GPU3.** Mode B (1 cold warmup forward + 1 timed warmed forward) at N = 1024,
V0 then K2B, same GPU. Script `hostA-modeb-syn.sh`, cells `$KB/trials/hostA/P-CZ/9b0l-syn1024-P620/<V>/modeB-r1`.
GPU0 and GPU3 now run at the same time on one host, so host CPU and IO are shared between the two chains.
Both first cells started at 17:31 and compiled FastFold `fastfold_layer_norm_cuda` (cold task-scoped cache on the network filesystem) at the
same moment. At 17:33 hostA's IO pressure read full avg10 = 61, from my own `cc1plus` processes, so the Mode A T5 and cold-forward
figures of these cells include that shared compile contention. Mode B iter1 (warm) does not.

| Variant | N | Run | Outcome | Notes |
|---|---|---|---|---|
| V0 | 1024 | L3 N1024-r2 (GPU0, Mode A) | VOID (IO stall, K9) | Stuck in the cold JIT build of DeepSpeed4Science EvoformerAttention (Microsoft, CUTLASS) from 17:41. GPU at 15338 MiB and 0 % util until the deadline kill (card at 14 MiB by 20:02). As of 21:07, `trial.json` still had no outcome (write to the network filesystem pending) |
| V0 | 1024 | P-CZ modeB-r1 (GPU3, Mode B) | VOID (IO stall, K9) | Same build stall. Card at 14 MiB by 20:37 after the 10800 s deadline. No iteration ran, so no M2 |

The 17:30 relaunch therefore produced no measurement. From about 19:02, `ssh hostA` timed out or closed on login,
so the remaining cells cannot be supervised.

**Relaunch, 2026-11-11 08:50.** hostA was rebooted again (08:36–08:39) and came back on the same kernel 7.0.0-43
and driver 595.99.02. All four GPUs were idle, IO pressure was 0, and none of my processes survived. Before the
relaunch I checked the working tree: `git diff c544c6b` sha256 equals the snapshot `diff_sha256`, all 48
non-docs untracked files match, and `pixi.lock`, `pyproject.toml` and `uv.lock` match.
- GPU0: `$KB/logs/hostA-gpu0-chain-5.sh`, log `hostA-ladder-gpu0-r3.log`. Order: V0 N1024-r3, K2B N1024-r2,
  then N = 1536 and 2048 as r1. A TIMEOUT now aborts the chain for review instead of marking the variant stopped.
- GPU3: `$KB/logs/hostA-gpu3-chain-3.sh`, log `hostA-modeb-syn-gpu3-r2.log`. Mode B modeB-r2, V0 then K2B. It
  starts 900 s after GPU0, so the two cold extension builds do not overlap at launch.

| Variant | N | Run | Outcome | Cell (N, P) | M3 NVML peak MiB | T5 s | Cold forward s (EMPairformer / diffusion / fitting / confidence) | Notes |
|---|---|---|---|---|---|---|---|---|
| V0 | 1024 | L3 N1024-r3 (GPU0) | SUCCESS, exit 0, 0 missing, no ERR files | match (1024, 620) | 15338 | 1686.1 | 1228.5 (1039.8 / 28.4 / 149.3 / 11.0) | N_atom 12228; raw output finite; M4 RSS 8.08 GiB. GPU3's Mode B started its cold build at 09:05, during this cell's EMPairformer. vs V0 r1 (old driver): coord median 1.12 Å, max 32.3 Å, logits mean \|Δ\| 0.071 (descriptive) |
| K-TA-K2B | 1024 | L3 N1024-r2 (GPU0) | SUCCESS, exit 0, 0 missing, no ERR files | match (1024, 620) | 15338 | 1483.6 | 1176.3 (995.9 / 26.8 / 142.9 / 10.7) | scope_coverage 1.0; same APB cueq→torch `ineligible_shape` rescues as V0 (480 / 2400 / 20); M4 RSS 8.13 GiB; T0 model-ready 275.5 s (V0 r3: 417.1 s, first launch after the reboot). Its EMPairformer overlapped GPU3's V0 Mode B warm forward. vs V0 r3: coord median 1.80 Å, RMS 9.06, max 67.4 Å, logits mean \|Δ\| 0.092; vs V0 r1: 1.43 / 11.10 / 62.2 Å, 0.073 (descriptive). All three dumps finite |
| V0 | 1536 | L3 N1536-r1 (GPU0, 09:51–10:30) | **GPU_OOM**, 10 outputs missing | (1536, 620) entered the model | 22338 | 2315.3 | EMPairformer, diffusion (52.6 s) and fitting (157.5 s) finished; OOM in the confidence head | `confidence.py:272` `memory_efficient_forward` → pairformer block → `primitives.py:153`: tried 4.50 GiB with 15.97 GiB allocated and 5.55 GiB reserved but unallocated, 1.73 GiB free of 23.55 GiB. M1 18.72 GiB, M2 21.52 GiB. GPU3 ran K2B Mode B during this cell |
| K-TA-K2B | 1536 | L3 N1536-r1 (GPU0, 10:30–11:08) | **GPU_OOM**, 10 outputs missing | (1536, 620) entered the model | 22338 | 2232.0 | EMPairformer finished; OOM in the confidence head | Same frame (`primitives.py:153`), the same 4.50 GiB request, and the same 15.97 / 5.55 GiB allocated / reserved-unallocated as V0. M1 20104832512 B and M2 23104323584 B, byte-identical to V0 N1536-r1. scope_coverage 1.0. The confidence head is outside fpf_k2b's replacement scope (cuEq triatt there) |
| V0 | 1280 | L3 N1280-r1 (**GPU3**, 10:31–11:03, refinement) | SUCCESS, exit 0, 0 missing, no ERR files | match (1280, 620) | 20932 | 1863.7 | 1570.5 (1367.8 / 38.9 / 147.4 / 16.4) | N_atom 14323; raw output finite; M1 18.14 GiB, M2 20.14 GiB; T0 258.8 s. GPU0 ran K2B N1536 at the same time |
| K-TA-K2B | 1280 | L3 N1280-r1 (GPU0, 11:09–11:40, refinement) | SUCCESS, exit 0, 0 missing, no ERR files | match (1280, 620) | 20932 | 1808.7 | 1521.4 (1314.7 / 37.4 / 152.8 / 16.5) | scope_coverage 1.0; M1 and M2 byte-identical to V0 N1280-r1 (19474915840 / 21630025728 B); T0 252.3 s; raw output finite. vs V0 N1280-r1: coord median 0.53 Å, RMS 1.99 Å, max 39.3 Å, logits mean \|Δ\| 0.050 (descriptive). GPU3 ran V0 N1408 at the same time |
| V0 | 1280 | L3 N1280-r2 (GPU3, 11:54–12:27, repeat 2) | SUCCESS, exit 0, 0 missing, no ERR files | match (1280, 620) | 20932 | 1939.4 | 1657.1 (1454.1 / 40.4 / 145.9 / 16.7) | M1 and M2 byte-identical to r1; raw output finite. Host shared with other users' GPU0 training and GPU2 job (K11), so the +6% EMPairformer time vs r1 is not attributable. vs r1: coord median 0.50 Å, RMS 1.86 Å, max 38.8 Å, logits mean \|Δ\| 0.050. K2B r1 vs V0 r1 / r2: median 0.53 / 0.37 Å, RMS 1.99 / 2.16 Å, i.e. the same size as V0's own repeat difference (`$KB/trials/hostA/L3/9b0l-P620/pairwise-N1280-r1r2.txt`, descriptive) |
| V0 | 1280 | L3 N1280-r3 (GPU3, 13:01–13:33, repeat 3) | SUCCESS, exit 0, 0 missing, no ERR files | match (1280, 620) | 20932 | 1912.6 | 1666.2 (1448.6 / 40.3 / 158.6 / 17.6) | M1 and M2 byte-identical to r1 and r2; raw output finite; same allowed V0 rescues (cueq→torch APB). Host-contended (K11). Pairwise (`pairwise-N1280-r1r2r3.txt`, descriptive). V0 repeat pairs: median 0.33–0.50 Å, RMS 1.24–1.86 Å. K2B r1 vs V0 r1–r3: median 0.32–0.53 Å, RMS 1.91–2.16 Å. Logits mean \|Δ\| is 0.050–0.052 in all six pairs |
| V0 | 1344 | L3 N1344-r1 (GPU3, 12:27–13:01, refinement) | **GPU_OOM**, exit 0, 10 outputs missing | match (1344, 620) | 20992 | 2000.6 | EMPairformer 1537.9, diffusion 43.0, fitting 152.5; OOM in the confidence head | Same frame as at 1408/1536 (`confidence.py:272` → `empairformer.py:355` → `kernel_routing.py:258` → `primitives.py:153`). Tried 3.45 GiB with 16.41 GiB allocated, 3.48 GiB reserved but unallocated and 3.36 GiB free. M1 20395201536 B, M2 21695037440 B. Host-contended (K11) |
| K-TA-K2B | 1280 | L3 N1280-r2 (GPU3, 14:06–14:37, repeat 2) | SUCCESS, exit 0, 0 missing, no ERR files | match (1280, 620) | 20932 | 1810.1 | 1540.9 (1330.2 / 37.9 / 155.2 / 16.5) | scope_coverage 1.0 (fpf_k2b 1920 EMPairformer triatt; cueq 40 confidence, 80 msa); M1/M2 byte-identical to V0 N1280. Host-contended (K11) |
| K-TA-K2B | 1280 | L3 N1280-r3 (GPU3, 14:37–15:08, repeat 3) | SUCCESS, exit 0, 0 missing, no ERR files | match (1280, 620) | 20932 | 1827.7 | 1546.0 (1335.6 / 37.9 / 154.7 / 16.7) | scope_coverage 1.0, same served backends as r2; M1/M2 byte-identical to V0 N1280. Host-contended (K11) |
| K-TA-K2B | 1344 | L3 N1344-r1 (GPU3, 13:33–14:06, paired level) | **GPU_OOM**, exit 0, 10 outputs missing | match (1344, 620) | 20992 | 1900.9 | EMPairformer 1433.0, diffusion 40.8, fitting 159.8; OOM in the confidence head | Byte-identical to V0 N1344-r1: M1 20395201536 B, M2 21695037440 B, M3 and the 3.45 GiB request all match. fpf_k2b served all 1920 EMPairformer triatt calls (scope 1.0). The 36 confidence-head triatt calls (cueq, outside scope) are where it fails. The −6.8% EMPairformer time vs V0 is host-contended and not attributable (K11) |
| V0 | 1408 | L3 N1408-r1 (GPU3, 11:03–11:38, refinement) | **GPU_OOM**, 10 outputs missing | (1408, 620) entered the model | 22682 | 2074.5 | EMPairformer, diffusion and fitting finished; OOM in the confidence head | Same frame (`primitives.py:153`): tried 3.78 GiB with 18.05 GiB allocated and 3.81 GiB reserved but unallocated, 1.40 GiB free. M1 20.88 GiB, M2 21.85 GiB |

**First failure: V0 at N = 1536.** This is one cell, so it is a provisional resource observation. The failing
request (4.50 GiB) is smaller than the reserved-but-unallocated pool (5.55 GiB), so allocator fragmentation is part
of this OOM. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` might move the boundary. That is a runtime-setting
change outside V0, so it was not tried. Chain-5 marks V0 stopped and continues with K2B at N = 1536 on GPU0.

**Boundary refinement for V0, GPU3** (same A5000 model and 24 GB; `$KB/logs/hostA-gpu3-chain-4.sh`, log
`hostA-refine-gpu3-r1.log`, started 10:31). One bisection step inside (1024, 1536): N = 1280 first, then 1408
on SUCCESS or 1152 otherwise. Same `hostA-ladder.sh` cell protocol, tag r1, timeout 10800 s.
K2B failed at N = 1536 in the same place. Its refinement runs on GPU0 (`$KB/logs/hostA-gpu0-chain-6.sh`, log
`hostA-refine-gpu0-r1.log`, started 11:09) at the same points: N = 1280, then 1408 on SUCCESS.

At the failure point the two variants are byte-identical in M1, M2 and the OOM request. The binding allocation is in
the confidence head, where fpf_k2b is not used. So fpf_k2b cannot move this boundary. Moving it needs a change in the
confidence-head pairformer or in the allocator.

**Second refinement step and repeats (V0, GPU3).** V0 succeeds at 1280 and fails at 1408. A new synthetic level, N = 1344,
was built on workstation CPU with `$KB/logs/synth-refine-2.sh` (same `synth_inputs` + `freeze-det` commands as
`synth-refine-1.sh`; rc 0; N_point 620; DNA chain #3 truncated, so no query-only MSA dir).
`$KB/logs/hostA-gpu3-chain-5.sh` (log `hostA-refine-gpu3-r2.log`, launched 11:39) runs 1280 r2, then 1344 r1.
If 1344 succeeds it runs 1344 r2 and r3; otherwise 1280 r3. Any repeat failure stops the chain. At 11:40 both
chains waited in the IO guard: another user's job on GPU2 was reading a 16 GB file from the network filesystem (D state, IO full avg10 ≈ 50).
That job was not touched. V0 N1280-r2 started on GPU3 at 11:54.

**K2B N1408-r1 did not run (GPU0 taken).** Another user's training job started on GPU0 at about 11:48. The ladder's own BUSY check
passed at 11:53:59, apparently before that process held a CUDA context. `run_trial`'s idle check then refused (rc 3,
"GPU 0 is not idle", 6.9 GB in use). No trial dir was created, so this is not a cell outcome. GPU0 was left to that user.
The ladder treated the missing `trial.json` like a failed cell (LADDER-STOP). `hostA-ladder.sh` now exits 8 ("NOTRIAL") in
that case, so a chain aborts for review instead of branching as if the cell had failed (K11). The K2B refinement moved to GPU3, queued
behind the V0 chain: `$KB/logs/hostA-gpu3-chain-7.sh` (log `hostA-refine-gpu3-r3.log`) runs 1408 r1. If that succeeds it runs
1408 r2 and r3; if not, it runs 1344 r1 (then r2 and r3) or 1280 r2 and r3. From 11:48 GPU3 cells share the host with
that GPU0 training job and with the GPU2 job, so their T5 and stage times can be inflated. This contention affects timing
only. Outcome and M1/M2 are per device.

**V0 N1344-r1 GPU_OOM (13:01).** The provisional V0 boundary on a 24 GB A5000 at P = 620 now lies in (1280, 1344).
chain-5 continues with 1280 r3. chain-7 had not started a cell. It was cancelled (own PGID, 13:05) and replaced by
`$KB/logs/hostA-gpu3-chain-8.sh` (log `hostA-refine-gpu3-r4.log`). chain-8 tests K2B at 1344 first, because 1344 is now
the paired level: if r1 succeeds it runs 1344 r2 and r3, then 1408 r1; otherwise it runs 1280 r2 and r3. The 1344 → 1312
refinement (about 2%) was not run.

**V0 at N = 1280: 3/3 consecutive successes (r1 11:03, r2 11:54, r3 13:01, GPU3).** Each cell checked out:
- cell `match` (1280, 620)
- exit 0, 0 missing outputs, no ERR files
- raw outputs finite
- identical M1 (19474915840 B) and M2 (21630025728 B)
- NVML peak 20932 MiB of 24564 MiB

The first failure is N = 1344 (1 GPU_OOM, confidence head; 1408 and 1536 OOM in the same frame). So the V0 support boundary for
*this cell* is 1280 ≤ N_max < 1344. "This cell" means: 9b0l-derived synthetic input, fixed P = 620, RTX A5000 24 GB, driver 595.99.02,
full V0 settings, Mode A, cold cache. Scope of the claim:
- It is runtime capacity evidence only. Synthetic inputs are never accuracy evidence. The CUDA output gate is pending P0-5,
  and the pairwise diffs above are descriptive.
- The failing side rests on single OOM cells. The granularity is 64 tokens.
- It covers one P level (620). The amendment-1 matrix (min/med/max of the manifest and 2× max) was not run.
- It covers one GPU type.

**K2B N1344-r1 GPU_OOM (14:06), byte-identical to V0.** K2B succeeds at 1280 and fails at 1344, the same bracket as V0.
At 1344 and 1536, M1, M2, M3 and the OOM request match V0 to the byte. fpf_k2b therefore does not change the support
boundary at 64-token resolution, as predicted by where the OOM occurs. chain-8 continues with K2B 1280 r2 and r3. That gives
the candidate the same 3-success check at the shared level.

**K2B at N = 1280: 3/3 consecutive successes (r1 11:09 GPU0; r2 14:06 and r3 14:37, GPU3).** Each cell checked out:
- cell `match` (1280, 620)
- complete outputs, finite
- fpf_k2b served all 1920 EMPairformer triatt calls
- M1/M2 byte-identical to V0

chain-8 ended CHAIN-DONE 15:07:58. Both variants therefore have the same support bracket, **1280 ≤ N_max < 1344**, for this cell
(same scope limits as above).

N = 1280 summary, 3 cells per variant (Mode A, cold; compile included; every cell shared the host with another GPU job):

| Variant | EMPairformer s (r1 / r2 / r3; mean) | Forward T3 s (r1 / r2 / r3) | T5 s (r1 / r2 / r3) | M1 / M2 GiB | M3 MiB |
|---|---|---|---|---|---|
| V0 | 1367.8 / 1454.1 / 1448.6; 1423.5 | 1571.6 / 1658.2 / 1666.2 | 1863.7 / 1939.4 / 1912.6 | 18.14 / 20.14 (all 3) | 20932 (all 3) |
| K2B | 1314.7 / 1330.2 / 1335.6; 1326.8 | 1521.4 / 1540.9 / 1546.0 | 1808.7 / 1810.1 / 1827.7 | 18.14 / 20.14 (all 3) | 20932 (all 3) |

- **Speed.** Every K2B EMPairformer time is below every V0 time. The mean is −96.7 s (−6.8 %). The forward is −5.9 %
  (mean 1536.1 vs 1632.0 s). The cleanest pair ran at the same time on GPU0 and GPU3 (V0 r1 vs K2B r1) and gives −3.9 %.
  These are cold cells, contention differed between them, and compile is included. So this is a consistent provisional
  observation, not a warmed speedup claim. The warm Mode B numbers are −2.5 % at N = 805 and −2.7 % at N = 1024.
- **Memory.** M1, M2 and M3 are identical for both variants in all six cells.
- **Outputs (descriptive, `$KB/trials/hostA/L3/9b0l-P620/pairwise-N1280-all.txt`).** All six dumps are finite.
  - V0 repeat pairs: coord median 0.33–0.50 Å, RMS 1.24–1.86 Å.
  - K2B repeat pairs: median 0.32–0.49 Å, RMS 1.19–2.01 Å.
  - V0-vs-K2B pairs: median 0.28–0.53 Å, RMS 1.70–2.19 Å.
  - Logits mean \|Δ\| is 0.050–0.052 in all 15 pairs.

  The cross-variant spread overlaps the within-variant spread, but sits at its upper end in RMS. No G-struct verdict is
  possible without sealed tolerances (P0-5).

**What sets the boundary.** The OOM frame is `empairformer.py:170` (`z += self.pair_transition(z)` in the
confidence-head pairformer) → `primitives.py:153` (`b = self.linear_no_bias_b(y)` in `Transition`). For N < 3200 that
code uses one chunk (`chunk_num = 1 if size < 3200 else 8`), so it allocates two full fp32 hidden tensors of N² × 4·c_z.
Each OOM request equals one of them exactly: N² × 512 × 4 B = 3.45 / 3.78 / 4.50 GiB at N = 1344 / 1408 / 1536.
At 1344 it failed with 3.36 GiB free plus 3.48 GiB reserved but unallocated, so fragmentation decides the exact point.
None of the available trimul/triatt candidates touches this allocation, and neither do M-CHUNK-* (`chunk_size` is not passed
to `Transition`). Two untested levers could move it:
- (a) Chunk `Transition` by memory rather than at the fixed 3200 threshold. This is a source change and a new variant, and
  it needs its own declared policy and review. It is row-wise, so it is exact in math but not bitwise.
- (b) `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. This is a runtime-setting change outside V0.

Neither was run. Both are setting decisions, and (a) also requires editing source, which is not allowed while GPU jobs run.

**2026-11-16 follow-up.** The user chose (a) as the priority option, and (a) is now variant `M-TCHUNK-mem` ([AMENDMENT-2.md](AMENDMENT-2.md) A2.1).
- CPU tests: 21 run, OK, 1 skipped (the CUDA equivalence test). Command: `CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 pixi run -e default --frozen python -m unittest discover -s tests/kernel_bench -p test_routing.py -v`.
- No GPU result yet. At 15:07 all four hostA GPUs ran other users' jobs.
- `$KB/logs/hostA-chain-9.sh` started at 15:21 (hostA PID <n>; log `$KB/logs/hostA-chain-9.log`). It waits until GPU3 or GPU0 is idle on three polls 5 min apart, for up to 48 h. It then runs:
  - a provenance snapshot;
  - the DS4Sci op_bench recheck;
  - the CUDA unit tests;
  - the M-TCHUNK-mem ladder.
- P0-5: the user chose sealed tolerances plus G-op and the trunk comparison. `$KB/logs/hostA-chain-10.sh` runs the 9b0l calibration (D_rep, D_seed, D_swap) and the N 1280 stage-peak attribution after chain-9 (LEDGER P0-5).
- **2026-11-17 GPU run (LEDGER K14):** other users' jobs on hostA GPU0 cut both chains short. Measured:
  - DS4Sci (Microsoft DeepSpeed4Science) tri_att_start at N 96, the path that crashed before K5: SUCCESS, G-op pass, 0.986× V0 op time (`$KB/trials/hostA/L1/r3/tri_att_start.n96.jsonl`).
  - CUDA unit tests: 3 OK, including chunked == unchunked for M-TCHUNK-mem.
  - Not measured: the other 9 DS4Sci cells, the M-TCHUNK-mem ladder, the stage peaks and P0-5. V0 s101-r1 hit GPU_OOM because another process held 14.51 GiB on the same GPU, so it is void.
  - 2026-11-18 (LEDGER K15): the DS4Sci sweep, the stage peaks and P0-5 are running on hostB RTX A6000s, and the M-TCHUNK-mem ladder waits for a hostA A5000. They use `$KB/logs/gpu-chain.sh`, which flags contended cells and reruns them. P0-5 tolerances from hostB are A6000 tolerances (see K15).
  - 2026-11-19 (LEDGER K17): everything on hostB finished SUCCESS.
    - **DS4Sci (Microsoft DeepSpeed4Science) triangle attention** after the K5 fix: it runs and passes G-op at N 96 and 256. It takes 1.23–1.88× the NVIDIA cuEq time at N 256–2048 and saves only 2–3 % peak memory. It is not a speed or capacity option.
    - **Stage peaks** at N 1280, V0, M1 in GB: trunk 15.38, diffusion 11.57, confidence 19.47. The confidence head is the peak stage, as on hostA. On the 48 GB A6000, M-TCHUNK-mem does not chunk (identical to the byte), so its capacity effect still needs the A5000 ladder.
    - **P0-5**: same-seed CUDA repeats differ by 0.72–0.95 Å Ca RMSD (max sample), ≤ 1.5 pLDDT and 1.2 % trunk error, with Kendall τ down to 0.657. Different seeds differ by 10–15 Å. The DS4Sci swap sits inside the repeat noise. The pre-fixed tolerance rule makes V0 fail its own τ gate and leaves the Ca RMSD gate vacuous (14.5 Å), so `tolerances.yaml` is not sealed (decision needed).
    - **M-TCHUNK-mem ladder** (LEDGER K19): the user does not require exact GPU parity. The ladder runs on hostB A6000 with a ballast process that leaves 24550 MiB free, the same as an idle hostA A5000, next to a V0 ladder under the same ballast. Results will be labelled 'A6000 + ballast', not A5000.
    - **Ladder result** (LEDGER K20, 9b0l synthetic, P 620, A6000 + ballast). Calibration first: with a ballast that leaves 24550 MiB free (nvidia-smi free on an idle A5000), V0 fits 1344 but the real hostA A5000 OOMs there. The ballast was recalibrated to 24084 MiB, the most one process can use on the A5000 according to its OOM text.
      - At 24084 on hostC: V0 runs 1280 three times (SUCCESS, M1 19.47 GB byte-identical to hostA) and OOMs at 1344. The OOM text matches hostA's (requests 3.45 GiB, 16.41 GiB allocated by PyTorch). So V0 has 1280 ≤ N_max < 1344, the same as the real A5000.
      - **M-TCHUNK-mem at 24084:** 1280, 1344, 1408 and 1536 SUCCESS; 1536 repeated three times, all SUCCESS and clean, with M1 23.23 GB byte-identical across the three runs. 1664 GPU_OOM. **So 1536 ≤ N_max < 1664**: at least 20 % more tokens and 44 % more N² than V0, on an emulated 24 GB card.
      - The 1536 margin is about 172 MiB (NVML peak minus ballast is 23912 MiB, against 24084 MiB free).
      - The 24550 run gave the same order one rung higher for V0 (1344 ≤ N_max < 1408) and the same M-TCHUNK-mem boundary (1536 ≤ N_max < 1664, 1536 run once).
      - **The next limit** at 1664, in both settings, is the NVIDIA cuEq triangle multiplication (`fused_sigmoid_gated_dual_gemm`, `empairformer.py:144`). It hits after all 5 samples were fitted, so it is [A] in the confidence-head pairformer, plus 2.5 GiB of reserved-but-unallocated memory.
      - **Correctness:** M-TCHUNK-mem only changes the row-chunk count of a per-row MLP, which is exact up to GEMM rounding. The CUDA test `test_cuda_chunked_matches_unchunked` (8 chunks vs 1) passed. The ladder inputs are synthetic and have no reference structure.
      - **Real-structure gate with chunking engaged (LEDGER K21, hostC A6000, real 9b0l, seed 101, one run):** with a ballast leaving 11500 MiB free, M-TCHUNK-mem succeeded. A logging hook showed k = 2 at 998 of 1020 pair `Transition` calls and at all 30 MSA-module calls. Against hostB V0 s101-r1 it **passes every sealed gate**: max Ca RMSD 1.09 Å (V0 r2 vs r1: 0.95), trunk s/z 1.11 % / 1.19 % (V0 r2: 1.14 % / 1.20 %), pLDDT Δ 0, top-1 match, τ 0.68 (V0 r2: 0.73). So the differences are the size of V0's own run-to-run noise. M1 was 8.81 GB vs 9.77 GB for unballasted V0. At 10500 MiB, V0 OOMs in an EMPairformer `Transition` (T5 534 s), while M-TCHUNK-mem (k 2 or 4) gets through the trunk, diffusion and confidence head and OOMs only in the final `pde` `torch.cat` (`cryozeta.py:409`), with 2.47 GiB reserved but unallocated. These are single runs; it is not a capacity claim. Caveat: cross-host (hostC vs hostB); there is no hostC V0 reference because V0 OOMed at 10500.
      - **Past the 1664 limit (LEDGER K22, hostC A6000 + 24084 ballast, 2026-11-22).** There are three variants on top of M-TCHUNK-mem. `-ES` adds the PyTorch caching-allocator setting `expandable_segments:True` through the process env. `-TMV4C` replaces the confidence-site trimul (NVIDIA cuEq `fused_sigmoid_gated_dual_gemm`) with Anthropic FlashPairformer `fpf_trimul_v4` (`uplifting-biomolecular-modeling` @ f4f62fa). `-ES-TMV4C` combines both.
        - **Expandable segments was never on before.** `runner/inference.py:69` sets it, but CUDA is already initialized by the module imports by then. A hostC probe saw `is_expandable=False` segments, so V0 and every earlier trial ran without it. The variant sets it before Python starts, and the CryoZeta source is not changed.
        - **Ladders** (9b0l synthetic, P 620):
          - `-ES`: 1536 ≤ N_max < 1664 (1536 3/3). 1664 still OOMs at the cuEq confidence trimul, but with 283 MiB reserved-but-unallocated instead of 2.53 GiB.
          - `-TMV4C`: 1536 ≤ N_max < 1664 (1536 3/3). 1664 now gets past every confidence trimul and OOMs at `confidence.py:234` `torch.stack(pde_preds)` (3.30 GiB) with 5.16 GiB reserved but unallocated.
          - **`-ES-TMV4C`: 1664 ≤ N_max < 1792**, with 1664 SUCCESS 3/3, clean, M1 22.20 GB in all three, margin 276 MiB. 1792 OOMs inside `fpf_trimul_v4` at the confidence site. It requests 1.53 GiB (one fp32 [N,N,128] tensor) with 174 MiB fragmented, so the memory is really short.
          - So on an emulated 24 GB card the combination goes from 1536 to 1664 tokens (+8 %, +17 % N²), and from V0's 1280 it is +30 % tokens and +69 % N². Neither change does this alone.
        - **Peaks at N 1536** (one run each; M1 allocated / M2 reserved): M-TCHUNK-mem 23.23 / 24.68 GB; `-ES` 23.16 / 23.92; `-TMV4C` 20.20 / 23.47; `-ES-TMV4C` 20.14 / 20.94. `fpf_trimul_v4` at the confidence site cuts the allocated peak by about 3 GB, and expandable segments cuts the reserve.
        - **Correctness, real 9b0l, seed 101, one run each, vs hostB V0 s101-r1:**
          - `-ES-TMV4C` at ballast 11500, with chunking engaged (k = 2), **passes every sealed gate**. Max Ca RMSD 1.36 Å, trunk s/z 1.10 % / 1.19 %, pLDDT Δ 1.5 (V0 r2 vs r1: 1.5), top-1 match, τ 0.85. The counters show the confidence trimul served by `fpf_trimul_v4` 40×, while EMPairformer and MSA stay on cuEq.
          - `-ES` at 10500, where M-TCHUNK-mem OOMed in K21, **SUCCESS and passes every gate** (Ca RMSD 1.23 Å, τ 0.54). M2 was 10.23 GB.
        - **Next limit:** the fp32 pair tensors of the confidence-head pairformer. One more copy can also be avoided: the per-sample `pae`/`pde` lists and their stacked copy coexist, 2 × 3.30 GiB at 1664. Writing each sample into a preallocated output would remove one copy, but it is a source change and is not done.
      - **Past the 1792 limit (LEDGER K23, hostC A6000 + 24084 ballast, 2026-11-23).** Two variants on top of `-ES-TMV4C` (kernels unchanged: Anthropic FlashPairformer `fpf_trimul_v4` @ f4f62fa at the confidence site, NVIDIA cuEq elsewhere).
        - `-OFF` sets `CRYOZETA_CONF_OFFLOAD_MIN_N=0`. A default-preserving source change in `confidence.py` makes CryoZeta's own `N > 2000` threshold, which moves each sample's `pae`/`pde` logits to the host, configurable. With 0 the offload is always on. Values do not change. Preallocation was rejected because the preallocated outputs would occupy 1.53 GiB during the last sample's pairformer, exactly where 1792 OOMed.
        - `-OFF-BF16C` also passes `--skip_amp.confidence_head false`, so the confidence head runs under bf16 autocast instead of fp32. This is lossy.
        - **Ladders** (9b0l synthetic, P 620, from 1792): **both 1920 ≤ N_max < 2048**, with 1920 SUCCESS 3/3, clean. M1 23.79 GB and M2 24.87 GB are identical across all six runs, and the margin is about 14 MiB.
          - 2048 OOMs after about 5 min in **trunk** cycle 0, at the EMPairformer trimul (NVIDIA cuEq `_cuequivariance_triangular_mult`, `empairformer.py:144`). It requests 1 GiB with 151 MiB reserved but unallocated, so the memory is really short.
          - At 1792 the two variants also have identical peaks (M1 21.73 / M2 24.18 GB), so the confidence head no longer binds, and bf16 in it brings **no** capacity. It only shortens the confidence head: 41.6 s fp32 vs 26.1 s bf16 at 1920.
        - **Correctness, real 9b0l, seed 101, ballast 11500, one run each, vs hostB V0 s101-r1:** both **pass every sealed gate**.
          - `-OFF`: max Ca RMSD 0.77 Å, pLDDT Δ 1.25, trunk s/z 1.10 % / 1.19 %, top-1 match, τ 0.64.
          - `-OFF-BF16C`: Ca RMSD 0.86 Å, **pLDDT Δ 2.75**, which is within the 5.5 gate but above the V0 repeat noise of 1.5. Trunk 1.12 % / 1.19 %, top-1 match, τ 0.62. One seed cannot separate the bf16 effect from diffusion noise.
        - So on an emulated 24 GB card `-OFF` goes from 1664 to 1920 tokens (+15 %, +33 % N²). Against V0's 1280 that is +50 % tokens and +125 % N². These are comparisons of lower bounds. `-OFF` is exact and is the one to keep; `-OFF-BF16C` is not recommended.
        - **Next limit:** the trunk trimul (NVIDIA cuEq) at 2048. Not tried: `fpf_trimul_v4` at the trunk site, where TM-V4 already passed every gate at the empairformer site on hostB, or a chunked trimul.
      - **Past 1920 (LEDGER K24, in progress, 2026-11-23).** What binds at 2048 comes from a CUDA memory snapshot at the `-OFF` OOM: 22.51 GiB live, of which NVIDIA cuEq trimul internals 6.00 GiB and the unused `bond_mask` input feature 4.40 GiB. The options list is [NO-RETRAIN-options.md](NO-RETRAIN-options.md).
        - X1 `-OFF-NBM` (`CRYOZETA_DROP_BOND_MASK=1`, exact) and X2 `TMV4A-OFF` (Anthropic FlashPairformer `fpf_trimul_v4` @ f4f62fa at all three trimul sites instead of NVIDIA cuEq) each **pass every sealed gate** (real 9b0l, seed 101, ballast 11500, one run each, vs hostB V0 s101-r1). M1 / M2: `-OFF` 8.72 / 10.25 GB, X1 7.87 / 9.52, X2 8.76 / 9.67.
        - X1 at **N 2048: one SUCCESS** (hostC, M1 21.25 GB, M2 22.79 GB), flagged CONTENDED. One run, not a capacity claim.
        - `-TMV4A-OFF-NBM-K2B` (X1 + X2 + Anthropic FlashPairformer `fpf_k2b` triangle attention at the empairformer site) **fails G-rank-top1** (hostD). V0's top-1 by `recall_ccmask_ca` is 1.33485 and the candidate's is 1.33501, a near-tie flip, but it fails under the sealed rule. Not rerun. K2B leaves capacity unchanged, so it stays out of the capacity stack.
        - The hostD N 2048 ladders (X1, X2) were **voided** by a shared-filesystem outage at about 15:09 (`Errno 107` writing `inference.log`). They are neither failures nor capacity evidence.
        - **X1+X2 `-TMV4A-OFF-NBM` (hostB GPU0, 2026-11-24/25): 2304 ≤ N_max < 2432.** Kernels: Anthropic FlashPairformer `fpf_trimul_v4` @ f4f62fa at all three trimul sites (no NVIDIA cuEq trimul), plus X1.
          - Gate: real 9b0l, seed 101, ballast 11500, vs hostB V0 s101-r1, **passes every sealed gate**. Max Ca RMSD 0.97 Å, trunk s/z 1.10 % / 1.19 %, top-1 match.
          - Ladder (9b0l synthetic, P 620, ballast 24084, step 128): 2048, 2176 and 2304 SUCCESS, clean. **2304 SUCCESS 3/3, clean**, with M1 23.62 GB and M2 24.35 GB identical in all three. T5 4153 / 4294 / 4322 s, measured under ballast, so not a speed figure.
          - 2432 OOMs about 77 min in, in the **confidence head** pairformer trimul (`confidence.py:275` → `empairformer.py:144` → `fpf_trimul_v4`). It requests 2.82 GiB with 251 MiB reserved but unallocated, so the memory is really short. The trunk got through 2432.
          - So on an emulated 24 GB card, X1+X2 goes from `-OFF`'s 1920 to 2304 tokens (+20 %, +44 % N²). Against V0's 1280 that is +80 % tokens and +224 % N². These are comparisons of lower bounds.
          - **Next limit:** the confidence-head pairformer's fp32 pair tensors at about 2432. X3/X4 (`-X34`) moved off hostC GPU0 on 2026-11-25 11:35 (gate on hostB GPU2, ladder on hostC GPU1).
        - **X1–X4 `-X34` gate (hostB GPU2, 2026-11-25 11:47–12:13): fails G-rank-top1.** Real 9b0l, seed 101, ballast 11500, vs hostB V0 s101-r1. SUCCESS, clean, M3 7578 MiB. Max Ca RMSD 0.99 Å, trunk s/z 1.13 % / 1.20 %, confidence abs max 0.004, τ 0.62: those gates pass. Top-1 by `recall_ccmask_ca` is svd_0.8 sample 2 (1.334515) and teaser sample 2 is second (1.334273). V0's top-1 is teaser sample 2 (1.334852) with svd_0.8 sample 2 second (1.334603). So the same sample wins, and the two fitting methods swap places on a 0.00024 gap, which is the same size as V0's own 0.00025 gap. X3/X4 only free tensors and move modules between devices, so this is diffusion nondeterminism. It still fails under the sealed rule. Like K2B, it was not rerun. Output in `$KB/analysis/gates/M-TCHUNK-mem-ES-TMV4A-OFF-NBM-X34-s101-r1.json`.
        - **Repeats r2, r3** (hostB GPU2, 13:30–14:28, same setup, approved after r1 failed): both SUCCESS, clean, M3 7578 MiB, and both **pass every sealed gate** vs V0 s101-r1. r2: max Ca RMSD 1.77 Å, pLDDT Δ 2.75, trunk s/z 1.10 % / 1.18 %, top-1 teaser sample 2 (1.335106), τ 0.62. r3: max Ca RMSD 0.81 Å, pLDDT Δ 1.25, trunk s/z 1.08 % / 1.19 %, top-1 teaser sample 2 (1.334445), τ 0.66. So top-1 holds in 2 of 3 runs; the r1 flip is a near-tie. r2's pLDDT Δ is within the 5.5 gate but above V0's repeat noise of 1.5. Under the sealed rule r1 still fails. **2026-11-26, the user accepted X1–X4 as gated ("算通过") from these three runs.** The sealed verdict for r1 stays FAIL, so report the acceptance with it.
        - **X1–X4 ladder (hostC GPU1, ballast 24084, step 128, from 2048):** 2048 SUCCESS (M3 19992 MiB), 2176 SUCCESS (20296), 2304 SUCCESS (22314), all clean. 2432 GPU_OOM after about 73 min, at the same site as X1+X2 (`confidence.py:275` → `empairformer.py:144` → `fpf_trimul_v4`). It requests 2.82 GiB with 2.25 GiB free. PyTorch-allocated at the OOM was 20.68 GiB, against X1+X2's 21.52 GiB: X3/X4 save 0.84 GiB there, about 0.57 GiB short of fitting. X4 brings the heads back before the confidence head runs, so mainly X3 helps at this point. At step 128 this is the same bracket as X1+X2, **2304 ≤ N_max < 2432**. **2304 repeated 3/3** (SUCCESS, clean, M3 22314 MiB each; the chain ended 18:28, REPEAT-OK). On the emulated 24 GB card, X1–X4 capacity is 2304 ≤ N_max < 2432, the same as X1+X2 at this step. The user saw no need for a 2368 cell ("没必要了"), so the bracket stays at step 128.
        - X3 and X4 are now in the source as default-off switches (`CRYOZETA_FREE_EMBED_INPUTS`, `CRYOZETA_HEADS_LATE`, exact; 1.43 + 0.81 GiB at 2048 by measured sizes). Planned, each as gate then ladder from 2048: X1+X2 (`-TMV4A-OFF-NBM`, on hostB GPU0) and X1-X4 (`-X34`, on hostC GPU0). The X1-alone ladder is dropped; X1+X2 covers it. Both have run (results above, 2026-11-25). hostD cannot be used: since its 17:05 reboot on 2026-11-23 it cannot shared-`mmap` files on the shared network mount (ENODEV), which pixi and safetensors need (LEDGER K24).
      - **Real 80 GB A100 (LEDGER K25, in progress since 2026-11-24).** an A100-80GB partition, a preemptible QOS (4 h cap), one GPU per job, no ballast, same 9b0l synthetic inputs (P 620). The code is commit `d310d7f`, branch `kernel-bench-memory` on kiharalab/CryoZeta.
        - Variants: V0, and the stable `-ES-TMV4C-OFF-NBM` (NVIDIA cuEq, plus Anthropic FlashPairformer `fpf_trimul_v4` at the confidence site, plus X1).
        - r1 at V0 2048–2816 and stable 3072–4096 (step 256), and a real-9b0l gate for each. Then refine by 128, and 3/3 at the highest success.
        - The first start (2026-11-25 10:28) failed in preflight on a setup gap: TEASER++ was not built and the CCD files were missing. Both are fixed and the cells resubmitted (LEDGER K25). A run that hits the 4 h preemptible limit counts as TIMEOUT, never SUCCESS.
        - **Real 9b0l, seed 101 (2026-11-26):** V0 SUCCESS, M3 13371 MiB, 943 s; stable SUCCESS, M3 10321 MiB, 913 s. Gated on CPU with the sealed K18 rule. Same-card (A100 stable vs A100 V0): all pass, Ca RMSD max 1.29 Å, confidence max 0.75, trunk s 1.11 % / z 1.19 %, top-1 match, τ 0.85. Cross-card, report only: A100 stable vs A6000 (hostB) V0 all pass (1.84 Å, 1.25, τ 0.81); A100 V0 vs A6000 V0 all pass (1.05 Å, 2.0, τ 0.85), so card-to-card drift is inside the A6000-calibrated noise band.
        - **Ladder r1 (2026-11-26), M3 peak / T5 wall:**
          - V0: 2048 SUCCESS 49383 MiB / 2189 s; 2304 SUCCESS 66781 / 2562; 2560 SUCCESS 80977 / 3334; 2816 GPU_OOM (asked for 15.12 GiB with 14.79 free). 2688 SUCCESS 70957 / 3467; 2560 r2/r3 SUCCESS; 2688 r2/r3 SUCCESS. **V0 on a real 80 GB A100: 2688 ≤ N_max < 2816, 3/3 at 2688.**
          - Stable: 3072 SUCCESS 53499 / 4497; 3328 SUCCESS 46323 / 5527; 3584 SUCCESS 53239 / 6463; 3840 SUCCESS 60861 / 7501; 4096 SUCCESS 68921 / 7989; 4352 SUCCESS 77421 / 9562 (added at the user's call); 4608 GPU_OOM (asked 10.12 GiB, 4.92 free), flagged CONTENDED by the M3 − M2 rule although no other app was on the GPU (the OOM message accounts for all used memory). 4352 r2/r3 SUCCESS (77421 / 9841 and 9700). **Stable on a real 80 GB A100: 4352 ≤ N_max < 4608, 3/3 at 4352**, 1.6× the V0 limit. M3 is not monotone in N because `M-TCHUNK-mem` sizes its chunks from free memory (not checked).
      - **Not measured on a real A5000.** All of this is A6000 + ballast, qualitative per the user (K19).
    - **Gates sealed** (LEDGER K18, user-approved rule chosen after reading the data): Ca RMSD ≤ 2.36 Å, confidence ≤ 5.5, trunk ≤ 2.39 %, top-1 must match, τ report-only. They are A6000-calibrated.
    - **K2B** (FlashPairformer `fpf_k2b`, Anthropic `uplifting-biomolecular-modeling` @ f4f62fa) and **TM-V4** (`fpf_trimul_v4`, same source) on the empairformer site **pass every gate** on hostB: 9b0l, seed 101, vs V0. Max-sample Ca RMSD is 1.14 / 1.12 Å, trunk error 1.19 %, top-1 kept, the same as V0's own repeat noise. Scope: one entry, one seed, A6000.
    - **Multi-entry gate, X1+X2 vs V0 (LEDGER K27, 2026-11-28, A6000, seed 101, one run each).** Four ligand-free real entries, N_token 816–1740 (9Y6N, 10DN, 36PP, 9X28). Every cell SUCCESS. X1+X2 peak memory −27 / −28 / −19 / −29 %, wall −5 / −4 / 0 / +5 %.
      - Sealed K18 gate: 9X28 passes all. 9Y6N fails only top-1 (a near-tie). 36PP fails coords (2 of 5 samples, 17.4 and 4.7 Å) and top-1, trunk passes. 10DN fails coords (37–38 Å, all samples), trunk z (45.6 %) and top-1.
      - V0 against its own repeat (one clean repeat per entry, plus one CONTENDED on 9Y6N and 10DN):
        - 9Y6N: V0 flips the same near-tie in 1 of 2 repeats.
        - 36PP: V0 flips the same top-1 too. Both V0 runs agree on samples 1 and 4, so X1+X2's 17 / 5 Å there is its own. Map-fit recall stays in V0's band.
        - 10DN: V0 is bimodal. The first run is 45.7 % (trunk z) and 38 Å from both repeats, which agree to 1.3 %. X1+X2 sits with the repeats but 6.2 % off, with one sample at 13 Å.
      - Reading: the top-1 flips are V0 noise. The 36PP and 10DN deviations are not, so X1+X2 (`fpf_trimul_v4` at the msa and empairformer sites) is not shown equivalent on real multi-entry data. The stable K25 variant (`fpf_trimul_v4` at the confidence site only) was not run here. On 2026-11-28 the user accepted X2 anyway (LEDGER K28).
- The open brainstorm on raising N_token is in [EXPLORATION-token-capacity.md](EXPLORATION-token-capacity.md). It contains no measurements beyond the ones cited here.

(b) is still not run.

At N = 1024, K2B's differences from V0 (median 1.4–1.8 Å, RMS 9–11 Å) are larger than V0's run-to-run difference
(median 1.12 Å, RMS 4.3 Å; one pair). At N = 805 they were the same size. With one cell per variant this
cannot be attributed. A G-struct verdict still needs sealed tolerances (P0-5), and the Mode B pairs below
add same-process V0 and K2B iteration pairs at N = 1024.

Mode B M2 check at N = 1024 (GPU3, `$KB/trials/hostA/P-CZ/9b0l-syn1024-P620/<V>/modeB-r2`, summary via `modeb_summary.py`):

| Variant | Outcome | iter0 cold s (compile, new cache) | M1 / M2 cold GiB | warm T3 s | warm EMPairformer / diffusion / fitting / confidence s | M1 / M2 warm GiB | M3 MiB | Served (empairformer) |
|---|---|---|---|---|---|---|---|---|
| V0 | SUCCESS, exit 0, 0 missing, cell match | 1188.0 (9, 74) | 12.15 / 14.68 | 885.1 | 703.2 / 27.0 / 146.0 / 7.8 | 12.15 / 13.16 | 15338 | NVIDIA cuEq trimul 1920, cuEq triatt 1920; APB cueq→torch `ineligible_shape` (baseline rescue, out of scope) |
| K-TA-K2B | SUCCESS, exit 0, 0 missing, cell match, scope_coverage 1.0 | 1184.6 (12, 98) | 12.15 / 14.68 | 883.2 | 684.5 (−18.7, −2.7 %) / 27.0 / 162.8 / 7.8 | 12.15 / 16.77 | 17472 | FlashPairformer `fpf_k2b` triatt 1920 (in scope, no fallback); cuEq trimul 1920; same APB rescues |

K2B started at 09:46:19, ended at 10:27:06 (MODEB-DONE). GPU0 ran V0 N1536-r1 on the same host for the whole cell.

- **Warm speed.** EMPairformer is 18.7 s (−2.7 %) faster with fpf_k2b at N = 1024, close to the −2.5 % at N = 805.
  Warm T3 is about equal (883.2 vs 885.1 s) because fitting, which these kernels do not touch, took 16.8 s longer.
  n = 1 per arm, so this is an observation, not a speedup claim.
- **Memory.** Live tensors (M1) are the same for both variants, 12.15 GiB, cold and warm, and also in the
  Mode A cells (`M1_invocation` 13041636352 B for V0 r1, V0 r3 and K2B r2). Only the allocator's reserve
  differs: K2B's warm M2 is 16.77 GiB vs 13.16 GiB, and its M3 is 17472 vs 15338 MiB. So fpf_k2b does not
  need more live memory at N = 1024. The extra reserve is caching or fragmentation, not demand. It can still
  matter at the capacity edge, before the allocator frees cached blocks on OOM.
- **Output differences (descriptive, `$KB/trials/hostA/P-CZ/9b0l-syn1024-P620/pairwise-modeB-r2.txt`).** All four
  dumps are finite. V0 it0 vs V0 it1 (same process): coord median 1.07 Å, RMS 3.94 Å, max 47.9 Å, logits mean
  \|Δ\| 0.063. The four V0-vs-K2B pairs: median 0.63–1.10 Å, RMS 2.50–7.26 Å, max 28.8–51.8 Å, logits 0.053–0.068.
  K2B it0 vs it1: 1.07 / 5.11 / 41.4 Å, 0.064. V0-vs-K2B is the same size as V0's own forward-to-forward
  difference. The larger Mode A pair above (RMS 9–11 Å) therefore looks like run-to-run spread, but no G-struct
  verdict without sealed tolerances (P0-5).

What can be said so far:

- **Speed (new driver).** K2B's cold EMPairformer at N = 1024 is 995.9 s vs V0's 1039.8 s (−4.2 %), and the whole
  cold forward is 1176.3 vs 1228.5 s. That is one cell each with compile included and with GPU3 active on the same host,
  so it is not a speedup claim. Warm numbers come from Mode B.
- **Memory (new driver).** K2B r2 and V0 r3 both peak at 15338 MiB NVML in Mode A, with the same M1 (12.15 GiB)
  and M2 (14.68 GiB). In Mode B, K2B's warm reserve is higher (see above) but its live tensors are the same.
  The r1 excess was allocator reserve that varies run to run, not live memory.
- **Memory (r1, old driver, superseded).** K2B's NVML peak was about 3 GB above V0's at N = 1024. At N = 805 the two were equal
  (13132 vs 13130 MiB). The jump appears at 14:19, during the first EMPairformer cycle and before another
  user's process (ppunuru, 444 MiB) joined GPU0 at about 14:24. NVML is device-wide, and the r1 cell was killed
  before writing M1/M2, so caching-allocator reserve and Triton autotune workspace could not be separated from live tensors.
  The Mode B check at N = 1024 above resolves it: the same live memory, a variable reserve.

## Revision under test

| Item | Value |
|---|---|
| Audited baseline | `c544c6bbd0366951d8b4e0bc4c0da547836f7b8f` (branch `kernel-bench`, HEAD) |
| Implementation | Uncommitted working-tree diff plus new untracked files. Both are hashed in `$KB/provenance/workstation-20261106-gaps/` (`diff_vs_baseline.patch`, `snapshot.json` → `diff_sha256`, `untracked_sha256`), taken after the last edit of this file; this file is part of the hashed diff, so the hashes live only in `snapshot.json`. Earlier snapshots (`workstation-20261106-1234`, `-final`, `-followup`) are kept and are superseded. |
| Environment | CryoZeta Pixi `default`, frozen lock: torch 2.7.1+cu126, triton 3.3.0, cuequivariance 0.9.1, Python 3.12. The `pixi.lock`, `uv.lock` and `pyproject.toml` sha256 values match the plan. No dependency changed. |
| Raw records | `$KB` (outside git): `logs/`, `trials/cpu-smoke/`, `provenance/` |

## CPU validation (contract only)

**Amendment-1 follow-up** ([AMENDMENT-1.md](AMENDMENT-1.md)). The last two sections of this file hold the review-pass and re-review CLI checks and the final test log.

**Previous pass: 27/27 unit tests pass** (`$KB/logs/unittest-3.log`). They cover:
- each harness's `--help`, invalid config and dry-run, plus refusal of an existing `--out`;
- outcome classification, timeout, strict scope, N/A coverage for an empty scope, and the V0 cueq→DS4Sci and cueq→torch rescues;
- loading every variant, with every `cli_args` resolved through the real config parser;
- M-CLEAR per-stack block-clear counts at N = 2000 and 2001;
- the C-CAT cache;
- the FPF K2B and TriMul-v4 adapter contracts against fp64 references;
- `op_bench` keeping `expand()` views.

**`op_bench` CPU smoke** (`$KB/trials/cpu-smoke/op_bench_cpu.{sh,log,jsonl}`):
- Setup: real checkpoint, fp32, N = 24, N_point = 16, synthetic seeded inputs, `--strict`.
- The CPU timings and `speed_ratio` values are **not performance data**.

| Module (block 0) | Baseline → candidate | Outcome | G-op |
|---|---|---|---|
| EM `tri_mul_out` | torch → torch | SUCCESS | pass |
| EM `tri_att_start` | torch → lma | SUCCESS | pass |
| EM `outer_row_product_mean` | einsum → chunked | SUCCESS | pass |
| EM `residue_row_attn` | torch → lma | SUCCESS | pass |
| EM `point_column_attn` | torch → lma | SUCCESS | pass |
| EM `attention_pair_bias` | torch → torch | SUCCESS | pass |
| Diffusion `attention_pair_bias` (N_sample 2) | torch → torch | SUCCESS | pass |
| EM `tri_att_start`, strict cueq on CPU | torch → cueq | **UNSUPPORTED** (`ineligible_shape`), not SUCCESS | n/a |

The smoke found one harness bug, now fixed. `op_bench` gave diffusion APB a z without the sample dimension, which crashed both arms with an `AssertionError` in `Attention`. The pipeline actually passes a stride-0 `expand` of shape [N_sample, N, N, c_z]. `synth_inputs` now builds that view, and `clone` keeps such views instead of materializing N_sample copies, which would overstate peak memory.

The CPU checks do not cover cueq, DS4Sci, TileLang or FPF kernel numerics. Those need a GPU (L1).

## Findings from source inspection

- **M-SDC changes RNG consumption.** `sample_diffusion` (`generator.py:145-238`) draws the init noise, the per-step noise and the augmentation once per chunk. With chunk 1 (V0), sample k gets different random numbers than it does with chunk 5. The plan's precondition for M-SDC ("sample RNG consumption must stay the same") therefore does not hold. **M-SDC-5 is deferred** (amendment 1, A1.3): it is out of the sample-paired benchmark and winner selection, and the harness refuses both its trial and its gates. The earlier suggestion to judge it against D_seed is withdrawn; a future study needs its own reviewed protocol.
- **Inference support points are capped** (amendment 1, A1.1): P = min(n_detected, N_token, 2000). The plan's "not capped" statement was wrong, and the old `synth_inputs` did not reduce P below the detection count.
- **FPF on the 5090 is unverified by construction** (plan §3.1). There is no cc 12.0 tuning row, so it uses `SAFE_ROWS`. The TMA cells need Triton ≥ 3.4, but the `default` env has 3.3.0. Label every 5090 FPF result as U.

## Speed track: recommendation

**No end-to-end speed winner.** The kernels kept are kept for memory:
- Real entries (LEDGER K27, A6000, 816–1740 tokens): X1+X2 changes wall time by −5 % to +5 % against V0.
- Real 9b0l (N 805) on the 80 GB A100 (LEDGER K25): the stable variant takes 913 s, V0 943 s.
- Exact speed work still open: S2, parallel fitting ([NO-RETRAIN-options.md](NO-RETRAIN-options.md)).

## Capacity track: recommendation

**For long inputs use `M-TCHUNK-mem-ES-TMV4A-OFF-NBM-X34` (X1–X4).** The user accepted it on 2026-11-28 (LEDGER K28). It is opt-in: see README, "Recommended configuration".

Measured limits, synthetic 9b0l at P 620. Each cell gives "largest N that succeeded ≤ N_max < smallest N that failed":

| Card | V0 | Stable (`-TMV4C-OFF-NBM`) | X1+X2, X1–X4 |
|---|---|---|---|
| 24 GB (A5000; A6000 with ballast) | 1280 ≤ N_max < 1344 | not laddered with NBM (`-TMV4C-OFF` without it: 1920) | 2304 ≤ N_max < 2432 |
| A100 80 GB | 2688 ≤ N_max < 2816 | 4352 ≤ N_max < 4608 | not laddered |

- What binds at the top is the confidence-head triangle multiplication (`fpf_trimul_v4`, confidence site): at 2432 for X1+X2 on 24 GB, and at 4608 for the stable variant on the A100. There the trunk already gets through.
- Real inputs carry P = min(n_detected, N_token, 2000) points. Above 2000 tokens that is up to 2000 points, against the 620 used here. The real-target ladder on the A100 (9ey0 2268, 9nb5 3036, 6n1q 4088, 8ygm 4752, all P 2000) is LEDGER K28.

## What it takes to produce numbers

1. Authorize a GPU type and an allocation or session (R1).
2. Authorize the code transfer (R2):
   - commit `kernel-bench` and push to GitHub;
   - `git pull` on the cluster;
   - run `pixi run -e default --frozen build-teaser` there (R4);
   - fetch the assets and check them with `provenance snapshot --assets-sha` (R3).
3. In the allocation, run a smoke first:
   - `v0_regression --route` with the V0 GPU flags (README, "V0 regression"): the CUDA-pending half of the gaps pass;
   - a single `run_trial --dry-run`, then one V0 P-CZ trial on 9b0l (README, Phase 0).
4. Then P0-2 detection freeze, then P0-3 through P0-8, then L1 → L2 → L3 as above.

## Reproduction (CPU)

```bash
cd $REPO; KB=/path/to/kernel-bench; P="pixi run -e default --frozen"
CUDA_VISIBLE_DEVICES= CRYOZETA_ASSETS_DIR=$KB/assets $P python -m unittest discover -s tests/kernel_bench -p "test_*.py" -v
CUDA_VISIBLE_DEVICES= $P bash $KB/trials/cpu-smoke/op_bench_cpu.sh > $KB/trials/cpu-smoke/op_bench_cpu.log 2>&1
$P python -m cryozeta.kernel_bench.provenance snapshot --repo . --assets $KB/assets \
    --assets-sha $KB/provenance/assets.sha256 --out $KB/provenance/$(hostname)-$(date +%Y%m%d-%H%M)
```

The GPU commands for every phase are in [README.md](README.md).

## Implementation review pass (CPU, 2026-11-06)

The findings and their resolutions are listed in [LEDGER.md](LEDGER.md) ("Implementation review"). Everything below is contract evidence on CPU, not performance or accuracy data.

**P axis, end to end through the real featurizer** (`$KB/trials/cpu-followup/cli_checks-2.{sh,log}`, fresh tree `r2/`):

| Case | Detection | Requested (N_token, P) | Model input (`record_input`) | `check_cell` |
|---|---|---|---|---|
| `r2/sub` | 900 points, seeded subset | (600, 400) | N_token 600, P 400, `p` [400, 400, 114], `pz` [400, 600, 125] | match |
| `r2/tile` | 250 points, tiled | (600, 600) | N_token 600, P 600, `p` [600, 600, 114], `pz` [600, 600, 125] | match |

- **Labels and caches.** Both dry-runs label the trial `synthetic: true` without `--synthetic`, and place `CUEQ_TRITON_CACHE_DIR`/`CUDA_CACHE_PATH` under the trial's own cache directory.
- **No `em_file`.** Neither generated entry carries `em_file`.

**Refusals exercised through the CLIs** (same log):

| Input | Exit | Message |
|---|---|---|
| `synth_inputs --n-point 601` at N_token 600 | 2 | "not realizable" |
| Variant setting `--seeds` | 1 | "harness-controlled settings cannot be set by a variant" |
| Variant setting `CUEQ_TRITON_CACHE_DIR` | 1 | "harness-controlled settings cannot be set by a variant" |
| Variant with chunk 5 and no `sample_paired: false` | 1 | "cannot be sample_paired (amendment 1, A1.3)" |
| Frozen tree with one extra file | 2 | "differs from its freeze-det manifest (tree_sha256)" |
| `compare_outputs` with `M-SDC-5` | 2 | "not sample-paired" |
| `compare_outputs` with V0, no `trial.json` | 2 | "needs run_trial outputs" |
| `compare_outputs` to an existing `--out` | 2 | "results are never overwritten" |
| `op_bench` with an unknown module | 1 | `AttributeError` traceback, no record written (loud but unformatted) |

All five CLIs exit 0 on `--help` (`synth_inputs`, `run_trial`, `compare_outputs`, `op_bench`, `freeze-det`).

**`op_bench` CPU smoke rerun** (`$KB/trials/cpu-smoke/op_bench_cpu-2.{sh,log,jsonl}`; the first run's files are kept unchanged):
- The same 8 cases as above give the same outcome classes.
- G-op passes for the 7 SUCCESS pairs.
- The strict CPU cueq case is UNSUPPORTED, with no G-op and no `speed_ratio`.

**Side effect found and fixed.** The helper `featurize.py` set `_COUNTERS_PATH`, so `kernel_routing`'s atexit dump wrote `./unused` into the repo root. The file was moved to `r2/featurize-tile.counters.json` and the helper now clears the path. A rerun left nothing behind (logged).

## Re-review fixes (CPU, 2026-11-06)

Reviewer C's findings and resolutions are rows C1–C6 in [LEDGER.md](LEDGER.md). CLI evidence: `$KB/trials/cpu-followup/cli_checks-3.{sh,log}`, fresh tree `r3/`. It reads the frozen `r2/sub` tree without writing to it. The `trial.json` files under `r3/cmp/` are hand-written fixtures that exercise argument checks; they are not run results.

| Input | Exit | Message |
|---|---|---|
| Variant `cli_args: ["--seeds=7"]` | 1 | "cli_args must be '--flag value' pairs (no '=' form, no spaces inside a value)" |
| Variant `--load_checkpoint` (prefix of `--load_checkpoint_path`) | 1 | "harness-controlled settings (or abbreviations of them)": `['--load_checkpoint']` |
| Variant `[--use_opm_tilelang, "false --seeds 7"]` (re-split by the config parser) | 1 | same, `['--seeds']` |
| Variant `--infer_setting.sample_diffusion_chunk 5` (prefix) | 1 | same, `['--infer_setting.sample_diffusion_chunk']` |
| Variant env `CRYOZETA_SYNC_STAGE_TIMER` | 1 | same, `['CRYOZETA_SYNC_STAGE_TIMER']` |
| `V0` and `M-LMA` dry-run (happy path) | 0 | plan written |
| `compare_outputs`, candidate outcome NUMERICAL | 2 | "gates need two SUCCESS trials; outcome {'cand': 'NUMERICAL'}" |
| `compare_outputs`, ref seed 102 vs cand 101 | 2 | "ref and cand trials are not paired (…)" |
| `compare_outputs`, ref ran `M-LMA` | 2 | "--ref ran M-LMA, not the candidate's baseline V0" |
| `compare_outputs --seed 7` | 2 | "--seed 7 / --name 9b0l are not the trials' seed and entries" |
| `compare_outputs`, paired SUCCESS fixtures, unsealed template | 0 | `gates: "not evaluated (tolerances unsealed)"`; no outputs to compare, so nothing is reported as passing |

A NaN delta now fails G-coord, G-conf and G-trunk (`MSdc.test_sealed_needs_calibration`, `MSdc.test_confidence_keeps_nan`). Under P-LC, variants with `cli_args` were a run-time CRASH (C5); the gaps pass fixed this (next section).

Test log of this pass: 49/49 (`$KB/logs/unittest-6.log`); superseded by the gaps section below (`unittest-9`, 56/56). Besides an open3d ImportWarning and two warnings from unmodified upstream data code, the full run prints one ResourceWarning, from the vendored FPF `triatt_k2b.py:504` (third-party code, left unchanged). Earlier logs are kept.

## Gaps follow-up (CPU, 2026-11-06)

Items, findings G1–G6, reviewer D's D1–D5 and reviewer E's E1–E2 are in [LEDGER.md](LEDGER.md) ("Gaps follow-up"). Everything here ran on CPU with `CUDA_VISIBLE_DEVICES=` empty. It is correctness and contract evidence: **no GPU process ran, and no latency, memory or token-length number exists.**

**V0 numerical regression vs `c544c6b`** (`v0_regression`, `$KB/trials/v0-regression/`). Each run uses the same checkpoint, inputs, settings and seed 101 in every arm; the baseline is a `git archive` of `c544c6b`, and the working tree was not reset. Fixtures (synthetic, never accuracy evidence): `fix2/s96_real` is 9b0l cut to 96 tokens (protein chain kept to 85 residues) with 64 detection points and a query-only MSA; `fix3/s96_msa` is the same entry with the real 9b0l protein MSA cropped to those residues. N_cycle 2, N_sample 2, N_step 5.

| Run | Settings | `patched`, `v0trial`, `repeat` vs baseline | Route arms (strict, every site reached) |
|---|---|---|---|
| run2 | s96_real, every `use_*` false | bitwise (`run2-recompare.json`) | OPM bitwise; rowcol max \|diff\| 3.22e-4; APB failed under the old cueq request (G3); TriMul not reached |
| run4 | s96_real, cueq TriMul on, cueq APB requested (served torch: ineligible shape) | bitwise (16 files) | APB identity, OPM einsum, rowcol LMA, TriMul C-CAT at 3 sites: 0 routing problems; bitwise except rowcol (3.22e-4) |
| run5 | `fix3/s96_msa`, every `use_*` false | bitwise | APB, OPM, TriMul (C-CAT vs torch) bitwise; rowcol 1.158e-3 |
| run6 | run4 settings, final `src` tree | bitwise (16 files); `cpu_shim` true | APB identity, OPM einsum, TriMul C-CAT at 3 sites: bitwise; rowcol LMA 3.22e-4; 0 routing problems; `route_missing` [] |

Row/column LMA is a different summation order, so it is reported, not gated, here. On CPU, "cueq" TriMul ran cuEquivariance's non-CUDA path.

**Unit tests and CLI checks.** 56/56 pass, none skipped: `$KB/logs/unittest-8.log` and, after reviewer E's fixes, `$KB/logs/unittest-9.log` (final; `CRYOZETA_ASSETS_DIR` set). Seven tests are new since `unittest-6`: `V0Regression` (comparator; route selection, including mixed sites), `CCatForward`, `PlcStageArgs` (two), `Cli.test_plc_stage_args` and `Inputs.test_query_only_msa`. `Variants.test_guards` and `test_one_attention_algorithm` were extended. `unittest-7` failed and is kept: one stale assertion (the old preflight message) and `RealFeaturizer` skipped because `CRYOZETA_ASSETS_DIR` was unset; both fixed. `cli_checks-5` (`$KB/trials/cpu-followup/cli_checks-5.{sh,log}`, fresh tree `r5/`):

| Input | Exit | Message / result |
|---|---|---|
| P-LC and P-CZ variant: `--use_lma true` with DS4Sci/cueq attention on, or `--use_memory_efficient_kernel true` | 2 | "preflight …: error: at most one attention algorithm …"; no trial directory |
| P-LC and P-CZ variant `--use_flash true` | 2 | "use_flash: no config passes it to the model (a silent no-op)" |
| P-CZ variant `--atom_arrays_dir /x` | 2 | "unrecognized arguments: --atom_arrays_dir /x" |
| P-LC and P-CZ variant `--model.input_embedder.p_dim 3` | 1 | "harness-controlled settings … cannot be set by a variant" |
| P-LC `M-LMA`, `M-CHUNK-128`, `V0`; P-CZ `V0` (dry-run) | 0 | stage tails `['--','--use_lma','true']`, `['--','--infer_setting.chunk_size','128']`, `[]` |
| `cryozeta-cycle-predict` directly: `-- --use_memory_efficient_kernel true`, `-- --use_flash true`, `-- --use_lma true --use_memory_efficient_kernel true` | 2 | the same parser errors; no `--dump_root` created |
| `M-SDC-5` | 2 | "M-SDC-5 is deferred" (unchanged) |

Reviewer E (LEDGER, E1–E2) re-reviewed the D fixes. Its two findings were fixed after run6, in parse-time checks only, and checked in `cli_checks-6` (`$KB/trials/cpu-followup/cli_checks-6.{sh,log}`, fresh tree `r6/`):

| Input | Exit | Message / result |
|---|---|---|
| P-CZ variant `--use_memory_efficient_kernel true` with DS4Sci/cueq off; `cryozeta-cycle-predict -- --use_memory_efficient_kernel true` with natives off | 2 | "use_memory_efficient_kernel: Attention has no such kernel and runs torch attention (a silent no-op)"; no trial or dump directory |
| P-CZ and P-LC variant `--num_workers 8`; P-CZ `--num_work 8` (prefix) | 1 | "harness-controlled settings (or abbreviations of them) cannot be set by a variant" |
| P-LC `M-LMA`; P-CZ `V0`, `M-LMA` (dry-run) | 0 | plan written |

**CUDA-pending** (needs an authorized GPU):
- `v0_regression --route` with the V0 GPU flags and env (README, "V0 numerical regression"): DS4Sci + cueq attention, `fast_layernorm`, no CPU shim;
- CUDA kernel numerics: cueq TriMul/attention, DS4Sci, tilelang OPM, `fast_layernorm`, FPF K2B, TriMul-v4;
- the non-dry-run `run_trial` path, `record_input` inside a GPU forward, and a real P-LC run.
