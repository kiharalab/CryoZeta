# What is left without retraining (capacity and speed)

Date: 2026-11-23 (LEDGER K24). The user asked for this list. It updates the 2026-11-16 brainstorm in
[EXPLORATION-token-capacity.md](EXPLORATION-token-capacity.md) with what has since been measured. Every item keeps the pretrained
weights. Evidence labels as there: [M] measured, [A] arithmetic from measured numbers, [L] literature, [H] hypothesis.
Nothing marked [A] or [H] is a capacity or speed claim.

Kernel sources:
- NVIDIA cuEquivariance (cuEq);
- Microsoft DeepSpeed4Science (DS4Sci);
- Anthropic FlashPairformer (FPF) from `uplifting-biomolecular-modeling` @ f4f62fa, with `fpf_trimul_v4` (TM-V4) and `fpf_k2b` (K2B).

## Where things stand

All capacity numbers are for an emulated 24 GB card: an A6000 with a ballast leaving 24084 MiB, as in K20. The input is
9b0l synthetic with P 620. Each value is the highest N with 3/3 successes [M].

| Variant (cumulative) | N_max |
|---|---|
| V0 | 1280 |
| + memory-sized Transition chunks (`M-TCHUNK-mem`), or `expandable_segments`, or TM-V4 at the confidence site (each alone) | 1536 |
| `-ES-TMV4C` | 1664 |
| `-ES-TMV4C-OFF` (pae/pde to host) | 1920 |
| `-TMV4A-OFF-NBM` (X1 + X2: no `bond_mask`, Anthropic FPF TM-V4 at every trimul site) | **2304** |
| `-TMV4A-OFF-NBM-X34` (+ X3, X4) | 2304 |

`-OFF` passes every sealed gate on real 9b0l (K23).

**What binds at 2048** [M]. A CUDA memory snapshot at the K23 OOM (`$KB/analysis/k24-snapshots/snap-hostC-2619435.pickle`) shows 22.51 GiB
live while cycle-0 trunk trimul requests 1 GiB:

| Live tensor | GiB | Needed there? |
|---|---|---|
| NVIDIA cuEq trimul internals (`triangular_multiplicative_update.py:58`) | 6.00 | Yes. TM-V4 needs about 25 % less (K24 op probe) |
| `bond_mask` input feature, [N_atom, N_atom] int64, N_atom ≈ 24300 | 4.40 | **No.** It is a training label; no inference code reads it |
| Model weights, fp32: EMPairformer 1.67, diffusion 0.76, confidence 0.05 | 2.49 | Diffusion and confidence weights: not during the trunk |
| Pairformer working tensors (`empairformer.py` 169/470/552/749/766) | 4.50 | Yes |
| `z_init`, `z` (bf16 [N, N, 128]) | 2.00 | Yes (recycling state) |
| `pz_init`, `pz` (`cryozeta.py:207/223`) | 1.22 | Yes (recycling state) |
| Raw `pz` [620, N, 125] and `p` [620, 620, 99] fp32 input features | 0.73 | **No** after the input embedder |
| Embedded `pz_inputs`, `p_inputs` (`embedders.py:97-98`) | 0.70 | **No** after `pz_init`/`p_init` (`cryozeta.py:207-208`) |

**Where the time goes at N 1920** (`-OFF` r1) [M]. Total forward is 2544 s:
- trunk (EMPairformer, N_cycle 10): 2285 s, 90 %;
- fitting: 131 s;
- diffusion: 86 s;
- confidence head: 42 s.

At N 1024 (hostA A5000, warm) the forward is 885 s, of which fitting is 146 s (16.5 %).

## Capacity: exact (values unchanged up to rounding)

| ID | Change | Saves at N 2048 | Status |
|---|---|---|---|
| X1 | Drop `bond_mask` before moving features to the GPU (`CRYOZETA_DROP_BOND_MASK=1`, `runner/inference.py`, default off) | 4.40 GiB [M], growing as N_atom² (about 7 GiB at N 2560 [A]) | **Gate passed** (K24, real 9b0l: M1 8.72 → 7.87 GB vs `-OFF`). N 2048 r1 SUCCESS on hostC (one run, CONTENDED). hostD ladder voided by a shared-filesystem outage. The solo rerun is dropped; the X1+X2 ladder covers it |
| X2 | Anthropic FPF TM-V4 at every trimul site instead of NVIDIA cuEq | Op peak 12312 → 9232 MiB (−3.0 GiB) [M] | **Gate passed** (K24, real 9b0l, all three sites). Its solo ladder was voided by the hostD outage and is not rerun; X1+X2 together (`-TMV4A-OFF-NBM`) is queued instead: gate, then ladder, on hostB GPU0 (waiting for the GPU) |
| X3 | `del pz_inputs, p_inputs` after line 208, and delete the raw `pz`/`p` features after the input embedder | 1.43 GiB [M sizes] | **Implemented** (default off): `CRYOZETA_FREE_EMBED_INPUTS=1`, `model/cryozeta.py`. In `-TMV4A-OFF-NBM-X34`: gate r1 fails G-rank-top1 on a near-tie, r2/r3 pass; the user accepted it as gated (K24). Ladder 2304 ≤ N_max < 2432, the same as X1+X2: X3/X4 save 0.84 GiB at the 2432 OOM, about 0.57 GiB short |
| X4 | Keep `diffusion_module` and `confidence_head` on the CPU during the trunk and move them in afterwards. The code already moves the trunk out after use | 0.81 GiB [M sizes]; about 0.1 s of H2D [A] | **Implemented** (default off): `CRYOZETA_HEADS_LATE=1`, `model/cryozeta.py`. In `-X34` with X3. The heads come back before the confidence head runs, so it does not help where 2432 binds |
| X5 | Move recycling state (`z_init`, `pz_init`) to pinned host memory during each cycle's pairformer and bring it back for the next cycle | 1.6 GiB [A]; about 0.15 s per cycle at about 12 GB/s [A] | After X3/X4, if the next snapshot still shows the trunk binding |
| X6 | MSA features (`msa`, `has_deletion`, `deletion_value`) to host during the pairformer (only the MSA module reads them) | Part of the 5.36 GiB `to_device` group that is not `bond_mask`; small [A] | Low priority |
| X7 | Diffusion pair-bias cache in bf16, or recompute it (EXPLORATION A4) | 1536 → 768 B/pair (about 3 GiB at 2048) [A] | Only once diffusion becomes the peak stage; today the trunk peaks |
| X8 | Unified memory: PyTorch `CUDAPluggableAllocator` over `cudaMallocManaged` (A5) | Capacity set by host RAM | Zero model change; speed cost unknown. AF3 reached 4352 tokens on 40 GB with XLA unified memory [L] |
| X9 | Out-of-core pair state, OpenFold `offload_inference` style (A6) | O(N·tile) on the GPU | Large engineering effort [L] |
| — | Chunked trimul (OpenFold inplace `_inference_forward`, chunk 256) | Op peak 8848 MiB, slightly below TM-V4 | **Rejected** [M]: 3.9× slower than NVIDIA cuEq at the op (345 vs 88.5 ms) |
| — | Multi-GPU pair parallelism (DAP) | N² / GPUs | Out of scope: the user said one card is enough |

Host RAM (not GPU): the featurizer builds `bond_mask` as a dense float64 N_atom² array, then makes float32 and int64 copies.
K26 skips it under `CRYOZETA_DROP_BOND_MASK=1` (commit 450f664 on `kernel-bench-memory`). The 39 other features are byte-identical,
and peak RSS while building features falls from 9.08 to 7.36 GiB at N 2048 [M, CPU, one run]. The earlier estimate of about 12 GB [A] was too high.

[A] The four exact trunk items X1, X3, X4 and X2 together remove about 9 GiB of the 22.5 GiB live at the 2048 OOM. Whether that
moves N_max to about 2300–2500 depends on which stage peaks next (diffusion, confidence, or the Pairformer working set).
Only the ladders can tell.

## Capacity: approximate (gate with the sealed P0-5 gates)

| ID | Change | Status |
|---|---|---|
| B1 | Confidence head in bf16 (`-OFF-BF16C`) | **Tried, rejected** (K23): no capacity once the trunk binds; pLDDT Δ 2.75 vs 1.5 noise |
| B2 | Diffusion in bf16 (`--skip_amp.sample_diffusion false`) | Untried. Only useful once diffusion peaks |
| B8 | Cap MSA depth (N_msa 5038 here) and support points P at large N | Untried. Cuts the MSA module and `pz` (P × N) terms |
| B9 | int8 storage of z | Untried. No GPU implementation found [L] |
| B4/B5 | Sparse pair (kNN) or chain-pair block sparsity | Emulate with pair masks at N 805 first (EXPLORATION) |
| B6 | Spatial context-window staging: extend P-LC `cycle_predict.py` to chain plus spatial neighbours, all in the map frame | The route to N far beyond one card; each stage fits |
| B7 | Symmetry: one asymmetric unit plus neighbours, expanded by the map's point group | N divided by the copy number for symmetric maps |

## Speed

The trunk is 90 % of the time, and it scales with N_cycle and roughly N³.

| ID | Change | Exact? | Effect |
|---|---|---|---|
| S1 | Anthropic FPF K2B triangle attention instead of NVIDIA cuEq | Rounding-level alone (passes every gate on hostB); combined with X1+X2 it fails G-rank-top1 (K24) | EMPairformer −6.8 % at N 1280 (hostA, 3 runs each), −2.7 % at 1024, −2.5 % at 805 [M]. Same capacity [M]. Kept out of the capacity stack after the K24 top-1 failure |
| S2 | Fitting in parallel over the 10 sample × threshold fits (CPU numpy/scipy `FitModelPoints`, now sequential), overlapped with the GPU confidence head, which reads the raw diffusion coordinates, not the fitted ones | Overlap with the confidence head: exact. Parallel fits: exact only if the global Python `random` draws (`fitting.py:659`, one `random.choice` per residue, in sample order) are made up front in the same order; otherwise rounding-free but a different random stream | Up to about 100 s at N 1920 [A]; about 16 % of the forward at N 1024, where it matters most [M share] |
| S3 | Fewer recycles: the harness and `cycle_predict` use N_cycle 10, `configs_base` uses 4. Or stop recycling early once s/z stop changing | No | 10 → 4 removes about 60 % of the trunk, about −54 % of the forward at N 1920 [A]. The largest lever; quality must be gated. This is the user's call |
| S4 | Anthropic FPF TM-V4 at the trunk | Rounding-level | Op 0.70× of NVIDIA cuEq [M], but 960 trimul calls per forward are only about 85 s of about 2770 s trunk at 2048 [A]: about −1 %. A memory lever, not a speed lever |
| S5 | Confidence head in bf16 | No | 42 → 26 s at N 1920 [M]; rejected with B1 |
| S6 | N_step / N_sample | No | Diffusion is 86 s (3.4 %) at 1920: little to gain |
| S7 | Vectorize the per-element `distance > 4` loop (`cryozeta.py:586-597`, one GPU sync per residue per sample) | Exact | Under 1 s [A]: cleanup, not a lever |

## Suggested order

1. K24: X1 and X2 each pass the gates. The combination with K2B (S1), `-TMV4A-OFF-NBM-K2B`, **fails G-rank-top1** (a near-tie flip, see LEDGER K24), so K2B is left out. Done: `-TMV4A-OFF-NBM` (X1+X2) and `-TMV4A-OFF-NBM-X34` (X1-X4) are both gated (X1-X4 by the user's call) at 2304 ≤ N_max < 2432.
   K27/K28: the user accepted X2 despite the K27 multi-entry deviations, so X1–X4 is the recommended configuration (README). The A100 real-target limits of V0 and X1–X4 are in LEDGER K28.
2. A new snapshot at the new limit.
3. S2: parallel fitting (exact speed).
4. S3: a recycle-count study on real entries, only if the user accepts approximate speed.
5. Beyond about 2.5k tokens on one card: B6 staging, B7 for symmetric maps.
