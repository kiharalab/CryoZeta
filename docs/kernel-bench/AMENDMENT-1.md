# Plan amendment 1: effective support points (P) and M-SDC-5

| | |
|---|---|
| Version | 1, 2026-11-06 |
| Amends | "Kernel plan for long token inference" (read-only reference: `cryoastra/reports/Kernel plan for long token inference.md`) |
| Branch / baseline | `kernel-bench` / `c544c6bbd0366951d8b4e0bc4c0da547836f7b8f` |
| Scope | Only the plan text cited below changes. Everything else in the plan stands. |

## A1.1 P is capped at inference

**Withdrawn plan text** (§1.3 Detection artifacts): "At inference N_point is not capped."

**Replacement:** inference uses P = min(n_detected, N_token, 2000) support points.
- `data/json_to_feature.py:338-341` requests `min(N_token, 2000)` support points.
- `utils/em_utils.py:sample_support_points_v2` prunes the detection cloud down to that request. It never adds points.
- `MAX_SUPPORT_POINTS = 200` is still training-only, as the plan said.

So P is not independent of N_token. Before this amendment, `synth_inputs` left a detection with more points than requested untouched. The model then saw min(n_detected, N_token, 2000) points, not the requested P, and a nominal N × P sweep would have measured repeated shapes under different labels.

CPU evidence (real featurizer, `tests/kernel_bench/test_points.py`):
- A 40-point detection, requested P = 25, at N_token = 30 gave P = 30 with the old generator.
- With the fix it gives P = 25: `em_support_points` [25, 3], `p` [25, 25, 114], `pz` [25, 30, 125].

**Fix** (the V0 cap and V0 behavior are unchanged):
- **Generator.** `synth_inputs --n-point P` writes exactly P detection points.
  - When the detection has more points, it takes a seeded subset (`--seed`, sorted indices). The same indices apply to every point-aligned field and to both axes of `interpolate_confidence`, and each point keeps its own cluster id.
  - When it has fewer, it uses translated copies, as before: clusters are offset and the interpolation confidence is block-diagonal.
  - `synthetic.json` records the requested, written and expected P, the mode, `n_detected`, the seed and an `indices_sha256`.
- **Cap.** A request for P > min(N_token, 2000) exits 2 ("not realizable"). It is never clamped.
- **Expectation.** The `freeze-det` manifest gives N_token and N_point = min(n_detected, N_token, 2000) per entry. Non-P-LC `run_trial` refuses (exit 2) a detection tree without it.
- **Verification.** `kernel_routing.record_input`, called in `InferenceRunner.predict` and a no-op unless counters are on, records each forward's model input: N_token, P, finite-point count and the `p`/`pz` shapes. `run_trial` compares it with the manifest (`trial.json` → `cell`).
- **New outcome class INVALID_CELL.**
  - A mismatch replaces any class, because an OOM at another P is no evidence about this cell.
  - A missing record only blocks SUCCESS: a failure before the forward keeps its class.
  - The original class is kept in `cell.outcome_before_check`.

## A1.2 Capacity matrix

**Withdrawn plan text** (§7.1 Matrix): "N_point: four levels chosen after P0-2: the minimum, median and maximum observed over the five entries, and 2× the maximum." It applied every level at every N.

**Replacement:**
- **Levels.** The P levels are still min, median and max of the P0-2 manifest's N_point over the five entries, plus 2× max.
- **Eligibility.** A cell (N_token, level) is run only if the level ≤ min(N_token, 2000). Otherwise the cell is listed as ineligible. It is not run, not clamped and not a failure.
- **Dedup.** Cells deduplicate on the effective (N_token, P). One run represents every level that maps to it and names them all.
- **Tooling.** `synth_inputs.capacity_cells(n_tokens, levels)` returns the eligible cells and the ineligible list. The command that feeds it from the manifest is in the README.

Consequences:
- **2× max.** It is ineligible at every N whenever max > 1000.
- **Search start.** A level cannot be carried below N_token = P, so the max-N search for that level starts at N ≥ P.
- **Max token length** (§7) is reported per eligible level.

P-LC trials run detection per stage and so have no frozen expectation. Their `cell` is "n/a (P-LC)", and they are not capacity-matrix cells.

**Going over the cap.** An experiment with P > min(N_token, 2000) changes the V0 support-point policy. It needs its own declared policy and review. It is not part of the unchanged-baseline kernel comparison, and it is not implemented.

## A1.3 M-SDC-5 is deferred

**Plan precondition** (§3.5, M-SDC row): "Sample RNG consumption must stay the same."

This precondition fails by inspection. In `sample_diffusion` (`model/generator.py:145-238`), chunk 5 groups the init-noise, per-step-noise and augmentation draws differently from chunk 1 (V0). At the same seed, sample k therefore receives different noise, so the trajectories are not sample-aligned.

**Withdrawn:** the previous pass's suggestion to judge M-SDC-5 against seed-to-seed spread (D_seed).

**Replacement:**
- **Deferred.** M-SDC-5 has status deferred. It is excluded from the sample-paired benchmark and from winner selection.
- **Metadata.** `variants/M-SDC-5.yaml` carries `status: deferred`, `sample_paired: false` and a `deferred_reason`. `kernel_routing.load_variant` validates these keys.
- **`run_trial`** refuses any non-active variant (exit 2), so a deferred variant cannot become a benchmark arm.
- **`compare_outputs`** evaluates gates only with `--variant`. It refuses `--tolerances` for a variant with `sample_paired: false`. No seed-spread or other threshold is substituted.
- **Enforced in content, not just metadata** (added after the implementation review):
  - `run_trial` refuses any variant that sets `sample_diffusion_chunk_size` ≠ 1 without declaring `sample_paired: false`, so a renamed copy of M-SDC-5 cannot enter the paired benchmark.
  - A deferred variant defaults to `sample_paired: false`.
  - `compare_outputs --variant` must match the `variant_id` (and the sha256, when recorded) in the candidate's `trial.json`.
- **Winner selection.** No automated winner-selection code exists. The exclusion is enforced by the two refusals above and by this document.
- **V0 unchanged.** The V0 sampler, RNG order and scientific settings are untouched: `generator.py` is not in the diff.

**Future M-SDC study.** It needs one of the following, reviewed separately:
- a multi-seed/distributional protocol, with its own statistics and thresholds;
- an explicitly declared noise-alignment design, such as pre-drawn per-sample noise, reviewed as a sampler change.

Neither is implemented or launched in this pass.
