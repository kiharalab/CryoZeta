# Inference kernels and baselines

`cryozeta-inference` and `cryozeta-cycle-predict` default to pure Triton triangle attention and eligible BioKernels triangle multiplication. Pair-bias attention uses PyTorch. Existing invocations should drop the `--use_cuequivariance_*` flags. Measurements with cuEquivariance below are historical reference data.

| Option | Default | Eligible calls |
| --- | --- | --- |
| `--infer_setting.triangle_attention_backend triton` | `triton` | Pure Triton triangle attention in eager FP16/BF16 CUDA inference on SM80 or later. The kernel checks Q/K/V and bias shapes at runtime. The tuned 1,980-token path uses A6000 and Triton 3.9.0. |
| `--infer_setting.triangle_attention_backend default` | `triton` | Bounded native PyTorch triangle attention. |
| `--infer_setting.triangle_attention_backend fpf_k2b` | `triton` | FlashPairformer K2B for BF16 with 1,980 tokens, four heads, and head width 32 on RTX A6000 with PyTorch `2.15.0+cu132`. |
| `--infer_setting.use_fpf_trimul true` | `false` | FlashPairformer TriMul v4 for BF16 at 1,980 tokens and 128 channels on the same A6000 stack. This is tried before BioKernels when enabled. |
| `--infer_setting.use_biokernels_trimul true` | `true` | Forward-only BioKernels TriMul with bounded contraction and strided LayerNorm on RTX A6000 with PyTorch `2.15.0+cu132`. This selects contiguous single-example pair activations with 128 input and hidden channels, 128–4,096 tokens, and BF16 CUDA autocast. |
| `--infer_setting.use_triton_confidence_transition true` | `false` | FP32 confidence pair Transition on SM86 with 128 channels and a 4× expansion in eager CUDA inference. Requires `--infer_setting.use_confidence_head true` to run. |
| `--infer_setting.transition_chunk_budget_mib 1024` | `0` | Fixed per-chunk budget for the normalized input and two hidden projections in native CUDA Transition. `0` uses the existing chunking rules. The budget excludes the output, temporary workspaces, and other live model tensors. |

Calls outside the selected kernels' supported configurations use PyTorch. The flags also apply to each stage of `cryozeta-cycle-predict`. Changing attention or multiplication backends can change floating-point results and fitted-model rankings.

## BioKernels forward triangle multiplication

This port adapts the tiled inference composition from BioKernels `37e3d6e48b476487d30fc53ca1b7dc4e132cd6c3` and its strided LayerNorm forward refinement. The implementation uses the two private projection buffers for bounded contraction, normalizes strided contraction tiles without transposing them to contiguous storage, and writes each gated output tile into its final destination. It uses the existing checkpoint weights and biases. See [`biokernels_trimul.py`](../src/cryozeta/kernels/biokernels_trimul.py).

To benchmark pure Triton triangle attention against native PyTorch with checkpoint weights and seeded synthetic inputs, run `python -m cryozeta.kernel_bench.triangle_attention --checkpoint PATH --module empairformer_stack.blocks.0.tri_att_start`.

### Native PyTorch TriMul comparison

On Dove (RTX A6000, PyTorch `2.15.0+cu132`), the three-backend replay uses the native, in-place PyTorch inference path as the reference. Each backend receives the same synthetic BF16 activation, first-block checkpoint weights, and dense Boolean mask. The input buffer is restored before each call, outside measurement. The median of seven timed calls follows two warmups, alternating backend order. Extra allocated peak excludes the live input. Error uses the full output at 1,980 tokens and 16 evenly spaced rows at the larger lengths. Values separated by a slash are outgoing and incoming.

| Tokens | Backend | Time (ms) | Extra allocated peak (GiB) | Relative L2 vs native PyTorch |
| ---: | --- | ---: | ---: | ---: |
| 1,980 | Native PyTorch | 190.31 / 193.06 | 2.25 / 2.13 | 0 / 0 |
| 1,980 | cuEquivariance | 44.72 / 41.12 | 5.61 / 5.61 | 0.628% / 0.694% |
| 1,980 | BioKernels | 72.23 / 67.84 | 2.12 / 2.12 | 0.053% / 0.063% |
| 3,072 | Native PyTorch | 492.33 / 503.85 | 4.69 / 4.50 | 0 / 0 |
| 3,072 | cuEquivariance | 156.26 / 158.78 | 13.50 / 13.50 | 0.598% / 0.663% |
| 3,072 | BioKernels | 220.94 / 211.25 | 4.70 / 4.70 | 0.052% / 0.065% |
| 4,096 | Native PyTorch | 923.62 / 953.63 | 7.75 / 7.50 | 0 / 0 |
| 4,096 | cuEquivariance | 312.27 / 322.06 | 24.00 / 24.00 | 0.581% / 0.643% |
| 4,096 | BioKernels | 416.24 / 421.52 | 8.26 / 8.26 | 0.053% / 0.070% |

### 9BH4 checkpoint inference

For the native PyTorch backend arm, use the same checkpoint and sampling options with these flags:

```sh
--infer_setting.triangle_attention_backend default \
--infer_setting.use_biokernels_trimul false \
--infer_setting.use_fpf_trimul false \
--infer_setting.use_confidence_head false
```

The native arm uses the current repository's PyTorch fallback for these three operations. Other model optimizations are shared across the runs.

All four runs used the same checkpoint, input, seed 101, one recycle, and five diffusion samples with 20 steps each, with confidence disabled. The isolated BioKernels arm changed only TriMul relative to cuEquivariance. The cuEq-free arm also used pure Triton triangle attention and PyTorch pair-bias attention. Both BioKernels arms dispatched TriMul 200 times. The cuEq-free arm dispatched pure Triton attention 1,600 times. The native and cuEq-free benchmarks guarded against calls to the three cuEquivariance backends.

| Metric | Native PyTorch | cuEq | Isolated BioKernels TriMul | BioKernels, no cuEq |
| --- | ---: | ---: | ---: | ---: |
| Feature-to-export time excluding checkpoint load (s) | 1,986.64 | 574.75 | 583.73 | 558.67 |
| Peak allocated (MiB) | 20,769.23 | 20,768.72 | 20,770.92 | 20,770.92 |
| Peak reserved (MiB) | 28,494 | 26,468 | 28,384 | 28,384 |
| Same-sample aligned all-atom RMSD vs native PyTorch (Å) | 0 | 0.035–0.099 | 0.056–0.105 | 0.035–0.098 |
| Highest-ranked TEASER sample | 4 | 3 | 0 | 4 |
| Highest fitted `recall_ccmask_ca` | 0.7789 | 0.7964 | 0.7886 | 0.7769 |
| Highest-ranked fitted structures, aligned all-atom RMSD vs native PyTorch (Å) | 0 | 0.976 | 0.990 | 0.035 |
| Highest-ranked fitted structures, original map frame RMSD vs native PyTorch (Å) | 0 | 67.269 | 71.814 | 0.045 |

The highest-ranked comparisons pair each run's own top export. The native PyTorch and cuEq-free arms selected the same diffusion sample. For comparison with cuEquivariance, the isolated and cuEq-free BioKernels arms had same-sample aligned ranges of 0.030–0.073 and 0.045–0.092 Å, respectively. Their top exports differed from cuEquivariance's by 0.864 and 0.982 Å after alignment and by 64.213 and 67.271 Å in the original map frame. Larger token counts were tested at the operator level.

The default configuration after cuEquivariance removal was rerun on Dove with a guard that rejects all cuEquivariance package lookups and imports. It completed 200 BioKernels TriMul and 1,600 Triton triangle attention calls in 545.84 s feature-to-export, with 20,770.92 MiB allocated and 28,384 MiB reserved. Its coordinates and complete output directory matched the earlier cuEq-free arm exactly. An 8T5O CLI smoke test also completed on the A6000 with PyTorch `2.7.1+cu118`.

A second controlled comparison held cuEquivariance triangle attention and pair-bias attention fixed while replacing only native, in-place PyTorch TriMul with BioKernels. Feature-to-export time fell from 663.39 to 583.73 s. Peak allocated was 20,769.23 versus 20,770.92 MiB and peak reserved was 28,494 versus 28,384 MiB. Same-sample all-atom RMSD after alignment was 0.034–0.068 Å. The top TEASER fit changed from sample 3 to sample 0, with `recall_ccmask_ca` changing from 0.8038 to 0.7886. Comparing the two top exports gave 0.869 Å after alignment and 64.302 Å in the original map frame.

## Default confidence and fixed-budget Transition

The confidence head is off by default in `cryozeta-inference` and `cryozeta-cycle-predict`. Default output omits PAE, pLDDT, PDE, resolved logits, and confidence JSON. Raw structures follow diffusion sample order. Superimposed structures are ranked by map-fit score, with unfitted fallbacks ordered by sample index. Use `--infer_setting.use_confidence_head true` to reproduce historical confidence runs, adding `--need_atom_confidence true` in standalone inference when full atom-level JSON is needed. The confidence timings later in this document are historical runs with the head enabled.

In the earlier cuEquivariance configuration, 9BH4 with seed 101 and confidence disabled took 574.75 s feature-to-export, excluding checkpoint loading, with peaks of 20,768.72 MiB allocated and 26,468 MiB reserved. The five structures matched the historical cuEquivariance reference sample by sample at 0.000 Å RMSD. Superimposed exports and `scores.csv` were byte-identical. Raw export indices reflect the switch to diffusion sample order.

With the 1,024 MiB budget and confidence still disabled, the same input took 577.52 s and reached the same allocated and reserved peaks. Its full output directory matched the confidence-off reference byte for byte. The budget reduces individual Transition peaks, so its impact on a full run depends on which operation sets the peak. Changing the GEMM row count can change rounding on other shapes or stacks.

The isolated pair Transition used the real checkpoint's first EMPairformer block weights and synthetic contiguous BF16 pair activations on the same stack. Times are median CUDA event times over five calls after warmup. Extra allocated peak is measured relative to the live input and weights. All three paired comparisons were bitwise equal.

| Tokens | Existing chunks | 1,024 MiB chunks | Time (ms) | Extra allocated peak (MiB) |
| ---: | ---: | ---: | ---: | ---: |
| 1,536 | 1 | 6 | 36.29 → 38.09 | 5,192.25 → 1,608.25 |
| 1,980 | 4 | 9 | 62.99 → 63.47 | 3,118.92 → 1,989.38 |
| 2,304 | 1 | 12 | 81.14 → 85.77 | 11,672.25 → 2,328.25 |

## Historical Dove measurements

The attention and multiplication replays use captured 9BH4 activations and the model's weights on an RTX A6000 with CUDA 13.2, PyTorch `2.15.0+cu132` and Triton 3.9.0. Reported times are warmed median CUDA-event times, with backend order alternated over seven repetitions. Additional peak memory is the maximum PyTorch CUDA memory allocated above the live inputs. The relative L2 error is `||candidate - reference||₂ / ||reference||₂`.

### Triangle attention

The Q/K/V slice is BF16 with 256 rows, four heads, 1,980 tokens and head width 32. Values are row / column. Each backend uses the same input, mask, timing and memory protocol. The reference is bounded native PyTorch attention.

| Backend | Time (ms) | Additional peak allocated (MiB) | Relative L2 vs PyTorch | Max absolute error vs PyTorch |
| --- | ---: | ---: | ---: | ---: |
| PyTorch | 785.82 / 794.03 | 276.44 / 276.44 | 0 / 0 | 0 / 0 |
| cuEquivariance | 38.50 / 38.53 | 633.09 / 633.09 | 0.004112 / 0.002773 | 0.078125 / 0.074219 |
| FlashPairformer K2B | 8.17 / 8.30 | 273.73 / 273.73 | 0.004112 / 0.002773 | 0.078125 / 0.074219 |
| Pure Triton | 14.24 / 14.34 | 153.97 / 153.97 | 0.0000553 / 0.0000504 | 0.03125 / 0.015625 |

K2B matched 98.29% / 98.22% of the cuEquivariance output bit patterns. Its relative L2 difference versus cuEquivariance was 0.0289% / 0.0342% at the attention output, 0.0666% / 0.0746% after the gated output projection, and 0.0139% / 0.0110% after the residual addition. Pure Triton matched about 99.98% of the native PyTorch output bit patterns. Its optimized path packs the shared bias, which adds about 30 MiB compared with the unpacked pure Triton implementation.

### Triangle multiplication and confidence Transition

The multiplication input is BF16 `[1980, 1980, 128]`. Values are outgoing / incoming. These timings measure the update before residual addition.

| Backend | Time (ms) | Additional peak allocated (GiB) | Relative L2 vs PyTorch | Max absolute error vs PyTorch |
| --- | ---: | ---: | ---: | ---: |
| PyTorch | 164.73 / 164.86 | 5.62 / 5.62 | 0 / 0 | 0 / 0 |
| cuEquivariance | 44.57 / 42.39 | 5.61 / 5.61 | 0.005991 / 0.005401 | 0.0625 / 0.0625 |
| FlashPairformer TriMul v4 | 41.59 / 41.72 | 3.75 / 3.75 | 0.006158 / 0.005460 | 0.09375 / 0.046875 |

TriMul v4 differed from cuEquivariance by 0.282% / 0.278% relative L2 at the update and 0.0411% / 0.0368% after the residual addition. The fused residual path had maximum absolute errors of 2 / 1 in the internal tensor.

The separate FP32 confidence Transition experiment at 1,980 tokens measured 116.07 ms and 1,914 MiB of additional peak allocation with Triton versus 156.30 ms and 3,098 MiB with PyTorch. In the corresponding 9BH4 checkpoint comparison, coordinates, rankings, 25 CIFs and `scores.csv` matched exactly. The maximum difference across sampled confidence tensor values was `1.72e-5`.

### End-to-end impact on 9BH4

The same checkpoint, input and seed 101 were used for every arm. Each used one recycle and five diffusion samples with 20 steps each. The reference is the cuEquivariance prediction from `0ed1de5`. The FP32 Triton confidence Transition was enabled in every arm to isolate attention and multiplication. The timings exclude checkpoint loading. Peak allocated measures live CUDA tensors during prediction. Peak reserved includes cached blocks held by PyTorch's allocator.

#### RMSD from the previous prediction

Compare each candidate to the reference at the same diffusion sample index. The representative atoms are the 1,980 centers selected by the model's `ca_mask`. All-atom comparisons use all 16,134 atoms. Kabsch alignment removes global translation and rotation. The map-coordinate comparison uses the exported top fitted model as written.

| RMSD in Å | K2B only | TriMul v4 only | Pure Triton attention CLI | K2B + TriMul v4 CLI |
| --- | ---: | ---: | ---: | ---: |
| Same-sample representative atoms after alignment, min–max | 0.022–0.034 | 0.022–0.038 | 0.016–0.031 | 0.031–0.061 |
| Same-sample all atoms after alignment, min–max | 0.027–0.057 | 0.026–0.078 | 0.032–0.046 | 0.044–0.095 |
| Same-sample all atoms before alignment, min–max | 0.246–1.237 | 0.665–2.371 | 0.436–1.684 | 0.591–1.210 |
| Top fitted model in map coordinates, all atoms | 0.150 | 7.652 | 0.073 | 0.389 |
| Top fitted model after alignment, all atoms | 0.029 | 0.026 | 0.036 | 0.044 |

All four candidate runs selected TEASER sample 3 as the top fit. The independently replaced TriMul run illustrates the distinction between geometry and placement. Its top fit differed by 0.026 Å after alignment and by 7.652 Å in the original map frame.

#### Timing, memory and fitting order

The independent K2B-only and TriMul-only measurements replaced eligible cuEquivariance calls of length 1,980 in the measurement harness and also allowed FP32 confidence calls. The CLI configurations use the BF16 scope in the option table. The two CLI runs also confirmed 20 successful FP32 Triton confidence Transition calls each.

| Measurement | cuEquivariance | K2B only | TriMul v4 only | Pure Triton attention CLI | K2B + TriMul v4 CLI |
| --- | ---: | ---: | ---: | ---: | ---: |
| Backend calls replaced | — | 1,920 | 240 | 1,600 | 1,600 / 200 |
| Feature-to-export time (s) | 615.96 | 579.73 | 611.15 | 588.07 | 570.24 |
| Peak allocated (GiB) | 22.43 | 22.43 | 22.33 | 22.43 | 22.43 |
| Peak reserved (GiB) | 34.46 | 34.46 | 31.46 | 34.46 | 36.45 |
| Kendall τ across 15 fitting scores | 1 | 0.676 | 0.524 | 0.638 | 0.410 |
| Byte-identical CIFs, ranked confidence summaries and score CSV | 31 / 31 | 0 / 31 | 0 / 31 | 0 / 31 | 0 / 31 |

The raw confidence order was `[0, 1, 2, 4, 3]` in every run. All five fitted exports used TEASER. The reference and K2B-only order was `[3, 0, 1, 2, 4]`. It changed to `[3, 0, 4, 1, 2]` with TriMul alone, `[3, 0, 2, 1, 4]` with pure Triton attention, and `[3, 1, 2, 4, 0]` with both FlashPairformer kernels. These measurements cover one checkpoint, input and seed on the specified stack.
