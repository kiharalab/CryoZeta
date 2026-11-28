# Optional inference kernels

Enable these options on `cryozeta-inference` or `cryozeta-cycle-predict`. They change floating-point arithmetic and can change coordinate files and fitting ranks. The defaults use the regular inference backends, with cuEquivariance controlled by the existing `--use_cuequivariance_*` flags.

| Option | Default | Eligible calls |
| --- | --- | --- |
| `--infer_setting.triangle_attention_backend triton` | `default` | Pure Triton triangle attention in eager FP16/BF16 CUDA inference on SM80 or later. The kernel checks Q/K/V and bias shapes at runtime. The tuned 1,980-token path uses A6000 and Triton 3.9.0. |
| `--infer_setting.triangle_attention_backend fpf_k2b` | `default` | FlashPairformer K2B for eligible cuEquivariance triangle attention calls. Use `--use_cuequivariance_attention true`. The measured configuration is BF16 with 1,980 tokens, four heads and head width 32 on A6000 with PyTorch `2.15.0+cu132`. |
| `--infer_setting.use_fpf_trimul true` | `false` | FlashPairformer TriMul v4 in place of cuEquivariance multiplication when `--use_cuequivariance_multiplicative_update true`. The measured configuration is BF16 at 1,980 tokens and 128 channels on the same A6000 stack. |
| `--infer_setting.use_triton_confidence_transition true` | `false` | FP32 confidence pair Transition on SM86 with 128 channels and a 4× expansion in eager CUDA inference. This enables fusion and the bounded native fallback when the fused kernel declines. |

For an existing invocation, append one triangle attention option. The TriMul and confidence Transition flags can be set independently. Calls outside the selected kernel's supported configuration use the regular route. These switches also work across the stages of `cryozeta-cycle-predict`.

## Dove measurements

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
