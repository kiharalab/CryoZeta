# Plan amendment 2: memory-based Transition chunking (M-TCHUNK-mem) and the K5 DS4Sci fix

Date: 2026-11-16. Both follow user decisions after the hostA L3 result (LEDGER K5, K12). Source edits were made with no
GPU job of mine running. The working tree differs from `hostA-20261110-gpu2/snapshot.json` from this point on;
`$KB/logs/hostA-chain-9.sh` takes a new snapshot before its first GPU step.

## A2.1 M-TCHUNK-mem

**Why.** The L3 boundary for V0 and K-TA-K2B (1280 ≤ N_max < 1344; 9b0l synthetic, P = 620, A5000 24 GB) is the
confidence-head pair `Transition` (`empairformer.py:170` → `model/modules/primitives.py`, `linear_no_bias_b`).
Below N_token 3200, V0 runs it as one chunk, so two fp32 hidden tensors of N² × 4·c_z are live at once (K12).

**Policy.** `docs/kernel-bench/variants/M-TCHUNK-mem.yaml`, layer L2, empty kernel scope, env
`CRYOZETA_TRANSITION_CHUNK=memory`. Without that env var the code path is V0's.

**Mechanism.** This is `transition_chunks` in `model/modules/primitives.py`, eval mode, CUDA only.
- free = `mem_get_info` free + (reserved − allocated).
- The chunk count is the smallest power of two k with rows·c·(1 + 2n)·4 B ≤ k · 0.5 · free, capped at one row per chunk.
- It uses max(k, V0's count), so it never chunks less than V0.
- Each chunk runs the same row-wise ops into the same preallocated output.
- The result is exact in math. It is not bitwise, because GEMM shapes and algorithm choice change.

Known ceilings:
- The fp32 costing overestimates bf16 trunk calls by 2×. That only adds chunks.
- The 0.5 factor is a fragmentation margin, not a measured value.
- The next allocation that fails after this one is unknown until the ladder runs.

**Checks.**
- CPU, `tests/kernel_bench/test_routing.py`:
  - `TransitionChunks.test_count` covers the chunk count: 8.32 GB of live tensors at N = 1344 → 4 chunks with 4 GiB free.
  - `test_cuda_chunked_matches_unchunked` asserts that the chunked path equals the unchunked path (rtol 1e-5, atol 1e-4) on CUDA. It is skipped on CPU and queued on hostA.
- GPU: the L3 ladder in `hostA-chain-9.sh`.
  - 1280 r1 is paired with V0 1280 r1–r3.
  - Then 1344, 1408, 1536, 1664, 1792, 1920 and 2048 r1, until the first non-SUCCESS.
  - Then r2 and r3 at the highest success.
  - A capacity claim needs 3 consecutive successes at that level. A G-struct verdict waits on P0-5.

**References** (notes: `$KB/notes/transition_chunking_refs.md`). None sizes Transition chunks from free GPU memory; all use fixed tables or trial-and-error.
- AlphaFold 3 (google-deepmind/alphafold3 @ 81151fb, Apache-2.0 code), fixed by token count:
  - `model_config.py:36-40`: `pair_transition_shard_spec = ((2048, None), (None, 1024))`;
  - `network/modules.py:503-511`: `mapping.sharded_apply` over rows;
  - `docs/performance.md:230-246`: a smaller-memory table for A100 40 GB.
- OpenFold (aqlaboratory/openfold @ be2ec18, Apache-2.0), retry on OOM:
  - `pair_transition.py:62-73`: `chunk_layer` over rows;
  - `utils/chunk_utils.py:345-431`: `ChunkSizeTuner`, a power-of-two search that retries on `RuntimeError`.
  - CryoZeta already carries both in `openfold_local/utils/chunk_utils.py`, but `Transition` does not use them.
- OpenFold3 (@ 6f2d059): `SwiGLUTransition` uses `chunk_layer`.
- Boltz-2 (jwohlwend/boltz @ b1ebfc4, MIT) chunks the hidden dimension, and only in the MSA-module z-transition: `layers/transition.py:47-76`.
- Protenix (bytedance/Protenix @ 85767b8) keeps the same fixed `1 if size < 3200 else 8`: `primitives.py:211`.
- ColossalAI AutoChunk (`colossalai/autochunk`) is a torch.fx memory-estimating compiler, too heavy for one module.

## A2.2 K5: DS4Sci bias dtype

The kernel is DS4Sci EvoformerAttention (Microsoft DeepSpeed4Science).
- `_deepspeed_evo_attn` in `openfold_local/model/primitives.py` now casts every bias to q's dtype before
  `DS4Sci_EvoformerAttention`. That is a no-op when the dtypes already match.
- Before the fix, under bf16 autocast, q was bf16 but a bias built from an fp32 mask stayed fp32, and the kernel raised
  `expected scalar type BFloat16 but found Float`.
- The fix changes nothing where c544c6b ran.
- It affects the case that crashed: V0 triangle attention at N ≤ 100, where NVIDIA cuEq hands over to DS4Sci (`CUEQ_TRIATTN_FALLBACK_THRESHOLD`).
- The same function serves V0's row attention, which already ran.

Checks:
- CPU: `Ds4sciDtype.test_biases_follow_q` passes.
- GPU: queued in `hostA-chain-9.sh` as op_bench `$KB/trials/hostA/L1/r3`, tri_att_start/end × N {96, 256, 805, 1280, 2048}. End to end, the D_swap cell of `hostA-chain-10.sh` runs the whole model with DS4Sci triangle attention. The 2400 s timeout covers the fresh-cache DS4Sci build.

## A2.3 24 GB emulation on an A6000 (2026-11-19, LEDGER K19)

User decision: exact GPU parity is not needed, and qualitative cross-GPU comparisons are acceptable.
- The A2.1 ladder therefore runs on hostB A6000s (48 GB) with `run_trial --ballast-free-mib 24550`. A helper process holds device memory until 24550 MiB stay free, which equals an idle hostA A5000 (24564 total − 14 used).
- `transition_chunks` reads `mem_get_info`, so it sees the reduced free memory, and an OOM happens at the same device-free level.
- V0 runs its own ladder under the same ballast as the same-setup baseline.
- Labels read 'A6000 + ballast'. Clock, bandwidth, allocator history and the one-time match of free memory differ from a real A5000.
