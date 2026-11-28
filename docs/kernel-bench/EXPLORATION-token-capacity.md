# Open exploration: raising N_token on one 24 GB GPU

Date: 2026-11-16. This is a brainstorm requested by the user. Ideas need not be feasible, and some quality loss is acceptable.
Nothing here is a capacity or speed claim. Status as of 2026-11-23 (what was measured, what is left): [NO-RETRAIN-options.md](NO-RETRAIN-options.md). Evidence labels:
- [M] measured here;
- [A] arithmetic from measured numbers;
- [L] literature, with sources in `$KB/notes/token_scaling_refs.md` (Transition references: `$KB/notes/transition_chunking_refs.md`);
- [H] hypothesis.

## Grounding

- [M] The boundary for V0 and K2B is 1280 ≤ N_max < 1344 (9b0l synthetic, P 620, N_msa 5038, A5000). The OOM is in the confidence-head pair `Transition`, which runs in fp32 because `skip_amp.confidence_head` is True.
- [M] Live-tensor peak M1 is 13.04 GB at N = 1024 and 19.47 GB at N = 1280.
- [A] Those two points fit M1 ≈ 1.6 GB + 10.9 KB · N², about 21 copies of an fp32 z per pair.
  - The unchunked Transition transient (LN out plus two hidden tensors, fp32) accounts for 4.6 KB of that.
  - The other ≈ 6.3 KB per pair is unattributed. The stage-peak runs in `hostA-chain-10.sh` will measure it.
  - If nothing else peaks, chunking only the Transition would put the arithmetic ceiling at ≈ 1650–1800. This is [A], not a prediction.
- [M] Sample ranking does not use the confidence head. It ranks on `recall + ccc_mask − num_big_distances` per registration method (`cryozeta.py:656-705`). So the confidence head that sets today's boundary affects only the reported pLDDT, PAE, pTM and PDE, never the selected coordinates.
- [L] Fused attention kernels (NVIDIA cuEq, DS4Sci from Microsoft DeepSpeed4Science, FPF K2B) never materialize the N×N logits per row. Attention memory is therefore already O(N²·H). The memory floor is the pair representation itself and its N²×c intermediates: Transition, the triangle-multiplication projections, the OPM, and the confidence head.

## Tier A: exact or near-exact (no model change; quality equal up to rounding)

| ID | Idea | Memory effect | Status / prior work |
|---|---|---|---|
| A1 | Memory-sized Transition chunks (`M-TCHUNK-mem`) | Divides the 4.6 KB/pair transient by k | Done; ladder queued. AF3 / OpenFold / Boltz use fixed tables [L] |
| A2 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | At the N = 1344 OOM, 3.48 GiB was reserved but unallocated [M] | Env-only variant, 1 YAML |
| A3 | Chunk whatever stage peaks next (confidence-head trimul and OPM-like terms, the diffusion conditioning) | Unknown until stage peaks are measured | After chain-10 |
| A4 | Diffusion pair-bias cache: 24 blocks × 16 heads, fp32 = 1536 B/pair (6.3 GB at N 2048) [A]. Cache it in bf16, or recompute it per block | 768 B or ≈ 0 per pair; recompute costs time | Source edit |
| A5 | Unified memory: PyTorch `CUDAPluggableAllocator` over `cudaMallocManaged`, so overflow pages to host RAM | Capacity set by host RAM; speed depends on how much overflows | AF3 reached 4,352 tokens on an A100 40 GB with XLA unified memory [L]. No PyTorch bio model found [L]. Zero model change |
| A6 | Out-of-core pair state: keep z in pinned host memory and stream row tiles. Triangle mult is a tiled GEMM over k. Triangle attention needs only the H×N² bias globally | GPU holds O(N·tile·c + H·N²) | OpenFold `offload_inference` moves m and z to CPU between sub-ops; 4,600-residue claim on "a single A100" [L]. Large engineering effort |

## Tier B: approximate, pretrained weights (quality loss measured by the sealed P0-5 gates)

| ID | Idea | Why it helps | Notes |
|---|---|---|---|
| B1 | Confidence head at large N: run it in bf16 (`--skip_amp.confidence_head false`, a CLI-only variant), on a token subset or crop, or skip it | It is today's bottleneck. Coordinates are unchanged because ranking does not use it [M] | You lose or degrade pLDDT/PAE only. Cheapest real gain |
| B2 | Diffusion in bf16 (`--skip_amp.sample_diffusion false`) | Halves the diffusion activations and the bias cache | Affects coordinates; gate with G-struct |
| B3 | Sliding-window or local triangle attention / pair-bias attention (sequence window, or spatial kNN) | Mostly **time**: O(N²·w) instead of O(N³). Little memory gain with fused kernels (see Grounding) | No prior work found [L]. AF3/Protenix windows only atom attention (32 queries × 128 keys) [L] |
| B4 | Sparse pair representation on a neighbour graph (kNN or distance cutoff from the map / first pass); triangle updates only over graph triangles | z memory O(N·k·c) instead of O(N²·c). This is what lifts the memory floor | No prior work found [L]. Needs new kernels |
| B5 | Chain-pair block sparsity: drop z blocks between chains whose map footprints do not touch | Large assemblies: memory ∝ number of contacting chain pairs, not N² | Coarse version of B4. No prior work found [L] |
| B6 | Spatial context-window staging: extend P-LC (`cycle_predict.py`, one chain per stage, no inter-chain context) to stages of chain + spatial neighbours ≤ N_max, overlapping, keeping only each stage's core chains | Unbounded total N; each stage fits the card | Stitching is nearly free because every stage is registered into the same map frame. HierAFold (ICLR 2026) does PAE-based split and refine for AF3 without a map [L]; CombFold assembles AF-Multimer subcomplexes [L] |
| B7 | Symmetry: predict one asymmetric unit plus its neighbours, then expand with the map's point-group operators | N divided by the copy number, for C/D/T/O/I assemblies, which are common in cryo-EM | No AF3-class implementation found [L] |
| B8 | Cap MSA depth (N_msa 5038 here) and support points P at large N | MSA-module and `pz` (P×N) terms | Expected quality cost is small to moderate [H] |
| B9 | int8 storage of z with dequantize in the kernels (A5000 has no fp8) | 2× vs bf16, 4× vs fp32 | LightNobel: TM-score Δ < 0.001 at up to 9,945 residues, but on ESMFold and a custom accelerator [L]. No GPU version found |

B3, B4 and B5 can be **emulated with pair masks** at N 805 on the real 9b0l before any kernel is written. Masked triangle updates ignore masked k, so a mask approximates the quality cost of sparsity, though not its memory saving. G-struct and G-rank vs D_seed then tell whether the idea is worth building.

## Tier C: needs retraining or distillation [L]

- Drop triangle attention: Pairmixer (runs ≈ 30% longer sequences), MiniFold.
- Linear triangle attention or chunk-pooled triangle update: Protenix-Mini+, ≈ 3% LDDT drop, applied to 10 of 16 blocks.
- Multi-residue (coarse) tokens in the trunk with residue-level diffusion: no AF3-class prior found.
- Low-rank pair representation: no prior found.

## Suggested order for one A5000

1. A1 (queued).
2. Stage peaks (queued).
3. A2.
4. B1 bf16.
5. A3/A4 on whatever peaks next.
6. A5 as the zero-change fallback.

For a few thousand tokens and up: B6 (it builds on P-LC and the map frame), B7 for symmetric maps, and B5/B4 after mask emulation shows tolerable quality.
