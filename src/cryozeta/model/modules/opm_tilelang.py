# Copyright 2026 KiharaLab, Purdue University.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import torch
import torch.nn.functional as F
try:
    import tilelang as tl
    from tilelang import language as T
except ImportError:
    tl = None
    T = None


def opm_fwd_kernel(
    batch, seq_len, length, c_m, c_z, c_hidden,
    block_N=16,
    block_S=4,
    threads=256,
    in_dtype="float32",
):
    """OPM forward kernel with mixed-precision support.

    Supports FP32 and BF16 input paths. When in_dtype="bfloat16", A and B
    are loaded as BF16 (halving global memory bandwidth), with FP32
    accumulation for full precision. W, Bias, Out, Norm are always FP32.

    Key optimizations:
    1. Mixed precision: BF16 A/B loads + FP32 accumulation (US-007).
    2. Loop reorder: s_local -> h -> ci(unrolled) reduces B_shared reads 4x
       (US-006).
    3. T.unroll for block_c=4 inner loops for instruction-level parallelism.
    4. A/B shared memory in in_dtype with H+1 padding for bank conflicts.
    5. Inputs padded to block-aligned sizes — no bounds checks in hot loops.
    6. W[Z, block_c*H] tile staged in shared memory for projection phase.
    7. Z-tiled projection: process block_z Z values per W tile load.
    """
    Bat, S, N, C, Z, H = batch, seq_len, length, c_m, c_z, c_hidden
    H_pad = H + 1  # Pad H dimension to reduce shared memory bank conflicts

    # N and S are guaranteed to be multiples of block_N and block_S (padded in wrapper)
    grid_x = N // block_N
    grid_y = N // block_N
    grid_z = Bat
    s_blocks = S // block_S

    block_c = 4
    block_z = 32
    z_blocks = (Z + block_z - 1) // block_z
    w_tile_cols = block_c * H  # 128

    @T.prim_func
    def main(
        A: T.Tensor((Bat, N, S, H), in_dtype),
        B_buf: T.Tensor((Bat, N, S, H), in_dtype),
        W: T.Tensor((Z, H * H), "float32"),
        Bias: T.Tensor((Z,), "float32"),
        Out: T.Tensor((Bat, N, N, Z), "float32"),
        Norm: T.Tensor((Bat, N, N), "float32"),
    ):
        with T.Kernel(grid_x, grid_y, grid_z, threads=threads) as (bj, bi, bb):
            T.use_swizzle(10)
            tx = T.get_thread_binding()

            ii = tx // block_N
            jj = tx % block_N
            t_i = bi * block_N + ii
            t_j = bj * block_N + jj

            # Accumulator for output [Z]
            acc = T.alloc_local((Z,), "float32")
            for z in T.serial(Z):
                acc[z] = 0.0

            # Cooperative load parameters for A/B
            # BF16: vec_width=8 (16 bytes), FP32: vec_width=4 (16 bytes) — both 128-bit loads
            ab_elems = block_N * block_S * H
            ab_vec_width = 8 if in_dtype == "bfloat16" else 4
            ab_elems_vec = ab_elems // ab_vec_width
            ab_dim_H_vec = H // ab_vec_width
            ab_load_iters = (ab_elems_vec + threads - 1) // threads

            # Cooperative load parameters for W
            w_elems = block_z * w_tile_cols
            w_vec_width = 4
            w_elems_vec = w_elems // w_vec_width
            w_cols_vec = w_tile_cols // w_vec_width
            w_load_iters = (w_elems_vec + threads - 1) // threads

            # Process c values in blocks of block_c
            c_blocks = H // block_c
            for c_blk in T.serial(c_blocks):
                c_base = c_blk * block_c

                # temp_rows: accumulate outer product for block_c c-values
                temp_rows = T.alloc_local((block_c, H), "float32")
                for ci in T.serial(block_c):
                    for h in T.serial(H):
                        temp_rows[ci, h] = 0.0

                # === Phase 1: S-loop with vectorized loads + H+1 padding ===
                for s_blk in T.serial(s_blocks):
                    s_base = s_blk * block_S

                    A_shared = T.alloc_shared((block_N, block_S, H_pad), in_dtype)
                    B_shared = T.alloc_shared((block_N, block_S, H_pad), in_dtype)

                    # Cooperative vectorized load A (no bounds check — padded inputs)
                    for vec_idx in T.serial(ab_load_iters):
                        flat_vec = tx + vec_idx * threads
                        if flat_vec < ab_elems_vec:
                            ln = flat_vec // (block_S * ab_dim_H_vec)
                            ls = (flat_vec % (block_S * ab_dim_H_vec)) // ab_dim_H_vec
                            lh_base = (flat_vec % ab_dim_H_vec) * ab_vec_width
                            for v in T.vectorized(ab_vec_width):
                                A_shared[ln, ls, lh_base + v] = A[bb, bi * block_N + ln, s_base + ls, lh_base + v]

                    # Cooperative vectorized load B
                    for vec_idx in T.serial(ab_load_iters):
                        flat_vec = tx + vec_idx * threads
                        if flat_vec < ab_elems_vec:
                            ln = flat_vec // (block_S * ab_dim_H_vec)
                            ls = (flat_vec % (block_S * ab_dim_H_vec)) // ab_dim_H_vec
                            lh_base = (flat_vec % ab_dim_H_vec) * ab_vec_width
                            for v in T.vectorized(ab_vec_width):
                                B_shared[ln, ls, lh_base + v] = B_buf[bb, bj * block_N + ln, s_base + ls, lh_base + v]

                    # Compute with reordered loops: s_local -> h -> ci(unrolled)
                    # Reads B_shared once per h, reuses across block_c ci values.
                    a_vals = T.alloc_local((block_c,), "float32")
                    for s_local in T.serial(block_S):
                        # Pre-load block_c A values for this s_local
                        for ci in T.unroll(block_c):
                            a_vals[ci] = A_shared[ii, s_local, c_base + ci]
                        # Inner loop: read B once, multiply with all ci
                        for h in T.serial(H):
                            val_b = B_shared[jj, s_local, h]
                            for ci in T.unroll(block_c):
                                temp_rows[ci, h] += a_vals[ci] * val_b

                # === Phase 2: Projection — W tile in shared memory ===
                for z_blk in T.serial(z_blocks):
                    z_base = z_blk * block_z

                    W_shared = T.alloc_shared((block_z, w_tile_cols), "float32")

                    # Cooperative vectorized W load
                    for vec_idx in T.serial(w_load_iters):
                        flat_vec = tx + vec_idx * threads
                        if flat_vec < w_elems_vec:
                            wz = flat_vec // w_cols_vec
                            wc_base = (flat_vec % w_cols_vec) * w_vec_width
                            gz = z_base + wz
                            g_col = c_base * H + wc_base
                            if gz < Z and g_col + w_vec_width <= H * H:
                                for v in T.vectorized(w_vec_width):
                                    W_shared[wz, wc_base + v] = W[gz, g_col + v]
                            else:
                                for v in T.serial(w_vec_width):
                                    if gz < Z and (g_col + v) < H * H:
                                        W_shared[wz, wc_base + v] = W[gz, g_col + v]
                                    else:
                                        W_shared[wz, wc_base + v] = 0.0

                    # Project temp_rows through W_shared into acc (unrolled ci)
                    for ci in T.unroll(block_c):
                        for zz in T.serial(block_z):
                            gz = z_base + zz
                            if gz < Z:
                                val_proj = T.alloc_local((1,), "float32")
                                val_proj[0] = 0.0
                                for d in T.serial(H):
                                    val_proj[0] += temp_rows[ci, d] * W_shared[zz, ci * H + d]
                                acc[gz] += val_proj[0]

            # Write output with normalization and bias
            norm_val = Norm[bb, t_i, t_j]
            for z in T.serial(Z):
                Out[bb, t_i, t_j, z] = (acc[z] + Bias[z]) / norm_val

    return main


@functools.lru_cache(maxsize=32)
def get_opm_kernel(B, S, N, C, Z, H, in_dtype="float32"):
    if tl is None:
        raise ImportError("tilelang is not installed")
    kernel_func = opm_fwd_kernel(B, S, N, C, Z, H, block_N=16, block_S=4, threads=256, in_dtype=in_dtype)
    return tl.compile(kernel_func, target="cuda")


def opm_tilelang(a, b, weight, bias, norm):
    is_3d = False
    if a.dim() == 3:
        is_3d = True
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
        norm = norm.unsqueeze(0)

    B, N, S, H = a.shape
    Z = weight.shape[0]

    original_dtype = a.dtype

    # Determine kernel input dtype: BF16 fast path or FP32
    if original_dtype == torch.bfloat16:
        in_dtype = "bfloat16"
        # A and B stay as BF16 — halves global memory bandwidth
        # W, bias, norm must be FP32 for the kernel
        weight = weight.float()
        bias = bias.float()
        norm = norm.float()
    else:
        in_dtype = "float32"
        # Convert everything to FP32 (handles FP16 and other dtypes)
        if original_dtype != torch.float32:
            a = a.float()
            b = b.float()
        weight = weight.float()
        bias = bias.float()
        norm = norm.float()

    # Pad N and S to multiples of block sizes for clean tiling
    block_N, block_S = 16, 4
    N_orig = N
    N_pad = ((N + block_N - 1) // block_N) * block_N
    S_pad = ((S + block_S - 1) // block_S) * block_S

    if N_pad != N or S_pad != S:
        a_p = torch.zeros(B, N_pad, S_pad, H, device=a.device, dtype=a.dtype)
        a_p[:, :N, :S, :] = a
        b_p = torch.zeros(B, N_pad, S_pad, H, device=b.device, dtype=b.dtype)
        b_p[:, :N, :S, :] = b
        norm_p = torch.ones(B, N_pad, N_pad, device=norm.device, dtype=torch.float32)
        norm_p[:, :N, :N] = norm
        a, b, norm = a_p, b_p, norm_p
        N, S = N_pad, S_pad

    kernel = get_opm_kernel(B, S, N, H, Z, H, in_dtype=in_dtype)
    out = torch.empty((B, N, N, Z), device=a.device, dtype=torch.float32)
    kernel(a, b, weight, bias, out, norm)

    # Unpad output if needed
    if N != N_orig:
        out = out[:, :N_orig, :N_orig, :]

    if original_dtype != torch.float32:
        out = out.to(original_dtype)

    if is_3d:
        out = out.squeeze(0)
    return out


def opm_chunked(a, b, weight, bias, norm, chunk_size=16):
    """Memory-efficient OPM using cuBLAS with N1-chunking.

    Drop-in replacement for opm_tilelang() — same signature.
    Memory: chunk_size * N * H^2 per chunk (vs N^2 * H^2 for full materialization).

    Runs entirely in the input dtype (BF16 or FP32). Under BF16 this matches
    the original PyTorch OPM behavior (einsum + F.linear both in BF16).
    """
    is_3d = False
    if a.dim() == 3:
        is_3d = True
        a, b, norm = a.unsqueeze(0), b.unsqueeze(0), norm.unsqueeze(0)

    B, N, S, H = a.shape
    Z = weight.shape[0]
    out = torch.empty(B, N, N, Z, device=a.device, dtype=a.dtype)

    for n1 in range(0, N, chunk_size):
        n1_end = min(n1 + chunk_size, N)
        C = n1_end - n1
        a_chunk = a[:, n1:n1_end]                              # [B, C, S, H]
        outer = torch.einsum("bcsh,bnsk->bcnhk", a_chunk, b)  # [B, C, N, H, H]
        outer = outer.reshape(B, C, N, H * H)                 # [B, C, N, H^2]
        out[:, n1:n1_end] = F.linear(outer, weight, bias)      # [B, C, N, Z]

    # Normalize after all chunks (matches native _opm path exactly)
    out /= norm.unsqueeze(-1)

    if is_3d:
        out = out.squeeze(0)
    return out
