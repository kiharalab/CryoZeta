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

import torch
import torch.nn.functional as F


def _use_reoriented_opm(a, b, weight, bias, norm, chunk_size):
    if not (
        chunk_size == 16
        and a.shape == b.shape == (1980, 1980, 128)
        and a.is_cuda
        and bias is not None
        and b.device == a.device == weight.device == bias.device == norm.device
        and a.dtype == b.dtype == norm.dtype == torch.bfloat16
        and weight.shape == (128, 128 * 128)
        and bias.shape == (128,)
        and weight.dtype == bias.dtype == torch.float32
        and norm.ndim == 0
        and a.is_contiguous()
        and b.is_contiguous()
        and weight.is_contiguous()
        and bias.is_contiguous()
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.bfloat16
        and torch.__version__ == "2.15.0+cu132"
        and torch.backends.cuda.preferred_blas_library().name == "Cublas"
        and torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is True
    ):
        return False
    return torch.cuda.get_device_name(a.device) == "NVIDIA RTX A6000"


def _opm_reoriented(a, b, weight, bias, norm):
    n, s, h = a.shape
    chunk_size = 16
    # Group contraction columns by hidden channel, then residue, so the
    # projection can consume the BF16 outer product without a layout copy.
    packed_b = b.permute(1, 2, 0).contiguous().view(1, s, h * n)
    lhs = a.new_empty(1, chunk_size * h, s)
    raw = a.new_empty(1, chunk_size * h, h * n)
    outer_view = raw.view(chunk_size, h * h, n)
    out = a.new_empty(n, n, weight.shape[0])
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        if end - start < chunk_size:
            # cuBLAS can select a different projection reduction for tails.
            del packed_b, lhs, raw, outer_view
            sequence_major_b = b.transpose(0, 1).contiguous()
            outer = torch.einsum("csh,snd->cnhd", a[start:end], sequence_major_b)
            out[start:end] = F.linear(outer.flatten(-2), weight, bias)
            del sequence_major_b, outer
            break
        lhs.view(chunk_size, h, s).copy_(a[start:end].transpose(1, 2))
        # The route requires BF16 operands because out= bypasses autocast.
        torch.bmm(lhs, packed_b, out=raw)
        projected = torch.baddbmm(
            bias[:, None], weight.expand(chunk_size, -1, -1), outer_view
        )
        out[start:end] = projected.transpose(1, 2)
        del projected
    out /= norm.view(1, 1)
    return out


def opm_chunked(a, b, weight, bias, norm, chunk_size=16):
    """Memory-efficient OPM using cuBLAS with N1-chunking.

    Memory: chunk_size * N * H^2 per chunk (vs N^2 * H^2 for full materialization).

    Runs entirely in the input dtype (BF16 or FP32). Under BF16 this matches
    the original PyTorch OPM behavior (einsum + F.linear both in BF16).
    """
    if _use_reoriented_opm(a, b, weight, bias, norm, chunk_size):
        return _opm_reoriented(a, b, weight, bias, norm)

    is_3d = False
    if a.dim() == 3:
        is_3d = True
        a, b, norm = a.unsqueeze(0), b.unsqueeze(0), norm.unsqueeze(0)

    B, N, _, H = a.shape
    Z = weight.shape[0]
    out = torch.empty(B, N, N, Z, device=a.device, dtype=a.dtype)

    # Batched GEMM over right-hand residues writes each H x H outer
    # product contiguously for the output projection. Keep the batch at one
    # to share the left operand as a stride-zero view across residues.
    grouped = (
        B == 1
        and a.shape == b.shape
        and H == 128
        and Z == 128
        and 1024 <= N <= 4096
        and 1024 <= a.shape[2] <= 4096
        and chunk_size == 16
        and a.is_cuda
        and a.dtype == b.dtype == torch.bfloat16
        and a.is_contiguous()
        and b.is_contiguous()
        and not torch.is_grad_enabled()
        and torch.cuda.get_device_capability(a.device) == (12, 0)
    )
    if (
        a.is_cuda
        and not torch.is_grad_enabled()
        and a.dtype in (torch.float16, torch.bfloat16)
        and N >= 512
        and N > chunk_size
        and not grouped
    ):
        # einsum flattens N and H for its right-hand GEMM operand. Store them
        # adjacently once so each row chunk can reuse that same matrix view.
        b = b.transpose(1, 2).contiguous().transpose(1, 2)

    for n1 in range(0, N, chunk_size):
        n1_end = min(n1 + chunk_size, N)
        C = n1_end - n1
        a_chunk = a[:, n1:n1_end]  # [B, C, S, H]
        if grouped:
            lhs = a_chunk[0].transpose(1, 2).reshape(C * H, -1).contiguous()
            outer = torch.matmul(lhs, b[0])  # [N, C * H, H]
            projected = F.linear(outer.view(N, C, H * H), weight, bias)
            out[0, n1:n1_end] = projected.transpose(0, 1)
            continue
        outer = torch.einsum("bcsh,bnsk->bcnhk", a_chunk, b)  # [B, C, N, H, H]
        outer = outer.reshape(B, C, N, H * H)  # [B, C, N, H^2]
        out[:, n1:n1_end] = F.linear(outer, weight, bias)  # [B, C, N, Z]

    # Normalize after all chunks (matches native _opm path exactly)
    out /= norm.unsqueeze(-1)

    if is_3d:
        out = out.squeeze(0)
    return out
