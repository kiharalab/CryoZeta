# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn

from cryozeta.model.modules.opm import opm_chunked
from cryozeta.openfold_local.model.primitives import Linear
from cryozeta.openfold_local.utils.precision_utils import is_fp16_enabled


class OuterProductMean(nn.Module):
    """
    Implements Algorithm 10.
    """

    def __init__(self, c_m, c_z, c_hidden, eps=1e-3):
        """
        Args:
            c_m:
                MSA embedding channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Hidden channel dimension
        """
        super().__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = Linear(c_m, c_hidden)
        self.linear_2 = Linear(c_m, c_hidden)
        self.linear_out = Linear(c_hidden**2, c_z, init="final")

    def _project_by_sequence(self, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Write projections directly in contraction order [N_res, N_seq, H].
        # Only one slice of normalized input is needed at a time.
        n_seq, n_res, _ = m.shape
        a = m.new_empty(n_res, n_seq, self.c_hidden)
        b = torch.empty_like(a)
        for start in range(0, n_seq, 32):
            end = min(start + 32, n_seq)
            normalized = self.layer_norm(m[start:end])
            a[:, start:end] = self.linear_1(normalized).transpose(0, 1)
            b[:, start:end] = self.linear_2(normalized).transpose(0, 1)
        return a, b

    def _forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            mask:
                [*, N_seq, N_res] MSA mask
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        unmasked = (
            mask is None
            and isinstance(self.eps, (float, int))
            and m.is_cuda
            and not torch.is_grad_enabled()
            and m.dtype == torch.bfloat16
            and 0 < m.shape[-3] <= 4096
            and not torch.compiler.is_compiling()
            and (
                not torch.is_autocast_enabled("cuda")
                or torch.get_autocast_dtype("cuda") == m.dtype
            )
        )
        if mask is None and not unmasked:
            mask = m.new_ones(m.shape[:-1])

        # The tiled projection only uses PyTorch LayerNorm and Linear.
        if (
            unmasked
            and m.ndim == 3
            and m.shape[-1] == self.c_m
            and 1024 <= m.shape[0]
            and 1024 <= m.shape[1] <= 4096
            and self.c_m == 256
            and self.c_hidden == 128
        ):
            a, b = self._project_by_sequence(m)
        else:
            # [*, N_seq, N_res, C_m]
            ln = self.layer_norm(m)

            # [*, N_seq, N_res, C]
            if mask is not None:
                mask = mask.unsqueeze(-1)
            a = self.linear_1(ln)
            if mask is not None:
                a = a * mask

            b = self.linear_2(ln)
            if mask is not None:
                b = b * mask

            del ln

            # Transpose to [*, N_res, N_seq, C]
            a = a.transpose(-2, -3).contiguous()
            b = b.transpose(-2, -3).contiguous()

        # Norm: [*, N_res, N_res]
        if mask is None:
            # Retain the two BF16 roundings of the all-ones contraction and
            # subsequent epsilon addition (important for odd sequence sizes).
            norm = m.new_full((), m.shape[-3]) + self.eps
        else:
            norm = torch.einsum("...abc,...adc->...bdc", mask, mask)
            norm = (norm + self.eps).squeeze(-1)

        return opm_chunked(a, b, self.linear_out.weight, self.linear_out.bias, norm)

    def forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        if is_fp16_enabled():
            with torch.amp.autocast("cuda", enabled=False):
                return self._forward(m.float(), mask, chunk_size, inplace_safe)
        else:
            return self._forward(m, mask, chunk_size, inplace_safe)
