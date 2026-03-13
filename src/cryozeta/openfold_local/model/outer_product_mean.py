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

import os
from functools import partial

import torch
import torch.nn as nn

from cryozeta.openfold_local.model.primitives import Linear
from cryozeta.openfold_local.utils.chunk_utils import chunk_layer
from cryozeta.openfold_local.utils.precision_utils import is_fp16_enabled
from cryozeta.model.modules.opm_tilelang import opm_chunked

try:
    from cryozeta.model.modules.opm_tilelang import opm_tilelang
    _tilelang_available = True
except (ImportError, OSError, RuntimeError):
    _tilelang_available = False


class OuterProductMean(nn.Module):
    """
    Implements Algorithm 10.
    """

    def __init__(self, c_m, c_z, c_hidden, eps=1e-3, use_tilelang=False):
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
        self.use_tilelang = use_tilelang and _tilelang_available

        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = Linear(c_m, c_hidden)
        self.linear_2 = Linear(c_m, c_hidden)
        self.linear_out = Linear(c_hidden**2, c_z, init="final")

    def _opm(self, a, b):
        # [*, N_res, N_res, C, C]
        outer = torch.einsum("...bac,...dae->...bdce", a, b)

        # [*, N_res, N_res, C * C]
        outer = outer.reshape((*outer.shape[:-2], -1))

        # [*, N_res, N_res, C_z]
        outer = self.linear_out(outer)

        return outer

    @torch.jit.ignore
    def _chunk(self, a: torch.Tensor, b: torch.Tensor, chunk_size: int) -> torch.Tensor:
        # Since the "batch dim" in this case is not a true batch dimension
        # (in that the shape of the output depends on it), we need to
        # iterate over it ourselves
        a_reshape = a.reshape((-1, *a.shape[-3:]))
        b_reshape = b.reshape((-1, *b.shape[-3:]))
        out = []
        for a_prime, b_prime in zip(a_reshape, b_reshape):
            outer = chunk_layer(
                partial(self._opm, b=b_prime),
                {"a": a_prime},
                chunk_size=chunk_size,
                no_batch_dims=1,
            )
            out.append(outer)

        # For some cursed reason making this distinction saves memory
        if len(out) == 1:
            outer = out[0].unsqueeze(0)
        else:
            outer = torch.stack(out, dim=0)

        outer = outer.reshape(a.shape[:-3] + outer.shape[1:])

        return outer

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
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        # [*, N_seq, N_res, C_m]
        ln = self.layer_norm(m)

        # [*, N_seq, N_res, C]
        mask = mask.unsqueeze(-1)
        a = self.linear_1(ln)
        a = a * mask

        b = self.linear_2(ln)
        b = b * mask

        del ln

        # Transpose to [*, N_res, N_seq, C]
        a = a.transpose(-2, -3).contiguous()
        b = b.transpose(-2, -3).contiguous()

        # Norm: [*, N_res, N_res]
        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)
        norm = (norm + self.eps).squeeze(-1)

        if self.use_tilelang:
            return opm_tilelang(
                a, b, self.linear_out.weight, self.linear_out.bias, norm
            )

        if os.environ.get("USE_OPM_CHUNKED", "1") == "1":
            return opm_chunked(
                a, b, self.linear_out.weight, self.linear_out.bias, norm
            )

        # Original einsum path
        if chunk_size is not None:
            outer = self._chunk(a, b, chunk_size)
        else:
            outer = self._opm(a, b)
        outer = outer / norm.unsqueeze(-1)
        return outer

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
