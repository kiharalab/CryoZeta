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

from functools import partial, partialmethod

import torch
import torch.nn as nn

from cryozeta.model import kernel_routing as kr
from cryozeta.openfold_local.model.primitives import Attention, LayerNorm, Linear
from cryozeta.openfold_local.utils.chunk_utils import chunk_layer
from cryozeta.openfold_local.utils.tensor_utils import (
    permute_final_dims,
)


class TriangleAttention(nn.Module):
    def __init__(self, c_in, c_hidden, no_heads, starting=True, inf=1e9):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Overall hidden channel dimension (not per-head)
            no_heads:
                Number of attention heads
        """
        super().__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf

        self.layer_norm = LayerNorm(self.c_in)

        self.linear = Linear(c_in, self.no_heads, bias=False, init="normal")

        self.mha = Attention(
            self.c_in, self.c_in, self.c_in, self.c_hidden, self.no_heads
        )

    @torch.jit.ignore
    def _chunk(
        self,
        x: torch.Tensor,
        biases: list[torch.Tensor],
        chunk_size: int,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_cuequivariance_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        use_fpf_k2b: bool = False,
    ) -> torch.Tensor:
        "triangle! triangle!"
        mha_inputs = {
            "q_x": x,
            "kv_x": x,
            "biases": biases,
        }

        return chunk_layer(
            partial(
                self.mha,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cuequivariance_attention=use_cuequivariance_attention,
                use_lma=use_lma,
                use_fpf_k2b=use_fpf_k2b,
            ),
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(x.shape[:-2]),
            _out=x if inplace_safe else None,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_cuequivariance_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """kernel-bench routing seam: the V0 route comes from the flags; an in-scope
        replacement from CRYOZETA_KERNEL_VARIANT overrides them (kernel_routing)."""
        v0 = (
            "cueq" if use_cuequivariance_attention
            else "ds4sci" if use_deepspeed_evo_attention
            else "lma" if use_lma
            else "torch"
        )
        with kr.op("triatt", v0, x) as call:
            use_fpf_k2b = call is not None and call.requested == "fpf_k2b"
            if call is not None and call.in_scope and not use_fpf_k2b:
                p = call.requested
                use_cuequivariance_attention = p == "cueq"
                use_deepspeed_evo_attention = p == "ds4sci"
                use_lma = p == "lma"
                use_memory_efficient_kernel = False
            return self._forward(
                x,
                mask=mask,
                chunk_size=chunk_size,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cuequivariance_attention=use_cuequivariance_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                use_fpf_k2b=use_fpf_k2b,
            )

    def _forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_cuequivariance_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        use_fpf_k2b: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, I, J, C_in] input tensor (e.g. the pair representation)
        Returns:
            [*, I, J, C_in] output tensor
        """
        if mask is None:
            # [*, I, J]
            mask = x.new_ones(
                x.shape[:-1],
            )

        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        # [*, I, J, C_in]
        x = self.layer_norm(x)

        # cuEquivariance (and fpf_k2b) get raw mask; other paths get inf-scaled mask
        if use_cuequivariance_attention or use_fpf_k2b:
            mask_bias = mask[..., :, None, None, :]
        else:
            # [*, I, 1, 1, J]
            mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # [*, H, I, J]
        triangle_bias = permute_final_dims(self.linear(x), (2, 0, 1))

        # [*, 1, H, I, J]
        triangle_bias = triangle_bias.unsqueeze(-4)

        biases = [mask_bias, triangle_bias]

        if chunk_size is not None:
            x = self._chunk(
                x,
                biases,
                chunk_size,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cuequivariance_attention=use_cuequivariance_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                use_fpf_k2b=use_fpf_k2b,
            )
        else:
            x = self.mha(
                q_x=x,
                kv_x=x,
                biases=biases,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cuequivariance_attention=use_cuequivariance_attention,
                use_lma=use_lma,
                use_fpf_k2b=use_fpf_k2b,
            )

        if not self.starting:
            x = x.transpose(-2, -3)

        return x


# Implements Algorithm 13
TriangleAttentionStartingNode = TriangleAttention


class TriangleAttentionEndingNode(TriangleAttention):
    """
    Implements Algorithm 14.
    """

    __init__ = partialmethod(TriangleAttention.__init__, starting=False)
