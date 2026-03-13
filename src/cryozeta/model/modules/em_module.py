# Copyright (C) 2026 KiharaLab, Purdue University
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import sys
from functools import partialmethod

import torch
import torch.nn as nn

from cryozeta.model.modules.primitives import Transition
from cryozeta.openfold_local.model.dropout import DropoutRowwise
from cryozeta.openfold_local.model.msa import MSAAttention
from cryozeta.openfold_local.model.primitives import LayerNorm, Linear
from cryozeta.openfold_local.model.triangular_attention import TriangleAttention
from cryozeta.openfold_local.model.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from cryozeta.openfold_local.utils.chunk_utils import chunk_layer
from cryozeta.openfold_local.utils.tensor_utils import add


# Openfold version for partial functions: https://github.com/aqlaboratory/openfold/blob/447670c03d00534007b3f1f51ef5be9b19efaca8/openfold/model/evoformer.py
class MSATransition(nn.Module):
    """
    Feed-forward network applied to MSA activations after attention.

    Implements Algorithm 9
    """

    def __init__(self, c_m, n):
        """
        Args:
            c_m:
                MSA channel dimension
            n:
                Factor multiplied to c_m to obtain the hidden channel
                dimension
        """
        super().__init__()

        self.c_m = c_m
        self.n = n

        self.layer_norm = LayerNorm(self.c_m)
        self.linear_1 = Linear(self.c_m, self.n * self.c_m, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.n * self.c_m, self.c_m, init="final")

    def _transition(self, m, mask):
        m = self.layer_norm(m)
        m = self.linear_1(m)
        m = self.relu(m)
        m = self.linear_2(m) * mask
        return m

    @torch.jit.ignore
    def _chunk(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        return chunk_layer(
            self._transition,
            {"m": m, "mask": mask},
            chunk_size=chunk_size,
            no_batch_dims=len(m.shape[:-2]),
        )

    def forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA activation
            mask:
                [*, N_seq, N_res, C_m] MSA mask
        Returns:
            m:
                [*, N_seq, N_res, C_m] MSA activation update
        """
        # DISCREPANCY: DeepMind forgets to apply the MSA mask here.
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        mask = mask.unsqueeze(-1)

        if chunk_size is not None:
            m = self._chunk(m, mask, chunk_size)
        else:
            m = self._transition(m, mask)

        return m


class PointResidueTransition(MSATransition):
    """
    Point-Residue transition network applied before row/column outer product mean.

    """

    def __init__(self, c_pz, n):
        """
        Args:
            c_pz:
                Point-residue channel dimension
            n:
                Factor multiplied to c_pz to obtain the hidden channel
                dimension
        """
        super().__init__(c_pz, n)


class ResidueRowAttentionWithPairBias(MSAAttention):
    """
    Implementation for Support row-wise gated self-attention with residue pair bias.
    """

    def __init__(self, c_pz, c_z, c_hidden, no_heads, inf=1e9):
        """
        Args:
            c_z:
                residue representation channel dimension
            c_p:
                point pair representation channel dimension, new added, should be in config.py
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of heads
            inf:
                A very large number
        """
        super().__init__(c_pz, c_hidden, no_heads, pair_bias=True, c_z=c_z, inf=inf)


##Modified by Yuanyuan
class PointColumnAttentionWithPairBias(nn.Module):
    """
    Implementation for Support column-wise gated self-attention with point pair bias.
    """

    def __init__(self, c_pz, c_p, c_hidden, no_heads, inf=1e9):
        """
        Args:
            c_pz:
                point-residue pair representation channel dimension, new added, should be in config.py
            c_p:
                point pair representation channel dimension, new added, should be in config.py
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of heads
            inf:
                A very large number
        """
        super().__init__()
        self.c_pz = c_pz
        self.c_p = c_p
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf
        self.column_att = MSAAttention(
            c_in=c_pz,
            c_hidden=c_hidden,
            no_heads=no_heads,
            pair_bias=True,
            c_z=c_p,
            inf=inf,
        )

    def forward(
        self,
        pz: torch.Tensor,
        p: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        use_flash: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            pz:
                [*, N_point, N_res, C_pz] Point Residue embedding
            p:
                [*, N_point, N_point, C_z] Residue embedding
            mask:
                [*, N_seq, N_res] MSA mask
            chunk_size:
                Size of chunks into which the inputs are split along their
                batch dimensions. A low value decreases memory overhead at the
                cost of slower execution. Chunking is not performed by default.
        """
        # [*, N_res, N_point, C_pz]
        pz = pz.transpose(-2, -3)
        if mask is not None:
            mask = mask.transpose(-1, -2)

        pz = self.column_att(
            m=pz,
            z=p,
            mask=mask,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            use_flash=use_flash,
        )

        # [*, N_point, N_res, C_pz]
        pz = pz.transpose(-2, -3)
        if mask is not None:
            mask = mask.transpose(-1, -2)
        return pz


class InterMultiplicativeUpdate(nn.Module):
    """
    Implements Algorithm 5 proposed by CryoZeta.
    """

    def __init__(self, c_z, c_p, c_hidden, c_pz, _outcomming=True):
        """
        Args:
            c_z:
                residue representation channel dimension
            c_p:
                point pair representation channel dimension, new added, should be in config.py
            c_hidden:
                Hidden channel dimension
            c_pz:
                output channel dimension, same as c_pz
            _outcomming:
                Whether the update is outcomming or incoming
        """
        super().__init__()

        self.c_z = c_z
        self.c_p = c_p
        self.c_hidden = c_hidden
        self.c_pz = c_pz
        self._outcomming = _outcomming

        # define initial projection linear layers
        self.linear_l_p = Linear(
            self.c_p, self.c_hidden
        )  # linear projection taking W_left as output
        self.linear_r_p = Linear(
            self.c_z, self.c_hidden
        )  # linear projection taking W_right as output
        # define gated linear layers
        self.linear_l_g = Linear(self.c_p, self.c_hidden, init="gating")
        self.linear_r_g = Linear(self.c_z, self.c_hidden, init="gating")

        # define final projection linear layers
        self.linear_z = Linear(self.c_hidden, self.c_pz, init="final")

        # define layer norm layers
        self.layer_norm_in_l = LayerNorm(self.c_p)
        self.layer_norm_in_r = LayerNorm(self.c_z)
        self.layer_norm_out = LayerNorm(self.c_hidden)

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        z: torch.Tensor,
        p: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            z:
                [*, N_res, N_res, C_z], residue pair representation

            p:
                [*, N_point, N_point, C_z], point pair representation
            mask:
                [*, N_res, N_res, pair mask
        Returns:
            [*, N_point, N_res, C_pz] output tensor
        """
        if mask is None:
            mask = z.new_ones(z.shape[:-1])
        mask = mask.unsqueeze(-1)

        x_l = self.linear_l_p(self.layer_norm_in_l(p))
        x_r = mask * self.linear_r_p(self.layer_norm_in_r(z))

        g_l = self.sigmoid(self.linear_l_g(x_l))
        g_r = self.sigmoid(self.linear_r_g(x_r))

        x_l = g_l * x_l
        x_r = g_r * x_r
        if self._outcomming:
            z_l = torch.einsum("...ijk->...ik", [x_l])
            z_r = torch.einsum("...ijk->...ik", [x_r])
        else:
            z_l = torch.einsum("...ijk->...jk", [x_l])
            z_r = torch.einsum("...ijk->...jk", [x_r])
        # out = torch.zeros(x_point.shape[1], x_res.shape[1], self.c_z)
        out = torch.einsum("...ik,...jk->...ijk", z_l, z_r)
        out = self.linear_z(self.layer_norm_out(out))
        return out


class InterMultiplicativeUpdateOutcomming(InterMultiplicativeUpdate):
    """
    Implements Algorithm 5 proposed by CryoZeta with agg_method=outcomming.
    """

    __init__ = partialmethod(InterMultiplicativeUpdate.__init__, _outcomming=True)


class InterMultiplicativeUpdateIncoming(InterMultiplicativeUpdate):
    """
    Implements Algorithm 5 proposed by CryoZeta with agg_method=incoming.
    """

    __init__ = partialmethod(InterMultiplicativeUpdate.__init__, _outcomming=False)


## Reference openfold version: https://github.com/aqlaboratory/openfold/blob/447670c03d00534007b3f1f51ef5be9b19efaca8/openfold/model/evoformer.py
class PairUpdate(nn.Module):
    """
    Implementing pair update module, combined triangle layers by Yuanyuan
    """

    def __init__(
        self,
        c_z,
        no_heads_pair,
        c_hidden,
        c_hidden_pair_att,
        transition_n,
        pair_dropout,
        eps=1e-3,
        inf=1e9,
    ):
        """
        Args:
            c_z:
                point or residue representation channel dimension
            no_heads_pair:
                number of attention heads for pair attention
            c_hidden:
                Hidden channel dimension
            c_hidden_pair_att:
                Hidden channel dimension for pair attention
            transition_n:
                number of transition layers
            pair_dropout:
                dropout rate for pair attention
            eps:
                small value for numerical stability
            inf:
                large value for numerical stability
        """
        super().__init__()
        self.c_z = c_z
        self.no_heads_pair = no_heads_pair
        self.c_hidden = c_hidden
        self.c_hidden_pair_att = c_hidden_pair_att
        self.transition_n = transition_n
        self.pair_dropout = pair_dropout
        self.eps = eps
        self.inf = inf

        # stacking triangle layers
        self.tri_multi_out = TriangleMultiplicationOutgoing(c_z, c_hidden)
        self.tri_multi_in = TriangleMultiplicationIncoming(c_z, c_hidden)
        self.tri_attention_start = TriangleAttention(
            c_z, c_hidden_pair_att, no_heads_pair, inf=inf
        )
        self.tri_attention_end = TriangleAttention(
            c_z, c_hidden_pair_att, no_heads_pair, inf=inf
        )
        self.pair_transition = Transition(c_z, transition_n)
        self.pair_dropout_row = DropoutRowwise(pair_dropout)

    def forward(
        self,
        input_tensor: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cuequivariance_attention: bool = False,
        use_cuequivariance_multiplicative_update: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: int | None = None,
        _offload_inference: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x_point:
                [*, N_point, N_point, C_z] input tensor 1
            x_res:
                [*, N_res, N_res, C_z] input tensor 2
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_point, N_res, C_z] output tensor
        """
        z = input_tensor
        if _attn_chunk_size is None:
            _attn_chunk_size = chunk_size

        tri_multi_update = self.tri_multi_out(
            z, pair_mask, inplace_safe=inplace_safe, _add_with_inplace=True,
            use_cuequivariance_multiplicative_update=use_cuequivariance_multiplicative_update,
        )
        if not inplace_safe:
            z = z + self.pair_dropout_row(tri_multi_update)
        else:
            z = tri_multi_update

        del tri_multi_update

        tri_multi_update = self.tri_multi_in(
            z,
            mask=pair_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
            use_cuequivariance_multiplicative_update=use_cuequivariance_multiplicative_update,
        )
        if not inplace_safe:
            z = z + self.pair_dropout_row(tri_multi_update)
        else:
            z = tri_multi_update

        del tri_multi_update

        z = add(
            z,
            self.pair_dropout_row(
                self.tri_attention_start(
                    z,
                    mask=pair_mask,
                    chunk_size=_attn_chunk_size,
                    use_memory_efficient_kernel=False,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_cuequivariance_attention=use_cuequivariance_attention,
                    use_lma=use_lma,
                    inplace_safe=inplace_safe,
                )
            ),
            inplace=inplace_safe,
        )

        z = z.transpose(-2, -3)
        if inplace_safe:
            input_tensor = z.contiguous()
            z = input_tensor

        z = add(
            z,
            self.pair_dropout_row(
                self.tri_attention_end(
                    z,
                    mask=pair_mask.transpose(-1, -2),
                    chunk_size=_attn_chunk_size,
                    use_memory_efficient_kernel=False,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_cuequivariance_attention=use_cuequivariance_attention,
                    use_lma=use_lma,
                    inplace_safe=inplace_safe,
                )
            ),
            inplace=inplace_safe,
        )

        z = z.transpose(-2, -3)
        if inplace_safe:
            input_tensor = z.contiguous()
            z = input_tensor

        z = add(
            z,
            self.pair_transition(z),
            inplace=inplace_safe,
        )

        if _offload_inference and inplace_safe:
            device = z.device
            del z
            assert sys.getrefcount(input_tensor) == 2
            input_tensor = input_tensor.to(device)
            z = input_tensor

        return z
