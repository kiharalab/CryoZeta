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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class ConvBlock3d(nn.Module):
    """
    ## ResNet Block
    """

    def __init__(self, channels: int, out_channels=None, kernel_size=3):
        """
        :param channels: the number of input channels
        :param out_channels: is the number of out channels. defaults to `channels`.
        """
        super().__init__()
        # `out_channels` not specified
        if out_channels is None:
            out_channels = channels

        # First normalization and convolution
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv3d(channels, out_channels, 3, padding=1),
        )

        # Final convolution layer
        self.out_layers = nn.Sequential(
            normalization(out_channels),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                groups=out_channels,
            ),
        )

        # `channels` to `out_channels` mapping layer for residual connection
        if out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv3d(channels, out_channels, 1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Initial convolution
        h = self.in_layers(x)
        # Final convolution
        h = self.out_layers(h)
        # Add skip connection
        return self.skip_connection(x) + h


class BasicTransformerBlock(nn.Module):
    """Basic Transformer Layer"""

    def __init__(self, d_model: int, n_heads: int, d_head: int):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of an attention head
        """
        super().__init__()
        # Self-attention layer and pre-norm layer
        self.attn1 = SelfAttention(d_model, n_heads, d_head)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = FeedForward(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        """
        :param x: are the input embeddings of shape `[batch_size, depth * height * width, d_model]`
        """
        # Self attention
        x = self.attn1(self.norm1(x)) + x
        # Feed-forward network
        x = self.ff(self.norm2(x)) + x

        return x


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super().__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_z = torch.arange(z, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros(
            (x, y, z, self.channels * 3),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        self.cached_penc = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return self.cached_penc


class FeedForward(nn.Module):
    """
    ### Feed-Forward Network
    """

    def __init__(self, d_model: int, d_mult: int = 4):
        """
        :param d_model: is the input embedding size
        :param d_mult: is multiplicative factor for the hidden layer size
        """
        super().__init__()
        self.net = nn.Sequential(
            GeGLU(d_model, d_model * d_mult),
            nn.Linear(d_model * d_mult, d_model),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class GeGLU(nn.Module):
    """
    ### GeGLU Activation

    $$\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$$
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        # Combined linear projections $xW + b$ and $xV + c$
        self.proj = nn.Linear(d_in, d_out * 2)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor):
        # Get $xW + b$ and $xV + c$
        x, gate = self.proj(x).chunk(2, dim=-1)
        # $\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$
        return x * self.gelu(gate)


class SelfAttention(nn.Module):
    """
    ### Self Attention Layer
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_head: int,
        use_flash_attention: bool = True,
    ):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of an attention head
        """
        super().__init__()

        self.n_heads = num_heads
        self.d_head = dim_head
        d_attn = dim_head * num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_attn, bias=False)
        self.o_proj = nn.Linear(d_attn, d_model)

        if use_flash_attention:
            try:
                from flash_attn import flash_attn_func

                self.flash = True
                self._flash_attention = flash_attn_func
            except ImportError:
                self.flash = False
        else:
            self.flash = False

    def forward(self, x: torch.Tensor):
        """
        :param x: are the input embeddings of shape `[batch_size, depth * height * width, d_model]`
        """
        batch_size, seq_len, _ = x.size()

        # Get query, key and value vectors
        qkv = self.qkv_proj(x)  # Shape: (batch_size, seq_len, 3*d_attn)
        q, k, v = qkv.chunk(3, dim=-1)  # Shape: (batch_size, seq_len, d_attn)

        q = q.view(batch_size, seq_len, self.n_heads, self.d_head)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head)

        if self.flash:
            output = self._flash_attention(q, k, v)
        else:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            output = F.scaled_dot_product_attention(q, k, v)
            output = output.permute(0, 2, 1, 3)

        return self.o_proj(output.reshape(batch_size, seq_len, -1))


class GroupNorm32(nn.GroupNorm):
    """
    ### Group normalization with float32 casting
    """

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    """
    ### Group normalization

    This is a helper function, with fixed number of groups.
    """
    return GroupNorm32(32, channels)


class UpSample3d(nn.Module):
    """
    ### Up-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # Apply a convolution to refine the upsampled features
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.scale_factor = 2

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, depth, height, width]`
        """
        # Upsample and then apply convolution
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        return self.conv(x)


class DownSample3d(nn.Module):
    """
    ## Down-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # $3 \times 3 \times 3$ convolution with stride length of $2$ to down-sample by a factor of $2$
        self.op = nn.Conv3d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width, depth]`
        """
        # Apply convolution
        return self.op(x)


class SpatialTransformerBlock3d(nn.Module):
    """
    ## Spatial Transformer
    """

    def __init__(self, channels: int, n_heads: int, n_layers: int):
        """
        :param channels: is the number of channels in the feature map
        :param n_heads: is the number of attention heads
        :param n_layers: is the number of transformer layers
        """
        super().__init__()
        # Initial group normalization
        self.norm = torch.nn.GroupNorm(
            num_groups=32, num_channels=channels, eps=1e-6, affine=True
        )
        # Initial $1 \times 1$ convolution
        self.proj_in = nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0)

        self.positional_encoding = PositionalEncoding3D(channels)

        # Transformer layers
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(channels, n_heads, channels // n_heads)
                for _ in range(n_layers)
            ]
        )

        # Final $1 \times 1$ convolution
        self.proj_out = nn.Conv3d(
            channels, channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor):
        """
        :param x: is the feature map of shape `[batch_size, channels, depth, height, width]`
        """
        # Get shape `[batch_size, channels, depth, height, width]`
        b, c, h, w, d = x.shape
        # For residual connection
        x_in = x
        # Normalize
        x = self.norm(x)
        # Initial $1 \times 1$ convolution
        x = self.proj_in(x)
        # Transpose and reshape from `[batch_size, channels, depth, height, width]`
        # to `[batch_size, depth * height * width, channels]`
        x = x.permute(0, 2, 3, 4, 1)

        pos_emb = self.positional_encoding(x)
        x += pos_emb

        x = x.view(b, h * w * d, c)

        # Apply the transformer layers
        for block in self.transformer_blocks:
            x = block(x)
        # Reshape and transpose from `[batch_size, height * width * depth, channels]`
        # to `[batch_size, channels, height, width]`
        x = x.view(b, h, w, d, c).permute(0, 4, 1, 2, 3)
        # Final $1 \times 1$ convolution
        x = self.proj_out(x)
        # Add residual
        return x + x_in
