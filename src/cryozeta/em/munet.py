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

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from .blocks import (
    ConvBlock3d,
    DownSample3d,
    SpatialTransformerBlock3d,
    UpSample3d,
)

protein_ca_labels = ["BG", "N", "CA", "C", "O", "CB", "Others"]
protein_aa_labels = [
    "BG",
    "ALA",
    "VAL",
    "PHE",
    "PRO",
    "MET",
    "ILE",
    "LEU",
    "ASP",
    "GLU",
    "LYS",
    "ARG",
    "SER",
    "THR",
    "TYR",
    "HIS",
    "CYS",
    "ASN",
    "TRP",
    "GLN",
    "GLY",
]
dna_c1p_labels = ["BG", "C1'"]
rna_c1p_labels = ["BG", "C1'"]
dna_na_labels = ["BG", "A", "T", "C", "G"]
rna_na_labels = ["BG", "A", "U", "C", "G"]
atom_labels = ["BG", "BB", "CA", "CB", "sidechain", "C1'", "sugar", "phosphate", "base"]
residue_labels = [
    "background",
    "ALA",
    "VAL",
    "PHE",
    "PRO",
    "MET",
    "ILE",
    "LEU",
    "ASP",
    "GLU",
    "LYS",
    "ARG",
    "SER",
    "THR",
    "TYR",
    "HIS",
    "CYS",
    "ASN",
    "TRP",
    "GLN",
    "GLY",
    "DRNA-A",
    "DRNA-UT",
    "DRNA-C",
    "DRNA-G",
]


class MUNet(nn.Module):
    """
    ## Modern U-Net model
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 1,
        channels: int = 32,
        n_res_blocks: int = 1,
        attention_levels: list[int] | None = None,
        channel_multipliers: list[int] | None = None,
        n_heads: int = 2,
        tf_layers: int = 1,
        softmax: bool = True,
        multiclass: bool = True,
    ):
        """
        :param in_channels: is the number of channels in the input feature map
        :param out_channels: are the number of channels in the output feature map
        :param channels: is the base channel count for the models
        :param n_res_blocks: number of residual blocks at each level
        :param attention_levels: are the levels at which attention should be performed
        :param channel_multipliers: are the multiplicative factors for number of channels for each level
        :param n_heads: is the number of attention heads in the transformers
        :param tf_layers: is the number of transformer layers in the transformers
        """
        if channel_multipliers is None:
            channel_multipliers = [1, 2, 4]
        if attention_levels is None:
            attention_levels = [2]
        super().__init__()
        self.channels = channels
        self.softmax = softmax
        out_channels = n_classes

        # Number of levels
        levels = len(channel_multipliers)

        # Input half of the U-Net
        self.input_blocks = nn.ModuleList()
        # Initial $3 \times 3 \times 3$ convolution that maps the input to `channels`.
        self.input_blocks.append(nn.Conv3d(in_channels, channels, 3, padding=1))
        # Number of channels at each block in the input half of U-Net
        input_block_channels = [channels]
        # Number of channels at each level
        channels_list = [channels * m for m in channel_multipliers]

        # Prepare levels
        for i in range(levels):
            # Add the residual blocks and attentions
            for _ in range(n_res_blocks):
                # Residual block maps from previous number of channels to the number of
                # channels in the current level
                layers = [ConvBlock3d(channels, out_channels=channels_list[i])]
                channels = channels_list[i]
                # Add transformer
                if i in attention_levels:
                    layers.append(
                        SpatialTransformerBlock3d(channels, n_heads, tf_layers)
                    )
                else:
                    layers.append(ConvBlock3d(channels, kernel_size=5))
                # Add them to the input half of the U-Net and keep track of the number of channels of its output
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_channels.append(channels)
            # Down sample at all levels except last
            if i != levels - 1:
                self.input_blocks.append(DownSample3d(channels))
                input_block_channels.append(channels)

        # The middle of the U-Net
        self.middle_block = nn.Sequential(
            SpatialTransformerBlock3d(channels, n_heads, tf_layers),
        )

        # Second half of the U-Net
        self.output_blocks = nn.ModuleList([])
        # Prepare levels in reverse order
        for i in reversed(range(levels)):
            # Add the residual blocks and attentions
            for j in range(n_res_blocks + 1):
                # Residual block maps from previous number of channels plus the
                # skip connections from the input half of U-Net to the number of
                # channels in the current level.
                layers = [
                    ConvBlock3d(
                        channels + input_block_channels.pop(),
                        out_channels=channels_list[i],
                    )
                ]
                channels = channels_list[i]
                # Add transformer
                if i in attention_levels:
                    layers.append(
                        SpatialTransformerBlock3d(channels, n_heads, tf_layers)
                    )
                else:
                    layers.append(ConvBlock3d(channels, kernel_size=5))
                # Up-sample at every level after last residual block
                # except the last one.
                # Note that we are iterating in reverse; i.e. `i == 0` is the last.
                if i != 0 and j == n_res_blocks:
                    layers.append(UpSample3d(channels))
                # Add to the output half of the U-Net
                self.output_blocks.append(nn.Sequential(*layers))

        if not multiclass:
            self.final_blocks = nn.ModuleList([])

            for _ii in range(out_channels):
                self.final_blocks.append(
                    torch.nn.Conv3d(channels_list[0], 1, kernel_size=(1, 1, 1))
                )

        else:
            self.final_blocks = nn.ModuleList([])
            self.final_blocks.append(
                torch.nn.Conv3d(channels_list[0], 9, kernel_size=(1, 1, 1))
            )

    def forward(self, x: torch.Tensor, n_recycles=0):
        """
        :param x: is the input feature map of shape `[batch_size, channels, width, height]`
        """
        # To store the input half outputs for skip connections
        x_input_block = []

        # Input half of the U-Net
        for input_module in self.input_blocks:
            x = input_module(x)
            x_input_block.append(x)
        # Middle of the U-Net
        x = self.middle_block(x)
        # Output half of the U-Net
        for output_module in self.output_blocks:
            # dim=1 is the channel dimension
            x = torch.cat([x, x_input_block.pop()], dim=1)
            x = output_module(x)

        # Final Head
        if len(self.final_blocks) == 1:
            x = self.final_blocks[0](x)
            x = F.log_softmax(x, dim=1)
            return x
        else:
            out_x = []
            for final_module in self.final_blocks:
                out_x.append(final_module(x))
            return out_x

    def extract_features(self, x: torch.Tensor, n_recycles=0):
        """
        :param x: is the input feature map of shape `[batch_size, channels, width, height]`
        """
        # To store the input half outputs for skip connections
        x_input_block = []

        # Input half of the U-Net
        for input_module in self.input_blocks:
            x = input_module(x)
            x_input_block.append(x)
        # Middle of the U-Net
        x = self.middle_block(x)
        # Output half of the U-Net
        for output_module in self.output_blocks:
            # dim=1 is the channel dimension
            x = torch.cat([x, x_input_block.pop()], dim=1)
            x = output_module(x)

        # Final Head
        return x


class ResidueMUNet(nn.Module):
    """
    ## Modern U-Net model for residue-level prediction
    """

    def __init__(
        self,
        atom_mdl,
        in_channels: int = 1,
        n_classes: int = 1,
        channels: int = 32,
        n_res_blocks: int = 1,
        attention_levels: list[int] | None = None,
        channel_multipliers: list[int] | None = None,
        n_heads: int = 2,
        tf_layers: int = 1,
        softmax: bool = True,
        multiclass: bool = True,
    ):
        """
        :param atom_mdl: pretrained atom-level model (frozen)
        :param in_channels: is the number of channels in the input feature map
        :param out_channels: are the number of channels in the output feature map
        :param channels: is the base channel count for the models
        :param n_res_blocks: number of residual blocks at each level
        :param attention_levels: are the levels at which attention should be performed
        :param channel_multipliers: are the multiplicative factors for number of channels for each level
        :param n_heads: is the number of attention heads in the transformers
        :param tf_layers: is the number of transformer layers in the transformers
        """
        if channel_multipliers is None:
            channel_multipliers = [1, 2, 4]
        if attention_levels is None:
            attention_levels = [2]
        super().__init__()
        self.channels = channels
        self.softmax = softmax
        out_channels = [n_classes]

        self.atom_mdl = atom_mdl
        for param in self.atom_mdl.parameters():
            param.requires_grad = False

        # Number of levels
        levels = len(channel_multipliers)

        # Input half of the U-Net
        self.input_blocks = nn.ModuleList()
        # Initial $3 \times 3 \times 3$ convolution that maps the input to `channels`.
        self.input_blocks.append(nn.Conv3d(in_channels, channels, 3, padding=1))
        # Number of channels at each block in the input half of U-Net
        input_block_channels = [channels]
        # Number of channels at each level
        channels_list = [channels * m for m in channel_multipliers]

        # Prepare levels
        for i in range(levels):
            # Add the residual blocks and attentions
            for _ in range(n_res_blocks):
                # Residual block maps from previous number of channels to the number of
                # channels in the current level
                layers = [ConvBlock3d(channels, out_channels=channels_list[i])]
                channels = channels_list[i]
                # Add transformer
                if i in attention_levels:
                    layers.append(
                        SpatialTransformerBlock3d(channels, n_heads, tf_layers)
                    )
                else:
                    layers.append(ConvBlock3d(channels, kernel_size=5))
                # Add them to the input half of the U-Net and keep track of the number of channels of its output
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_channels.append(channels)
            # Down sample at all levels except last
            if i != levels - 1:
                self.input_blocks.append(DownSample3d(channels))
                input_block_channels.append(channels)

        # The middle of the U-Net
        self.middle_block = nn.Sequential(
            SpatialTransformerBlock3d(channels, n_heads, tf_layers),
        )

        # Second half of the U-Net
        self.output_blocks = nn.ModuleList([])
        # Prepare levels in reverse order
        for i in reversed(range(levels)):
            # Add the residual blocks and attentions
            for j in range(n_res_blocks + 1):
                # Residual block maps from previous number of channels plus the
                # skip connections from the input half of U-Net to the number of
                # channels in the current level.
                layers = [
                    ConvBlock3d(
                        channels + input_block_channels.pop(),
                        out_channels=channels_list[i],
                    )
                ]
                channels = channels_list[i]
                # Add transformer
                if i in attention_levels:
                    layers.append(
                        SpatialTransformerBlock3d(channels, n_heads, tf_layers)
                    )
                else:
                    layers.append(ConvBlock3d(channels, kernel_size=5))
                # Up-sample at every level after last residual block
                # except the last one.
                # Note that we are iterating in reverse; i.e. `i == 0` is the last.
                if i != 0 and j == n_res_blocks:
                    layers.append(UpSample3d(channels))
                # Add to the output half of the U-Net
                self.output_blocks.append(nn.Sequential(*layers))

        # Final Heads
        if not multiclass:
            self.final_blocks = nn.ModuleList([])

            for _ii in range(out_channels):
                self.final_blocks.append(
                    torch.nn.Conv3d(channels_list[0], 1, kernel_size=(1, 1, 1))
                )

        else:
            self.final_blocks = nn.ModuleList([])
            self.final_blocks.append(
                torch.nn.Conv3d(channels_list[0], n_classes, kernel_size=(1, 1, 1))
            )

    def forward(self, x: torch.Tensor, n_recycles=0):
        """
        :param x: is the input feature map of shape `[batch_size, channels, width, height]`
        """
        atm_feat = self.atom_mdl.extract_features(x)

        atom_pred = self.atom_mdl.final_blocks[0](atm_feat)
        atom_pred = F.softmax(atom_pred, dim=1)

        x_input_block = []

        # Input half of the U-Net
        for i, input_module in enumerate(self.input_blocks):
            if i == 0:
                x = input_module(x) + atm_feat
            else:
                x = input_module(x)
            x_input_block.append(x)
        # Middle of the U-Net
        x = self.middle_block(x)
        # Output half of the U-Net
        for output_module in self.output_blocks:
            # dim=1 is the channel dimension
            x = torch.cat([x, x_input_block.pop()], dim=1)
            x = output_module(x)

        # Final Head
        x = self.final_blocks[0](x)
        aa_pred = F.softmax(x, dim=1)

        output = torch.cat(
            [atom_pred, aa_pred], dim=1
        )  ## first 9 channel is for atom, rest is for aa

        return output


def get_detection_model(load_pretrained=True, compile=True):

    model = ResidueMUNet(atom_mdl=MUNet(n_classes=9), n_classes=25)

    if load_pretrained:
        model_path = "assets/cryozeta-detection-v0.0.1.safetensors"
        model.load_state_dict(load_file(model_path))

    if compile:
        model = torch.compile(model)

    return model
