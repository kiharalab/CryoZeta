# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Modifications Copyright 2026 KiharaLab, Purdue University.
#
# This file is included in a GPLv3-licensed project. The original
# code remains under Apache-2.0; the combined work is distributed
# under GPLv3.

import torch
import torch.nn as nn
from torch.nn import Linear

from cryozeta.model.modules.primitives import LinearNoBias


# Adapted From openfold.model.heads
class DistogramHead(nn.Module):
    """Implements Algorithm 1 [Line17] in AF3
    Computes a distogram probability distribution.
    For use in computation of distogram loss, subsection 1.9.8 (AF2)
    """

    def __init__(self, c_z: int = 128, no_bins: int = 64) -> None:
        """
        Args:
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
            no_bins (int, optional): Number of distogram bins. Defaults to 64.
        """
        super().__init__()

        self.c_z = c_z
        self.no_bins = no_bins

        self.linear = Linear(in_features=self.c_z, out_features=self.no_bins)

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # [*, N, N, C_z]
        """
        Args:
            z (torch.Tensor): pair embedding
                [*, N_token, N_token, C_z]

        Returns:
            torch.Tensor: distogram probability distribution
                [*, N_token, N_token, no_bins]
        """
        # [*, N, N, no_bins]
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits


class PointResidueClassHead(nn.Module):
    """
    For use in computation of point residue class loss, subsection 1.9.11
    """

    def __init__(self, c_p, c_out=10, **kwargs):
        """
        Args:
            c_p:
                Input channel dimension
            c_out:
                Number of distogram bins
        """
        super().__init__()

        self.c_p = c_p
        self.c_out = c_out

        self.linear = LinearNoBias(self.c_p, self.c_out)

    def forward(self, p):
        """
        Args:
            p:
                [*, N_point, N_res, C_p] point residue representation
        Returns:
            [*, N_point, N_res, C_out] logits
        """
        # [*, N_point, N_res, C_out]
        logits = self.linear(p)

        return logits


class PointNoiseHead(nn.Module):
    """
    Class for detecting noise points among support points. A noise point is defined
    as being at least 5Å away from any ground truth carbon alpha atoms.
    The input is a point residue representation which is then processed via a row-wise sum
    to be used as input for a linear classification layer.
    """

    def __init__(self, c_p, c_out=2):
        """
        Initializes the PointNoiseHead class which is used to detect noise points.

        Args:
            c_p (int): Input channel dimension after row-wise sum (features per point).
            c_out (int): Output dimension, typically 2 for binary classification (noise or not noise).
        """
        super().__init__()
        self.c_p = c_p
        self.c_out = c_out

        # Linear layer to process the input features and output the classification logits
        self.linear = nn.Linear(self.c_p, self.c_out)

    def forward(self, point_features):
        """
        Forward pass to compute logits for noise detection based on the row-wise summed input features.

        Args:
            point_features (torch.Tensor): [*, N_point, N_res, C_p] Input feature tensor for each support point,
                                           before row-wise summing.

        Returns:
            torch.Tensor: [*, N_point, C_out] Logits indicating the probability of each point being noise.
        """
        # Sum features across the residue dimension
        summed_features = torch.sum(point_features, dim=2)  # Sum across N_res dimension

        # Compute the logits using the linear layer
        logits = self.linear(summed_features)

        return logits
