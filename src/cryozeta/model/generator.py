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

from collections.abc import Callable
from itertools import pairwise
from typing import Any

import torch
from tqdm import tqdm

from cryozeta.model.utils import centre_random_augmentation


class InferenceNoiseScheduler:
    """
    Scheduler for noise-level (time steps)
    """

    def __init__(
        self,
        s_max: float = 160.0,
        s_min: float = 4e-4,
        rho: float = 7,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
    ) -> None:
        """Scheduler parameters

        Args:
            s_max (float, optional): maximal noise level. Defaults to 160.0.
            s_min (float, optional): minimal noise level. Defaults to 4e-4.
            rho (float, optional): the exponent numerical part. Defaults to 7.
            sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
        """
        self.sigma_data = sigma_data
        self.s_max = s_max
        self.s_min = s_min
        self.rho = rho
        print(f"inference scheduler {self.sigma_data}")

    def __call__(
        self,
        N_step: int = 200,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Schedule the noise-level (time steps). No sampling is performed.

        Args:
            N_step (int, optional): number of time steps. Defaults to 200.
            device (torch.device, optional): target device. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): target dtype. Defaults to torch.float32.

        Returns:
            torch.Tensor: noise-level (time_steps)
                [N_step+1]
        """
        step_size = 1 / N_step
        step_indices = torch.arange(N_step + 1, device=device, dtype=dtype)
        t_step_list = (
            self.sigma_data
            * (
                self.s_max ** (1 / self.rho)
                + step_indices
                * step_size
                * (self.s_min ** (1 / self.rho) - self.s_max ** (1 / self.rho))
            )
            ** self.rho
        )
        # replace the last time step by 0
        t_step_list[..., -1] = 0  # t_N = 0

        return t_step_list


def sample_diffusion(
    denoise_net: Callable,
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    noise_schedule: torch.Tensor,
    N_sample: int = 1,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    diffusion_chunk_size: int | None = None,
    inplace_safe: bool = False,
    attn_chunk_size: int | None = None,
    use_cuequivariance_attention_pair_bias: bool = False,
    pair_bias_cache_mib: int = 0,
    cache_atom_conditioning: bool = False,
) -> torch.Tensor:
    """Implements Algorithm 18 in AF3.
    It performances denoising steps from time 0 to time T.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        denoise_net (Callable): the network that performs the denoising step.
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, c_s]
        z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, c_z]
        noise_schedule (torch.Tensor): noise-level schedule (which is also the time steps) since sigma=t.
            [N_iterations]
        N_sample (int): number of generated samples
        gamma0 (float): params in Alg.18.
        gamma_min (float): params in Alg.18.
        noise_scale_lambda (float): params in Alg.18.
        step_scale_eta (float): params in Alg.18.
        diffusion_chunk_size (Optional[int]): Chunk size for diffusion operation. Defaults to None.
        inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
        attn_chunk_size (Optional[int]): Chunk size for attention operation. Defaults to None.
        pair_bias_cache_mib (int): Memory budget for reusable token attention biases, in MiB.
        cache_atom_conditioning (bool): Reuse reference atom features across diffusion steps.

    Returns:
        torch.Tensor: the denoised coordinates of x in inference stage
            [..., N_sample, N_atom, 3]
    """
    N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
    batch_shape = s_inputs.shape[:-2]
    device = s_inputs.device
    dtype = s_inputs.dtype

    # Pre-compute pair conditioning once (invariant across all denoising steps).
    # This avoids recomputing the [N_token, N_token, 2*c_z] cat + projection
    # 200 times, saving ~2GB peak VRAM per step at N=2000.
    cached_pair_z = denoise_net.compute_pair_conditioning(
        z_trunk=z_trunk,
        input_feature_dict=input_feature_dict,
        inplace_safe=inplace_safe,
    )

    denoise_options = {}
    if noise_schedule.numel() > 2 and (N_sample == 1 or diffusion_chunk_size == 1):
        if pair_bias_cache_mib > 0:
            biases = denoise_net.diffusion_transformer.prepare_pair_bias_cache(
                cached_pair_z,
                pair_bias_cache_mib * 2**20,
                use_cuequivariance_attention_pair_bias,
            )
            if biases:
                denoise_options["cached_pair_biases"] = biases
        if cache_atom_conditioning and cached_pair_z.is_cuda:
            encoder = denoise_net.atom_attention_encoder
            if not encoder.training and not torch.is_grad_enabled():
                denoise_options["cached_atom_conditioning"] = (
                    encoder.prepare_conditioning(
                        input_feature_dict,
                        s_trunk.unsqueeze(-3),
                        cached_pair_z.unsqueeze(-4),
                        inplace_safe,
                    )
                )

    noise_steps = tuple(pairwise(noise_schedule))

    def _chunk_sample_diffusion(chunk_n_sample):
        x_l = noise_schedule[0] * torch.randn(
            size=(*batch_shape, chunk_n_sample, N_atom, 3), device=device, dtype=dtype
        )

        for c_tau_last, c_tau in tqdm(
            noise_steps,
            desc="Diffusion denoising",
            leave=False,
            dynamic_ncols=True,
        ):
            x_l = (
                centre_random_augmentation(
                    x_input_coords=x_l, N_sample=1, no_center=False
                )
                .squeeze(dim=-3)
                .to(dtype)
            )

            gamma = float(gamma0) if c_tau > gamma_min else 0
            t_hat = c_tau_last * (gamma + 1)
            delta_noise_level = torch.sqrt(t_hat**2 - c_tau_last**2)
            x_noisy = x_l + noise_scale_lambda * delta_noise_level * torch.randn(
                size=x_l.shape, device=device, dtype=dtype
            )

            # Expand the denoising time to the sample batch.
            t_hat = (
                t_hat.reshape((1,) * (len(batch_shape) + 1))
                .expand(*batch_shape, chunk_n_sample)
                .to(dtype)
            )

            x_denoised = denoise_net(
                x_noisy=x_noisy,
                t_hat_noise_level=t_hat,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                chunk_size=attn_chunk_size,
                inplace_safe=inplace_safe,
                use_cuequivariance_attention_pair_bias=use_cuequivariance_attention_pair_bias,
                cached_pair_z=cached_pair_z,
                **denoise_options,
            )

            delta = (x_noisy - x_denoised) / t_hat[
                ..., None, None
            ]  # Line 9 of AF3 uses 'x_l_hat' instead, which we believe  is a typo.
            dt = c_tau - t_hat
            x_l = x_noisy + step_scale_eta * dt[..., None, None] * delta

        return x_l

    if diffusion_chunk_size is None:
        return _chunk_sample_diffusion(N_sample)

    chunks = []
    no_chunks = N_sample // diffusion_chunk_size + (
        N_sample % diffusion_chunk_size != 0
    )
    for i in tqdm(
        range(no_chunks),
        desc="Diffusion sample chunks",
        leave=False,
        dynamic_ncols=True,
    ):
        chunk_n_sample = min(diffusion_chunk_size, N_sample - i * diffusion_chunk_size)
        chunks.append(_chunk_sample_diffusion(chunk_n_sample))
    return torch.cat(chunks, dim=-3)
