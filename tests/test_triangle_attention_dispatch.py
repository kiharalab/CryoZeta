import math

import pytest
import torch

from cryozeta.openfold_local.model import primitives
from cryozeta.openfold_local.model.triangular_attention import TriangleAttention


@pytest.mark.parametrize("chunk_size", [None, 2])
@pytest.mark.parametrize("use_lma", [False, True])
def test_triangle_mask_matches_cpu_fallback(chunk_size, use_lma):
    torch.manual_seed(4)
    triangle = TriangleAttention(8, 4, 2, starting=False).eval()
    x = torch.randn(5, 5, 8)
    mask = torch.rand(5, 5) > 0.3
    with torch.no_grad():
        triangle.mha.linear_o.weight.normal_()
        expected = triangle(x, mask, chunk_size=chunk_size, use_lma=use_lma)
        if use_lma:
            actual = triangle(
                x,
                mask,
                chunk_size=chunk_size,
                use_lma=True,
                use_memory_efficient_kernel=True,
                use_deepspeed_evo_attention=True,
            )
        else:
            actual = triangle(
                x,
                mask,
                chunk_size=chunk_size,
                use_cuequivariance_attention=True,
            )
    assert torch.equal(actual, expected)
    assert torch.isfinite(actual).all()


def test_cuequivariance_runtime_failure_scales_query_and_converts_mask(monkeypatch):
    torch.manual_seed(5)
    triangle = TriangleAttention(8, 4, 2).eval()
    x = torch.randn(5, 5, 8)
    mask = torch.rand(5, 5) > 0.3

    def fail(q, k, v, bias, raw_mask):
        assert torch.equal(raw_mask, mask[..., :, None, None, :])
        expected_q, _, _ = triangle.mha._prep_qkv(
            triangle.layer_norm(x), triangle.layer_norm(x)
        )
        assert torch.equal(q / math.sqrt(triangle.c_hidden), expected_q)
        raise OSError("cuEquivariance unavailable at runtime")

    with torch.no_grad():
        triangle.mha.linear_o.weight.normal_()
        expected = triangle(x, mask)
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
        monkeypatch.setattr(primitives, "cueq_would_fall_back", lambda *args: False)
        monkeypatch.setattr(primitives, "_cuequivariance_attn", fail)
        actual = triangle(x, mask, use_cuequivariance_attention=True)
    assert torch.equal(actual, expected)
