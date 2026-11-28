import pytest
import torch

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
        actual = triangle(
            x,
            mask,
            chunk_size=chunk_size,
            use_lma=use_lma,
            use_memory_efficient_kernel=True,
            use_deepspeed_evo_attention=True,
        )
    assert torch.equal(actual, expected)
    assert torch.isfinite(actual).all()
