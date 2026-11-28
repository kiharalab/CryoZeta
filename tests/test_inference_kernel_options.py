import pytest
import torch

from cryozeta.kernels import flashpairformer
from cryozeta.model.modules.confidence import ConfidenceHead
from cryozeta.openfold_local.model import primitives
from cryozeta.openfold_local.model.triangular_attention import TriangleAttention
from cryozeta.openfold_local.model.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
)


@pytest.mark.parametrize("backend", ["triton", "fpf_k2b"])
def test_experimental_attention_preserves_cpu_fallback(backend):
    layer = TriangleAttention(8, 4, 2).eval()
    x = torch.randn(5, 5, 8)
    with torch.no_grad():
        layer.mha.linear_o.weight.normal_()
        expected = layer(x, chunk_size=2, use_cuequivariance_attention=True)
        layer.mha.inference_triangle_backend = backend
        actual = layer(x, chunk_size=2, use_cuequivariance_attention=True)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("decline", [True, False])
def test_k2b_can_replace_cuequivariance_on_the_requested_path(monkeypatch, decline):
    calls = []

    def k2b(q, k, v, bias, mask):
        calls.append("k2b")
        assert q.ndim == 5 and bias.dtype == torch.float32
        return None if decline else v

    def cueq(*, q, k, v, bias, mask):
        calls.append("cueq")
        return v

    monkeypatch.setattr(flashpairformer, "try_k2b", k2b)
    monkeypatch.setattr(primitives, "_cueq_triangle_attention", cueq, raising=False)
    q, k, v = [torch.randn(3, 2, 5, 4) for _ in range(3)]
    bias = torch.randn(1, 2, 5, 5, dtype=torch.bfloat16)
    mask = torch.ones(3, 1, 1, 5, dtype=torch.bool)
    actual = primitives._cuequivariance_attn(q, k, v, bias, mask, use_fpf_k2b=True)
    assert torch.equal(actual, v.transpose(-2, -3))
    assert calls == (["k2b", "cueq"] if decline else ["k2b"])


def test_fpf_trimul_receives_residual_mode(monkeypatch):
    module = TriangleMultiplicationOutgoing(c_z=8, c_hidden=8).eval()
    z = torch.randn(5, 5, 8)
    mask = torch.ones(5, 5)

    def trimul(m, x, keep, *, residual):
        assert m is module and x is z and keep is mask and residual
        return x + 1

    monkeypatch.setattr(flashpairformer, "try_trimul", trimul)
    module.use_fpf_trimul = True
    actual = module(
        z,
        mask,
        inplace_safe=True,
        _add_with_inplace=True,
        use_cuequivariance_multiplicative_update=True,
    )
    assert torch.equal(actual, z + 1)


def test_confidence_transition_is_opt_in():
    default = ConfidenceHead(n_blocks=1)
    triton = ConfidenceHead(n_blocks=1, use_triton_transition=True)
    assert (
        default.pairformer_stack.blocks[0].pair_transition.inference_hidden_budget_bytes
        is None
    )
    assert (
        triton.pairformer_stack.blocks[0].pair_transition.inference_hidden_budget_bytes
        == 1024**3
    )
