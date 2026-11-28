import pytest
import torch

from cryozeta.kernels import flashpairformer
from cryozeta.model.modules.confidence import ConfidenceHead
from cryozeta.openfold_local.model.triangular_attention import TriangleAttention
from cryozeta.openfold_local.model.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)


@pytest.mark.parametrize("backend", ["triton", "fpf_k2b"])
def test_experimental_attention_preserves_cpu_fallback(backend):
    layer = TriangleAttention(8, 4, 2).eval()
    x = torch.randn(5, 5, 8)
    with torch.no_grad():
        layer.mha.linear_o.weight.normal_()
        expected = layer(x, chunk_size=2)
        layer.mha.inference_triangle_backend = backend
        actual = layer(x, chunk_size=2)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("decline", [True, False])
def test_k2b_dispatches_with_raw_mask_and_scales_pytorch_fallback(monkeypatch, decline):
    layer = TriangleAttention(8, 4, 2).eval()
    x = torch.randn(5, 5, 8)
    mask = torch.rand(5, 5) > 0.3
    calls = []

    def k2b(q, k, v, bias, keep):
        calls.append(True)
        assert torch.equal(keep, mask[..., :, None, None, :])
        assert torch.equal(q, unscaled_q)
        return None if decline else v

    monkeypatch.setattr(flashpairformer, "try_k2b", k2b)
    with torch.no_grad():
        layer.mha.linear_o.weight.normal_()
        normalized = layer.layer_norm(x)
        unscaled_q, _, v = layer.mha._prep_qkv(
            normalized, normalized, apply_scale=False
        )
        expected = (
            layer(x, mask)
            if decline
            else layer.mha._wrap_up(v.transpose(-2, -3), normalized)
        )
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
        layer.mha.inference_triangle_backend = "fpf_k2b"
        actual = layer(x, mask)
    assert calls == [True]
    assert torch.equal(actual, expected)


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
    )
    assert torch.equal(actual, z + 1)


@pytest.mark.parametrize(
    "layer", [TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming]
)
def test_biokernels_trimul_strided_mask_and_residual(layer):
    if (
        not torch.cuda.is_available()
        or torch.__version__ != "2.15.0+cu132"
        or torch.cuda.get_device_name() != "NVIDIA RTX A6000"
    ):
        pytest.skip("Requires the qualified BioKernels TriMul CUDA stack")

    z = torch.randn(1, 129, 129, 128, dtype=torch.bfloat16, device="cuda")
    mask = (torch.rand(1, 129, 258, device="cuda") > 0.1)[..., ::2]
    module = layer(c_z=128, c_hidden=128).eval().cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        module.linear_z.weight.normal_(std=128**-0.5)
        module.use_biokernels_trimul = True
        expected = module(z, mask.contiguous())
        actual = module(z, mask, inplace_safe=True, _add_with_inplace=True)
        assert torch.isfinite(expected).all() and expected.count_nonzero() > 0
        assert torch.equal(actual, z + expected)


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
