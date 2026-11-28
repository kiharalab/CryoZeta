"""Regression tests for portable OPM input-projection tiling.

CPU-only. Routing tests exercise the real forward gate end to end with the
projection helper and contraction mocked, using large shapes that satisfy
the real eligibility predicate (CUDA residency is emulated on CPU tensors).
The single numerical parity check compares both projection outputs of the
tiled helper against the untiled computation bitwise.
"""

import importlib

import pytest
import torch

opm_mod = importlib.import_module("cryozeta.openfold_local.model.outer_product_mean")
OuterProductMean = opm_mod.OuterProductMean

C_M = 256
C_Z = 128
C_HIDDEN = 128


def make_opm(dtype=torch.bfloat16):
    torch.manual_seed(0)
    opm = OuterProductMean(C_M, C_Z, C_HIDDEN)
    return opm.to(dtype)


@pytest.fixture
def fake_cuda(monkeypatch):
    """Make CPU tensors look CUDA-resident to the inference gates."""
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))


# ---------------- numerical parity (single small check) ----------------


def test_tiled_projection_matches_untiled_bitwise():
    opm = make_opm()
    torch.manual_seed(1)
    m = torch.randn(1031, 41, C_M, dtype=torch.bfloat16)  # ragged final tile
    with torch.no_grad():
        a, b = opm._project_by_sequence(m)
        ln = opm.layer_norm(m)
        a_ref = opm.linear_1(ln).transpose(0, 1).contiguous()
        b_ref = opm.linear_2(ln).transpose(0, 1).contiguous()
    assert torch.equal(a, a_ref)
    assert torch.equal(b, b_ref)
    assert a.shape == (41, 1031, C_HIDDEN)
    assert a.is_contiguous() and b.is_contiguous()


# ---------------- forward-gate routing ----------------


def route(opm, m, mask, monkeypatch):
    """Run the real forward with projection/contraction mocked; report routing."""
    calls = {}

    def fake_project(self, x):
        calls["tiled"] = calls.get("tiled", 0) + 1
        n_seq, n_res, _ = x.shape
        a = torch.zeros(n_res, n_seq, self.c_hidden, dtype=x.dtype)
        return a, torch.empty_like(a)

    def fake_contraction(a, b, weight, bias, norm):
        calls["contraction"] = calls.get("contraction", 0) + 1
        return a.new_empty(0)

    monkeypatch.setattr(OuterProductMean, "_project_by_sequence", fake_project)
    monkeypatch.setattr(opm_mod, "opm_chunked", fake_contraction)
    with torch.no_grad():
        out = opm(m, mask)
    return calls, out


@pytest.mark.parametrize("shape", [(1024, 1031), (1031, 1024), (1980, 1980)])
def test_large_unmasked_inputs_tile(shape, fake_cuda, monkeypatch):
    opm = make_opm()
    m = torch.empty(*shape, C_M, dtype=torch.bfloat16)
    calls, _ = route(opm, m, None, monkeypatch)
    assert calls.get("tiled") == 1
    assert calls.get("contraction") == 1


def test_masked_input_does_not_tile(fake_cuda, monkeypatch):
    opm = make_opm()
    m = torch.empty(1500, 1200, C_M, dtype=torch.bfloat16)
    mask = torch.ones(1500, 1200, dtype=torch.bfloat16)
    calls, _ = route(opm, m, mask, monkeypatch)
    assert "tiled" not in calls


def test_batched_input_does_not_tile(fake_cuda, monkeypatch):
    opm = make_opm()
    m = torch.empty(2, 1200, 1100, C_M, dtype=torch.bfloat16)
    calls, _ = route(opm, m, None, monkeypatch)
    assert "tiled" not in calls


def test_cpu_input_does_not_tile(monkeypatch):
    # No fake_cuda fixture: a genuinely CPU-resident input fails the
    # unmasked inference gate, so the untiled path must be used.
    opm = make_opm()
    m = torch.empty(1200, 1100, C_M, dtype=torch.bfloat16)
    calls, _ = route(opm, m, None, monkeypatch)
    assert "tiled" not in calls


def test_compiling_does_not_tile(fake_cuda, monkeypatch):
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    opm = make_opm()
    m = torch.empty(1500, 1200, C_M, dtype=torch.bfloat16)
    calls, _ = route(opm, m, None, monkeypatch)
    assert "tiled" not in calls


def test_small_input_does_not_tile(fake_cuda, monkeypatch):
    opm = make_opm()
    m = torch.empty(1200, 500, C_M, dtype=torch.bfloat16)
    calls, _ = route(opm, m, None, monkeypatch)
    assert "tiled" not in calls


def test_oversized_input_does_not_tile(fake_cuda, monkeypatch):
    opm = make_opm()
    m = torch.empty(1200, 4097, C_M, dtype=torch.bfloat16)
    calls, _ = route(opm, m, None, monkeypatch)
    assert "tiled" not in calls


def test_fp32_input_does_not_tile(fake_cuda, monkeypatch):
    opm = make_opm(torch.float32)
    m = torch.empty(1500, 1200, C_M, dtype=torch.float32)
    calls, _ = route(opm, m, None, monkeypatch)
    assert "tiled" not in calls
