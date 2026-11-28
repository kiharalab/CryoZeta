"""Routing, M-CLEAR and adapter-contract checks on CPU.

The FPF adapters run end to end through CryoZeta's routed modules with the vendored fp64 torch
references standing in for the Triton kernels: this checks layout, mask, scale, direction, weight
mapping and residual wiring, not kernel numerics (those need a GPU, op_bench).

  CUDA_VISIBLE_DEVICES= pixi run -e default --frozen python -m unittest discover -s tests/kernel_bench \
      -p test_routing.py -v
"""

import contextlib
import glob
import inspect
import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from cryozeta.model import kernel_providers as kp
from cryozeta.model import kernel_routing as kr
from cryozeta.openfold_local.model.triangular_attention import TriangleAttention
from cryozeta.openfold_local.model.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VARIANTS = os.path.join(REPO, "docs", "kernel-bench", "variants")
needs_docs = unittest.skipUnless(os.path.isdir(VARIANTS), "needs the local docs/kernel-bench variant YAMLs, which are not tracked")
F64 = torch.float64


def randomized(m, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.5)
    return m.double().eval()


class State(unittest.TestCase):
    """Restores the routing globals and the provider module cache after each test."""

    def setUp(self):
        self._saved = kr.VARIANT, kr._COUNTERS_PATH, dict(kp._MODS)
        kr.reset()

    def tearDown(self):
        kr.VARIANT, kr._COUNTERS_PATH = self._saved[0], self._saved[1]
        kp._MODS.clear()
        kp._MODS.update(self._saved[2])
        kr.reset()

    def call(self, op, v0, served, reason=None, site="empairformer"):
        tok = kr._SITE.set(site)
        try:
            with kr.op(op, v0, torch.zeros(1)):
                if reason:
                    kr.ineligible(reason)
                kr.served(served)
        finally:
            kr._SITE.reset(tok)

    def counters(self):
        return kr.snapshot()["counters"]


class Routing(State):
    def test_inert_default(self):
        kr.VARIANT, kr._COUNTERS_PATH = kr.Variant(), None
        with kr.op("triatt", "cueq", torch.zeros(1)) as c:
            self.assertIsNone(c)
        self.assertEqual(self.counters(), [])

    def test_strict(self):
        kr.VARIANT = kr.Variant(strict=True, replace={("triatt", "empairformer"): "fpf_k2b"})
        with self.assertRaises(kr.KernelUnsupported):
            self.call("triatt", "cueq", "cueq", reason="ineligible_hw")
        with self.assertRaises(kr.KernelFallback):
            self.call("triatt", "cueq", "cueq")
        self.call("triatt", "cueq", "fpf_k2b")
        self.call("triatt", "cueq", "cueq", site="msa")  # outside the scope: V0 route, no error

    def test_v0_composite_rescues(self):
        kr._COUNTERS_PATH = os.path.join(tempfile.mkdtemp(), "c.json")
        self.call("triatt", "cueq", "ds4sci", reason="ineligible_shape")  # allowed V0 rescue
        self.call("apb", "cueq", "torch", reason="ineligible_shape")  # allowed V0 rescue
        self.call("triatt", "cueq", "torch")  # not allowed
        self.call("trimul", "cueq", "torch")  # not allowed
        got = {(c["op"], c["served_backend"]): c["route_allowed"] for c in self.counters()}
        self.assertEqual(got, {("triatt", "ds4sci"): True, ("apb", "torch"): True,
                               ("triatt", "torch"): False, ("trimul", "torch"): False})

    def test_load_variant(self):
        bad = ["policy_id: V0", "variant_id: x\nfoo: 1", "variant_id: x\nclear_cache: sometimes",
               "variant_id: x\nstrict: 'yes'",
               "variant_id: x\nreplace: [{op: trimul, call_site: msa, provider: fpf_trimul_v4, min_tokens: 0}]",
               "variant_id: x\nreplace: [{op: triatt, call_site: msa, provider: cueq, min_tokens: 101}]",
               "variant_id: x\nreplace: [{op: trimul, call_site: msa, provider: fpf_trimul_v4, min_token: 101}]",
               "variant_id: x\nreplace: [{op: triatt, call_site: msa, provider: tilelang}]",
               "variant_id: x\nreplace: [{op: triatt, call_site: nowhere, provider: torch}]",
               "variant_id: x\nreplace: [{op: opm, call_site: msa, provider: einsum},"
               " {op: opm, call_site: msa, provider: chunked}]"]
        with tempfile.TemporaryDirectory() as d:
            for i, text in enumerate(bad):
                p = os.path.join(d, f"{i}.yaml")
                with open(p, "w") as f:
                    f.write(text)
                with self.subTest(text), self.assertRaises(ValueError):
                    kr.load_variant(p)
        self.assertEqual(kr.load_variant(None), kr.Variant())

    @needs_docs
    def test_shipped_variants_load(self):
        paths = glob.glob(os.path.join(VARIANTS, "*.yaml"))
        self.assertTrue(paths)
        for p in paths:
            with self.subTest(os.path.basename(p)):
                kr.load_variant(p)

    def test_clear_policy(self):
        for policy, below, above in (("legacy", False, True), ("always", True, True), ("never", False, False)):
            kr.VARIANT = kr.Variant(clear_cache=policy)
            self.assertEqual((kr.clear_between_blocks(2000, False), kr.clear_between_blocks(2001, False)),
                             (below, above), policy)
            self.assertFalse(kr.clear_between_blocks(5000, True))


@needs_docs
class VariantCli(unittest.TestCase):
    """Every shipped variant's cli_args resolve through the real config parser to the value it names."""

    def test_cli_args_resolve(self):
        from cryozeta.runner.inference import build_configs

        for p in sorted(glob.glob(os.path.join(VARIANTS, "*.yaml"))):
            v = kr.load_variant(p)
            pairs = list(zip(v.cli_args[::2], v.cli_args[1::2]))
            argv, sys.argv = sys.argv, ["test", "--use_interpolation", "false", *v.cli_args]
            try:
                cfg = build_configs()
            finally:
                sys.argv = argv
            for flag, want in pairs:
                got = cfg
                for k in flag[2:].split("."):
                    got = got[k]
                with self.subTest(variant=v.variant_id, flag=flag):
                    self.assertEqual(str(got).lower(), want.lower())


class OpBenchClone(unittest.TestCase):
    def test_expand_view_kept(self):
        from cryozeta.kernel_bench.op_bench import clone

        z = torch.randn(4, 4, 3).expand(5, 4, 4, 3)
        y = clone(z, torch.float64)
        self.assertEqual((y.shape, y.stride()[0], y.dtype), (z.shape, 0, torch.float64))
        self.assertTrue(torch.equal(y.float(), z))
        self.assertNotEqual(y.data_ptr(), z.data_ptr())


class _Stub(torch.nn.Module):
    def forward(self, *args, **kwargs):
        return args


class MClearPerStack(State):
    """Every stack routes its blocks through run_block with its own site label and the M-CLEAR flag."""

    EXPECTED = {"empairformer", "msa", "confidence", "other", "diffusion", "atom"}

    @classmethod
    def setUpClass(cls):
        from cryozeta.model.cryozeta import CryoZeta
        from cryozeta.runner.inference import build_configs

        argv, sys.argv = sys.argv, ["test", "--use_interpolation", "false"]
        try:
            configs = build_configs()
        finally:
            sys.argv = argv
        with torch.device("meta"):
            model = CryoZeta(configs)
        cls.stacks = {k: m for k, m in model.named_modules() if hasattr(m, "_prep_blocks") and hasattr(m, "kb_site")}

    def test_labels(self):
        sites = {k: m.kb_site for k, m in self.stacks.items()}
        self.assertEqual(set(sites.values()), self.EXPECTED, sites)
        for k, s in sites.items():
            self.assertEqual(s == "atom", ".atom_transformer." in k + ".", k)

    def test_forward_uses_policy(self):
        for k, m in self.stacks.items():
            src = inspect.getsource(type(m).forward)
            with self.subTest(k):
                self.assertIn("kr.clear_between_blocks(z.shape[-2], self.training)", src)
                self.assertIn("clear_cache_between_blocks=clear_cache_between_blocks", src)

    def test_block_clear_counts(self):
        for policy, n, want in (("legacy", 2000, 0), ("legacy", 2001, 2), ("always", 16, 2), ("never", 5000, 0)):
            kr.VARIANT = kr.Variant(clear_cache=policy)
            for k, m in self.stacks.items():
                kr.reset()
                blocks, m.blocks = m.blocks, torch.nn.ModuleList([_Stub(), _Stub()])
                try:
                    sig = inspect.signature(m._prep_blocks).parameters
                    args = {p: None for p, v in sig.items() if v.default is inspect.Parameter.empty}
                    for fn in m._prep_blocks(**args, clear_cache_between_blocks=kr.clear_between_blocks(n, False)):
                        fn()
                finally:
                    m.blocks = blocks
                with self.subTest(policy=policy, n=n, stack=k):
                    self.assertEqual(kr.snapshot()["block_clears"].get(m.kb_site, 0), want)


class CCat(unittest.TestCase):
    def test_cache(self):
        m = randomized(TriangleMultiplicationOutgoing(c_z=8, c_hidden=4)).float()
        w = kp.cat_weights(m)
        torch.testing.assert_close(w["p_in_weight"], torch.cat([m.linear_a_p.weight, m.linear_b_p.weight]))
        torch.testing.assert_close(w["g_in_bias"], torch.cat([m.linear_a_g.bias, m.linear_b_g.bias]))
        self.assertIs(kp.cat_weights(m)["p_in_weight"], w["p_in_weight"])  # reused
        with torch.no_grad():
            m.linear_a_p.weight.add_(1.0)  # in-place weight change -> rebuilt
        torch.testing.assert_close(kp.cat_weights(m)["p_in_weight"][:4], m.linear_a_p.weight)
        self.assertFalse([k for k in m.state_dict() if "_kb_" in k])
        m.double()
        self.assertEqual(kp.cat_weights(m)["p_in_weight"].dtype, F64)


class CCatForward(State):
    """C-CAT through TriMul.forward: strict ("trimul", "other") -> cueq_cached_cat serves, reuses one cached concat
    across calls and equals the V0 cueq path bit for bit. On CPU this exercises cuequivariance's CPU path; the CUDA
    kernel's numerics are covered only on a GPU (v0_regression / op_bench)."""

    def test_forward(self):
        g = torch.Generator().manual_seed(3)
        real, seen = kp.cat_weights, []

        def spy(m):
            w = real(m)
            seen.append(w["p_in_weight"])
            return w

        for cls in (TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming):
            for residual in (False, True):
                m = randomized(cls(c_z=16, c_hidden=8)).float()
                z = torch.randn(1, 12, 12, 16, generator=g)
                mask = (torch.rand(1, 12, 12, generator=g) > 0.2).float()
                kw = dict(inplace_safe=residual, _add_with_inplace=residual,
                          use_cuequivariance_multiplicative_update=True)
                kr.reset()
                seen.clear()
                with torch.no_grad():
                    kr.VARIANT = kr.Variant()
                    ref = m(z.clone(), mask, **kw)
                    kr.VARIANT = kr.Variant(strict=True, replace={("trimul", "other"): "cueq_cached_cat"})
                    with mock.patch.object(kp, "cat_weights", side_effect=spy) as cw:
                        outs = [m(z.clone(), mask, **kw) for _ in range(2)]
                c = [x for x in self.counters() if x["op"] == "trimul"]
                with self.subTest(cls=cls.__name__, residual=residual):
                    for out in outs:
                        torch.testing.assert_close(out, ref, rtol=0, atol=0)
                    self.assertEqual(cw.call_count, 2)
                    self.assertIs(seen[0], seen[1])  # cached, not re-concatenated
                    self.assertEqual([(x["call_site"], x["requested_backend"], x["served_backend"], x["count"])
                                      for x in c], [("other", "cueq_cached_cat", "cueq_cached_cat", 2)])
                    with torch.enable_grad(), self.assertRaises(kr.KernelUnsupported):  # inference-only cache
                        m(z.clone(), mask, **kw)


class PlcStageArgs(unittest.TestCase):
    """P-LC: inference flags after ``--`` reach every stage's configs through cycle_predict's real parsers (its
    argparse, then parse_configs per stage); malformed pairs, flags a stage sets itself, attention-algorithm conflicts
    and the unwired use_flash are refused in parse_args, before any stage work."""

    NATIVE_OFF = ["--use_deepspeed_evo_attention", "false", "--use_cuequivariance_attention", "false"]

    @classmethod
    def setUpClass(cls):
        from cryozeta.runner import cycle_predict

        cls.cp = cycle_predict
        cls.tmp = tempfile.TemporaryDirectory()
        cls.base = os.path.join(cls.tmp.name, "base.json")
        with open(cls.base, "w") as f:
            f.write("[]")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def parse(self, native=(), extra=()):
        argv = ["--base_json", self.base, "--load_checkpoint_path", "/x.safetensors", "--dump_root", self.tmp.name,
                *native, *(["--", *extra] if extra else [])]
        return self.cp.parse_args(argv)

    def stages(self, args):
        seen, d = [], Path(self.tmp.name)
        with mock.patch.object(self.cp, "inference_main", side_effect=seen.append), \
                mock.patch.dict(self.cp.configs_base):
            for i in (1, 2):
                self.cp.run_stage(stage_name=f"stage{i}", temp_json=Path(self.base), dump_dir=d / f"dump{i}",
                                  atom_arrays_dir=d / f"aa{i}", args=args)
        return seen

    def test_forwarded_to_every_stage(self):
        default = self.stages(self.parse())[0]
        cases = (([], ["--infer_setting.chunk_size", "128"], dict(chunk=128, lma=False, ds=True, cueq=True)),
                 ([], ["--infer_setting.chunk_size", "None"], dict(chunk=None, lma=False, ds=True, cueq=True)),
                 (self.NATIVE_OFF, ["--use_lma", "true"], dict(chunk=default.infer_setting.chunk_size, lma=True,
                                                                ds=False, cueq=False)))
        self.assertNotIn(default.infer_setting.chunk_size, (128, None))
        self.assertFalse(default.use_lma)
        for native, extra, want in cases:
            for i, c in enumerate(self.stages(self.parse(native, extra))):
                with self.subTest(extra=extra, stage=i):
                    self.assertEqual(dict(chunk=c.infer_setting.chunk_size, lma=c.use_lma,
                                          ds=c.use_deepspeed_evo_attention, cueq=c.use_cuequivariance_attention),
                                     want)
                    self.assertEqual((c.model.input_embedder.p_dim, c.load_checkpoint_path), (99, "/x.safetensors"))

    def test_refusals(self):
        bad = (["--seeds", "7"], ["--use_lma=true"], ["--use_lma"], ["--foo", "1"],
               ["--use_deepspeed_evo_attention", "false"], ["--model.input_embedder.p_dim", "5"],
               ["--em_file_dir", "/x"], ["--dump", "x"], ["--infer_setting.chunk_size", "abc"],
               ["--use_lma", "true", "--use_lma", "false"], ["--use_lma", "--seeds"], ["--use_lma", "true false"],
               ["--use_lma", "true"], ["--use_memory_efficient_kernel", "true"],  # native DS4Sci/cueq attention still on
               ["--use_flash", "true"], ["--atom_arrays_dir", "/x"])  # use_flash: no config reaches the model
        for extra in bad:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as e:
                self.parse(extra=extra)
            self.assertEqual(e.exception.code, 2, extra)
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
            self.parse(self.NATIVE_OFF, ["--use_lma", "true", "--use_memory_efficient_kernel", "true"])
        self.assertIn("at most one attention algorithm", err.getvalue())
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
            self.parse(self.NATIVE_OFF, ["--use_memory_efficient_kernel", "true"])  # alone: Attention has no branch
        self.assertIn("use_memory_efficient_kernel: Attention has no such kernel", err.getvalue())


class FpfAdapters(State):
    def test_trimul_v4(self):
        G = kp._load("trimul")
        kp._MODS["trimul"] = types.SimpleNamespace(
            supported=lambda *a, **k: (True, ""), pack_weights=lambda **kw: kw,
            TrimulUnsupported=G.TrimulUnsupported,
            trimul=lambda z, mask, *, direction, weights, residual: G.reference_torch(
                z, mask, direction=direction, residual=residual, dtype=F64, **weights))
        g = torch.Generator().manual_seed(1)
        for cls in (TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming):
            for residual in (False, True):
                m = randomized(cls(c_z=16, c_hidden=8))
                z = torch.randn(1, 12, 12, 16, generator=g, dtype=F64)
                mask = (torch.rand(1, 12, 12, generator=g) > 0.2).to(F64)
                kw = dict(inplace_safe=residual, _add_with_inplace=residual)
                kr.VARIANT = kr.Variant()
                ref = m(z.clone(), mask, **kw)
                kr.VARIANT = kr.Variant(strict=True, replace={("trimul", "other"): "fpf_trimul_v4"})
                with self.subTest(cls=cls.__name__, residual=residual):
                    torch.testing.assert_close(m(z.clone(), mask, **kw), ref)

    def test_k2b(self):
        K = kp._load("k2b")
        kp._MODS["k2b"] = types.SimpleNamespace(
            flash_supported=lambda *a: (True, ""),
            flash_triangle_attention=lambda q, k, v, bias, mask: K.reference_triangle_attention(
                q, k, v, bias, mask, dtype=F64))
        g = torch.Generator().manual_seed(2)
        for starting in (True, False):
            m = randomized(TriangleAttention(c_in=16, c_hidden=8, no_heads=2, starting=starting))
            x = torch.randn(1, 10, 10, 16, generator=g, dtype=F64)
            mask = (torch.rand(1, 10, 10, generator=g) > 0.2).to(F64)
            mask[..., 0] = 1  # no fully masked row
            mask[..., 0, :] = 1
            kr.VARIANT = kr.Variant()
            ref = m(x, mask)
            kr.VARIANT = kr.Variant(strict=True, replace={("triatt", "other"): "fpf_k2b"})
            with self.subTest(starting=starting):
                torch.testing.assert_close(m(x, mask), ref)

    @needs_docs
    @torch.no_grad()
    def test_x34_short_stage_uses_baseline(self):
        path = os.path.join(VARIANTS, "M-TCHUNK-mem-ES-TMV4A-OFF-NBM-X34.yaml")
        variant = kr.load_variant(path)
        m = randomized(TriangleMultiplicationOutgoing(c_z=16, c_hidden=8)).float()
        for site, n in (("msa", 63), ("empairformer", 94), ("confidence", 100)):
            z = torch.randn(1, n, n, 16)
            kr.VARIANT = kr.Variant()
            ref = m(z.clone())
            kr.VARIANT = variant
            with self.subTest(site=site, n=n):
                got = kr.run_block(site, False, m, z.clone())
                torch.testing.assert_close(got, ref, rtol=0, atol=0)
                c, = [c for c in self.counters() if c["call_site"] == site]
                self.assertEqual((c["requested_backend"], c["served_backend"], c["reason"]),
                                 ("torch", "torch", "below_min_tokens"))
                self.assertTrue(c["route_allowed"])
                self.assertFalse(c["in_replacement_scope"])
        with self.assertRaises(kr.KernelUnsupported):  # at the floor, CPU must still refuse FPF
            kr.run_block("confidence", False, m, torch.randn(1, 101, 101, 16))

    def test_cpu_refusals(self):
        """The real FPF support checks refuse CPU tensors; strict scope turns that into UNSUPPORTED."""
        kr.VARIANT = kr.Variant(strict=True, replace={("trimul", "other"): "fpf_trimul_v4",
                                                      ("triatt", "other"): "fpf_k2b"})
        z = torch.randn(1, 8, 8, 16)
        with self.assertRaises(kr.KernelUnsupported):
            randomized(TriangleMultiplicationOutgoing(c_z=16, c_hidden=8)).float()(z)
        with self.assertRaises(kr.KernelUnsupported):
            randomized(TriangleAttention(c_in=16, c_hidden=8, no_heads=2)).float()(z, torch.ones(1, 8, 8))


class Ds4sciDtype(unittest.TestCase):
    def test_biases_follow_q(self):
        """K5: under bf16 autocast q is bf16 but an fp32-mask bias stays fp32; the DS4Sci kernel needs one dtype."""
        from cryozeta.openfold_local.model import primitives as P

        seen = []

        def fake(q, k, v, biases):
            seen.extend(t.dtype for t in (q, k, v, *biases))
            return torch.zeros_like(q)

        bf = torch.bfloat16
        q = torch.randn(1, 4, 2, 4, 8, dtype=bf)
        biases = [torch.zeros(1, 4, 1, 1, 4), torch.zeros(1, 1, 2, 4, 4, dtype=bf)]
        with mock.patch.object(P, "ds4s_is_installed", True), \
                mock.patch.object(P, "DS4Sci_EvoformerAttention", fake, create=True):
            o = P._deepspeed_evo_attn(q, q, q, biases)
        self.assertEqual(set(seen), {bf})
        self.assertEqual((o.shape, o.dtype), (q.transpose(-2, -3).shape, bf))  # returns [*, Q, H, C]


class TransitionChunks(unittest.TestCase):
    def test_count(self):
        """M-TCHUNK-mem: confidence-head pair Transition at N = 1344 (c_z 128, n 4, fp32) needs 8.32 GB live."""
        from cryozeta.model.modules.primitives import transition_chunks

        rows, gib = 1344 * 1344, 2**30
        self.assertEqual(transition_chunks(rows, 128, 4, 64 * gib), 1)  # fits: V0 behavior
        self.assertEqual(transition_chunks(rows, 128, 4, 4 * gib), 4)   # 8.32e9 <= 4 * 0.5 * 4 GiB
        self.assertEqual(transition_chunks(5, 128, 4, 0), 5)             # capped at one row per chunk

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA only")
    def test_cuda_chunked_matches_unchunked(self):
        from cryozeta.model.modules import primitives as P

        m = randomized(P.Transition(c_in=128, n=4)).float().cuda()
        x = torch.randn(256, 256, 128, device="cuda")
        with torch.no_grad():
            ref = m(x)
            with mock.patch.dict(os.environ, {"CRYOZETA_TRANSITION_CHUNK": "memory"}), \
                    mock.patch.object(P, "transition_chunks", return_value=8) as tc:
                got = m(x)
        self.assertEqual(tc.call_count, 1)
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
