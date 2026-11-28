"""CPU regressions for the FPF TriMul warm probe's numerical admission gate."""

import math
import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

import torch

from cryozeta.third_party.fpf.opt_core.kernels.fpf_trimul_v4 import generic


class WarmProbe(unittest.TestCase):
    def run_probe(self, source=None, value=None):
        # Provider and class outputs match unless a single outgoing element is
        # corrupted. Keep the real FP64 statement and all error reductions.
        def output(z, origin, outgoing):
            out = z.double()
            if origin == source and outgoing:
                out[0, 0, 0] = value
            return out

        def provider(z, outgoing, *_args, **_kwargs):
            return output(z, "provider", outgoing)

        def class_reference(z, _mask, *, direction, **_kwargs):
            return output(z, "class", direction == "outgoing")

        reference_blocks = generic.reference_torch_blocks

        def blocks(z, mask, *, direction, **kwargs):
            kwargs["block_rows"] = 1
            for i, j, block in reference_blocks(z, mask, direction=direction, **kwargs):
                if source == "fp64" and direction == "outgoing" and i.start == j.start == 0:
                    block[0, 0, 0] = value
                yield i, j, block

        write_stamp = Mock(return_value="test-stamp")
        with (
            patch.multiple(
                generic,
                _PROBES={},
                _PROBE_MEM=generic._PROBE_MEM.copy(),
                COUNTS={"probe": {}},
                reference_torch=class_reference,
                reference_torch_blocks=blocks,
                probe_sizes=lambda *_args: [2],
                probe_stamp_facts=lambda *_args: {"test": True},
                probe_stamp_read=lambda _facts: None,
                probe_stamp_write=write_stamp,
            ),
            patch.object(generic.CELLS, "cell_for", return_value={}),
            patch.object(generic.CELLS, "run", lambda launch, cfg, _dev: launch(cfg)),
            patch.object(generic.K, "trimul_v4_forward", provider),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(torch, "autocast", return_value=nullcontext()),
        ):
            verdict = generic.probe(torch.device("cpu"), 2, 2, False, True)
            self.assertIs(generic.probe(torch.device("cpu"), 2, 2, False, True), verdict)
            self.assertEqual(write_stamp.call_count, int(verdict["ok"]))
        return verdict

    def test_finite_match(self):
        verdict = self.run_probe()
        self.assertTrue(verdict["ok"])
        self.assertEqual((verdict["ratio_max"], verdict["ratio_rms"]), (1.0, 1.0))

    def test_nonfinite_output_in_one_block_and_direction(self):
        for source in ("provider", "class", "fp64"):
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(source=source, value=value):
                    self.assertFalse(self.run_probe(source, value)["ok"])

    def test_block_max_keeps_nonfinite_error(self):
        z = torch.zeros(2, 2, 1)
        corrupted = z.clone()
        corrupted[0, 0, 0] = math.nan
        blocks = [(slice(i, i + 1), slice(None), z[i:i + 1].double()) for i in range(2)]
        with patch.object(generic, "reference_torch_blocks", return_value=blocks):
            stats = generic.probe_error_stats(
                corrupted, corrupted, z, None, direction="outgoing", raw={"w_ap": torch.ones(1, 1)}
            )
        self.assertFalse(math.isfinite(stats["max_s"]))
        self.assertFalse(math.isfinite(stats["max_c"]))

    def test_nonfinite_sum_of_squared_errors(self):
        # A finite class value can still overflow the FP64 error statistic.
        self.assertFalse(self.run_probe("class", 1e200)["ok"])

    def test_stamp_requires_passing_ratios(self):
        api = Mock()
        api.stamp_dirs.return_value = [("test", "test-dir")]
        valid = {"ok": True, "ratio_max": generic.SAME_CLASS_MAX, "ratio_rms": generic.SAME_CLASS_RMS}
        api.stamp_read.return_value = {"record": valid}
        with patch.object(generic, "_stamp_api", return_value=api):
            self.assertEqual(generic.probe_stamp_read({"test": True})["record"], valid)
        for field in ("ratio_max", "ratio_rms"):
            for value in (math.nan, math.inf, -math.inf, valid[field] + 0.01):
                record = {"ok": True, "ratio_max": 1.0, "ratio_rms": 1.0, field: value}
                with self.subTest(field=field, value=value), patch.object(generic, "_stamp_api", return_value=api):
                    api.stamp_read.return_value = {"record": record}
                    self.assertIsNone(generic.probe_stamp_read({"test": True}))
