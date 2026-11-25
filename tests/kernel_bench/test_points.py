"""Effective point-count (P) axis: synthetic detection subsampling/tiling and the real CPU
featurizer against the requested cell. Needs torch; the featurizer case also needs the CCD assets
(CRYOZETA_ASSETS_DIR), otherwise it is skipped.

  CRYOZETA_ASSETS_DIR=$KB/assets CUDA_VISIBLE_DEVICES= pixi run -e default --frozen \
      python -m unittest discover -s tests/kernel_bench -p test_points.py -v
"""

import os
import tempfile
import unittest

import torch

from cryozeta.kernel_bench import common as C
from cryozeta.kernel_bench.synth_inputs import tile_detection


def det(n0):
    """Every field encodes its source index i, so alignment after selection is checkable."""
    i = torch.arange(n0, dtype=torch.float32)
    return {"main_atom_coords": torch.stack([i, 2 * i, 3 * i], 1), "main_atom_probs": i[:, None].repeat(1, 3),
            "res_features": i[:, None].repeat(1, 28), "cluster_ids": torch.arange(n0) // 4,
            "point_extra": i.clone(), "emdb_id": "EMD-0",
            "interpolate_confidence": (i[:, None] * 1000 + i[None, :])[..., None, None].repeat(1, 1, 5, 3)}


class Subsample(unittest.TestCase):
    def test_aligned_fields_and_both_ic_axes(self):
        out, info = tile_detection(det(50), 20, seed=3)
        self.assertEqual(info["mode"], "subsampled")
        idx = out["main_atom_coords"][:, 0].long()
        self.assertEqual(len(idx), 20)
        self.assertTrue(bool((idx[1:] > idx[:-1]).all()))  # sorted, no duplicates
        f = idx.float()
        self.assertTrue(torch.equal(out["main_atom_coords"], torch.stack([f, 2 * f, 3 * f], 1)))
        self.assertTrue(torch.equal(out["main_atom_probs"][:, 0], f))
        self.assertTrue(torch.equal(out["res_features"][:, 5], f))
        self.assertTrue(torch.equal(out["point_extra"], f))  # unknown point-aligned key follows too
        self.assertTrue(torch.equal(out["cluster_ids"], idx // 4))  # each point keeps its own cluster
        self.assertEqual(out["emdb_id"], "EMD-0")
        ic = out["interpolate_confidence"]
        self.assertEqual(tuple(ic.shape), (20, 20, 5, 3))
        self.assertTrue(torch.equal(ic[..., 0, 0], f[:, None] * 1000 + f[None, :]))  # row and column axes

    def test_reproducible(self):
        a, ia = tile_detection(det(50), 20, seed=3)
        b, ib = tile_detection(det(50), 20, seed=3)
        c, ic = tile_detection(det(50), 20, seed=4)
        self.assertEqual(ia["indices_sha256"], ib["indices_sha256"])
        self.assertTrue(torch.equal(a["main_atom_coords"], b["main_atom_coords"]))
        self.assertNotEqual(ia["indices_sha256"], ic["indices_sha256"])

    def test_unchanged_and_bad_ic(self):
        out, info = tile_detection(det(20), 20)
        self.assertEqual(info["mode"], "unchanged")
        self.assertEqual(len(out["main_atom_coords"]), 20)
        bad = det(20)
        bad["interpolate_confidence"] = bad["interpolate_confidence"][:, :19]
        with self.assertRaises(ValueError):
            tile_detection(bad, 10)


class Tile(unittest.TestCase):
    def test_tiles_exactly(self):
        src = det(7)
        out, info = tile_detection(src, 20, gap=10.0)
        self.assertEqual((info["mode"], info["copies"]), ("tiled", 3))
        for k in ("main_atom_coords", "main_atom_probs", "res_features", "cluster_ids", "point_extra"):
            self.assertEqual(len(out[k]), 20, k)
        shift = info["x_shift_angstrom"]
        self.assertEqual(shift, 6.0 + 10.0)
        self.assertTrue(torch.allclose(out["main_atom_coords"][7], src["main_atom_coords"][0] + torch.tensor([shift, 0, 0])))
        cid = out["cluster_ids"]
        self.assertTrue(set(cid[:7].tolist()).isdisjoint(cid[7:14].tolist()))  # copies never share a cluster
        ic = out["interpolate_confidence"]
        self.assertEqual(tuple(ic.shape), (20, 20, 5, 3))
        self.assertTrue(torch.equal(ic[7:14, 7:14], src["interpolate_confidence"]))
        self.assertEqual(float(ic[:7, 7:].abs().sum()), 0.0)  # block-diagonal: no cross-copy confidence


def _assets():
    try:
        from cryozeta.configs.configs_data import data_configs

        return os.path.isfile(data_configs["ccd_components_file"])
    except Exception:  # noqa: BLE001
        return False


@unittest.skipUnless(_assets(), "CCD assets not found (set CRYOZETA_ASSETS_DIR)")
class RealFeaturizer(unittest.TestCase):
    """json_to_feature on CPU: requested P vs the model-input shapes, via the real record_input hook."""

    def features(self, d, name, detection):
        from cryozeta.data.json_to_feature import SampleDictToFeatures

        path = os.path.join(d, name, "CryoZeta-Detection")
        os.makedirs(path)
        torch.save(detection, os.path.join(path, f"{name}.pt"))
        entry = {"name": name, "sequences": [{"proteinChain": {"sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLI",
                                                                "count": 1}}]}
        return SampleDictToFeatures(entry, d).get_feature_dict()[0]

    def record(self, name, feats):
        from cryozeta.model import kernel_routing as kr

        saved = kr._COUNTERS_PATH
        kr._COUNTERS_PATH, kr._REC["inputs"] = "unused", []
        try:
            kr.record_input(name, feats)
            return list(kr._REC["inputs"])
        finally:
            kr._COUNTERS_PATH = saved
            kr.reset()

    def test_requested_p_reaches_the_model(self):
        src = {k: v for k, v in det(40).items() if k not in ("point_extra", "emdb_id")}
        src["main_atom_coords"] = torch.rand(40, 3, generator=torch.Generator().manual_seed(0)) * 40
        n_token, want = 30, 25
        with tempfile.TemporaryDirectory() as d:
            fixed = self.features(d, "sub", tile_detection(src, want, seed=0)[0])
            old = self.features(d, "old", src)  # earlier generator: detection left at 40 points
        self.assertEqual(tuple(fixed["em_support_points"].shape), (want, 3))
        self.assertEqual(tuple(fixed["p"].shape[:2]), (want, want))
        self.assertEqual(tuple(fixed["pz"].shape[:2]), (want, n_token))
        rec = self.record("sub", fixed)
        self.assertEqual(rec[0]["valid_points"], want)
        self.assertEqual(C.check_cell({"sub": (n_token, want)}, rec), ("match", ""))
        # old behaviour: the featurizer prunes 40 -> min(N_token, 2000) = 30, not the requested 25
        self.assertEqual(old["em_support_points"].shape[0], 30)
        status, detail = C.check_cell({"old": (n_token, want)}, self.record("old", old))
        self.assertEqual(status, "mismatch")
        self.assertIn("P=30", detail)


if __name__ == "__main__":
    unittest.main()
