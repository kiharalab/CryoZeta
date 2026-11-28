"""Exercise fitted CIF selection through the ordinary dumper and combine entrypoint."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from biotite.structure import AtomArray
from biotite.structure.io import pdbx

from cryozeta.runner.combine import main as combine
from cryozeta.runner.dumper import DataDumper


class Combine(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.name = "test36"
        self.input = self.root / "input.json"
        self.input.write_text(json.dumps([{"name": self.name}]))
        self.final = self.root / self.name / "CryoZeta-Final"

    def dump(self, model, seed, base, confidence, scores, legacy=False):
        stage = self.root / self.name / model
        raw = torch.tensor(base, dtype=torch.float32)[:, None, None].expand(-1, 1, 3)
        pred = {
            "summary_confidence": [{"ranking_score": x} for x in confidence],
            "coordinate": raw,
            "fit_scores": [(i, "teaser", score) for i, score in enumerate(scores)],
            "coordinate_teaser": raw + 10,
        }
        self.write_prediction(model, seed, pred)
        stage_scores = stage / "saved_data" / "scores.csv"
        stage_scores.parent.mkdir(exist_ok=True)
        with stage_scores.open("a") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["pdb_id", "sample_idx", "method", "recall_ccmask_ca"])
            writer.writerows((self.name, i, "teaser", score) for i, score in enumerate(scores))

        if legacy:
            (stage / f"seed_{seed}" / "scores.csv").unlink(missing_ok=True)
            torch.save(
                {"ca_mask": np.array([True]), "pred_all_coordinate": raw},
                stage / "saved_data" / f"output_dict_{self.name}.pt",
            )
        return stage

    def write_prediction(self, model, seed, pred):
        atoms = AtomArray(1)
        for key, val in {
            "label_entity_id": ["1"], "label_asym_id": ["A"], "chain_id": ["A"],
            "res_id": [1], "res_name": ["ALA"], "atom_name": ["CA"], "element": ["C"],
        }.items():
            atoms.set_annotation(key, np.array(val))
        DataDumper(str(self.root), model).dump(
            self.name, seed, pred, atoms, {"1": "polypeptide(L)"}
        )

    def select(self, seed, count):
        combine(str(self.root), str(self.input), count, str(seed))
        return [
            float(pdbx.CIFFile.read(self.final / f"{self.name}_sample_{rank}.cif")
                  .block["atom_site"]["Cartn_x"].as_array(float)[0])
            for rank in range(count)
        ]

    def test_different_fit_methods_across_internal_seeds(self):
        from unittest.mock import Mock
        from cryozeta.model.cryozeta import CryoZeta

        def block(method, x, confidence, fit_score):
            coord = torch.full((1, 1, 3), float(x))
            p = {key: coord for key in ("coordinate", "plddt", "pae", "pde", "resolved",
                                        "point_residue_logits", "point_noise_logits")}
            p.update({f"coordinate_{m}": None for m in ("teaser", "svd_0.8", "svd_0.4", "vesper", "superimposed")})
            p[f"coordinate_{method}"] = coord + 10
            p.update(summary_confidence=[{"ranking_score": confidence}], full_data=[], fit_scores=[(0, method, fit_score)])
            return p, {}, {}

        model = Mock()
        model._main_inference_loop.side_effect = [block("teaser", 1, 0, 9), block("svd_0.8", 2, 1, 8)]
        pred, _, _ = CryoZeta.main_inference_loop(model, {}, 1, N_model_seed=2)
        self.write_prediction("CryoZeta", 12, pred)
        self.assertEqual(self.select(12, 2), [11, 12])

    def test_seed_local_scores_both_models_and_stable_ties(self):
        self.dump("CryoZeta", 34, [100, 200], [0, 1], [99, 98])
        self.dump("CryoZeta", 12, [1, 2], [0, 1], [7, 6])
        self.dump("CryoZeta-Interpolate", 12, [3, 4], [1, 1], [3, 7])
        self.assertEqual(self.select(12, 3), [11, 14, 12])

    def test_recover_legacy_confidence_ranks_by_coordinates(self):
        self.dump("CryoZeta-Interpolate", 72, [1, 2, 3], [7, 10, 10], [3, 2, 1], legacy=True)
        self.assertEqual(self.select(72, 3), [11, 12, 13])

    def test_v0_gate_checks_new_scores_against_legacy_artifacts(self):
        import shutil

        from cryozeta.kernel_bench.v0_regression import compare_baseline

        self.dump("CryoZeta", 12, [1, 2], [0, 1], [7, 6], legacy=True)
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "baseline"
            shutil.copytree(self.root, archive)
            stage = self.root / self.name / "CryoZeta"
            table = stage / "seed_12" / "scores.csv"

            def compare():
                return compare_baseline(str(archive), str(self.root))

            self.assertFalse(compare()["outputs_match"])
            header = "pdb_id,sample_idx,method,recall_ccmask_ca,cif_rank\n"
            table.write_text(header + f"{self.name},0,teaser,7,1\n{self.name},1,teaser,6,0\n")
            result = compare()
            self.assertTrue(result["outputs_match"])
            self.assertFalse(result["bitwise"])
            table.write_text(header + f"{self.name},0,teaser,7,0\n{self.name},1,teaser,6,1\n")
            self.assertFalse(compare()["outputs_match"])

    def test_reject_duplicate_legacy_coordinates(self):
        self.dump("CryoZeta", 72, [2, 2], [7, 10], [2, 1], legacy=True)
        with self.assertRaisesRegex(ValueError, "coordinate"):
            self.select(72, 1)
        self.assertFalse(self.final.exists())

    def test_reject_appended_legacy_scores(self):
        for scores in ([2, 1], [1, 3]):
            self.dump("CryoZeta", 72, [1, 2], [7, 10], scores, legacy=True)
        with self.assertRaisesRegex(ValueError, "duplicate|repeated"):
            self.select(72, 1)
        self.assertFalse(self.final.exists())

    def test_reject_unscoped_legacy_scores_with_multiple_seeds(self):
        stage = self.dump("CryoZeta", 72, [1, 2], [7, 10], [2, 1], legacy=True)
        (stage / "seed_8").mkdir()
        with self.assertRaisesRegex(ValueError, "seed identifiers"):
            self.select(72, 1)
        self.assertFalse(self.final.exists())

    def test_reject_missing_legacy_coordinates_before_copy(self):
        self.dump("CryoZeta", 72, [1, 2], [7, 10], [2, 1], legacy=True)
        stage = self.dump("CryoZeta-Interpolate", 72, [3, 4], [7, 10], [2, 1], legacy=True)
        (stage / "seed_72" / "predictions" / "atom_arrays" / f"{self.name}_sample_1.npz").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "unfitted coordinates"):
            self.select(72, 2)
        self.assertFalse(self.final.exists())

    def test_replacing_one_seed_replaces_its_scores(self):
        self.dump("CryoZeta", 12, [1, 2], [0, 1], [9, 7])
        self.dump("CryoZeta", 12, [3, 4], [0, 1], [5, 6])
        self.assertEqual(self.select(12, 2), [14, 13])
