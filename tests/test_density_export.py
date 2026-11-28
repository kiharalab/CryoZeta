import numpy as np
import torch

from cryozeta.runner import dumper


def test_density_filter_uses_coordinates_of_each_export(tmp_path, monkeypatch):
    written = {}

    def record_structure(*, output_fpath, pred_coordinate, **kwargs):
        written[output_fpath] = pred_coordinate.tolist()

    monkeypatch.setattr(dumper, "save_structure_cif", record_structure)
    writer = dumper.DataDumper(str(tmp_path))
    writer.dump(
        "target",
        101,
        {
            "coordinate": torch.tensor([[10, 11], [20, 21]]),
            "coordinate_superimposed": torch.tensor([[110, 111], [120, 121]]),
            "summary_confidence": [{"ranking_score": 0.1}, {"ranking_score": 0.9}],
        },
        atom_array=np.array([0, 1]),
        entity_poly_type={},
        atom_masks={
            "coordinate": [np.array([False, True]), np.array([True, False])],
            "coordinate_superimposed": [np.array([True, True]), np.array([False, True])],
        },
    )

    root = tmp_path / "target" / "seed_101"
    assert written[str(root / "predictions" / "target_sample_0.cif")] == [20]
    assert written[str(root / "predictions_superimposed" / "target_sample_0.cif")] == [121]
    assert written[str(root / "predictions_superimposed" / "target_sample_1.cif")] == [110, 111]
