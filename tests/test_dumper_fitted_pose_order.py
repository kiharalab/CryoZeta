import pytest
import torch

from cryozeta.runner import dumper


@pytest.mark.parametrize("include_fits", [True, False])
@pytest.mark.parametrize("confidence_enabled", [True, False])
def test_superimposed_pose_order_and_distinct_npz_paths(
    tmp_path, monkeypatch, include_fits, confidence_enabled
):
    written = {}

    def record_structure(
        *, output_fpath, pred_atom_array_npz_path, pred_coordinate, **kwargs
    ):
        assert pred_atom_array_npz_path not in written
        written[pred_atom_array_npz_path] = pred_coordinate.item()

    monkeypatch.setattr(dumper, "save_structure_cif", record_structure)
    output = tmp_path / "output"
    arrays = tmp_path / "arrays"
    writer = dumper.DataDumper(str(output), atom_arrays_dir=str(arrays))
    pred = {
        "coordinate": torch.tensor([10, 20, 30, 40]),
        "coordinate_svd_0.4": torch.tensor([110, 120, 130, 140]),
        "coordinate_superimposed": torch.tensor([310, 320, 330, 340]),
        "summary_confidence": [
            {"ranking_score": score} for score in (0.7, 0.8, 0.1, 0.9)
        ],
    }
    if include_fits:
        pred["superimposed_fits"] = [
            {"method": "svd_0.4", "sample_idx": 2, "score": 0.5},
            {"method": None, "sample_idx": 3, "score": None},
            {"method": "teaser", "sample_idx": 0, "score": 0.7},
            {"method": None, "sample_idx": 1, "score": None},
        ]
    if not confidence_enabled:
        del pred["summary_confidence"]
    writer.dump("target", 101, pred, atom_array=object(), entity_poly_type={})

    def values(suffix):
        return [
            written[str(arrays / f"target_seed_101_sample_{rank}{suffix}.npz")]
            for rank in range(4)
        ]

    assert values("") == ([40, 20, 10, 30] if confidence_enabled else [10, 20, 30, 40])
    assert values("_svd_0.4") == (
        [140, 120, 110, 130] if confidence_enabled else [110, 120, 130, 140]
    )
    if include_fits:
        assert values("_superimposed") == (
            [330, 310, 320, 340] if confidence_enabled else [330, 310, 340, 320]
        )
    else:
        assert values("_superimposed") == (
            [340, 320, 310, 330] if confidence_enabled else [310, 320, 330, 340]
        )
    summaries = list(output.rglob("*summary_confidence*.json"))
    assert len(summaries) == (4 if confidence_enabled else 0)
