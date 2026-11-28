"""Regression tests for entity handling in combine_stages on multi-chain stages."""

import numpy as np
import gemmi
from biotite.structure import AtomArray

from cryozeta.data.utils import load_atom_array_npz, save_atom_array_npz
from cryozeta.runner.combine_stages import combine_npz_to_cif


def _make_chain(chain_id: str, entity_id: str, n_atoms: int = 3) -> AtomArray:
    arr = AtomArray(n_atoms)
    arr.coord = np.tile(np.arange(n_atoms, dtype=float)[:, None], (1, 3))
    arr.set_annotation("atom_name", np.array(["N", "CA", "C"][:n_atoms]))
    arr.set_annotation("element", np.array(["N", "C", "C"][:n_atoms]))
    arr.set_annotation("res_name", np.array(["MET"] * n_atoms))
    arr.set_annotation("res_id", np.ones(n_atoms, dtype=int))
    arr.set_annotation("chain_id", np.array([chain_id] * n_atoms))
    arr.set_annotation("label_entity_id", np.array([entity_id] * n_atoms))
    arr.set_annotation("hetero", np.zeros(n_atoms, dtype=bool))
    return arr


def test_load_atom_array_npz_set_annotation_updates_getattr(tmp_path):
    arr = _make_chain("A", "1")
    npz_path = tmp_path / "a.npz"
    save_atom_array_npz(str(npz_path), arr)
    loaded = load_atom_array_npz(str(npz_path))
    assert np.unique(loaded.label_entity_id).tolist() == ["1"]

    # set_annotation must be reflected by attribute reads: a plain setattr in
    # the loader used to create a shadowing attribute that kept returning the
    # stale value after set_annotation.
    loaded.set_annotation("label_entity_id", np.array(["7"] * len(loaded)))
    assert np.unique(loaded.label_entity_id).tolist() == ["7"]


def test_combine_keeps_distinct_entities_across_stages(tmp_path):
    # Real per-stage NPZs all start entity ids at "1"; the combined CIF must
    # remap them to globally unique ids instead of merging the molecules.
    p1 = tmp_path / "s1.npz"
    p2 = tmp_path / "s2.npz"
    save_atom_array_npz(str(p1), _make_chain("A", "1"))
    save_atom_array_npz(str(p2), _make_chain("A", "1"))
    out = tmp_path / "combined.cif"

    combine_npz_to_cif(dump_root=tmp_path, npz_paths=[p1, p2], output_cif=out)

    table = (
        gemmi.cif.read(str(out))
        .sole_block()
        .find("_atom_site.", ["label_asym_id", "label_entity_id"])
    )
    pairs = sorted({(row[0], row[1]) for row in table})
    assert [c for c, _ in pairs] == ["A", "B"]
    assert [e for _, e in pairs] == ["1", "2"]
