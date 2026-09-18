"""GPU-free unit tests for multi-chain stage packing in cycle_predict."""

import json
from pathlib import Path

from cryozeta.runner.cycle_predict import (
    extract_chains_sorted_by_length,
    generate_stage_json,
    get_sequence_length,
    pack_chains_into_stages,
)


def _write_base_json(tmp_path: Path, sequences: list[dict], name: str = "9b0l") -> Path:
    base_json = tmp_path / "base.json"
    base_json.write_text(json.dumps([{"name": name, "sequences": sequences}]))
    return base_json


def _protein(length: int, count: int = 1) -> dict:
    return {"proteinChain": {"sequence": "A" * length, "count": count}}


def _dna(length: int) -> dict:
    return {"dnaSequence": {"sequence": "G" * length, "count": 1}}


def _rna(length: int) -> dict:
    return {"rnaSequence": {"sequence": "U" * length, "count": 1}}


def _stage_entities(stage: list[tuple[int, int, dict]]) -> list[tuple[int, int]]:
    return [(seq_idx, copy_idx) for seq_idx, copy_idx, _ in stage]


def _stage_residues(stage: list[tuple[int, int, dict]]) -> int:
    return sum(get_sequence_length(entry) for _, _, entry in stage)


def _write_9b0l_sequences(tmp_path: Path) -> Path:
    # 9b0l-like entry: protein 520, rna 247, dna 23/11/4 (805 residues total)
    return _write_base_json(
        tmp_path,
        [_dna(11), _protein(520), _dna(4), _dna(23), _rna(247)],
    )


def test_budget_600_packs_round_robin_across_entities(tmp_path):
    base_json = _write_9b0l_sequences(tmp_path)
    chains = extract_chains_sorted_by_length(base_json)
    stages = pack_chains_into_stages(chains, 600)

    assert len(stages) == 2
    assert all(_stage_residues(stage) <= 600 for stage in stages)
    # Round-robin takes one copy per entity first; the RNA strand (247) does
    # not fit after the protein (520), so it forms the second stage.
    assert _stage_entities(stages[0]) == [(1, 0), (3, 0), (0, 0), (2, 0)]
    assert _stage_residues(stages[0]) == 558
    assert _stage_entities(stages[1]) == [(4, 0)]
    assert _stage_residues(stages[1]) == 247


def test_default_budget_fits_whole_complex_in_one_stage(tmp_path):
    base_json = _write_9b0l_sequences(tmp_path)
    chains = extract_chains_sorted_by_length(base_json)
    stages = pack_chains_into_stages(chains, 2800)

    assert len(stages) == 1
    assert len(stages[0]) == 5
    assert _stage_residues(stages[0]) == 805


def test_homomultimer_copies_fill_stage_round_robin(tmp_path):
    # Chain A length 600 count=2 + chain B length 280 count=2
    base_json = _write_base_json(
        tmp_path, [_protein(600, count=2), _protein(280, count=2)], name="6n1q"
    )
    chains = extract_chains_sorted_by_length(base_json)

    # Budget 900: each stage holds one copy per entity.
    stages = pack_chains_into_stages(chains, 900)
    assert len(stages) == 2
    assert _stage_entities(stages[0]) == [(0, 0), (1, 0)]
    assert _stage_entities(stages[1]) == [(0, 1), (1, 1)]

    # Default budget: all four copies packed into a single stage.
    stages = pack_chains_into_stages(chains, 2800)
    assert len(stages) == 1
    assert _stage_entities(stages[0]) == [(0, 0), (1, 0), (0, 1), (1, 1)]
    assert _stage_residues(stages[0]) == 1760


def test_oversized_chain_gets_its_own_stage(tmp_path):
    base_json = _write_base_json(tmp_path, [_protein(3000), _protein(100)], name="big")
    chains = extract_chains_sorted_by_length(base_json)
    stages = pack_chains_into_stages(chains, 2800)

    assert len(stages) == 2
    assert _stage_entities(stages[0]) == [(1, 0)]
    assert _stage_entities(stages[1]) == [(0, 0)]


def test_generate_stage_json_writes_multiple_sequences(tmp_path):
    base_json = _write_9b0l_sequences(tmp_path)
    chains = extract_chains_sorted_by_length(base_json)
    stages = pack_chains_into_stages(chains, 2800)

    em_pt = tmp_path / "em.pt"
    em_pt.write_bytes(b"")
    json_path, stage_name = generate_stage_json(
        base_json=base_json,
        seq_entries=[entry for _, _, entry in stages[0]],
        stage_index=1,
        out_dir=tmp_path / "stage_jsons",
        em_pt=em_pt,
    )

    assert stage_name == "9b0l_stage_1"
    data = json.loads(json_path.read_text())
    assert len(data) == 1
    assert len(data[0]["sequences"]) == 5
    assert all(
        entry[chain_type]["count"] == 1
        for entry in data[0]["sequences"]
        for chain_type in entry
    )
    assert data[0]["em_file"] == str(em_pt.resolve())
