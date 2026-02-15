# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Modifications Copyright 2026 KiharaLab, Purdue University.
#
# This file is included in a GPLv3-licensed project. The original
# code remains under Apache-2.0; the combined work is distributed
# under GPLv3.

# pylint: disable=C0114,C0301
import os

data_configs = {
    "num_dl_workers": 16,
    "test_ref_pos_augment": True,
    "msa": {
        "enable": True,
        "enable_rna_msa": True,
        "prot": {
            "pairing_db": "uniref100",
            "non_pairing_db": "mmseqs_other",
            "pdb_mmseqs_dir": "/scratch/gilbreth/ykagaya/CryoZeta/project_assets/CryoZeta/msa/prot",
            "seq_to_pdb_idx_path": "",
            "indexing_method": "sequence",
        },
        "rna": {
            "seq_to_pdb_idx_path": "",
            "rna_msa_dir": "/scratch/gilbreth/ykagaya/CryoZeta/project_assets/CryoZeta/msa/rna",
            "indexing_method": "sequence",
        },
        "strategy": "random",
        "merge_method": "dense_max",
        "min_size": {
            "train": 1,
            "test": 2048,
        },
        "max_size": {
            "train": 16384,
            "test": 16384,
        },
        "sample_cutoff": {
            "train": 2048,
            "test": 2048,
        },
    },
    "template": {
        "enable": False,
    },
    "ccd_components_file": os.path.join("assets", "components.v20240608.cif"),
    "ccd_components_rdkit_mol_file": os.path.join(
        "assets", "components.v20240608.cif.rdkit_mol.pkl"
    ),
}
