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

import os
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from os.path import exists as opexists, join as opjoin
from typing import Any

import numpy as np
from biotite.structure import AtomArray
from loguru import logger

from cryozeta.data.constants import STD_RESIDUES, rna_order_with_x
from cryozeta.data.msa_utils import (
    FeatureDict,
    add_assembly_features,
    clip_msa,
    convert_monomer_features,
    get_identifier_func,
    load_and_process_msa,
    make_sequence_features,
    merge_features_from_prot_rna,
    msa_parallel,
    pair_and_merge,
    rna_merge,
)
from cryozeta.data.tokenizer import TokenArray

MSA_MAX_SIZE = 16384


# Common function for train and inference
def process_single_sequence(
    pdb_name: str,
    sequence: str,
    raw_msa_paths: list[str] | None,
    seq_limits: list[str] | None,
    msa_entity_type: str = "prot",
    msa_type: str = "non_pairing",
) -> FeatureDict:
    """
    Processes a single sequence to generate sequence and MSA features.

    Args:
        pdb_name (str): The name of the PDB entry.
        sequence (str): The input sequence.
        raw_msa_paths (Optional[list[str]]): List of paths to raw MSA files.
        seq_limits (Optional[list[str]]): List of sequence limits for different databases.
        msa_entity_type (str): The type of MSA entity, either "prot" or "rna". Defaults to "prot".
        msa_type (str): The type of MSA, either "non_pairing" or "pairing". Defaults to "non_pairing".

    Returns:
        FeatureDict: A dictionary containing the sequence and MSA features.

    Raises:
        AssertionError: If `msa_entity_type` is not "prot" or "rna".
    """
    assert msa_entity_type in ["prot", "rna"]
    num_res = len(sequence)

    if msa_entity_type == "prot":
        sequence_features = make_sequence_features(
            sequence=sequence,
            num_res=num_res,
        )
    elif msa_entity_type == "rna":
        sequence_features = make_sequence_features(
            sequence=sequence,
            num_res=num_res,
            mapping=rna_order_with_x,
            x_token="N",
        )

    msa_features = load_and_process_msa(
        pdb_name=pdb_name,
        msa_type=msa_type,
        raw_msa_paths=raw_msa_paths,
        seq_limits=seq_limits,
        input_sequence=sequence,
        msa_entity_type=msa_entity_type,
    )
    sequence_features.update(msa_features)
    return sequence_features


# Common function for train and inference
def tokenize_msa(
    msa_feats: Mapping[str, np.ndarray],
    token_array: TokenArray,
    atom_array: AtomArray,
) -> dict[str, np.ndarray]:
    """
    Tokenize raw MSA features.

    Args:
        msa_feats (Dict[str, np.ndarray]): raw MSA features.
        token_array (TokenArray): token array of this bioassembly
        atom_array (AtomArray): atom array of this bioassembly

    Returns:
        Dict[str, np.ndarray]: the tokenized MSA features of the bioassembly.
    """
    token_center_atom_idxs = token_array.get_annotation("centre_atom_index")
    # res_id: (asym_id, residue_index)
    # msa_idx refers to the column number of a residue in the msa array
    res_id_2_msa_idx = {
        (msa_feats["asym_id"][idx], msa_feats["residue_index"][idx]): idx
        for idx in range(msa_feats["msa"].shape[1])
    }

    restypes = []
    col_idxs_in_msa = []
    col_idxs_in_new_msa = []
    for token_idx, center_atom_idx in enumerate(token_center_atom_idxs):
        restypes.append(STD_RESIDUES[atom_array.cano_seq_resname[center_atom_idx]])
        if (
            res_id := (
                atom_array[center_atom_idx].asym_id_int,
                atom_array[center_atom_idx].res_id,
            )
        ) in res_id_2_msa_idx:
            col_idxs_in_msa.append(res_id_2_msa_idx[res_id])
            col_idxs_in_new_msa.append(token_idx)

    num_msa_seq, _ = msa_feats["msa"].shape
    num_tokens = len(token_center_atom_idxs)

    restypes = np.array(restypes)
    col_idxs_in_new_msa = np.array(col_idxs_in_new_msa)
    col_idxs_in_msa = np.array(col_idxs_in_msa)

    # msa
    # For non-amino acid tokens, copy the token itself
    feat_name = "msa"
    new_feat = np.repeat(restypes[None, ...], num_msa_seq, axis=0)
    new_feat[:, col_idxs_in_new_msa] = msa_feats[feat_name][:, col_idxs_in_msa]
    msa_feats[feat_name] = new_feat

    # has_deletion, deletion_value
    # Assign 0 to non-amino acid tokens
    for feat_name in ["has_deletion", "deletion_value"]:
        new_feat = np.zeros((num_msa_seq, num_tokens), dtype=msa_feats[feat_name].dtype)
        new_feat[:, col_idxs_in_new_msa] = msa_feats[feat_name][:, col_idxs_in_msa]
        msa_feats[feat_name] = new_feat

    # deletion_mean
    # Assign 0 to non-amino acid tokens
    feat_name = "deletion_mean"
    new_feat = np.zeros((num_tokens,))
    new_feat[col_idxs_in_new_msa] = msa_feats[feat_name][col_idxs_in_msa]
    msa_feats[feat_name] = new_feat

    # profile
    # Assign one-hot enbedding (one-hot distribution) to non-amino acid tokens corresponding to restype
    feat_name = "profile"
    new_feat = np.zeros((num_tokens, 32))
    new_feat[np.arange(num_tokens), restypes] = 1
    new_feat[col_idxs_in_new_msa, :] = msa_feats[feat_name][col_idxs_in_msa, :]
    msa_feats[feat_name] = new_feat
    return msa_feats


# Common function for train and inference
def merge_all_chain_features(
    pdb_id: str,
    all_chain_features: dict[str, FeatureDict],
    asym_to_entity_id: dict,
    is_homomer_or_monomer: bool = False,
    merge_method: str = "dense_max",
    max_size: int = 16384,
    msa_entity_type: str = "prot",
) -> dict[str, np.ndarray]:
    """
    Merges features from all chains in the bioassembly.

    Args:
        pdb_id (str): The PDB ID of the bioassembly.
        all_chain_features (dict[str, FeatureDict]): Features for each chain in the bioassembly.
        asym_to_entity_id (dict): Mapping from asym ID to entity ID.
        is_homomer_or_monomer (bool): Indicates if the bioassembly is a homomer or monomer. Defaults to False.
        merge_method (str): Method used for merging features. Defaults to "dense_max".
        max_size (int): Maximum size of the MSA. Defaults to 16384.
        msa_entity_type (str): Type of MSA entity, either "prot" or "rna". Defaults to "prot".

    Returns:
        dict[str, np.ndarray]: Merged features for the bioassembly.
    """
    all_chain_features = add_assembly_features(
        pdb_id,
        all_chain_features,
        asym_to_entity_id=asym_to_entity_id,
    )
    if msa_entity_type == "rna":
        np_example = rna_merge(
            all_chain_features=all_chain_features,
            merge_method=merge_method,
            msa_crop_size=max_size,
        )
    elif msa_entity_type == "prot":
        np_example = pair_and_merge(
            is_homomer_or_monomer=is_homomer_or_monomer,
            all_chain_features=all_chain_features,
            merge_method=merge_method,
            msa_crop_size=max_size,
        )
    np_example = clip_msa(np_example, max_num_msa=max_size)
    return np_example


class InferenceMSAFeaturizer:
    # Now we only support protein msa in inference

    @staticmethod
    def process_prot_single_sequence(
        sequence: str,
        description: str,
        is_homomer_or_monomer: bool,
        msa_dir: str | None,
        pairing_db: str,
        msa_entity_type: str = "prot",
    ) -> FeatureDict:
        """
        Processes a single protein sequence to generate sequence and MSA features.

        Args:
            sequence (str): The input protein sequence.
            description (str): Description of the sequence, typically the PDB name.
            is_homomer_or_monomer (bool): Indicates if the sequence is a homomer or monomer.
            msa_dir (Union[str, None]): Directory containing the MSA files, or None if no pre-computed MSA is provided.
            pairing_db (str): Database used for pairing.

        Returns:
            FeatureDict: A dictionary containing the sequence and MSA features.

        Raises:
            AssertionError: If the pairing MSA file does not exist when `is_homomer_or_monomer` is False.
        """
        # For non-pairing MSA
        if msa_dir is None:
            # No pre-computed MSA was provided, and the MSA search failed
            raw_msa_paths = []
        else:
            raw_msa_paths = [
                opjoin(
                    msa_dir,
                    "mmseqs_other_hits.a3m"
                    if msa_entity_type == "prot"
                    else "rnacentral.a3m",
                )
            ]
        pdb_name = description

        sequence_features = process_single_sequence(
            pdb_name=pdb_name,
            sequence=sequence,
            raw_msa_paths=raw_msa_paths,
            seq_limits=[-1],
            msa_entity_type=msa_entity_type,
            msa_type="non_pairing",
        )
        if not is_homomer_or_monomer:
            # Separately process the pairing MSA
            raw_msa_path = opjoin(msa_dir, "uniref100_hits.a3m")
            assert opexists(raw_msa_path), (
                f"No pairing-MSA of {pdb_name} (please check {raw_msa_path})"
            )

            all_seq_msa_features = load_and_process_msa(
                pdb_name=pdb_name,
                msa_type="pairing",
                raw_msa_paths=[raw_msa_path],
                seq_limits=[-1],
                identifier_func=get_identifier_func(
                    pairing_db=pairing_db,
                ),
                handle_empty="raise_error",
            )
            sequence_features.update(all_seq_msa_features)

        return sequence_features

    @staticmethod
    def get_inference_msa_features_for_assembly(
        bioassembly: Sequence[Mapping[str, Mapping[str, Any]]],
        entity_to_asym_id: Mapping[str, set[int]],
        msa_entity_type: str = "prot",
    ) -> FeatureDict:
        """
        Processes the bioassembly to generate MSA features for protein entities in inference mode.

        Args:
            bioassembly (Sequence[Mapping[str, Mapping[str, Any]]]): The bioassembly containing entity information.
            entity_to_asym_id (Mapping[str, set[int]]): Mapping from entity ID to asym ID integers.

        Returns:
            FeatureDict: A dictionary containing the MSA features for the protein entities.

        Raises:
            AssertionError: If the provided precomputed MSA path does not exist.
        """
        if msa_entity_type == "rna":
            chain_type = "rnaSequence"
        elif msa_entity_type == "prot":
            chain_type = "proteinChain"
        else:
            raise ValueError(f"Invalid msa_entity_type: {msa_entity_type}")

        entity_to_asym_id_int = dict(entity_to_asym_id)
        asym_to_entity_id = {}
        entity_id_to_sequence = {}
        # In inference mode, the keys in bioassembly is different from training
        # Only contains protein entity, many-to-one mapping
        entity_id_to_sequence = {}
        for i, entity_info_wrapper in enumerate(bioassembly):
            entity_id = str(i + 1)
            entity_type = next(iter(entity_info_wrapper.keys()))
            entity_info = entity_info_wrapper[entity_type]

            if entity_type == chain_type:
                # Update entity_id_to_sequence
                entity_id_to_sequence[entity_id] = entity_info["sequence"]

                # Update asym_to_entity_id
                for asym_id_int in entity_to_asym_id_int[entity_id]:
                    asym_to_entity_id[asym_id_int] = entity_id
        if len(entity_id_to_sequence) == 0:
            # No protein entity
            return None
        is_homomer_or_monomer = (
            len(set(entity_id_to_sequence.values())) == 1
        )  # Only one protein sequence

        if msa_entity_type == "rna":
            is_homomer_or_monomer = True

        sequence_to_entity = defaultdict(list)
        for entity_id, seq in entity_id_to_sequence.items():
            sequence_to_entity[seq].append(entity_id)

        sequence_to_features: dict[str, dict[str, Any]] = {}
        msa_sequences = {}
        msa_dirs = {}
        for idx, (sequence, entity_id_list) in enumerate(sequence_to_entity.items()):
            msa_info = bioassembly[int(entity_id_list[0]) - 1][chain_type]["msa"]
            msa_dir = msa_info.get("precomputed_msa_dir", None)
            if msa_dir is not None:
                assert opexists(msa_dir), (
                    f"The provided precomputed MSA path of entities {entity_id_list} does not exists: \n{msa_dir}"
                )
                msa_dirs[idx] = msa_dir
            else:
                pairing_db_fpath = msa_info.get("pairing_db_fpath", None)
                non_pairing_db_fpath = msa_info.get("non_pairing_db_fpath", None)
                assert pairing_db_fpath is not None, (
                    "Path of pairing MSA database is not given."
                )
                assert non_pairing_db_fpath is not None, (
                    "Path of non-pairing MSA database is not given."
                )
                assert msa_info["pairing_db"] in ["uniprot", "", None], (
                    f"Using {msa_info['pairing_db']} as the source for MSA pairing "
                    f"is not supported in online MSA searching."
                )

                msa_info["pairing_db"] = "uniprot"
                msa_sequences[idx] = (sequence, pairing_db_fpath, non_pairing_db_fpath)
        if len(msa_sequences) > 0:
            msa_dirs.update(msa_parallel(msa_sequences))

        for idx, (sequence, entity_id_list) in enumerate(sequence_to_entity.items()):
            if len(entity_id_list) > 1:
                logger.info(
                    f"Entities {entity_id_list} correspond to the same sequence."
                )
            msa_info = bioassembly[int(entity_id_list[0]) - 1][chain_type]["msa"]
            msa_dir = msa_dirs[idx]

            description = f"entity_{'_'.join(map(str, entity_id_list))}"
            sequence_feat = InferenceMSAFeaturizer.process_prot_single_sequence(
                sequence=sequence,
                description=description,
                is_homomer_or_monomer=is_homomer_or_monomer,
                msa_dir=msa_dir,
                pairing_db=msa_info["pairing_db"],
                msa_entity_type=msa_entity_type,
            )
            sequence_feat = convert_monomer_features(sequence_feat)
            sequence_to_features[sequence] = sequence_feat
            if msa_dir and opexists(msa_dir) and idx in msa_sequences.keys():
                if (msa_save_dir := msa_info.get("msa_save_dir", None)) is not None:
                    if opexists(dst_dir := opjoin(msa_save_dir, str(idx + 1))):
                        shutil.rmtree(dst_dir)
                    shutil.copytree(msa_dir, dst_dir)
                    for fname in os.listdir(dst_dir):
                        if not fname.endswith(".a3m"):
                            os.remove(opjoin(dst_dir, fname))
                else:
                    shutil.rmtree(msa_dir)

        all_chain_features = {
            asym_id_int: deepcopy(
                sequence_to_features[entity_id_to_sequence[entity_id]]
            )
            for asym_id_int, entity_id in asym_to_entity_id.items()
            if seq in sequence_to_features
        }
        if len(all_chain_features) == 0:
            return None

        np_example = merge_all_chain_features(
            pdb_id="test_assembly",
            all_chain_features=all_chain_features,
            asym_to_entity_id=asym_to_entity_id,
            is_homomer_or_monomer=is_homomer_or_monomer,
            merge_method="dense_max",
            max_size=MSA_MAX_SIZE,
            msa_entity_type=msa_entity_type,
        )

        return np_example

    def make_msa_feature(
        bioassembly: Sequence[Mapping[str, Mapping[str, Any]]],
        entity_to_asym_id: Mapping[str, Sequence[str]],
        token_array: TokenArray,
        atom_array: AtomArray,
        enable_rna_msa: bool = True,
    ) -> dict[str, np.ndarray] | None:
        """
        Processes the bioassembly to generate MSA features for protein entities in inference mode and tokenizes the features.

        Args:
            bioassembly (Sequence[Mapping[str, Mapping[str, Any]]]): The bioassembly containing entity information.
            entity_to_asym_id (Mapping[str, Sequence[str]]): Mapping from entity ID to asym ID strings.
            token_array (TokenArray): Token array of the bioassembly.
            atom_array (AtomArray): Atom array of the bioassembly.

        Returns:
            Optional[dict[str, np.ndarray]]: A dictionary containing the tokenized MSA features for the protein entities,
                or an empty dictionary if no features are generated.
        """
        prot_msa_feats = InferenceMSAFeaturizer.get_inference_msa_features_for_assembly(
            bioassembly=bioassembly,
            entity_to_asym_id=entity_to_asym_id,
            msa_entity_type="prot",
        )

        if enable_rna_msa:
            rna_msa_feats = (
                InferenceMSAFeaturizer.get_inference_msa_features_for_assembly(
                    bioassembly=bioassembly,
                    entity_to_asym_id=entity_to_asym_id,
                    msa_entity_type="rna",
                )
            )
        else:
            rna_msa_feats = None

        np_chains_list = []
        if prot_msa_feats is not None:
            np_chains_list.append(prot_msa_feats)
        if rna_msa_feats is not None:
            np_chains_list.append(rna_msa_feats)
        if len(np_chains_list) == 0:
            return {}

        msa_feats = merge_features_from_prot_rna(np_chains_list)

        if msa_feats is None:
            return {}

        msa_feats = tokenize_msa(
            msa_feats=msa_feats,
            token_array=token_array,
            atom_array=atom_array,
        )
        return {
            k: v
            for (k, v) in msa_feats.items()
            if k
            in ["msa", "has_deletion", "deletion_value", "deletion_mean", "profile"]
        }
