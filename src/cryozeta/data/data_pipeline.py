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

from collections import defaultdict

from biotite.structure import AtomArray


class DataPipeline:
    @staticmethod
    def get_label_entity_id_to_asym_id_int(atom_array: AtomArray) -> dict[str, int]:
        """
        Get a dictionary that associates each label_entity_id with its corresponding asym_id_int.

        Args:
            atom_array (AtomArray): AtomArray object

        Returns:
            dict[str, int]: label_entity_id to its asym_id_int
        """
        entity_to_asym_id = defaultdict(set)
        for atom in atom_array:
            entity_id = atom.label_entity_id
            entity_to_asym_id[entity_id].add(atom.asym_id_int)
        return entity_to_asym_id
