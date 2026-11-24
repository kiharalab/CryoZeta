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

# pylint: disable=C0114
from cryozeta.configs.extend_types import ListValue, RequiredValue

inference_configs = {
    "seeds": ListValue([101]),
    "dump_dir": "./output",
    "need_atom_confidence": False,
    "input_json_path": RequiredValue(str),
    "load_checkpoint_path": RequiredValue(str),
    "num_workers": 0,
    "use_msa": True,
    "enable_rna_msa": True,
    "overwrite": False,
}
