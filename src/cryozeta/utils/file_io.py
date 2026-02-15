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

import gzip
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from cryozeta.utils.torch_utils import map_values_to_list

PANDAS_NA_VALUES = [
    "",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    # "NA",
    "NULL",
    "NaN",
    "n/a",
    "nan",
    "null",
]


def read_indices_csv(csv: str | Path) -> pd.DataFrame:
    """
    Read a csv file without the content changing.

    Args:
        csv (Union[str, Path]): A csv file path.

    Returns:
        pd.DataFrame : A pandas DataFrame.
    """
    df = pd.read_csv(csv, na_values=PANDAS_NA_VALUES, keep_default_na=False, dtype=str)
    return df


def load_gzip_pickle(pkl: str | Path) -> Any:
    """
    Load a gzip pickle file.

    Args:
        pkl (Union[str, Path]): A gzip pickle file path.

    Returns:
        Any: The loaded data.
    """
    with gzip.open(pkl, "rb") as f:
        data = pickle.load(f)
    return data


def dump_gzip_pickle(data: Any, pkl: str | Path):
    """
    Dump a gzip pickle file.

    Args:
        data (Any): The data to be dumped.
        pkl (Union[str, Path]): A gzip pickle file path.
    """
    with gzip.open(pkl, "wb") as f:
        pickle.dump(data, f)


def save_json(data: dict, output_fpath: str | Path, indent: int = 4):
    """
    Save a dictionary to a JSON file.

    Args:
        data (dict): The dictionary to be saved.
        output_fpath (Union[str, Path]): The output file path.
        indent (int, optional): The indentation level for the JSON file. Defaults to 4.
    """
    data_json = data.copy()
    data_json = map_values_to_list(data_json)
    with open(output_fpath, "w") as f:
        if indent is not None:
            json.dump(data_json, f, indent=indent)
        else:
            json.dump(data_json, f)
