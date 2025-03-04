"""
@brief Script to enforce the usual pre-processing steps for the raw data (reply
    inference, message id imputation, de-aliasing, bot identification, source
    imputation).
@author Arjun Ashok (arjun3.ashok@gmail.com)
@creation-date February 2025
"""

# ------------- Environment Setup ------------- #
# external packages -- none for now
import pandas as pd
import numpy as np
from tqdm import tqdm

# built-in modules
import sys
from pathlib import Path
from json import dump, load

# DECAL modules
from decalfc.utils import *
from decalfc.abstractions.rawdata import _load_data, _load_paths, _save_data, pre_process_data


# ------------- Auxiliary Utility ------------- #
def clean_special_chars(df: pd.DataFrame) -> pd.DataFrame:
    """Removes all special character sequences (e.g. "##", "/", etc.) from any
    column and replaces the string with a dash instead.

    Args:
        df (pd.DataFrame): dataframe to clean.

    Returns:
        pd.DataFrame: cleaned dataframe.
    """
    
    # for every column, clean all substrings
    for col in df.columns:
        if df.dtypes[col] != "object":
            continue
        df[col] = df[col].str.replace(r"/", "-").replace("##", "-")

    # export df
    return df


# ------------- Primary Utility ------------- #
def pre_process_raw_data(
    incubator: str, load_versions: dict[str, str], save_versions: dict[str, str]
) -> None:
    """Wrapper for the rawdata method to pre-process data; handles loading and 
    saving without additional specification for ease of script use.

    Args:
        incubator (str): incubator to reference.
        load_versions (dict[str, str]): versions to use for the pre-processing.
        save_versions (dict[str, str]): versions to save post-processing.
    """
    
    # load data
    paths = _load_paths(incubator, load_versions)
    data_lookup = _load_data(paths)
    
    # clean strings
    data_lookup = {k: clean_special_chars(v) for k, v in data_lookup.items()}
    
    # pre-process data
    pre_process_data(
        data_lookup=data_lookup, incubator=incubator, copy=False,
        save_versions=save_versions
    )


# ------------- Run as Script ------------- #
if __name__ == "__main__":
    kwargs = parse_input(sys.argv)
    pre_process_raw_data(**kwargs)

