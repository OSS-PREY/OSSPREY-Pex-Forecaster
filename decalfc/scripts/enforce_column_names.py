"""
@brief Script to enforce a consistent column name across all datasets as 
    dictated by the field-mappings in the params dictionary.
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
from decalfc.abstractions.rawdata import _load_data, _load_paths, _save_data

# constants & setup parallel processing


# ------------- Primary Utility ------------- #
def enforce_naming_scheme(
    incubator: str, field_mappings: dict[str, str | dict[str, str]]=None,
    versions: dict[str, list[str]]=None
) -> None:
    """Wrapper to enforce a consistent column name across all datasets as 
    dictated by the field-mappings in the params dictionary.

    Args:
        incubator (str): incubator to process.
        field_mappings (dict[str, str], optional): field mappings to enforce 
            across all datasets. Defaults to params dictionary mappings.
        versions (dict[str, list[str]]): versions of the incubator's datasets to
            use; should be dtype: datasets names. Defaults to 0 and 0.
    """
    
    # load in the mappings if required
    if field_mappings is None:
        field_mappings = params_dict["field-mappings"][incubator]
    
    # infer versions if required
    if versions is None:
        versions = {
            "tech": ["0"],
            "social": ["0"]
        }
    
    # traverse datasets one at a time, enforcing the naming scheme
    log("Enforcing Column Names", "new")
    
    for dtype, datasets in tqdm(versions.items()):
        for ds in datasets:
            # load in data
            df = pd.read_parquet(
                Path(params_dict["dataset-dir"]) / f"{incubator}_data" /
                f"{params_dict['augmentations'][incubator][dtype][ds]}.parquet"
            )
            
            # grab and enforce naming scheme
            if len(datasets) > 1:
                names = field_mappings[dtype][incubator][ds]
            else:
                names = field_mappings[dtype]
            
            df.rename(columns=names, inplace=True)
            
            # drop any ignore columns
            df.drop(
                columns=[col for col in df.columns if col.startswith("[IGNORE]")],
                inplace=True
            )

            # save the data
            df.to_parquet(
                Path(params_dict["dataset-dir"]) / f"{incubator}_data" /
                f"{params_dict['augmentations'][incubator][dtype][ds]}.parquet"
            )

    # done
    log("all datasets updated with consistent column names!")
    return


# ------------- Run as Script ------------- #
if __name__ == "__main__":
    kwargs = parse_input(sys.argv)
    enforce_naming_scheme(**kwargs)

