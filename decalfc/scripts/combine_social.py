"""
@brief Script to combine multiple social network mediums (issues, emails, etc.)
    into one centralized social dataset.
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


# ------------- Auxiliary Utility ------------- #



# ------------- Primary Utility ------------- #
def combine_social_mediums(
    incubator: str, social_versions: dict[str, str]=None,
    save_version: dict[str, int]=None
) -> pd.DataFrame:
    """Wrapper to truncate the social and technical data to only the time under
    incubation.

    Args:
        incubator (str): incubator to process.
        social_versions (list[str], optional): versions of the incubator's 
            social datasets to use. Assumes all data has the standardized column
            names. Defaults to ["0i", "0e"], i.e. the raw issues and emails data
            to combine.
        save_version (str, optional): version to save the combined data as. 
            Defaults to None.

    Returns:
        pd.DataFrame: post-combination social dataset.
    """

    # infer dates lookup if needed
    if not isinstance(dates, dict):
        # enforce path object
        dates = Path(dates)
        
        # open json
        with open(dates, "r") as f:
            dates = load(f)

    # infer versions if needed
    if versions is None:
        versions = ["0i", "0e"]
    versions = [str(v) for v in versions]
    
    # load the data
    sdfs = [
        pd.read_parquet(
            Path(params_dict["augmentations"][incubator][version])
        ) for version in versions
    ]
    
    # convert 
    

# ------------- Run as Script ------------- #
if __name__ == "__main__":
    kwargs = parse_input(sys.argv)
    combine_social_mediums(**kwargs)

