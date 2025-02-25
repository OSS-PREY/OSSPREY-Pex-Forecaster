"""
@brief Script to enforce the incubation dates for a given dataset.
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
def merge_st_end_dates(df: pd.DataFrame, dates: pd.DataFrame) -> pd.DataFrame:
    """Merges the start and end dates into columns "st_date" and "end_date" in 
    the dataframe for efficient group-transform downstream.

    Args:
        df (pd.DataFrame): social or technical data.
        dates (pd.DataFrame): dates lookup with three columns (project, st_date,
            end_date).

    Returns:
        pd.DataFrame: merged dataframe with start and end date information for 
            every row.
    """
    
    pass


# ------------- Primary Utility ------------- #
def truncate_incubation_time(
    incubator: str, dates: Path | str | dict[str, dict[str, str]],
    versions: dict[str, str]=None, save_versions: dict[str, int]=None
) -> dict[str, pd.DataFrame]:
    """Wrapper to truncate the social and technical data to only the time under
    incubation.

    Args:
        incubator (str): incubator to process.
        dates (Path | str | dict): path to incubation dates lookup (Path, str) 
            or the actual lookup itself.
            
            Lookup format: {
                "start-incubation"/"end-incubation": {
                    "project_name": "date_str"
                }}
        versions (dict[str, str], optional): versions of the incubator's dataset
            to use; assumes the `date` field is present in both social and 
            technical data for this version. Defaults to the default versions.
        save_versions (dict[str, int]): versions to save the truncated dataset
            to. If not provided, only returns a data lookup. Defaults to None.
    
    Returns:
        dict[str, pd.DataFrame]: lookup of "tech"/"social" to the respective 
            datasets post-truncation.
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
        versions = dict(zip(
            ["tech", "social"],
            params_dict["default-versions"][incubator]
        ))
    versions = {k: str(v) for k, v in versions.items()}
    
    # load the data
    rawdata_paths = _load_paths(incubator, versions)
    print(rawdata_paths)
    dlookup = _load_data(rawdata_paths)
    tdf = dlookup["tech"]
    sdf = dlookup["social"]
    
    # ensure the data types for the columns we operate on
    tdf["date"] = pd.to_datetime(tdf["date"])
    sdf["date"] = pd.to_datetime(sdf["date"])
    
    # convert lookup into a useable format
    dates = pd.DataFrame(dates)
    dates["start-incubation"] = pd.to_datetime(dates["start-incubation"])
    dates["end-incubation"] = pd.to_datetime(dates["end-incubation"])
    dates.rename(
        columns={"start-incubation": "st_date", "end-incubation": "end_date"},
        inplace=True
    )
    print(dates)
    

# ------------- Run as Script ------------- #
if __name__ == "__main__":
    kwargs = parse_input(sys.argv)
    truncate_incubation_time(**kwargs)

