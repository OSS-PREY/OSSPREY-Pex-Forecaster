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
    
    # return the merged dataframe without the excess project column
    merged = pd.merge(
        df, dates, left_on="project_name", right_on="project", how="left"
    )
    merged.drop(columns="project", inplace=True)
    return merged

def check_sufficient_data(mtdf: pd.DataFrame, msdf: pd.DataFrame) -> dict[str, pd.DataFrame | dict[str, tuple[str, str]]]:
    """Checks whether we have data to cover the full incubation period for a 
    given project. If we lack the data, we'll note the project down and remove 
    its data. If we have extra data, we'll truncate to only fit the project 
    timeline.
    
    NOTE: To consider social and tech overlap, we'll only require there to be 
    technical contributions. We can argue for using the max of either one, but 
    we typically require technical contributions data to 

    Args:
        mtdf (pd.DataFrame): merged tech dataframe with the start and end dates 
            for the incubation time period.
        msdf (pd.DataFrame): merged social dataframe with the start and end 
            dates for the incubation time period.

    Returns:
        dict[str, pd.DataFrame | dict[str, tuple[str, str]]]: dictionary with 
            the following structure:
            {
                "data": pd.DataFrame // truncated dataframe
                "missing": { // any missing projects + some metadata
                    "project_name": [data end date, incubation end date]
                }
            }
    """
    
    # auxiliary fn
    def is_valid_project(proj_tdf: pd.DataFrame, proj_sdf: pd.DataFrame) -> bool:
        """Indicator function to test if a project contains the necessary 
        requirements to be kept. Requires tech and social data for a given 
        project simultaneously.
        """
        
        # 
    


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
    combine_social_mediums(**kwargs)

