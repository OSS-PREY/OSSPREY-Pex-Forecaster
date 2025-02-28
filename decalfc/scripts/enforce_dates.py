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

def check_sufficient_data(mtdf: pd.DataFrame, msdf: pd.DataFrame) -> dict[str, pd.DataFrame | dict[str, dict[str, str]]]:
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
                "mtdf": pd.DataFrame // truncated tech dataframe
                "msdf": pd.DataFrame // truncated social dataframe
                "missing": { // any missing projects + some metadata
                    "project_name": {
                        "data_end_date": max tracked,
                        "data_st_date": min tracked,
                        "inc_end_date": incubation end date,
                        "inc_st_date": incubation start date
                    }
                }
            }
    """
    
    # auxiliary fn
    def is_valid_project(proj_tdf: pd.DataFrame, proj_sdf: pd.DataFrame, ndays_tol: int=30) -> bool:
        """Indicator function to test if a project contains the necessary 
        requirements to be kept. Requires tech and social data for a given 
        project simultaneously. Allows for some number of days of tolerance.
        
        Returns True if the project is valid, otherwise False.
        """
        
        """
        if (proj_tdf.date.max().date)< proj_tdf.end_date.iloc[0].date:
            proj_end_date = proj_tdf.date.max().date
        else:
            proj_end_date = proj_tdf.end_date.iloc[0].date
        
        if (proj_tdf.date.min().date)> proj_tdf.st_date.iloc[0].date:
            proj_end_date = proj_tdf.date.min().date
        else:
            proj_end_date = proj_tdf.st_date.iloc[0].date
        
        """
        
        # check if the end dates are within the tolerance or better
        if pd.isnull(proj_tdf.end_date.iloc[0].date()) or pd.isnull(proj_tdf.st_date.iloc[0].date()):
            return False
        
        if (proj_tdf.date.max().date() - proj_tdf.end_date.iloc[0].date()).days < ndays_tol:
            return False

        # check if the start dates are within the tolerance or better
        if (proj_tdf.date.min().date() - proj_tdf.st_date.iloc[0].date()).days > ndays_tol:
            return False

        # passed checks
        return True

    
    # only keep overlapping projects
    overlap_projects = set(mtdf.project_name) & set(msdf.project_name)
    mtdf = mtdf[mtdf.project_name.isin(overlap_projects)]
    msdf = msdf[msdf.project_name.isin(overlap_projects)]
    
    # lookups to build
    valid_projects = list()
    invalid_info = dict()
    
    # traverse valid projects
    for (proj, proj_tdf), (_, proj_sdf) in zip(mtdf.groupby("project_name"), msdf.groupby("project_name")):
        if is_valid_project(proj_tdf, proj_sdf):
            valid_projects.append(proj)
        else:
            invalid_info[proj] = {
                "data_end_date": proj_tdf.date.max(),
                "data_st_date": proj_tdf.date.min(),
                "inc_end_date": proj_sdf.end_date.iloc[0],
                "inc_st_date": proj_sdf.st_date.iloc[0]
            }
    
    # remove invalids
    mtdf = mtdf[mtdf.project_name.isin(valid_projects)]
    msdf = msdf[msdf.project_name.isin(valid_projects)]
    
    # return lookup
    return {
        "mtdf": mtdf,
        "msdf": msdf,
        "missing": invalid_info
    }

def truncate_data(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Executes the truncation as requested for all projects.

    Args:
        df (pd.DataFrame): dataframe to truncate.

    Returns:
        tuple[pd.DataFrame, int]: cleaned (no excess cols)and truncated
            dataframe, number of projects truncated
    """
    
    # grab some tracker info to check statistics
    orig_end_dates = df.groupby("project_name")["date"].max().to_dict()
    orig_st_dates = df.groupby("project_name")["date"].min().to_dict()
    
    # enforce incubation st-end dates
    df = df[(df.st_date <= df.date) & (df.date <= df.end_date)]
    
    # compare tracker info
    trunc_end_dates = df.groupby("project_name")["date"].max().to_dict()
    trunc_st_dates = df.groupby("project_name")["date"].min().to_dict()
    
    ndiffs = len(orig_end_dates) - len(trunc_end_dates) # number of projects truncated out
    for proj in trunc_end_dates: # number of projects with different dates
        if ((orig_end_dates[proj] != trunc_end_dates[proj]) or
            (orig_st_dates[proj] != trunc_st_dates[proj])):
            ndiffs += 1
    
    # drop extraneous columns
    df = df.drop(columns=["end_date", "st_date"])
    
    # export stats
    return df, ndiffs


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
    
    log("Truncate to Incubation Period", "new")

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
    log("loading data...")
    rawdata_paths = _load_paths(incubator, versions)
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
    dates.reset_index(names="project", inplace=True)
    dates.rename(
        columns={"start-incubation": "st_date", "end-incubation": "end_date"},
        inplace=True
    )

    # merge the lookup
    log("integrating start and end dates...")
    mtdf = merge_st_end_dates(tdf, dates)
    msdf = merge_st_end_dates(sdf, dates)
    
    # check that we keep only the projects with enough data
    log("checking projects with sufficient, overlapping data...")
    csd_ret = check_sufficient_data(mtdf, msdf)
    mtdf, msdf = csd_ret["mtdf"], csd_ret["msdf"]
    validity_info = csd_ret["missing"]
    
    # enforce the truncation restriction
    log("enforcing truncation...")
    mtdf, ntruncated_tech = truncate_data(mtdf)
    msdf, ntruncated_social = truncate_data(msdf)
    
    # update report info and print the report
    validity_info.update({
        "num_projects_truncated_tech": ntruncated_tech,
        "num_projects_truncated_social": ntruncated_social
    })
    log(log_type="summary")
    log(validity_info, log_type="report")
    
    # save the files
    if save_versions is not None:
        log("\nsaving data...")
        _save_data(
            {"tech": mtdf, "social": msdf}, incubator, new_version=save_versions
        )


# ------------- Run as Script ------------- #
if __name__ == "__main__":
    kwargs = parse_input(sys.argv)
    truncate_incubation_time(**kwargs)

