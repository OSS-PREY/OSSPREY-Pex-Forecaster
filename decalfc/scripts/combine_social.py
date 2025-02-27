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
from decalfc.abstractions.rawdata import _load_data, _load_paths, _save_data, infer_replies

# constants & setup parallel processing
__ISSUES_COLUMN_MAPPER__ = {
    "project_name": "project_name",
    "sender_name": "sender_name",
    "message_id": "message_id",
    "date": "date",
    "body": "body",
    "subject": "subject",
    "in_reply_to": "in_reply_to",
    "month": "month"
}
__EMAILS_COLUMN_MAPPER__ = {
    "project_name": "project_name",
    "sender_name": "sender_name",
    "message_id": "message_id",
    "date": "date",
    "body": "body",
    "subject": "subject",
    "reply_to": "in_reply_to",
    "month": "month"
}
__SUBSTR_LOOKUP__ = {
    "i": __ISSUES_COLUMN_MAPPER__,
    "e": __EMAILS_COLUMN_MAPPER__
}


# ------------- Auxiliary Utility ------------- #
def ensure_reply_information(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures the reply field (a) exists and (b) contains valid mappings to the
    message_id field.

    Args:
        df (pd.DataFrame): dataframe (typically the issues data) to process.

    Returns:
        pd.DataFrame: reply-imputed dataframe.
    """
    
    # check column exists
    if "in_reply_to" not in df.columns:
        df["in_reply_to"] = None
    
    # wrap call to underlying reply inference technique
    df = infer_replies({"social": df}, copy=False)["social"]
    return df

def ensure_cols(df: pd.DataFrame, version: str) -> pd.DataFrame:
    """Ensures the columns in the dataframe align with the other social formats.
    Infers the alignment from the translations in the substr lookup; the substr 
    maps to the end of the version string.
    
    Args:
        df (pd.DataFrame): dataframe to align.
        version (str): version of the social data.

    Returns:
        pd.DataFrame: reference to the same underlying dataframe.
    """
    
    # infer the lookup
    social_type = None
    for substr in __SUBSTR_LOOKUP__:
        if version.endswith(substr):
            social_type = substr
            break
    
    if social_type is None:
        raise ValueError(
            f"Invalid version string {version}. Should end with one of {list(__SUBSTR_LOOKUP__.keys())}"
        )
    
    lookup = __SUBSTR_LOOKUP__[social_type]
    
    # drop non-essential columns
    drop_cols = [col for col in df.columns if col not in lookup]
    df.drop(columns=drop_cols, inplace=True)
    
    # rename columns
    df.rename(columns=lookup, inplace=True)
    
    # add the source column
    df["source"] = social_type
    df["source"] = df["source"].astype("category")

    # export
    return df


# ------------- Primary Utility ------------- #
def combine_social_mediums(
    incubator: str, social_versions: list[str]=None,
    save_version: str="0"
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
            Path(params_dict["dataset-dir"]) / f"{incubator}_data" /
            f"{params_dict['augmentations'][incubator]['social'][versions]}.parquet"
        ) for version in versions
    ]
    
    # ensure replies prior to merge
    sdfs = [ensure_reply_information(df) for df in sdfs]
    
    # ensure column alignment
    sdfs = [ensure_cols(df, v) for df, v in zip(sdfs, versions)]
    
    # concatenate the dataframes by matching columns
    social_df = pd.concat(sdfs, ignore_index=True)
    
    # export as the save version
    social_df.to_parquet(
        Path(params_dict["dataset-dir"]) / f"{incubator}_data" /
        f"{params_dict['augmentations'][incubator]['social'][save_version]}.parquet"
    )
    

# ------------- Run as Script ------------- #
if __name__ == "__main__":
    kwargs = parse_input(sys.argv)
    combine_social_mediums(**kwargs)

