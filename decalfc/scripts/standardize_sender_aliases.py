"""
@brief Script to enforce a consistent sender alias across multiple datasets 
    (including all tech and social data).
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
def load_aliases(aliases: Path | str | pd.DataFrame) -> pd.DataFrame:
    """Utility for loading in the aliases data and cleaning it up:
        - removing special characters in names
        - extracting the relevant part of the email addresses and alias IDs

    Args:
        aliases (Path | str | pd.DataFrame): alias lookup or path to alias 
            lookup. Expects the following columns: name, email_addr, alias_id,
            person_id.

    Returns:
        pd.DataFrame: loaded and partially cleaned alias lookup.
    """
    
    # load if needed
    if not isinstance(aliases, pd.DataFrame):
        aliases = pd.read_csv(aliases)
    
    # only keep the relevant part of each field
    aliases.email_addr = aliases.email_addr.str.split("@").str[0]
    aliases.email_addr = aliases.email_addr.str.split(" at ").str[0]
    aliases.alias_id = aliases.alias_id.str.split("@").str[0]
    aliases.alias_id = aliases.alias_id.str.split(" at ").str[0]
    
    # remove special character names
    pd.set_option("future.no_silent_downcasting", True)
    special_char_locations = aliases.name.str.contains(
        r"[^\x00-\x7F]", regex=True
    ).fillna(False)
    pd.set_option("future.no_silent_downcasting", False)
    aliases.loc[special_char_locations, "name"] = pd.NA
    
    # export aliases
    return aliases

def aggregate_aliases(lookup: pd.DataFrame, alias_field: str="alias_id", col_priority: list[str]=None) -> pd.DataFrame:
    """Utility for cleaning up the lookup table to ensure one column which 
    combines the best information from each of the alternate columns in the 
    specified order.

    Args:
        lookup (pd.DataFrame): lookup of data with the different aliases used 
            across the datasets.
        alias_field (str): field that was used as the alias previously.
        col_priority (list[str]): priority of columns to use for the centralized
            alias.

    Returns:
        pd.DataFrame: cleaned alias lookup, i.e. alias to centralized alias 
            mapping.
    """
    
    # infer fields
    if col_priority is None:
        col_priority = ["name", "email_addr", "alias_id", "person_id"]
    
    # merge aliases into one column
    pd.set_option("future.no_silent_downcasting", True)
    lookup["sender_name"] = lookup[col_priority].bfill(axis=1).iloc[:, 0]
    pd.set_option("future.no_silent_downcasting", False)
    
    # for each unique sender name, accumulate its aliases (one per row)
    lookup = lookup.groupby("sender_name")[alias_field].apply(
        lambda x: list(x)
    ).reset_index()
    lookup = lookup.explode(alias_field)
    
    # drop extraneous columns
    keep_cols = [alias_field, "sender_name"]
    lookup.drop(columns=lookup.columns.difference(keep_cols), inplace=True)
    
    # return
    return lookup

def update_aliases(
    df: pd.DataFrame, aliases: pd.DataFrame, data_alias_field: str,
    lookup_alias_field: str
) -> pd.DataFrame:
    """Updates the original dataset's with the sender name aliases. Final 
    aliases will be under the `sender_name` column.

    Args:
        df (pd.DataFrame): dataset to update.
        aliases (pd.DataFrame): aliases lookup.
        data_alias_field (str): column to match with the aliases in the raw 
            data.
        lookup_alias_field (str): alias field in the aliases lookup.

    Returns:
        pd.DataFrame: updates dataset without extraneous alias column 
            (data_alias_field).
    """
    
    # merge on the alias field with the new sender names
    df = df.merge(
        aliases, how="left", left_on=data_alias_field,
        right_on=lookup_alias_field
    )
    
    # drop extraneous columns
    df.drop(data_alias_field, axis=1, inplace=True)
    
    # export df
    return df


# ------------- Primary Utility ------------- #
def enforce_consistent_aliases(
    incubator: str, aliases: str | Path | pd.DataFrame,
    load_save_versions: dict[str, str], alias_field: str="alias_id",
    col_priority: list[str]=None,
) -> None:
    """Wrapper to truncate the social and technical data to only the time under
    incubation.

    Args:
        incubator (str): incubator to process.
        aliases (str | Path | pd.DataFrame): aliases lookup or path to aliases
            lookup. Expects the following columns: name, email_addr, alias_id,
            person_id.
        load_save_versions (dict[str, dict[str, str]]): lookup of the dtype
            to the version to load (key) and the version to save as (value).
        alias_field (str, optional): field that was used as the alias 
            previously.
        col_priority (list[str], optional): priority of columns to use for the 
            centralized alias. Defaults to None.
    """

    # load in the aliases
    log("loading and cleaning aliases...")
    aliases = load_aliases(aliases)
    
    # setup the fields to pass in
    log("aggregating aliases...")
    cleaned_aliases = aggregate_aliases(aliases, alias_field, col_priority)
    
    # load in the data to update and save one at a time to avoid OOM error
    for dtype, version_pkg in load_save_versions.items():
        for load_version, save_version in version_pkg.items():
            # load data
            log(f"processing {dtype} data for version {load_version} --> {save_version}...")
            df = pd.read_parquet(
                Path(params_dict["dataset-dir"]) / f"{incubator}_data" /
                f"{params_dict['augmentations'][incubator][dtype][load_version]}.parquet"
            )
            
            # update aliases
            df = update_aliases(df, cleaned_aliases, alias_field, alias_field)
            
            # save data
            df.to_parquet(
                Path(params_dict["dataset-dir"]) / f"{incubator}_data" /
                f"{params_dict['augmentations'][incubator][dtype][save_version]}.parquet"
            )

    # all done
    log("all data updated with consistent aliases!")
    return


# ------------- Run as Script ------------- #
if __name__ == "__main__":
    # wrap main
    kwargs = parse_input(sys.argv)
    enforce_consistent_aliases(**kwargs)

