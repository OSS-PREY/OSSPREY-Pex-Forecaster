"""
    @brief defines the raw data class for easily interfacing with the dataset. 
           Primary utility comes from easily experimenting with format and 
           bundling all relevant information in one object.
    @author Arjun Ashok (arjun3.ashok@gmail.com)
    @creation-date January 2024
    @version 0.1.0
"""


# Imports
import pandas as pd
import numpy as np
from tqdm import tqdm
from pandarallel import pandarallel
from jellyfish import jaro_winkler_similarity

import json
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from itertools import accumulate
from typing import Any
from multiprocessing import Pool

from decalfc.utils import *


# Auxiliary Functions
def _load_paths(incubator: str, versions: dict[str, str], ext: str = "parquet") -> dict[str, str]:
    """
        Loads the paths from the params dictionary and the previously defined 
        user variables.
    """

    # return lookup for paths
    dataset_dir = Path(params_dict["dataset-dir"])
    return {
        data_type: dataset_dir / f"{incubator}_data/{paths[versions[data_type]]}.{ext}"
        for data_type, paths in params_dict["augmentations"][incubator].items()
    }

def _load_data(paths: dict[str, str], dtypes: dict=None) -> dict[str, pd.DataFrame]:
    """
        Loads the paths from the params dictionary and the previously defined 
        user variables.
    """

    # return lookup for dataframes
    if all(Path(path).exists() for _, path in paths.items()):
        ## load data in parquet
        log("loading parquet files")
        ret = dict()
        
        for data_type, path in paths.items():
            log(f"loading in {data_type}")
            ret[data_type] = pd.read_parquet(path, engine=PARQUET_ENGINE)

        return ret
    
    # convert if possible
    ## swap extensions & preserve original
    paths = {k: Path(p) for k, p in paths.items()}
    mod_paths = {k: p.with_suffix(f".{'csv'.lstrip('.')}") for k, p in paths.items()}

    ## load as csv
    ret = {
        data_type: pd.read_csv(path, engine="c", low_memory=False)
        for data_type, path in mod_paths.items()
    }
    
    ## save to parquet
    for dt, df in ret.items():
        df.to_parquet(paths[dt], index=False)
    
    # export
    return ret

def _save_data(data_lookup: dict[str, pd.DataFrame], incubator: str, new_version: dict[str, str]):
    """
        Saves the rawdata (tech & social) by their new version ids.
    """

    # generate paths
    new_version = {k: str(v) for k, v in new_version.items()}   # enforce types
    paths = _load_paths(incubator, new_version, ext="parquet")
    print(paths)

    # save
    log(f"saving data for {incubator}: {new_version}")
    data_lookup["tech"].to_parquet(paths["tech"], engine=PARQUET_ENGINE, index=False)
    log(f"saved tech...")
    data_lookup["social"].to_parquet(paths["social"], engine=PARQUET_ENGINE, index=False)
    log(f"saved social...")
    log(f"done!")

def _memory_usage(data_lookup: dict[str, pd.DataFrame]) -> None:
    """
        Prints how much memory the lookup is using.

        @param data_lookup: lookup for dataframe
    """

    # print memory usage
    for dt, df in data_lookup.items():
        print(df.memory_usage(deep=True))

def _validate_data(data_lookup: dict[str, pd.DataFrame]) -> bool:
    """
        Validates the data by ensuring the following:
            1. Months start at 0
            2. Only projects with both social and technical data are kept
        
        If any changes are made, flags to save.

        @param data_lookup (dict[str, pd.DataFrame]): data lookup
        
        @returns bool: flag on whether or not to save data
    """
    
    # track if any changes are made
    flag = False
    
    # starting month
    def minimize_months(df: pd.DataFrame) -> bool:
        ## generate lookup
        min_months = df.groupby("project_name")["month"].min().to_dict()
        
        ## check passed
        if all(min_month == 0 for _, min_month in min_months.items()):
            return False
        
        ## otherwise enforce the min
        min_months = {proj: min_month for proj, min_month in min_months.items() if min_month > 0}
        
        for proj, min_month in min_months.items():
            log(f"correcting {proj} with {min_month=}...")
            df.loc[df["project_name"] == proj, "month"] -= min_month
        
        return True
            
    flag |= minimize_months(data_lookup["social"])
    flag |= minimize_months(data_lookup["tech"])
    
    # overlap
    ## get projects info
    s_set = set(data_lookup["social"]["project_name"].unique())
    t_set = set(data_lookup["tech"]["project_name"].unique())
    overlap = t_set                                                             # changed to only care about tech, not both tech and social
    total = s_set | t_set
    
    ## if there's a mismatch; note, we're fine with no social activity, not fine
    ## with no technical
    if len(total) != len(total & t_set):
        ### update flag
        flag |= True
        
        ### update projects set
        data_lookup["social"] = data_lookup["social"][
            data_lookup["social"]["project_name"].isin(t_set)
        ]
        
    ## report
    log((
        f"Altogether {len(total)} projects, and {len(overlap)} have tech "
        f"networks. Removing the {len(total) - len(overlap)} projects, i.e.: "
        f"{total - overlap}"
    ))
    
    # return if changes were made
    return flag


# Augmentations
def clean_file_paths(data_lookup: dict[str, pd.DataFrame], incubator: str=None, copy: bool=False) -> dict[str, pd.DataFrame]:
    """
    
        Cleans filepaths which contains artifacts from the commit messages.

        @paramd data_lookup (dict[str, pd.DataFrame]): lookup
        @param copy (bool, optional): whether to do inplace or not. Defaults to 
            False.

    Returns:
        dict[str, pd.DataFrame]: lookup
    """
    # aux fn
    def process_filename(filename):
        regex_str = r"^  - copied, (changed|unchanged) from r\d*, "
        rem = re.subn(regex_str, "", filename)
        if rem[1] > 1:
            print(":: ERROR :: double replacement w/", filename)
        return rem[0] #, rem[1]
    
    # diverge if copying
    if copy:
        data_lookup = {k: v.copy() for k, v in data_lookup.items()}
    
    # process
    df = data_lookup["tech"]
    old_file_names = df["file_name"].copy()
    
    log("processing file paths...")
    df["file_name"] = df["file_name"].parallel_apply(process_filename)

    # report
    num_changed = (df["file_name"] != old_file_names).sum()
    log("", "summary")
    print(f"Number of Filenames Changed = {num_changed}")
    
    # return reference
    return data_lookup

def clean_sender_names(data_lookup: dict[str, pd.DataFrame], incubator: str=None, copy: bool=False) -> dict[str, pd.DataFrame]:
    """
        Cleans sender names of any extraneous characters that may interfere with 
        network generation.

        @param data_lookup (dict[str, pd.DataFrame]): lookup
        @param copy (bool, optional): inplace or not. Defaults to False.

    Returns:
        dict[str, pd.DataFrame]: reference to lookup
    """

    # setup imputation
    log("setting up imputation...")
    field = "dealised_author_full_name"
    
    if copy:
        data_lookup = {k: v.copy() for k, v in data_lookup.items()}
    df = data_lookup["social"]
    
    # setup
    num_entries = df.shape[0]
    num_changed = 0

    # imputing
    log("removing extraneous characters in the sender names")
    for i, row in tqdm(df.iterrows()):
        # initialize
        name = str(row["sender_name"])

        # remove space & "#"
        if "#" in name:
            num_changed += 1
            name = name.strip("#").strip()
        if "\n" in name:
            num_changed += 1
            name = name.replace("\n", " ")

        # impute
        df.at[i, field] = name
    
    # summary
    log("", "summary")
    print(f"Number of {field} Entries Corrected: {num_changed}")
    print(f"Corrected {num_changed / num_entries}% of the data")
    return data_lookup

def impute_months(data_lookup: dict[str, pd.DataFrame], strat: str="month", incubator: str=None, copy: bool=False) -> dict[str, pd.DataFrame]:
    """
        Imputes the month field using the start date as a relative point in time:
            1. [strat="month"] Use the start month as the base, i.e. 3/17/2024 
               would make 3 the base month, so 4/01/2024 would still be month 1
            2. [strat="day"] Use the start day as the base and use a set 
               interval of 30 days, i.e. 3/17/2024 would make 4/16/2024 the 
               first day of month 1, etc.
    """

    # setup
    log(f"imputing months via {strat}", "new")
    if copy:
        data_lookup = {k: v.copy() for k, v in data_lookup.items()}

    for k, df in data_lookup.items():
        df["date"] = pd.to_datetime(df["date"]) #, format="%Y-%m-%d %H:%M:%S %Z")
        
        if df["date"].dt.tz is None:
            df["date"] = df["date"].dt.tz_localize("UTC")

    # ensure no NaT by removing any projects that don't have both social and 
    # tech info
    social_proj_set = set(data_lookup["social"]["project_name"].unique())
    tech_proj_set = set(data_lookup["tech"]["project_name"].unique())
    overlap_proj_set = set(social_proj_set & tech_proj_set)
    log(f"removing {len(social_proj_set | tech_proj_set) - len(overlap_proj_set)} projects for not having both social and technical information", "warning")

    data_lookup["social"] = data_lookup["social"][
        data_lookup["social"]["project_name"].isin(overlap_proj_set)
    ]
    data_lookup["tech"] = data_lookup["tech"][
        data_lookup["tech"]["project_name"].isin(overlap_proj_set)
    ]
    
    # get earliest entry for both datasets; note, we'll require commits to be 
    # present in the first month, so realistically we only use the tech network
    # minimum month
    log("getting start dates...")
    first_entry_date = data_lookup["tech"].groupby("project_name")["date"].min().reset_index()

    ## setup lookup
    first_entry_date.rename(columns={"date": "first_entry_date"}, inplace=True)
    first_entry_date["first_entry_date"] = pd.to_datetime(first_entry_date["first_entry_date"])

    # utility
    log("filling months...")
    def months_fill(df: pd.DataFrame, first_entry_date: pd.DataFrame, strat: str) -> pd.DataFrame:
        """
            Fills months using the strat.
        """

        # generate months
        match strat:
            case "month":
                # merge
                df = pd.merge(df, first_entry_date, on="project_name", how="left")
                df["first_month"] = df["first_entry_date"].dt.month
                df["first_year"] = df["first_entry_date"].dt.year

                # number of months since the first month
                df["month"] = (df["date"].dt.year - df["first_year"]) * 12 \
                    + (df["date"].dt.month - df["first_month"])
                
                # clean
                df.drop(columns=["first_month", "first_year", "first_entry_date"], inplace=True)
            
            case "day":
                min_dates = df.groupby("project_name")["date"].transform("min")
                df["month"] = (df["date"] - min_dates).dt.days // 30

            case _:
                print(f"<ERROR> failed to resolve strat {strat} chosen for imputing months")
        
        # output
        return df
    
    # execution
    data_lookup = {k: months_fill(df, first_entry_date, strat) for k, df in data_lookup.items()}
    
    # remove any social data prior to first commit
    data_lookup["social"] = data_lookup["social"][data_lookup["social"]["month"] >= 0]
    
    # export
    return data_lookup

def impute_messageid(
    data_lookup: dict[str, pd.DataFrame], incubator: str=None, copy: bool=True,
    force_impute: bool=False, strat: str="conserve", field: str="message_id",
    author_field: str=None
) -> dict[str, pd.DataFrame]:
    """Generates a unique messageid from project, sender, timestamp. This 
    inherently applies only to the social data.
    
    Args:
        data_lookup (dict[str, pd.DataFrame]): lookup of tech/social to 
            dataframe.
        incubator (str, optional): not needed, only here as an argument for 
            backwards compatability. Defaults to None.
        copy (bool, optional): whether to copy data prior to imputation. 
            Defaults to True.
        force_impute (bool, optional): whether to forcefully impute even if the
            field is already mostly filled (95% threshold). Defaults to False.
        strat (str, optional): one of {"conserve", "long"}.
            - "conserve" generates a memory conservative unique message id for 
            each communication (i.e. simple numbering)
            - "long" generates a unique id using other fields available in the 
            social dataset
            
            Defaults to "conserve".
    """

    # references
    if copy:
        data_lookup = {k: v.copy() for k, v in data_lookup.items()}
    df = data_lookup["social"]
    author_field = author_field if author_field else (
        params_dict["author-source-field"][incubator] if incubator is not None
        else "sender_name"
    )

    # enforce type (for later imputation of replies, etc.)
    if field not in df.columns:
        df[field] = None
    df[field] = df[field].astype(str)

    # check not overriding
    log("Imputing Message ID", "new")
    if len(df[field].unique()) / df.shape[0] > 0.95:
        # print warning
        print(f"<WARNING> :: message id field already has >95% unique IDs: {len(df['message_id'].unique()) / df.shape[0] * 100}%")

        # forcefully impute or not
        if not force_impute:
            print("not imputing. . .")
            return data_lookup
    
    # setup
    log("continuing imputation. . .")
    num_entries = df.shape[0]
    num_missing_bef = df[field].isna().sum()
    missing_field = 0
    
    # generate unique ids
    def gen_long_msg_id(row):
        # create unique id
        time = row["date"]
        project = row["project_name"]
        author = row[author_field]

        # missing
        if time == "" or project == "" or author == "":
            nonlocal missing_field
            missing_field += 1

        # export
        return f"<{time}={project}@{author}>".replace(" ", "")

    # apply
    if strat == "conserve":
        # craft ids via integers
        df[field] = pd.Series(range(df.shape[0])).astype(int)
        
        # add the brackets to be verified as replies
        df[field] = "<" + df[field].astype(str) + ">"
    else:
        df[field] = df.apply(gen_long_msg_id, axis=1).reset_index(drop=True)

    # report
    num_missing_aft = df[field].isna().sum()
    delta = num_missing_bef - num_missing_aft
    prop_before = num_missing_bef / num_entries * 100
    prop_after = num_missing_aft / num_entries * 100
    prop_delta = prop_after - prop_before

    log(log_type="summary")
    print(f"Number Missing {field} (before): {num_missing_bef}")
    print(f"Number Missing {field} (after): {num_missing_aft}")
    print(f"Number of {field} Entries Corrected: {delta}")
    print(f"Entries w/ Missing Field Info: {missing_field}")
    print(f"Corrected {prop_delta}% of the data, from {prop_before}% to {prop_after}% message IDs in place")

    # export
    return data_lookup

def infer_replies(
    data_lookup: dict[str, pd.DataFrame], incubator: str=None, copy: bool=True,
    force_impute: bool=False, strat: str="ONE"
) -> dict[str, pd.DataFrame]:
    """
        Generate reply information by grouping by project, subject then 
        associating replies with the reply before it. Note that we'll now assume
        that anyone who replies after another person, replies to every who has 
        replied in that thread.

        NOTE: uses message id for reply info

        Args:
            data_lookup (dict[str, pd.DataFrame]): lookup of tech/social to the 
                data.
            incubator (str, optional): name of the incubator. Not needed, only 
                here for backwards compatibility. Defaults to None.
            copy (bool, optional): whether to copy prior to imputation. Defaults
                to True.
            force_impute (bool, optional): whether to ignore warnings about 
                pre-existing reply information. Defaults to False.
            strat (str, optional): strategy to use, should be one of {"ONE", 
                "ALL"}. ONE only imputes the previous reply, ALL imputes all 
                previous replies. Defaults to ONE.
    """
    
    # auxiliary fn
    def basic_reply_inference(col: pd.Series) -> pd.Series:
        """Basic reply inferencing by drawing an edge with only the previous 
        person who is being replied to.
        """
        
        # shift down message id's by one
        return col.shift(1)
    
    def period_reply_inference(reply_col: pd.Series) -> list[str]:
        """A function to be used in conjunction with groupby and transform to 
        conduct reply inference on a period of a given project's data. Note that
        this generic formula can be applied regardless of if we 

        Args:
            reply_col (pd.Series): a period of a single project's reply data, 
                grouped by subject and sorted by date.

        Returns:
            list[str]: accumulate replies for the given thread of a single 
                project.
        """
        
        # custom accumulate logic
        def acc_fn(x: str, y: str) -> str:
            if not x:
                return y
            if not y:
                return x
            return f"{x} {y}"
        
        # accumulate the replies and return the column
        return list(accumulate(reply_col, acc_fn))

    # references
    if copy:
        data_lookup = {k: v.copy() for k, v in data_lookup.items()}
    df = data_lookup["social"]

    # utility for checking inference strategy (check number of potential replies)
    reply_freq = df.groupby(["project_name", "subject"], observed=True).size().reset_index(name="count")
    prop_replies_inference = reply_freq[reply_freq["count"] > 1]["count"].sum() / len(reply_freq)

    # check overriding; either if we can't impute more than exists or we're 
    # imputing when we probably don't need to
    log("Inferring Reply Information", "new")
    prop_filled = len(df["in_reply_to"].unique()) / df.shape[0]
    if prop_replies_inference <= prop_filled:
        # print warning
        log(
            f"inference strategy obtains less threads than already provided: {prop_replies_inference * 100}% < {prop_filled * 100}%",
            "warning"
        )
        
        # skip imputation if not forcing
        if not force_impute:
            log("not imputing. . .")
            return data_lookup

    if prop_filled > 0.95:
        # print warning
        log(
            f"in_reply_to field already has >95% unique replies: {len(df['in_reply_to'].unique()) / df.shape[0] * 100}%",
            "warning"
        )
        
        # skip imputation if not forcing
        if not force_impute:
            log("not imputing. . .")
            return data_lookup

    # inference
    log("continuing inference")
    field = "in_reply_to"
    impute_source_field = "message_id"

    # for later comparison
    missing_before = (df[field] == "").sum() + df[field].isnull().sum()
    num_entries = df.shape[0]
    prop_before = missing_before / num_entries * 100

    # sort and group for easy association
    df.sort_values(by=["project_name", "subject", "date"], inplace=True)
    grouped = df.groupby(["project_name", "subject"], observed=True)

    # transform to get the in_reply_to field imputed
    log("imputing from the source field...")
    df[field] = grouped[impute_source_field].transform(basic_reply_inference)
    
    # transform to accumulate the replies to accurately reflect the reply chain
    if strat == "ALL":
        log("accumulating the replies...")
        df[field] = grouped[field].transform(period_reply_inference)

    # remove self-referencing edges
    same_sender_mask = df[impute_source_field] == df[field]
    df.loc[same_sender_mask, field] = ""

    # report
    missing_after = (df[field] == "").sum() + df[field].isnull().sum()
    delta = missing_before - missing_after
    prop_after = missing_after / num_entries * 100
    prop_delta = prop_after - prop_before

    log(log_type="summary")
    print(f"Number of Missing Entries (before): {missing_before}")
    print(f"Number of Missing Entries (after): {missing_after}")
    print(f"Number of {field} Entries Imputed: {delta}")
    print(f"Percent Missing went from {prop_before}% to {prop_after}% for a delta of {prop_delta}%")

    # export
    return data_lookup

def clean_source_files(data_lookup: dict[str, pd.DataFrame], incubator: str=None, copy: bool=True) -> dict[str, pd.DataFrame]:
    """
        Ensure that only coding files are used for technical network generation 
        in the commits data by imputing the `is_coding` field.
    """

    # setup
    print("\n<Ensuring Definition for Code Source Files>")
    if copy:
        data_lookup = {k: v.copy() for k, v in data_lookup.items()}
    df = data_lookup["tech"]
    field = "is_coding"
    tech_file_path = Path(params_dict["ref-dir"]) / "programming-langs.json"

    # initialize if field doesn't exist
    if field not in df.columns:
       df[field] = 0

    # specify tech files
    with open(tech_file_path, "r") as f:
        programming_languages_extensions = json.load(f)

    coding_extensions = set([".mdtext"])
    for pl in programming_languages_extensions:
        if "extensions" not in pl:
            continue
        # filter out some data extensions, e.g., json
        if pl["type"] != "programming" and pl["type"] != "markup":
            continue
        coding_extensions = coding_extensions.union(set(pl["extensions"]))

    # for later comparison
    num_entries = df.shape[0]
    num_coding_bef = df[field].sum()
    num_changed = [0, 0, 0]

    # imputing
    def check_source(row):
        file = row["file_name"]

        try:
            ext = file[file.rindex("."): ]
        except:
            num_changed[0] += 1 if row[field] != 0 else 0
            return 0

        if ext in coding_extensions:
            num_changed[1] += 1 if row[field] != 1 else 0
            return 1
        else:
            num_changed[2] += 1 if row[field] != 0 else 0
            return 0

    df[field] = df[["file_name", field]].parallel_apply(check_source, axis=1)

    # report
    num_coding_aft = df[field].sum()
    delta = num_coding_aft - num_coding_bef
    prop_before = num_coding_bef / num_entries * 100
    prop_after = num_coding_aft / num_entries * 100
    prop_delta = prop_after - prop_before

    print("\n ::: SUMMARY ::: ")
    print(f"Number of Coding files (before): {num_coding_bef}")
    print(f"Number of Coding files (after): {num_coding_aft}")
    print(f"Number of {field} Entries Forced: {delta}")
    print(f"Number of {field} Entries Corrected: {sum(num_changed)}")
    print(f"Number changed: {num_changed[0]} skipped, {num_changed[1]} forced src, {num_changed[2]} forced non-src")
    print(f"Corrected {prop_delta}% of the data, from {prop_before}% to {prop_after}% source files considered")

    # export
    return data_lookup

def infer_bots(data_lookup: dict[str, pd.DataFrame], incubator: str, threshold: float=0.05, copy: bool=True) -> dict[str, pd.DataFrame]:
    """
        Uses heurstics about bot names and string matching to infer whether or 
        not a user is a bot. NOTE we only remove bots that contribute a 
        significant proportion of the data.
    """

    # setup imputation
    print("\n<Inferring Bots>")
    field = "is_bot"
    author_field = params_dict["author-source-field"][incubator]
    ref_dir = Path(params_dict["ref-dir"])
    dir_bot_def = ref_dir / f"{incubator}_bot_names.json"

    # specify bots
    try:
        with open(dir_bot_def, "r") as f:
            bot_names = json.load(f)
    except FileNotFoundError as fe:
        log(f"Failed to find reference file for bot names @ {dir_bot_def}", "warning")
        bot_names = {
            "substring-bots": set(),
            "project-bots": set()
        }

    bot_substrings = bot_names["substring-bots"]
    bot_specific = bot_names["project-bots"]
    bot_specific = set([bot.lower() for bot in bot_specific])

    # reading data
    if copy:
        data_lookup = {k: v.copy() for k, v in data_lookup.items()}
    tech_df = data_lookup["tech"]
    social_df = data_lookup["social"]

    tech_df["is_bot"] = 0
    social_df["is_bot"] = 0

    # grouping
    grouped_tech_df = tech_df.groupby(["project_name", author_field])
    grouped_social_df = social_df.groupby(["project_name", author_field])
    proj_activity_lookup = {
        "tech": tech_df.groupby("project_name").size().to_dict(),
        "social": social_df.groupby("project_name").size().to_dict()
    }

    # utility
    def process_bots(group, type_activity: dict[str, int], bot_substrings: list[str]) -> dict[str, dict[str, Any]]:
        """
            Utility to detect whether or not the sender is a bot, returning a 
            dict for {substring: int, outlier: int, proportion: float, 
            is-bot: int}
        """

        # match extension
        proj = str(group["project_name"].iloc[0])
        sender = str(group[author_field].iloc[0])

        # counts
        num_commits = group.shape[0]
        num_proj = type_activity[proj]
        prop = num_commits / num_proj

        # flags
        name_match = sender.lower() in bot_specific                         # specifically defined as bot (manually)
        sub_match = any(sub in sender.lower() for sub in bot_substrings)    # substring matches with a bot name
        outlier_match = prop > threshold                                    # proportion of commits matches

        # set true
        return {sender: {
            "substring": int(sub_match),
            "outlier": int(outlier_match),
            "proportion": prop,
            "is-bot": int((name_match) or (sub_match and outlier_match))
        }}
    
    def gen_bots_lookup(df: pd.DataFrame, type_activity: dict[str, int], bot_subs: list[str]) -> list[dict[str, dict[str, Any]]]:
        """
            Handles multiprocessing for the bot inference.
        """

        # multiprocessing
        bots = df.apply(
            process_bots, 
            type_activity=type_activity, 
            bot_substrings=bot_subs
        ).to_list()

        # close
        return bots

    def impute_bots(df: pd.DataFrame, bot_lookup: list[dict[str, dict[str, Any]]]) -> pd.DataFrame:
        """
            Uses the bot lookup to impute the `is_bot` field.
        """

        # lookup generation
        bots = {name for person_info in bot_lookup for name, info in person_info.items() if info["is-bot"] == 1}

        # utility
        def check_bot(sender):
            return 1 if sender in bots else 0
        
        # apply
        df[field] = df[author_field].apply(check_bot)
        return df

    # apply
    tech_bots = gen_bots_lookup(grouped_tech_df, proj_activity_lookup["tech"], bot_subs=bot_substrings)
    tech_df = impute_bots(tech_df, tech_bots)
    social_bots = gen_bots_lookup(grouped_social_df, proj_activity_lookup["social"], bot_subs=bot_substrings)
    social_df = impute_bots(social_df, social_bots)

    # report
    num_tech_bots = tech_df[tech_df["is_bot"] == 1] \
        .groupby(["project_name", author_field]) \
        .size() \
        .shape[0]
    num_social_bots = social_df[social_df["is_bot"] == 1] \
        .groupby(["project_name", author_field]) \
        .size() \
        .shape[0]

    print("\n ::: SUMMARY ::: ")
    print(f"Number of Tech Bots (after): {num_tech_bots}")
    print(f"Number of Social Bots (after): {num_social_bots}")
    
    # export
    with open(ref_dir / f"{incubator}_tech_bots_removed.json", "w") as f:
        json.dump(tech_bots, f, indent=4)
    with open(ref_dir / f"{incubator}_social_bots_removed.json", "w") as f:
        json.dump(social_bots, f, indent=4)
    return data_lookup

def dealias_senders(
    data_lookup: dict[str, pd.DataFrame], incubator: str, source_field: str="",
    sim_threshold: float=0.9, dev_threshold: int=6000, export: bool=False,
    copy: bool=True, **kwargs
) -> dict[str, pd.DataFrame]:
    """
        De-aliases committers and commenters using string similarity matching 
        and other heuristics.

        NOTE: need to manually specific source field in the lookup below.
    """

    # source_field lookup
    author_lookup = params_dict["author-source-field"]
    author_field = source_field if source_field != "" else author_lookup[incubator]
    output_field = "dealised_author_full_name"
    ref_dir = Path(params_dict["ref-dir"])
    
    # CACHING -- replace all pre-computed aliases #
    # The idea here is that any future aliases we use can equivalently be mapped
    # onto any aliases we've previously made. Thus, we save time by not needing
    # to re-make the same aliases while preserving the opportunity to make new
    # aliases
    alias_mapping_path = ref_dir / f"{incubator}_alias_mapping.json"

    def _enforce_aliases():
        with open(ref_dir / f"{incubator}_alias_mapping.json", "r") as f:
            alias_mapping = json.load(f)

        def _dealiasing(project_name, author_name):
            if (project_name in alias_mapping) and (author_name in alias_mapping[project_name]):
                return alias_mapping[project_name][author_name]
            return " ".join([name.capitalize() for name in author_name.split(" ")])
        
        # dealiasing
        df = data_lookup["tech"]
        df = df[(df["is_bot"] == False) & (df["is_coding"] == True) & (df[author_field] != "none")]
        df = df[df[author_field].notna()]
        df[output_field] = df.apply(lambda x: _dealiasing(x["project_name"], x[author_field]), axis=1)
        aft_num_tech = df[output_field].nunique()

        df = data_lookup["social"]
        df = df[(df["is_bot"] == False) & (df[author_field] != "none")]
        df = df[df[author_field].notna()]
        df[output_field] = df.apply(lambda x: _dealiasing(x["project_name"], x[author_field]), axis=1)
        aft_num_social = df[output_field].nunique()
        
        # returns
        return aft_num_tech, aft_num_social
    
    if alias_mapping_path.exists():
        # grab some debugging info
        bef_num_tech = data_lookup["tech"][author_field].unique().shape[0]
        bef_num_social = data_lookup["social"][author_field].unique().shape[0]
        
        # enforce the previous aliases we've found
        aft_num_tech, aft_num_social = _enforce_aliases(**kwargs)
        
        # give a summary of just the caching protocol
        print("======== CACHING SUMMARY ========")
        print(f"Unique Tech Devs (before): {bef_num_tech}")
        print(f"Unique Tech Devs (after): {aft_num_tech}")
        print(f" ::::::::: DELTA TECH = {aft_num_tech - bef_num_tech}")
        print(f"Unique Social Devs (before): {bef_num_social}")
        print(f"Unique Social Devs (after): {aft_num_social}")
        print(f" ::::::::: DELTA SOCIAL = {aft_num_social - bef_num_social}\n\n")
    
    # dealias functionality
    def indices_dict(lis):
        d = defaultdict(list)
        for i,(a,b) in enumerate(lis):
            d[a].append(i)
            d[b].append(i)
        return d

    def disjoint_indices(lis):
        d = indices_dict(lis)
        sets = []
        while len(d):
            que = set(d.popitem()[1])
            ind = set()
            while len(que):
                ind |= que 
                que = set([y for i in que 
                            for x in lis[i] 
                            for y in d.pop(x, [])]) - ind
            sets += [ind]
        return sets

    # union-find algo
    def disjoint_sets(lis):
        return [list(set([x for i in s for x in lis[i]])) for s in disjoint_indices(lis)]

    # process a name
    def process_name(name):
        try:
            # if it is an email, take only the user domain
            name = name.split("@")[0]
        except:
            print(name)
            raise KeyError
        # remove text within brakets and parentheses
        name = re.sub(r"[\(\[].*?[\)\]]", "", name)
        # remove non-alphanumeric chars
        name = re.sub("[^a-zA-Z ]+", "", name)
        if "$" in name:
            # some names are of this pattern: "sg $ $date: 2008/10/07 10:18:51 $"
            name = name.split("$")[0]
        return name.strip()

    def check_segments(name1, name2):
        name_segs_1 = name1.split(" ")
        name_segs_2 = name2.split(" ")

        if len(name_segs_1) == len(name_segs_2) == 2:
            first_name_1, last_name_1 = name_segs_1
            first_name_2, last_name_2 = name_segs_2

            # option 1: first name 1 compare to first name 2, last name 1 compare to last name 2
            # e.g., "robert yates" v.s. "robert butts"
            first_name_score = jaro_winkler_similarity(first_name_1, first_name_2)
            last_name_score = jaro_winkler_similarity(last_name_1, last_name_2)
            if first_name_score < 0.8 or last_name_score < 0.8:
                return False
            # option 2: first name 1 compare to last name 2, last name 1 compare to first name 2
            # e.g., "yates robert" v.s. "robert butts"
            else:
                first_name_score = jaro_winkler_similarity(first_name_1, last_name_2)
                last_name_score = jaro_winkler_similarity(last_name_1, first_name_2)
                if first_name_score > 0.8 and last_name_score > 0.8:
                    return True
        return True

    # setup processes
    kw_args = {
        "data_lookup": data_lookup,
        "incubator": incubator,
        "source_field": source_field,
        "sim_threshold": sim_threshold,
        "dev_threshold": dev_threshold,
        "export": export,
        "copy": copy
    }

    # PROCESS 1 -- generate aliases #
    def _dealiasing_gen_aliases(**kwargs):
        # read in
        if copy:
            data_lookup["tech"] = data_lookup["tech"].copy()
            data_lookup["social"] = data_lookup["social"].copy()
        
        tech_df = data_lookup["tech"][["project_name", author_field, "is_bot", "is_coding"]]
        social_df = data_lookup["social"][["project_name", author_field, "is_bot"]]
        bef_num_tech = tech_df[author_field].unique().shape[0]
        bef_num_social = social_df[author_field].unique().shape[0]
        
        # clear unusable
        print("clearing unusable rows...")
        tech_df = tech_df[(tech_df["is_bot"] == False) & (tech_df["is_coding"] == True) & \
                        (tech_df[author_field] != "none")]
        social_df = social_df[(social_df["is_bot"] == False) & (social_df[author_field] != "none")]
        # tech_df.query(f"is_bot == False and is_coding == True and {author_field} != 'none'", inplace=True)
        # social_df.query(f"is_bot == False and {author_field} != 'none'", inplace=True)

        tech_df = tech_df[tech_df[author_field].notna()]
        social_df = social_df[social_df[author_field].notna()]

        # processing
        print("processing...")
        tech_dict = tech_df.to_dict("records")
        social_dict = social_df.to_dict("records")
        committers = {}
        contributors = {}

        for commit in tqdm(tech_dict):
            project_name = commit["project_name"]
            sender_name = commit[author_field]
            if project_name not in committers:
                committers[project_name] = set()
            committers[project_name].add(sender_name)

        for email in tqdm(social_dict):
            project_name = email["project_name"]
            sender_name = email[author_field]
            if project_name not in contributors:
                contributors[project_name] = set()
            contributors[project_name].add(sender_name)

        # get projects set
        t_projects = set(committers.keys())
        s_projects = set(contributors.keys())
        projects = sorted([p for p in t_projects.intersection(s_projects) if not pd.isna(p)])

        project_alias_clustering = {}
        for project in tqdm(projects):
            clustering_pairs = []
            committer_names = set(committers[project])
            contributor_names = set(contributors[project])
            developer_names = list(committer_names.union(contributor_names))

            # skip if too big
            num_devs = len(developer_names)
            if num_devs > dev_threshold:
                continue

            for i in tqdm(range(len(developer_names)), leave=False):

                p1 = process_name(developer_names[i])
                
                for j in tqdm(range(i+1, len(developer_names)), leave=False):
                    # if it is an email, take only the user domain
                    p2 = process_name(developer_names[j])

                    # reslove issues that two different devs sharing same first name, 
                    # e.g., "robert ottaway", "robert sayre"
                    if not check_segments(p1, p2):
                        continue

                    jaro_winkler_similarity_score = jaro_winkler_similarity(p1, p2)
                    # sounding_match_score = any([match_rating_comparison(s1, s2) for s1 in name_segs_1 for s2 in name_segs_2])
                    # sounding_match_score = any([sounding_match_score, match_rating_comparison(p1, p2)])

                    # add pairs if:
                    # (1) if the score fall down to sim_threshold 
                    # (2) or if the score fall down to 0.82 then we use pronouncetion to help make decision
                    if jaro_winkler_similarity_score > sim_threshold: # or (jaro_winkler_similarity_score > 0.82 and sounding_match_score):
                        clustering_pairs.append([developer_names[i], developer_names[j]])
            
            project_alias_clustering[project] = disjoint_sets(clustering_pairs)

        with open(ref_dir / f"{incubator}_project_alias_clustering.json", "w") as f:
            json.dump(project_alias_clustering, f, indent = 4)


        # post-processing
        print("starting post-processing...")
        for project in tqdm(project_alias_clustering):
            # print(project, len(project_alias_clustering[project]))
            for n_index, n_cluster in tqdm(enumerate(project_alias_clustering[project]), leave=False):
                cluster = n_cluster[:]
                lowest_score = 0
                # continue checking if the avg score of any nodes is below sim_threshold
                while len(cluster) >= 2 and lowest_score < sim_threshold:
                    name_to_pop = None
                    lowest_score = float("inf")
                    score_dict = {}
                    for name_i in cluster:
                        p1 = process_name(name_i)
                        this_score = 0
                        for name_j in cluster:
                            if name_i == name_j: continue
                            p2 = process_name(name_j)
                            jaro_winkler_similarity_score = jaro_winkler_similarity(p1, p2)
                            this_score += jaro_winkler_similarity_score/(len(cluster)-1)
                        if this_score < lowest_score:
                            name_to_pop = name_i
                            lowest_score = this_score
                        score_dict[name_i] = this_score

                    if lowest_score < sim_threshold:
                        cluster.pop(cluster.index(name_to_pop))
                        score_dict.pop(name_to_pop)

                # continue checking segments in names of two-parts form, e.g., "robert ottaway", "robert sayre"
                flag = True
                while flag and cluster:
                    flag = False
                    # use cluster_copy to avoid affecting the for-loop
                    pop_set = set()
                    for name_i in cluster:
                        p1 = process_name(name_i)
                        for name_j in cluster:
                            if name_i == name_j: continue
                            p2 = process_name(name_j)
                            # if the two names cant be in same cluster
                            # pop the node with lowest avg. score
                            if check_segments(p1, p2):
                                continue
                            flag = True
                            if score_dict[name_i] < score_dict[name_j]:
                                pop_set.add(name_i)
                            else:
                                pop_set.add(name_j)

                    for name in pop_set:
                        cluster.pop(cluster.index(name))
                        score_dict.pop(name)

                # ingore large cluster or if it has fewer than two names in the cluster
                if len(cluster) < 2 or len(cluster) > 5:
                    project_alias_clustering[project].pop(n_index)
                    continue

                # manual check: cases that dont make sense
                elif "michael glauche" in cluster and "michael akerman" in cluster:
                    project_alias_clustering[project].pop(n_index)
                    continue
                elif "martin kool" in cluster and "martin vojtek" in cluster:
                    project_alias_clustering[project].pop(n_index)
                    continue
                elif "john arnold" in cluster and "john hofman" in cluster:
                    project_alias_clustering[project].pop(n_index)
                    continue
                elif "martin von gagern" in cluster and "martin weber" in cluster:
                    project_alias_clustering[project].pop(n_index)
                    continue
                else:
                    project_alias_clustering[project][n_index] = cluster

        with open(ref_dir / f"{incubator}_project_alias_clustering_filtered.json", "w") as f:
            json.dump(project_alias_clustering, f, indent = 4)

        # construct the alias to full name mapping
        # ideally, we can find a "regular" full name for each cluster. 
        # this does not tend to affect the result of de-aliasing, it is only for a better looking 
        # (e.g., "gerrie" will be mapped to "gerrie myburgh" but not "gerriem")

        print("alias mapping...")
        alias_mapping = {}
        for project in tqdm(project_alias_clustering):
            if project not in alias_mapping:
                alias_mapping[project] = {}
            
            for alias_list in project_alias_clustering[project]:
                longest_name = None 
                longest_name_length = float("-inf")
                list_of_regular_names = []

                for alias in alias_list:
                    length = len(alias)
                    if length > longest_name_length:
                        longest_name = alias 
                        longest_name_length = length

                    # exactly has two parts, which is a regular full name
                    if len(alias.split(" ")) == 2:
                        list_of_regular_names.append([alias, length])

                # first piority: longest name with two exactly parts (i.e., first name, last name)
                if list_of_regular_names:
                    final_name = sorted(list_of_regular_names, key=lambda x: x[1]).pop()[0]
                # second piority: the longest name in the list
                else: final_name = longest_name
                # capitalization
                final_name_list = [name.capitalize() for name in final_name.split(" ")]
                final_name = " ".join(final_name_list)
                alias_mapping[project].update({alias: final_name for alias in alias_list})

        with open(ref_dir / f"{incubator}_alias_mapping.json", "w") as f:
            json.dump(alias_mapping, f, indent=4)

        # returns
        return bef_num_tech, bef_num_social
    

    # PROCESS 2 -- enforce aliases
    def _dealiasing_enforce_aliases(**kwargs):
        with open(ref_dir / f"{incubator}_alias_mapping.json", "r") as f:
            alias_mapping = json.load(f)

        def _dealiasing(project_name, author_name):
            if (project_name in alias_mapping) and (author_name in alias_mapping[project_name]):
                return alias_mapping[project_name][author_name]
            return " ".join([name.capitalize() for name in author_name.split(" ")])
        
        # dealiasing
        tdf = data_lookup["tech"]
        # df.query(f"is_bot == False and is_coding == True and {author_field} != 'none'", inplace=True)
        tdf = tdf[(tdf["is_bot"] == False) & (tdf["is_coding"] == True) & (tdf[author_field] != "none")]
        tdf = tdf[tdf[author_field].notna()]
        tdf[output_field] = tdf.apply(lambda x: _dealiasing(x["project_name"], x[author_field]), axis=1)
        # tdf[output_field] = _dealiasing(tdf["project_name"], tdf[author_field])
        aft_num_tech = tdf[output_field].nunique()

        sdf = data_lookup["social"]
        # df.query(f"is_bot == False and {author_field} != 'none'", inplace=True)
        sdf = sdf[(sdf["is_bot"] == False) & (sdf[author_field] != "none")]
        sdf = sdf[sdf[author_field].notna()]
        # df[output_field] = _dealiasing(df["project_name"], df[author_field])
        sdf[output_field] = sdf.apply(lambda x: _dealiasing(x["project_name"], x[author_field]), axis=1)
        aft_num_social = sdf[output_field].nunique()

        # export
        if export:
            _save_data(data_lookup, incubator, new_version=kwargs["new_version"])

        # returns
        return {"tech": tdf, "social": sdf}, aft_num_tech, aft_num_social

    # conduct
    bef_num_tech, bef_num_social = _dealiasing_gen_aliases(**kw_args)
    bef_num_tech, bef_num_social = data_lookup["tech"]["sender_name"].nunique(), data_lookup["social"]["sender_name"].nunique()
    new_lookup, aft_num_tech, aft_num_social = _dealiasing_enforce_aliases(**kw_args)

    # report
    print("======== SUMMARY ========")
    print(f"Unique Tech Devs (before): {bef_num_tech}")
    print(f"Unique Tech Devs (after): {aft_num_tech}")
    print(f" ::::::::: DELTA TECH = {aft_num_tech - bef_num_tech}")
    print(f"Unique Social Devs (before): {bef_num_social}")
    print(f"Unique Social Devs (after): {aft_num_social}")
    print(f" ::::::::: DELTA SOCIAL = {aft_num_social - bef_num_social}")
    print("All done!")

    # return
    return new_lookup

def pre_process_data(
    data_lookup: dict[str, pd.DataFrame], incubator: str, copy: bool=False,
    save_versions: dict[str, int]=None, **kwargs
) -> None:
    """Pre-Processes the raw data via the usual steps (e.g. message_id 
    imputation, reply inference, source file cleaning, bot inference, and 
    de-aliasing).

    Args:
        data_lookup (dict[str, pd.DataFrame]): data lookup of tech/social to the
            respective data.
        incubator (str): incubator reference to use.
        copy (bool, optional): copy flag to pass into each computation. Defaults
            to False to avoid OOM errors.
        save_versions (dict[str, int], optional): version lookup of social/tech 
            to the respective version to use. Defaults to 1/1 for tech/social.
    """
    
    log("Pre-Processing Raw Data", "new")
    
    # infer save versions if needed
    if save_versions is None:
        save_versions = {"tech": 1, "social": 1}

    # validation
    if _validate_data(data_lookup):
        _save_data(
            data_lookup=data_lookup, incubator=incubator,
            new_version=save_versions
        )

    # cleaning; we'll cache prior to the de-aliasing step in case of OOM errors
    data_lookup = impute_messageid(
        data_lookup, incubator=incubator, copy=False, force_impute=True
    )
    data_lookup = infer_replies(data_lookup, incubator=incubator, copy=False)
    data_lookup = clean_source_files(data_lookup, incubator=incubator, copy=False)
    data_lookup = infer_bots(data_lookup, incubator=incubator, copy=False)
    _save_data(
        data_lookup=data_lookup, incubator=incubator, new_version=save_versions
    )
    data_lookup = dealias_senders(data_lookup, incubator=incubator, copy=False)
    _save_data(
        data_lookup=data_lookup, incubator=incubator, new_version=save_versions
    )


# Class; we'll avoid slots since we're loading large datasets in
@dataclass(slots=False)
class RawData:
    # data
    incubator: str                                                  # original incubator
    versions: dict[str, str] = field(default=None)                  # tech & social augmentation number
    gen_full: bool = field(default=False)                           # if base data, generate processed version
    ext: str = field(default="parquet")                             # data file type
    paths: dict[str, str] = field(init=False)                       # type (tech/social) : path
    data: dict[str, pd.DataFrame] = field(init=False, repr=False)   # type : df

    # post-initialization
    def __post_init__(self):
        # load data
        if self.versions is None:
            self.versions = dict(zip(
                ["tech", "social"],
                params_dict["default-versions"][self.incubator]
            ))
        self.versions = {k: str(v) for k, v in self.versions.items()}   # allow for integer inputs
        self.paths = _load_paths(incubator=self.incubator, versions=self.versions, ext=self.ext)

        dtypes = params_dict["dtypes"][self.incubator]
        self.data = _load_data(self.paths, dtypes=dtypes)
        
        # validate
        self.validate_data()

        # first-time loading
        base_set = all(v == "0" for k, v in self.versions.items())

        if base_set:
            # project incubation
            self.gen_proj_incubation()

        if self.gen_full and base_set:
            print("\n<Generating Processed Data>")

            # validation
            is_invalid, error_msg = self.validate_data()
            if is_invalid:
                print(f":: CAUTION :: invalid data; {error_msg}")
                exit(1)

            # cleaning
            data_lookup = impute_messageid(self.data, copy=False)
            data_lookup = infer_replies(data_lookup, copy=False)
            data_lookup = clean_source_files(data_lookup, incubator=self.incubator, copy=False)
            data_lookup = infer_bots(data_lookup, incubator=self.incubator, copy=False)
            _save_data(
                data_lookup=data_lookup, 
                incubator=self.incubator, 
                new_version={"tech": 1, "social": 1}
            )
            data_lookup = dealias_senders(data_lookup, incubator=self.incubator, copy=False)
            _save_data(
                data_lookup=data_lookup, 
                incubator=self.incubator, 
                new_version={"tech": 1, "social": 1}
            )
    

    # utility
    def validate_data(self):
        """
            Validates the rawdata loaded in.

            Since there are many combinations of fields we can use, we'll store 
            the minimum information we need (i.e. "replies"), and then their 
            values will store the individual fields that may be used to impute 
            (i.e. "message_id", "timestamp", "subject", or "in_reply_to", etc.).
        """

        # ensure some columns
        # for k, v in self.data.items():
        #     if "dealised_author_full_name" not in v.columns:
        #         v["dealised_author_full_name"] = v[params_dict["author-field"][self.incubator]]
        #     if "is_bot" not in v.columns:
        #         v["is_bot"] = 0
        #     if k == "tech" and "is_coding" not in v.columns:
        #         v["is_coding"] = 1
                
        # ensure data itself is valid
        if _validate_data(self.data):
            self.save_data()
        return False, "data is valid"
        
        return False, "data is valid"

        # aliases for all the key columns
        tech_aliases = {
            "month": set(["month", "timestamp"]),
            "project_name": set(["project", "proj_name", "project_name"]),
            "file_name": set(["file", "file_name", "file_path", "path", "filename"]),
            "sender_name": set(["sender_name", "sender_email"]),
            "date": set(["timestamp", "time", "date"]),
            "is_bot": set(["is_bot", "bot"]),
            "is_coding": set(["is_coding", "coding"]),
            "dealiased_author_full_name": set(["dealiased_author_full_name"])
        }
        social_aliases = {
            "month": set(["month", "timestamp"]),
            "project_name": set(["project", "proj_name", "project_name"]),
            "in_reply_to": set(["in_reply_to", "reply", "receiver", "receiver_email"]),
            "subject": set(["subject", "title"]),
            "sender_name": set(["sender_name", "sender_email"]),
            "date": set(["timestamp", "time", "date"]),
            "is_bot": set(["is_bot", "bot"]),
            "dealiased_author_full_name": set(["dealised_author_full_name"]),
            "message_id": set(["message_id"])
        }

        # 
        tech_key_cols = set().union(*tech_aliases.values())
        social_key_cols = set().union(*social_aliases.values())

        # keys :: required attributes
        # values :: a list of sets where in each set, all attributes must be 
        #           present in order to satisfy the validity condition
        required_columns = {
            "month": [
                set(["month"])
            ],
            "project": [
                set(["project"])
            ],
            "reply": [
                set(["in_reply_to"]),
                set(["timestamp", "subject"]),
                set(["sender_name", "receiver_name"])
            ],
            "sender": [
                set(["sender_name"]),
                set(["sender_email"])
            ]
        }

        imputable_cols = set([
            "is_coding",
            "is_bot",
            "message_id"
        ])


        # check the columns exist, otherwise force them to
        tech_missing_cols = set(tech_aliases.keys()) - set(self.data["tech"].columns)
        social_missing_cols = set(social_aliases.keys()) - set(self.data["social"].columns)
        
        for col in tech_missing_cols:
            self.data["tech"][col] = dtypes["tech"]

        # check the columns have valid data (heuristics-driven, not 
        # comprehensive)

    def gen_proj_incubation(self):
        """
            Generates the lookup for project lengths. Uses the maximum recorded 
            month for each project.
        """

        # generate lookups
        t_proj_incubation = self.data["tech"].groupby("project_name", observed=True)["month"].max().to_dict()
        s_proj_incubation = self.data["social"].groupby("project_name", observed=True)["month"].max().to_dict()
        all_proj = set(t_proj_incubation.keys()) | set(s_proj_incubation.keys())

        # merge & save
        proj_incubation = {k: max(t_proj_incubation.get(k, 0), s_proj_incubation.get(k, 0)) for k in all_proj}
        proj_incubation = dict(sorted(proj_incubation.items()))
        proj_incubation = {k: int(v) + 1 for k, v in proj_incubation.items()}

        with open(params_dict["incubation-time"][self.incubator], "w") as f:
            json.dump(proj_incubation, f, indent=4)

    def check_missing_data(self, cols: list[str]=None) -> None:
        """
            Distributions of missing data.

            @param cols: list of column names to check, defaults to all columns
        """

        # utility
        def missing_data_util(dtype: str, cols: list[str]=None, THRESHOLD_MISSING: float=0.05):
            # check args
            df = self.data[dtype]
            if cols == None:
                cols = list(df.columns)

            # reading data
            num_entries = df.shape[0]

            print(f"\n<SUMMARY for {dtype}>")
            for col in cols:
                # count missing entries
                missing_count = (df[col] == "").sum() + df[col].isnull().sum()

                # summary
                if missing_count / num_entries > THRESHOLD_MISSING:
                    print("\t\t<WARNING> ", end="")
                print(f"number of missing entries in \'{col}\': {missing_count} / {num_entries}")
        
        # social & tech
        missing_data_util(dtype="tech", cols=cols)
        missing_data_util(dtype="social", cols=cols)

    def save_data(self) -> None:
        """
            Saves data to specified format.
        """

        # save
        _save_data(self.data, self.incubator, self.versions)


# Utility
def gen_df_lookup(augs: dict[str, list[int]], params: dict[str, Any]) -> dict[str, RawData]:
    """
        Loads all datasets in accordingly.
    """

    # load dfs
    temp_zip = ["tech", "social"]
    dfs = dict()

    for source_name, aug_nums in augs.items():
        dfs[source_name] = RawData(
            source_name,
            dict(zip(temp_zip, [str(x) for x in aug_nums]))
        )
    
    # return lookup
    return dfs


# Testing
if __name__ == "__main__":
    # rd = RawData("github", {"tech": 3, "social": 4})
    rd = RawData("eclipse", {"tech": 1, "social": 1})
    rd.data = infer_replies(rd.data, rd.incubator, copy=False, force_impute=True)
    df = rd.data["social"]
    
    df = df[["in_reply_to", "subject", "project_name", "date", "message_id", "sender_name"]]
    df = df[~df.in_reply_to.isna()]
    df.sort_values(by=["project_name", "subject", "date"], inplace=True)
    df.to_csv("./temp.csv", index=False)
    
    # rd = RawData("eclipse", {"tech": 0, "social": 0})
    # rd.gen_proj_incubation()
    # rd.data = dealias_senders(rd.data, rd.incubator, copy=False)
    # rd.save_data()
    # rd.data = impute_messageid(rd.data, copy=False)
    # rd.data = infer_replies(rd.data, copy=False)
    # print(rd.data["social"][["message_id", "in_reply_to"]])

    # if input("save: ") == "y":
    #     rd.save_data()

    # print("reached")
    # rd.data = impute_months(rd.data, copy=False)
    # print(rd.data["social"][["project_name", "month", "date", "months_since_first"]])
    # print(rd.data["tech"][["project_name", "month", "date", "months_since_first"]])

    # import timeit
    # import polars as pl
    # print('testing polars')

    # t1 = timeit.timeit(lambda: pl.read_csv(rd.paths["tech"], dtypes={"ref_or_sha": pl.String}), number=1)
    # print(t1)
    # t2 = timeit.time()
    # l = pd.read_csv(rd.paths["tech"], engine="python")
    # t2 = timeit.time() - t2
    # t3 = timeit.time()
    # l = pd.read_csv(rd.paths["tech"], engine="c")
    # t3 = timeit.time() - t3
    # t4 = timeit.time()
    # l = pd.read_csv(rd.paths["tech"], engine=PARQUET_ENGINE)
    # t4 = timeit.time() - t4

    # rd.data["social"][[
    #     "project_name",
	# 	"date",
	# 	"month",
	# 	"message_id",
	# 	"sender_name",
	# 	"sender_email",
	# 	"in_reply_to",
	# 	"receiver_email",
	# 	"subject",
	# 	"from_commit",
	# 	"author_full_name",
	# 	"is_bot",
	# 	"dealised_author_full_name"]].head(100).to_csv("../temp.csv", index=False)
