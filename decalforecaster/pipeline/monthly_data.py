"""
    @brief Segments social & technical project data by months 
    @author Dr. Likang Yin (lykin@ucdavis.edu), Arjun Ashok 
            (arjun3.ashok@gmail.com)
    @creation-date (unknown, modified by Arjun @ later date)
"""


# ---------------- Environment Setup ---------------- #
# external packages
import pandas as pd
from tqdm import tqdm

# built-in modules
import os
import sys
import json
from math import floor

# DECAL modules
import decalforecaster.utils as util
from decalforecaster.utils import PARQUET_ENGINE
from decalforecaster.abstractions.rawdata import *


# ------------------- relative time generation ---------------------- #
def relative_time(cur_month: int, **kwargs) -> int:
    """
        Dispatches the selected relative time strategy.
    """

    #	Defines a lookup for functions that take in a given month & the total 
    #	number of months in a project and outputs the relative month.
    REL_TIME_APPR = {
        "default": default_time,
        "static": static_time,
        "dynamic": dynamic_time,
        "cluster": cluster_time
    }
    return REL_TIME_APPR[kwargs["strat"]](cur_month, **kwargs)


def static_time(month: int, **kwargs) -> int:
    """
        Given a static ratio, this function computes the relative month using a 
        simple proportion multiplication.

        NOTE this proportion should be found empirically through expected values, 
            although the actual number will likely vary
            
        Requires project name, reltime dictionary of ratios with project and apache.
    """

    # unpack args
    try:
        incubator = kwargs["incubator_name"]
        reltime_pack = kwargs["reltime_pack"]
    except:
        print("ERROR: missing args for static time strategy")
        exit(1)
    
    # calculations
    proportion = float(reltime_pack["apache"]) / float(reltime_pack[incubator])		# unpack proportion
    return floor(proportion * month)


def default_time(month: int, **kwargs) -> int:
    """
        Default strategy, 1 month is 1 month regardless of dataset.
    """

    # # wrap static time with 1 : 1 proportion
    # return static_time(month, proj_len, rel_time_pack=["0", "1", "1"])
    return month


def dynamic_time(month: int, **kwargs) -> int:
    """
        Dynamic strategies will be functions of time that vary across the time that 
        they are controlled in. Can be based on the proportion of months done in a 
        project.

        The idea here is that Github projects are usually very active in the first 
        few months, but activity drastically spreads out after the first year or 
        so.
    """

    # not implemented for now, still yet to be deemed useful
    return month


def cluster_time():
    """

    """

    #
    pass


# ------------------- segmentation utility ---------------------- #
def segment_data(df: pd.DataFrame, incubator: str, time_strat: str, output_dir: Path, author_field: str, ratios: dict) -> None:
    """
        Segments data by month.
    """

    # setup
    print(f"processing relative time with the [{time_strat}] strat...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # generate lookup
    df = dict(tuple(df.groupby("project_name")))
    for proj, proj_df in tqdm(df.items()):
        # apply relative time function
        num_months = proj_df["month"].max()
        proj_df["month"] = proj_df["month"].apply(
            relative_time, 
            strat=time_strat, 
            incubator_name=incubator, 
            reltime_pack=ratios
        )

    # save monthly df
    for project in tqdm(df):
        monthly_df_dict = dict(tuple(df[project].groupby("month")))
        for month in monthly_df_dict:
            monthly_df = monthly_df_dict[month]
            monthly_df = monthly_df[monthly_df[author_field].notna()]
            if monthly_df.empty: continue

            file_path = output_dir / "{}__{}.parquet".format(project, str(int(month)))
            monthly_df.to_parquet(file_path, engine=PARQUET_ENGINE, index=False)


# ---------------- script ---------------- #
def monthly_segmentation(args_dict):
    # setup
    print("\n<Segmenting Monthly Data>")
    params_dict = util._load_params()
    dataset_dir = Path(params_dict["dataset-dir"])

    # execute input
    util._log("reading in raw data...")
    rd = RawData(incubator=args_dict["incubator"], versions=args_dict["versions"])
    with open(params_dict["lifecycle-ratios"], "r") as f:
        lifecycle_dict = json.load(f)
    author_field = "dealised_author_full_name"
    
    # check clean trials
    util._log("clearing out previous trials...")
    monthly_data_dir = dataset_dir / f"{rd.incubator}_data" / "monthly_data/"
    util._clear_dir(monthly_data_dir)

    # relative time calculation
    util._log("figuring out relative time strategy...")
    if args_dict.get("reltime", "default") != "default":
        relative_time_info = args_dict["reltime"].split("-", 1)
        time_strat = relative_time_info[0]
        lifecycle_ratio = relative_time_info[1]
        ratios = {k: v["ratio"] for k, v in lifecycle_dict[lifecycle_ratio].items()}
    else:
        time_strat = "default"
        ratios = dict()

    t_output_dir = monthly_data_dir / f"{params_dict['tech-type'][rd.incubator]}/"
    s_output_dir = monthly_data_dir / f"{params_dict['social-type'][rd.incubator]}/"

    # segmentation
    util._log("segmenting...")
    segment_data(rd.data["tech"], rd.incubator, time_strat, t_output_dir, author_field=author_field, ratios=ratios)
    segment_data(rd.data["social"], rd.incubator, time_strat, s_output_dir, author_field=author_field, ratios=ratios)


if __name__ == "__main__":
    # user args
    args_dict = util._parse_input(sys.argv)
    monthly_segmentation(args_dict)

