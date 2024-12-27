"""
@brief Defines the project data abstraction for pushing a single project's data
    through the pipeline. Note the caching mechanism will work inherently via 
    the pre-defined saving mechanism since overlapping months will be 
    overwritten and new months won't be considered. We simply add functionality
    to ignore previous months.
@author Arjun Ashok (arjun3.ashok@gmail.com)
@creation-date December 2024
@version 0.1.0
"""


# ------------- Environment Setup ------------- #
# external packages -- none for now
import pandas as pd
import numpy as np
from tqdm import tqdm
from pandarallel import pandarallel

# built-in modules
from copy import deepcopy
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field

# DECAL modules
import decalforecaster.utils as util
from decalforecaster.abstractions.rawdata import clean_file_paths, \
    clean_sender_names, impute_months, impute_messageid, infer_replies, \
    infer_bots, clean_source_files, dealias_senders
from decalforecaster.pipeline.monthly_data import segment_data

# constants & setup parallel processing
NUM_PROCESSES = 6
PARQUET_ENGINE = "pyarrow"
pandarallel.initialize(nb_workers=NUM_PROCESSES, progress_bar=True)
params_dict = util._load_params()
tqdm.pandas()
INCUBATOR_ALIAS = "ospos"

IMPLEMENTED_TASKS = {
    "net-gen": None,
    "traj": None,
    "forecast": None,
    "pp-paths": clean_file_paths,
    "pp-names:": clean_sender_names,
    "pp-months": impute_months,
    "pp-msg-id": impute_messageid,
    "pp-is-coding": clean_source_files,
    "pp-replies": infer_replies,
    "pp-bots": infer_bots,
    "pp-de-alias": dealias_senders
}


# ------------- Helper Fn ------------- #
def _route_preprocesses(data: dict[str, pd.DataFrame], tasks: list[str], incubator: str="ospos", inplace: bool=True) -> None | dict[str, pd.DataFrame]:
    """Wraps all pre-processing steps requested into one neat package.

    Args:
        data (dict[str, pd.DataFrame]): data lookup
        tasks (list[str]): tasks, matching the specified keys, to execute.
        incubator (str, optional): incubator name to fall under. Defaults to 
            "OSPOS".
        inplace (bool, optional): whether to perform the action inplace or to 
            copy. Defaults to True.

    Returns:
        None | dict[str, pd.DataFrame]: returns data lookup if not inplace.
    """
    
    # copy if needed
    if not inplace:
        data = deepcopy(data)

    # iterate through the steps and route if needed
    for task in tasks:
        # skip downstream tasks
        if not task.startswith("pp"):
            continue
        
        # route
        IMPLEMENTED_TASKS[tasks](data, incubator=incubator, copy=False)
        
    # return if needed
    return data if not inplace else None


# ------------- Abstraction ------------- #
@dataclass(slots=False)
class ProjData:
    # data
    ## user specified
    proj_name: str = field(init=True, repr=True)                                # identifier for the project
    tdata: pd.DataFrame = field(init=True, repr=False)                          # tech df
    sdata: pd.DataFrame = field(init=True, repr=False)                          # social df
    tasks: list[str] = field(init=True, repr=True)                              # tasks to complete on the data
    
    ## inferred
    incubations: dict[str, int] = field(init=False, repr=False)                 # incubation time lookup
    gen_full: bool = field(default=False)                                       # if base data, generate processed version
    data: dict[str, pd.DataFrame] = field(init=False, repr=False)               # type : df
    netdata: pd.DataFrame = field(init=False, repr=False)                       # network data produced

    # post-initialization
    def __post_init__(self):
        # intialize the data
        self.data = {
            "tech": self.tdata,
            "social": self.sdata
        }
        
        # check if tasks are valid
        if self.tasks[0] == "ALL":
            self.tasks = list(IMPLEMENTED_TASKS.keys())
        
        # generate auxiliary information
        self.gen_proj_incubation()
        self.check_missing_data()

        # pre-processing
        _route_preprocesses(self.data, self.tasks)
            

    # internal utility
    def gen_proj_incubation(self):
        """
            Generates the lookup for project lengths. Uses the maximum recorded 
            month for each project.
        """

        # generate lookups
        t_proj_incubation = self.data["tech"].groupby("project_name")["month"].max().to_dict()
        s_proj_incubation = self.data["social"].groupby("project_name")["month"].max().to_dict()
        all_proj = set(t_proj_incubation.keys()) | set(s_proj_incubation.keys())

        # merge & save
        proj_incubation = {k: max(t_proj_incubation.get(k, 0), s_proj_incubation.get(k, 0)) for k in all_proj}
        proj_incubation = dict(sorted(proj_incubation.items()))
        proj_incubation = {k: int(v) for k, v in proj_incubation.items()}

        # save in memory itself
        self.incubations = proj_incubation


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
        pass
    
    
    # split by month
    def monthwise_split(self) -> None:
        """Wraps the first pipeline stage for splitting the data by month.
        """
        
        # setup
        util._log("Segmenting Monthly Data", "log")
        author_field = "dealised_author_full_name"
        time_strat = "default"
        ratios = dict()

        # segmentation; overwrite the previous data for the built-in caching 
        # (effectively)
        util._log("segmenting...")
        self.data["tech"] = segment_data(
            self.tdata, time_strat, author_field=author_field,
            ratios=ratios
        )
        self.data["social"] = segment_data(
            self.sdata, time_strat, author_field=author_field,
            ratios=ratios
        )
        
    
    # network generation
    def network_gen(self) -> None:
        """Generates the networks and caches in memory for quicker usage and no
        need for clearing cache later.
        """
        
        
        


# Testing
if __name__ == "__main__":
    # rd = RawData("github", {"tech": 3, "social": 4})
    jd = ProjData(
        proj_name="tester",
        tdata=pd.DataFrame(),
        sdata=pd.DataFrame(),
        tasks=[""]
    )
