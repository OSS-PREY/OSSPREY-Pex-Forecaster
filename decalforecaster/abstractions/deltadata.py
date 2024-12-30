"""
@brief Defines the delta data abstraction for pushing a single project's UPDATED
    data through the pipeline. Note the caching mechanism will work inherently 
    via the pre-defined saving mechanism since overlapping months will be 
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
from decalforecaster.utils import PARQUET_ENGINE, CSV_ENGINE
from decalforecaster.abstractions.rawdata import clean_file_paths, \
    clean_sender_names, impute_months, impute_messageid, infer_replies, \
    infer_bots, clean_source_files, dealias_senders
from decalforecaster.pipeline.monthly_data import segment_data
from decalforecaster.pipeline.create_networks import process_social_nets, \
    process_tech_nets

# constants & setup parallel processing
NUM_PROCESSES = 6
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

def _load_cached_data(proj_name: str) -> pd.DataFrame:
    """Helper function to load the cached data if possible.

    Args:
        proj_name (str): the name of the project's data to search for.

    Returns:
        pd.DataFrame: returns the cached network data if possible, otherwise 
            empty.
    """
    
    # generate the path
    path = Path(params_dict["delta-cache-dir"]) / f"{proj_name}.csv"
    
    # check the file
    if not path.exists():
        return pd.DataFrame()
    
    # load the data
    return pd.read_csv(path, engine=CSV_ENGINE)


# ------------- Abstraction ------------- #
@dataclass(slots=False)
class DeltaData:
    # data
    ## user specified
    proj_name: str = field(init=True, repr=True)                                # identifier for the project
    tdata: pd.DataFrame = field(init=True, repr=False)                          # tech df
    sdata: pd.DataFrame = field(init=True, repr=False)                          # social df
    tasks: list[str] = field(init=True, repr=True)                              # tasks to complete on the data
    
    ## either specified or inferred
    last_cached_month: int = field(init=True, default=-1)                       # last month previously calculated; MAY NOT BE REQUIRED
    
    ## inferred or loaded upon initialization
    incubations: dict[str, int] = field(init=False, repr=False)                 # incubation time lookup; MAY NOT BE REQUIRED
    data: dict[str, pd.DataFrame] = field(init=False, repr=False)               # type : df
    cached_netdata: pd.DataFrame = field(init=False, repr=False)                # cached network data from previous months
    
    ## potentially calculatec
    netdata: pd.DataFrame = field(init=False, repr=False)                       # network data produced (whole project history)
    forecasts: dict[int, float] = field(init=False, repr=True)                  # sustainability forecasts (for new months only)
    trajectories: dict[int, dict[str, list[float]]] = field(init=False)         # trajectories for the latest months as {month: {forecast_type: forecasts as a list}}

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
        
        # load cached data if possible
        ## IO
        self.cached_netdata = _load_cached_data(self.proj_name)
        
        ## set the correct cached month; maintain the correct limit
        if self.last_cached_month == -1:
            self.last_cached_month = self.cached_netdata.shape[0]
        else:
            self.cached_netdata = self.cached_netdata[self.cached_netdata["month"] < self.last_cached_month]
        
        util._log(f"using the months [0, {self.last_cached_month}) from the cache")
        
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
        
        # helper fn
        def segment_data(df: pd.DataFrame, author_field: str, save_dir: Path, start_month: int=0) -> None:
            """Segments data by month without considering relative time.
            Args:
                df (pd.DataFrame): data to segment by month.
                author_field (str): field to treat as the author field.
                save_dir (Path): path to save to.
                start_month (int): month to start number at. Defaults to 0.
                
            Returns:
                None
            """
            
            # make the directory if not already
            save_dir.mkdirs(exist="ok")
            
            # generate lookup for each project
            df = dict(tuple(df.groupby("project_name")))
            
            # save each project's month's data
            for project in tqdm(df):
                # generate the monthly dictionary
                monthly_df_dict = dict(tuple(df[project].groupby("month")))
                
                # save in memory
                for month in monthly_df_dict:
                    # grab current month and remove missing rows
                    monthly_df = monthly_df_dict[month]
                    monthly_df = monthly_df[monthly_df[author_field].notna()]
                    
                    # skip empty data
                    if monthly_df.empty: continue
                    
                    # save
                    file_path = save_dir / f"{project}__{str(int(month))}.parquet"
                    monthly_df.to_parquet(file_path, engine=PARQUET_ENGINE, index=False)
                    
            # end fn
            return

        # routing for the IO
        dataset_dir = Path(params_dict["dataset-dir"])
        monthly_data_dir = dataset_dir / f"{INCUBATOR_ALIAS}_data" / "monthly_data/"
        t_output_dir = monthly_data_dir / f"{params_dict['tech-type'][INCUBATOR_ALIAS]}/"
        s_output_dir = monthly_data_dir / f"{params_dict['social-type'][INCUBATOR_ALIAS]}/"

        # segmentation; overwrite the previous data for the built-in caching 
        # (effectively)
        util._log("segmenting...")
        segment_data(
            self.tdata, author_field=author_field, save_dir=t_output_dir, 
            start_month=self.last_cached_month
        )
        segment_data(
            self.sdata, author_field=author_field, save_dir=s_output_dir,
            start_month=self.last_cached_month
        )


    # network generation
    def gen_networks(self) -> None:
        """Generates the networks and caches in memory for quicker usage and no
        need for clearing cache later. Ensures to combine the new data and save
        to ensure updated cache for future work.
        """
        
        pass
        
        # create the edgelists
        
        # extract features
        
        # clear space for limiting memory usage
        
        # combine network data (vertical stacking, essentially)
        
        # cache the new combined data
        

        # end fn
        return

    
    # predictions & trajectories
    def gen_forecasts(self) -> dict[int, float]:
        """Generates the forecasts for all new months of data for export back.
        Caches the result within the object for easy tasks in the future.

        Returns:
            dict[int, float]: month number to sustainability forecast for that 
                month.
        """
        
        pass
    
    def gen_trajectories(self) -> dict[int, dict[str, list[float]]]:
        pass
    
    
    


# Testing
if __name__ == "__main__":
    dd = DeltaData(
        proj_name="tester",
        tdata=pd.DataFrame(),
        sdata=pd.DataFrame(),
        tasks=["ALL"]
    )
