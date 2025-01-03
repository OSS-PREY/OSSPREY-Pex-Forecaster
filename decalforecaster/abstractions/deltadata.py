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
from os import cpu_count
from copy import deepcopy
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field
from json import dump

# DECAL modules
import decalforecaster.utils as util
from decalforecaster.utils import PARQUET_ENGINE, CSV_ENGINE
from decalforecaster.abstractions.rawdata import clean_file_paths, \
    clean_sender_names, impute_months, impute_messageid, infer_replies, \
    infer_bots, clean_source_files, dealias_senders
from decalforecaster.abstractions.tsmodel import *
from decalforecaster.pipeline.create_networks import create_networks
from decalforecaster.pipeline.network_features import extract_features
from decalforecaster.pipeline.network_visualizations import net_vis_info

# constants & setup parallel processing
NUM_PROCESSES = cpu_count()
pandarallel.initialize(nb_workers=NUM_PROCESSES, progress_bar=True)
params_dict = util._load_params()
tqdm.pandas()
INCUBATOR_ALIAS = "ospos"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
def _route_preprocesses(data: dict[str, pd.DataFrame], tasks: list[str],
    incubator: str=INCUBATOR_ALIAS, inplace: bool=True
) -> None | dict[str, pd.DataFrame]:
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
        IMPLEMENTED_TASKS[task](data, incubator=incubator, copy=False)
        
    # return if needed
    return data if not inplace else None

def _gen_cache_path(proj_name: str, **kwargs) -> Path:
    """Generates a path to load/save the cached netdata.

    Args:
        proj_name (str): project name.

    Returns:
        Path: path object pointing to the dedicated path
    """
    
    # return the formulation
    path = Path(params_dict["delta-cache-dir"]) / f"{proj_name}.csv"
    return path

def _load_cached_data(proj_name: str) -> pd.DataFrame:
    """Helper function to load the cached data if possible.

    Args:
        proj_name (str): the name of the project's data to search for.

    Returns:
        pd.DataFrame: returns the cached network data if possible, otherwise 
            empty.
    """
    
    # generate the path
    path = _gen_cache_path(proj_name)
    
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
    incubator: str = field(init=True, default=INCUBATOR_ALIAS)                  # incubator to organize data by; may switch to project name
    last_cached_month: int = field(init=True, default=-1)                       # last month previously calculated; MAY NOT BE REQUIRED
    delta_args: dict[str, Any] = field(init=True, default=None)                 # args to use in the pipeline calls
    
    ## inferred or loaded upon initialization
    incubations: dict[str, int] = field(init=False, repr=False)                 # incubation time lookup; MAY NOT BE REQUIRED
    data: dict[str, pd.DataFrame] = field(init=False, repr=False)               # type : df
    cached_netdata: pd.DataFrame = field(init=False, repr=False)                # cached network data from previous months
    
    ## potentially calculated
    netdata: pd.DataFrame = field(init=False, repr=False)                       # network data produced (whole project history)
    forecasts: dict[int, float] = field(init=False, repr=True)                  # sustainability forecasts (for new months only)
    trajectories: dict[int, dict[str, list[float]]] = field(init=False)         # trajectories for the latest months as {month: {forecast_type: forecasts as a list}}
    net_vis: dict[str, list[list[str | int]]] = field(init=False, repr=False)   # network visualization info

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
        
        # check if args are valid
        if self.delta_args is None:
            self.delta_args = {
                "incubator": self.incubator
            }
        
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
        proj_incubation = {k: int(v) - self.last_cached_month + 1 for k, v in proj_incubation.items()}

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
        path = _gen_cache_path(self.proj_name)
        util._check_path(path)
        self.netdata.to_csv(path, index=False)

    def clean_disk(self) -> None:
        """To be run when the destructor is called. Clears any used storage on 
        disk to minimize concurrent disk usage.
        """
        
        # MONTHWISE SPLIT
        # clear disk usage to limit space requirement and prevent double 
        # writing (using the same data twice) or mis-association (using one 
        # project's data in another's)
        dataset_dir = Path(params_dict["dataset-dir"])
        monthly_data_dir = dataset_dir / f"{self.incubator}_data" / "monthly_data/"
        t_output_dir = monthly_data_dir / f"{params_dict['tech-type'][INCUBATOR_ALIAS]}/"
        s_output_dir = monthly_data_dir / f"{params_dict['social-type'][INCUBATOR_ALIAS]}/"
        
        util._clear_dir(dir=t_output_dir, skip_input=True)
        util._clear_dir(dir=s_output_dir, skip_input=True)
        
        # NETWORK GENERATION
        data_dir = Path(params_dict["dataset-dir"]) / f"{self.incubator}_data"
        network_dir = Path(params_dict["network-dir"])
        t_type = params_dict["tech-type"][INCUBATOR_ALIAS]
        s_type = params_dict["social-type"][INCUBATOR_ALIAS]
        net_path = network_dir / "netdata" / f"{self.incubator}-network-data.csv"
        
        util._clear_dir(data_dir, skip_input=True)
        util._clear_dir(network_dir / f"{self.incubator}_{t_type}", skip_input=True)
        util._clear_dir(network_dir / f"{self.incubator}_{s_type}", skip_input=True)
        
        util._check_dir(data_dir)
        util._del_file(net_path)


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
            save_dir.mkdir(parents=True, exist_ok=True)
            
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
                    
                    # offset each month to treat as a new project--helps avoid
                    # some padding that happens later on in the pipeline; save
                    # to file
                    file_path = save_dir / f"{project}__{str(int(month) - start_month)}.parquet"
                    monthly_df.to_parquet(file_path, engine=PARQUET_ENGINE, index=False)
                    
            # end fn
            return

        # routing for the IO
        dataset_dir = Path(params_dict["dataset-dir"])
        monthly_data_dir = dataset_dir / f"{self.incubator}_data" / "monthly_data/"
        t_output_dir = monthly_data_dir / f"{params_dict['tech-type'][INCUBATOR_ALIAS]}/"
        s_output_dir = monthly_data_dir / f"{params_dict['social-type'][INCUBATOR_ALIAS]}/"
        
        # clear disk usage to limit space requirement and prevent double 
        # writing (using the same data twice) or mis-association (using one 
        # project's data in another's)
        util._clear_dir(dir=t_output_dir, skip_input=True)
        util._clear_dir(dir=s_output_dir, skip_input=True)

        # segmentation; no longer need to overwrite, we simply treat this as new
        # data and we'll concatenate the old data with this
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
        
        # create the edgelists, extract features from the networks on disk
        create_networks(self.delta_args)
        extract_features(self.delta_args, self.incubations)
        
        # compile data into memory
        network_dir = Path(params_dict["network-dir"])
        net_path = network_dir / "netdata" / f"{self.incubator}-network-data.csv"
        self.netdata = pd.read_csv(net_path, engine=CSV_ENGINE)
        
        # combine network data (vertical stacking, essentially); ensure to 
        # remove the extra rows created and we make-up for the month offset
        self.netdata["month"] += self.last_cached_month
        self.netdata = pd.concat(
            [self.cached_netdata, self.netdata], axis="rows", ignore_index=True
        )
        
        # ensure column order
        self.netdata = self.netdata[[
            "s_num_nodes", "s_weighted_mean_degree", "s_num_component",
            "s_avg_clustering_coef", "s_largest_component", "s_graph_density",
            "t_num_dev_nodes", "t_num_file_nodes", "t_num_dev_per_file",
            "t_num_file_per_dev", "t_graph_density", "proj_name", "month",
            "st_num_dev", "t_net_overlap", "s_net_overlap"
        ]]
        
        # export to cache
        self.save_data()

        # end fn
        return
    
    def vis_networks(self) -> None:
        """Generates the network visualization for the PEX tool to display 
        accordian graphs, etc.
        """
        
        # check the networks match this project; if not, clear and generate 
        # networks
        
        ## grab directories
        network_dir = Path(params_dict["network-dir"])
        tech_type = params_dict["tech-type"][INCUBATOR_ALIAS]
        social_type = params_dict["social-type"][INCUBATOR_ALIAS]
        
        t_dir = network_dir / f"{self.incubator}_{tech_type}"
        s_dir = network_dir / f"{self.incubator}_{social_type}"
        
        ## check all files match
        if not all(f.name.startswith(self.proj_name) for f in t_dir.glob("**/*")):
            wrong_files = [f.name for f in t_dir.glob("**/*") if not f.name.startswith(self.proj_name)]
            raise ValueError(f"technical directory (\"{t_dir}\") contains non-matching project file (needs only {self.proj_name} files, got {wrong_files})")
        if not all(f.name.startswith(self.proj_name) for f in s_dir.glob("**/*")):
            wrong_files = [f.name for f in s_dir.glob("**/*") if not f.name.startswith(self.proj_name)]
            raise ValueError(f"social directory (\"{s_dir}\") contains non-matching project file (needs only {self.proj_name} files, got {wrong_files})")
        
        # generate visualizations
        self.net_vis = net_vis_info(self.delta_args)
        
        # update old visualizations if possible and re-store
        vis_path = Path(params_dict["network-dir"]) / "net_vis" / f"{self.incubator}.json"
        
        if vis_path.exists():
            with open(vis_path, "r") as f:
                old_vis = json.load(self.net_vis, f, indent=0)
        else:
            old_vis = {
                "tech": dict(),
                "social": dict()
            }

        print(self.net_vis, old_vis)
        self.net_vis["tech"].update(old_vis["tech"])
        self.net_vis["social"].update(old_vis["social"])
        self.net_vis["tech"] = dict(sorted(self.net_vis["tech"]))
        self.net_vis["social"] = dict(sorted(self.net_vis["social"]))
        
        with open(vis_path, "w") as f:
            json.dump(self.net_vis, f, indent=0)


    # predictions & trajectories
    def gen_forecasts(self, model_arch: str="BLSTM", **kwargs) -> dict[int, float]:
        """Generates the forecasts for all new months of data for export back.
        Caches the result within the object for easy tasks in the future.

        Args:
            model_arch (str, optional): model architecture to use {BLSTM, BGRU}.
                Defaults to "BLSTM".

        Returns:
            dict[int, float]: month number to sustainability forecast for that 
                month.
        """
        
        # aux functions
        drop_cols = [
            "proj_name", 
            "month",
            # "s_largest_component"   # hardcode the `c` strat for netdata
        ]
        
        def apply_augs(augs: str="cbn") -> None:
            """Temporarily hard-coded for the model-weights we use.

            Args:
                augs (str, optional): augmentations to use in the standard
                    grammar form. Defaults to "cbn".
            """
            
            # aux fn
            keep_nan = 1e-10
            
            def zscore_normalize(df):
                non_nrm_data = df[drop_cols]
                nrm_data = df.drop(columns=drop_cols)

                # normalize
                nrm_data = (nrm_data - nrm_data.mean()) / (nrm_data.std() + keep_nan)

                # merge & return
                return pd.concat([non_nrm_data, nrm_data], axis=1)
            
            def minmax_normalize(df):
                # split
                non_nrm_data = df[drop_cols]
                nrm_data = df.drop(columns=drop_cols)

                # normalize
                nrm_data = (nrm_data - nrm_data.min()) / (nrm_data.max() + keep_nan)

                # merge & return
                return pd.concat([non_nrm_data, nrm_data], axis=1)
            
            def actdev_normalize(df):
                """
                    Normalizes by the number of active developers per month.
                """

                # split
                non_nrm_data = df[drop_cols]
                nrm_data = df.drop(columns=drop_cols)

                # normalize; sum of both networks in conjunction per month
                # max_devs = (group["t_num_dev_nodes"] + group["s_num_nodes"])
                max_devs = df["st_num_dev"]
                max_devs = max_devs + (max_devs == 0)           # force to one
                nrm_data = nrm_data.div(max_devs, axis=0)

                # merge & return
                return pd.concat([non_nrm_data, nrm_data], axis=1)
            
            # router
            aug_router = {
                "j": None,
                "n": actdev_normalize,
                "m": minmax_normalize,
                "z": zscore_normalize,
                "a": None,
                "d": None,
                "u": None,
                "b": lambda d: d,
                "c": lambda d: d.drop(columns="s_largest_component")
            }
            
            # apply augs inplace
            for aug in augs:
                self.netdata = aug_router[aug](self.netdata)
        
        def gen_tensors() -> torch.Tensor:
            # ensure columns order
            self.netdata = self.netdata[[
                "s_num_nodes", "s_weighted_mean_degree", "s_num_component",
                "s_avg_clustering_coef", "s_graph_density", "t_num_dev_nodes",
                "t_num_file_nodes", "t_num_dev_per_file", "t_num_file_per_dev",
                "t_graph_density", "proj_name", "month", "st_num_dev",
                "t_net_overlap", "s_net_overlap"
            ]]
            
            # track in a dictionary format
            data_dict = {}

            # convert to list representation
            data_dict = self.netdata.drop(drop_cols, axis=1).values.tolist()

            # convert to tensor form
            return torch.tensor(data_dict)
        
        # apply augmentations and convert to tensor for use in the model
        num_months = self.netdata.shape[0]
        apply_augs()
        X = gen_tensors()
        
        # load in model
        model = None
        path = Path(params_dict["weights-dir"])
        hyperparams = {
            "input_size": X.shape[1],
            "hidden_size": 64,
            "num_classes": 2,
            "dropout_rate": 0.4,
            "learning_rate": 0.0001,
            "batch_size": 512,
            "num_epochs": 10,
            "num_layers": 1,
        }
                
        match model_arch:
            case "BLSTM":
                model = BRNN(**hyperparams).to(DEVICE)
                path = path / "BLSTM.pt"
            case "BGRU":
                model = BGNN(**hyperparams).to(DEVICE)
                path = path / "BGRU.pt"
            case _:
                raise ValueError(f"Failed to associate model to provided architecture \"{model_arch}\". Expected one of [BLSTM, BGRU]")
        
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval() 
        
        # generate forecasts
        fcs = dict()
        for i in range(1, num_months):
            # grab the first i months
            data = X[:i, ...]
            
            # transform data to use
            data = data.to(DEVICE)
            data = data.reshape(1, data.shape[0], -1)
            
            # generate raw prob forecast
            preds = model.predict(data)[:, 1].to(DEVICE)

            # concatenate lists
            fcs[i - 1] = float(preds.cpu().detach().numpy()[0])

        # save & export
        forecast_dir = Path(params_dict["forecast-dir"])
        util._check_dir(forecast_dir)
        
        with open(forecast_dir / f"{self.proj_name}.json", "w") as f:
            dump(fcs, f, indent=4)
            
        self.forecasts = fcs
    
    def gen_trajectories(self) -> dict[int, dict[str, list[float]]]:
        pass


# Testing
if __name__ == "__main__":
    data_dir = Path().cwd() / "data" / "ospos_data"
    
    # first batch
    dd = DeltaData(
        proj_name="spark",
        tdata=pd.read_parquet(data_dir / "commits.parquet"),
        sdata=pd.read_parquet(data_dir / "issues.parquet"),
        tasks=["ALL"]
    )
    dd.monthwise_split()
    dd.gen_networks()
    # dd.vis_networks()
    dd.gen_forecasts()
    
    # second batch
    dd = DeltaData(
        proj_name="spark",
        tdata=pd.read_parquet(data_dir / "test_commits.parquet"),
        sdata=pd.read_parquet(data_dir / "test_issues.parquet"),
        tasks=["ALL"]
    )
    dd.monthwise_split()
    dd.gen_networks()
    # dd.vis_networks()
    dd.gen_forecasts()
    
    (Path().cwd() / "network-data" / "caches" / "spark.csv").unlink()


