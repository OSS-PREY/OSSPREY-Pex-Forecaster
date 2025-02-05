"""
@brief API endpoints are routed through here, handling interface between 
    frontend and backend of the PEX tool.
@author Arjun Ashok (arjun3.ashok@gmail.com)
"""

# ------------- Environment Setup ------------- #
# external packages -- none for now
import pandas as pd

# built-in modules
import sys
import json
from pathlib import Path
from typing import Any

# DECAL modules
import decalforecaster.utils as util
from decalforecaster.utils import PARQUET_ENGINE, CSV_ENGINE
from decalforecaster.pipeline import *
from decalforecaster.abstractions.deltadata import *
# from decalforecaster.abstractions.projdata import *

# setup the App for communication
# no app, making local API


# ------------- API Endpoints ------------- #
# local vars for storage
data_pkg = {
    "proj_name": None,
    "tdata": None,
    "sdata": None,
    "month_range": [None, None]
}
tasks_pkg = list()

# receiver route
def compute_forecast(data: dict[str, str | pd.DataFrame | list[str] | list[int]]) -> dict[str, str | int]:
    """Wrapper API call for verifying the integrity of the input data and then 
    executing the necessary computation.
    
    The information expected in the request is as follows:
    {
        (project_name): str
        (tech_data): pd.DataFrame or dict, should be easily convertable into a 
            pandas dataframe, i.e. {column: data}.
        (social_data): pd.DataFrame or dict, should be easily convertable into a
            pandas dataframe, i.e. {column: data}.
        (tasks): list[str], match the tasks implemented.
        (month_range): list[int], length of two to define the inclusive month 
            range requested in the call.
    }
    
    Implemented Tasks (key to be matched with):
        - *(pp-paths): Pre-processing via File Path Cleansing
        - *(pp-names): Pre-processing via Sender Name Cleansing
        - *(pp-months): Pre-processing via Month Imputation
        - *(pp-msg-id): Pre-processing via Message ID Imputation
        - *(pp-is-coding): Pre-processing via Source File Identification
        - *(pp-replies): Pre-processing via Reply Inferencing
        - *(pp-bots): Pre-processing via Bot Inference
        - *(pp-de-alias): Pre-processing via De-Aliasing
        - (net-gen) Network Generation
        - (net-vis) Network Visualization
        - (forecast) Forecast predictions
        - (traj) Trajectories
        - (ALL): Shorthand to compute all pre-processing steps possible and then
            compute all downstream tasks.
        
        * All pre-processing tasks are done by default on any received data
    
    Verification Messages:
        (200) Data is successfully received
        (400) Failed to provide data with the request; invalid format for data
        (500) Unidentified error
    """
    
    # error handling if missing
    if not data:
        raise ValueError("No data provided")
    
    # error handling if incorrectly structured
    check_request_structure(data)
        
    # error handling if incorrectly formatted; attempt parsing
    attempt_request_parse(data)
    
    # dispatch tasks to complete
    router(tasks_pkg, data_pkg)
    
    # send back okay message and aynchronously compute?
    return {"message": "Data received successfully", "code": 200}


'''
## NOT SURE IF THESE WILL BE MORE USEFUL
# exporter route: forecasts
@app.route("/forecaster/predictions", methods=["GET"])
def export_forecasts():
    """Route for exporting the predictions for a given project as JSON object.
    
    Verification Messages:
    """
    pass

# exporter route: trajectories
@app.route("/forecaster/trajectories", methods=["GET"])
def export_trajectories():
    """Route for exporting the predictions for a given project as JSON object.
    
    Verification Messages:
    """
    pass

# exporter route: networks
@app.route("/forecaster/networks", methods=["GET"])
def export_networks():
    """Route for grabbing the raw data as a JSON dictionary. Specified format 
    should be easily convertable into a pandas dataframe, i.e. {column: data}.
    
    Verification Messages:
    """
    
    # TODO
    pass
'''

# ------------- Endpoint Helpers ------------- #
def attempt_request_parse(data: dict[str, Any]) -> None:
    """Wraps the parsing functionality for the data package with verified 
    entries. Checks the format of the data, essentially. Doesn't return an 
    error, only raises.

    Args:
        data (dict[str, Any]): data received
    """
    
    # attempt parsing
    try:
        # parse dataframes
        global data_pkg
        data_pkg["tdata"] = pd.DataFrame(data["tech_data"])
        data_pkg["sdata"] = pd.DataFrame(data["social_data"])
        
        # parse project name and month range
        data_pkg["proj_name"] = data["project_name"]
        data_pkg["month_range"] = data["month_range"]
        
        # parse tasks to complete
        global tasks_pkg
        tasks_pkg = data["tasks"]
        
    except ValueError as ve:
        raise ValueError(
            f"Data not parse-able by pandas: {str(ve)}"
        )

def check_request_structure(data: dict[str, Any]) -> None:
    """Wraps the parsing functionality for the data package with a valid 
    structure, i.e. keys expected.

    Args:
        data (dict[str, Any]): data received

    Returns:
        tuple[dict, int] | None: returns an error package to route back unless
            no error exists.
    """
    
    # setup the input validation
    needed_keys = ["project_name", "tech_data", "social_data", "tasks", "month_range"]
    val_types = [str, dict, dict, list, list]
    implemented_tasks = set(IMPLEMENTED_TASKS.keys())
    
    # check type of data
    if not isinstance(data, dict):
        raise ValueError("Data not formatted as a dictionary")
    
    # check all info is included
    if not all(key in data for key in needed_keys):
        raise ValueError(f"Missing some expected information; expected {needed_keys}, got {list(data.keys())}")
    
    # check all value types are valid
    if not all(isinstance(val, val_type) for val, val_type in zip(data.values(), val_types)):
        actual_types = [str(type(v)) for v in data.values()]
        raise ValueError(
            f"Values of are an unexpected type; expected {val_types}, got {actual_types}"
        )
    
    # check all tasks are implemented
    if not all(task in implemented_tasks for task in data["tasks"]):
        raise ValueError(
            f"Received un-implemented tasks; expected a subset of {implemented_tasks}, got {data['tasks']}"
        )
    
    # no error
    return None


# ------------- Internal Router ------------- #
def router(tasks: list[str], data: dict[str, Any]) -> list[str]:
    """Routes the calls we need to make plus intercepts any pre-computed calls 
    to avoid re-computing any information.

    Args:
        tasks (list[str]): tasks to complete.
        data (dict[str, Any]): data package from the request.

    Returns:
        list[str]: list of tasks completed. Will contain any subset of
            {"CACHED", "NETS", "FORECASTS", "TRAJECTORIES"}.
    """
    
    # auxiliary fn & data
    def check_cache(tasks: list[str], proj_name: str, end_month: int, start_month: int=0) -> dict[str, Any] | None:
        """Checks if the info requested has already been cached. If so, returns
        the requested data as if it has been computed fresh.

        Args:
            tasks (list[str]): tasks to complete.
            proj_name (str): project name/identifier.
            end_month (int): compute for [start_month, end_month].
            start_month (int): compute for [start_month, end_month]. Defaults to 
                0.

        Returns:
            dict[str, Any]: package of results with each key being a requested 
                task to its respective export package.
        """
        
        # load in the project's information if we have a previous cache #
        # available #

        ## expected paths
        params_dict = util._load_params()
        paths = {
            "net-gen": Path(params_dict["delta-cache-dir"]) / f"{proj_name}.csv",
            "net-vis": Path(params_dict["network-visualization-dir"]) / f"{proj_name}.json",
            "forecast": Path(params_dict["forecast-dir"]) / f"{proj_name}.json",
            "traj": Path(params_dict["trajectory-dir"]) / f"{proj_name}.json"
        }
        
        ## check all paths exist for the requested caches
        if not all(paths[task].exists() for task in tasks):
            return None
        
        ## load in the caches
        caches = dict()

        for task in tasks:
            ## check type of call
            if task == "net-gen":
                caches[task] = pd.read_csv(paths[task], engine=CSV_ENGINE)
            else:
                with open(paths[task], "r") as f:
                    caches[task] = json.load(f)
            
        # now that we have the caches, ensure each of them has enough info to 
        # fulfill the request
        for task, cache in caches.items():
            if len(cache) < end_month + 1:
                return None
        
        # export the valid caches for this result
        for task in tasks:
            if task == "net-gen":
                caches[task] = caches[task].head(end_month + 1)
            else:
                caches[task] = {k: v for k, v in caches[task].items() if start_month <= int(k) <= end_month}
        
        return caches
    
    def deliver_results(tasks: list[str], proj_name: str, pkg: dict[str, Any]) -> list[str]:
        """Handles the delivery of all the computed results for each task. 
        Exports them into an expected format.

        Args:
            tasks (list[str]): tasks requested.
            proj_name (str): identifier for the project operated on.
            pkg (dict[str, Any]): package of results where each key matches a
                task's key.
        
        Returns:
        {
            
            "tasks": list[str] -- list of tasks completed. Will contain any 
                subset of {"CACHED", "NETS", "FORECASTS", "TRAJECTORIES"}.
        """
        
        pass

    def dispatcher(tasks: list[str], data: dict[str, Any]) -> dict[str, Any]:
        """Dispatches the computation for all requested tasks via the DeltaData
        abstraction. Raises an error on failure.

        Args:
            tasks (list[str]): tasks requested.
            data (dict[str, Any]): data package to use in completing the 
                requests.

        Returns:
            dict[str, Any]: returns a dictionary of valid results if computation
                progressed as expected, otherwise raises an error.
        """
        
        # initialize the delta data
        dd = DeltaData(
            proj_name=data["proj_name"], tdata=data["tdata"],
            sdata=data["sdata"], tasks=tasks
        )
        
        # capture the results from the object, subset from all computations
        key_translator = {
            "net-gen": dd.__dict__.get("netdata", None),
            "net-vis": dd.__dict__.get("net_vis", None),
            "forecast": dd.__dict__.get("forecasts", None),
            "traj": dd.__dict__.get("trajectories", None)
        }
        dispatch_res = {task: key_translator[task] for task in tasks}
        
        # check that all tasks have been appropriately computed
        if any(dispatch_res[task] is None for task in tasks):
            compl_tasks = [task for task in tasks if dispatch_res[task] is not None]
            raise ValueError(f"Tasks failed to compute as expected; needed {tasks}, but only have data for {compl_tasks}")
    
    # check if the cache is available
    cached_result = check_cache(
        tasks=tasks, proj_name=data["proj_name"],
        end_month=data["month_range"][-1]
    )
    
    ## send out cached result if possible
    if cached_result is not None:
        delivered_tasks = deliver_results(
            tasks=tasks, proj_name=data["proj_name"], pkg=cached_result
        )
        return delivered_tasks
    
    # if not, compute a fresh result
    computed_result = dispatcher(tasks=tasks, data=data)
    delivered_tasks = deliver_results(
        tasks=tasks, proj_name=data["proj_name"], pkg=computed_result
    )
    return delivered_tasks


# ------------- Testing ------------- #
if __name__ == "__main__":
    test_data = {
        
    }
    compute_forecast(test_data)

