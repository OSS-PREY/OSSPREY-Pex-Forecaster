"""
    @brief Defines basic utility for use across all programs.
    @author Arjun Ashok (arjun3.ashok@gmail.com)
    @creation-date February 2024
"""

# --- Environment Setup --- #
## packages
import pandas as pd
import torch
from pandarallel import pandarallel
from tqdm import tqdm

## built-in modules
import argparse
import json
from os import cpu_count
import shutil
from ast import literal_eval
from pathlib import Path
from typing import Any

## constants
PARAMS_PATH = Path("./ref/params.json")
NUM_PROCESSES = cpu_count()
PARQUET_ENGINE = "pyarrow"
CSV_ENGINE = "python"
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else (
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
)

# --- Utility --- #
def load_params() -> dict[str, Any]:
    """
        Wrapper for loading in the params dictionary for centralized params.
    """

    # load & return
    with open(PARAMS_PATH, "r") as f:
        params_dict = json.load(f)
    return params_dict

def parse_input(argv: list[str], **kwargs) -> dict[str, Any]:
    """
        Centralized method for obtaining and parsing input into a readable 
        dictionary.

        Strategy Implemented: accept params as a kwargs list
    """

    # parse args
    parser = argparse.ArgumentParser(description="Parse keyword args.")
    parser.add_argument(
        "--kwargs",
        nargs="*",
        help="Pass key-value pairs as --kwargs key1=value1 key2=value2 ...",
    )
    args = parser.parse_args()
    
    # convert into a dictionary
    kwargs = dict()
    if not args.kwargs:
        return kwargs
    
    for arg in args.kwargs:
        # split into the k, v pairs
        try:
            key, value = arg.split("=", 1)
        except ValueError:
            raise ValueError(f"Error: Invalid argument format '{arg}'. Use key=value.")
        
        # parsing
        try:
            # evaluate as JSON str
            try:
                parsed_value = json.loads(value)
            except:
                # safely evaluate the string repr of the value
                parsed_value = literal_eval(value)
        except (ValueError, SyntaxError):
            # worst case, keep as a string
            parsed_value = value
        
        # store results
        kwargs[key] = parsed_value
    
    # export
    return kwargs

def check_dir(dir: str | Path) -> None:
    """
        Checks if a dir exists, makes it if it doesn't
    """

    # make the dir along with its parents
    Path(dir).mkdir(parents=True, exist_ok=True)

def check_path(path: str | Path) -> None:
    """
        Checks if the dir for a file exists, otherwise creates it.
    """

    dir = Path(path).parent
    check_dir(dir)

def clear_dir(dir: str | Path, skip_input: bool=False) -> None:
    """
        Clears the contents of a directory if it exists. Main purpose is to 
        avoid accessing data from previous trials when generating networks.
    """

    # check existence
    if not Path(dir).exists():
        return

    # failsafe
    if "_data" not in str(dir):
        print(f"<WARNING> attempting to delete potentially sensitive directory: {dir}")

        if not skip_input:
            resp = input("Continue [y/n]? ")
            if resp.lower() != "y":
                return
    
    # remove
    print("<Clearing Previous Trials>")
    shutil.rmtree(dir)

def del_file(path: str | Path) -> None:
    """
        Deletes the specified file.
    """
    
    # check path
    path = Path(path)
    
    if path.is_dir():
        raise ValueError(f"Path provided \"{path.absolute()}\" is a directory; use util._clear_dir() instead")
    
    # delete
    path.unlink()

def log(msg: str="", log_type: str="log", output: str="console",
    file_name: str="logger", check_verbosity: bool=True) -> None:
    """Logs a different message depending on the type fed in.

    Args:
        msg (str, optional): output to print/store to console or log file. 
            Defaults to "".
        log_type (str, optional): str, one of {"log", "warning", "error", 
            "note", "debug", "summary"}; "new" will default to a custom type of 
            log identifier. Defaults to "log".
        output (str, optional): buffer to print to, one of {"console", "file"}. 
            Defaults to "console".
        file_name (str, optional): filename to print to. Defaults to "logger".
        check_verbosity (bool, optional): condition to print or not, helpful for
            concise logging with verbosity checks. True indicates we do print, 
            False indicates we skip logging entirely. Defaults to True.
    """
    
    # check verbosity, only skip if we're logging to STDOUT
    if not check_verbosity and output == "file":
        return

    # auxiliary functions
    def log_file(info: str):
        with open(f"../{file_name}.log", "a") as f:
            f.write(f"{info}\n")

    # output type
    output_router = {
        "console": print,
        "file": log_file
    }

    # match type
    match log_type:
        case "log":
            output_router[output](f"Log> {msg.capitalize()}")
        
        case "warning":
            output_router[output](f"WARNING> {msg}")
        
        case "error":
            output_router[output](f"ERROR> {msg}")
        
        case "note":
            output_router[output](f"Note> {msg}")
        
        case "debug":
            output_router[output](f"\n < DEBUGGING > {msg}")
        
        case "summary":
            output_router[output]("\n::: < SUMMARY > :::")
        
        case "none":
            output_router[output](msg)

        # default to new
        case _:
            output_router[output](f"\n<{msg.title()}>")


# --- Environment Setup --- #
pandarallel.initialize(nb_workers=NUM_PROCESSES, progress_bar=True)
tqdm.pandas()
params_dict = load_params()


# --- Dependent Utility --- #
def reset_cache(proj_name: str, products: list[str]=None) -> None:
    """Clears all cached data products specified, defaults to all.
    
    Args:
        proj_name (str): _description_
        products (list[str], optional): end products to clear out. Should be
            a subset of {"net-gen", "net-vis", "forecast", "traj"}.
    """
    
    # path lookup for all products
    paths = {
        "net-gen": Path(params_dict["delta-cache-dir"]) / f"{proj_name}.csv",
        "net-vis": Path(params_dict["network-visualization-dir"]) / f"{proj_name}.json",
        "forecast": Path(params_dict["forecast-dir"]) / f"{proj_name}.json",
        "traj": Path(params_dict["trajectory-dir"]) / f"{proj_name}.json"
    }
    
    # check prods
    if products is None:
        products = list(paths.keys())
    
    # check if exists, delete if it does
    for prod in products:
        # lookup path
        path = paths[prod]
        
        # delete if it exists
        path.unlink(missing_ok=True)

