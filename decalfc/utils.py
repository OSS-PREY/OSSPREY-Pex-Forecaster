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
import shutil
from time import time
from pprint import pformat, pprint
from os import cpu_count
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
TRANSFER_STRATS = [
    "A{opt} --> A{t_opt}",
    "E{opt} --> A{t_opt}",
    "O{opt} --> A{t_opt}",
    "E{opt} + O{opt} --> A{t_opt}",
    "A{opt}^ + E{opt} --> A{t_opt}^^",
    "A{opt} --> E{t_opt}",
    "O{opt} --> E{t_opt}",
    "A{opt} + O{opt} --> E{t_opt}",
    "E{opt}^ --> E{t_opt}^^",
    "A{opt} + E{opt}^ --> E{t_opt}^^",
    "A{opt} --> G{t_opt}",
    "E{opt} --> G{t_opt}",
    "O{opt} --> G{t_opt}",
    "A{opt} + E{opt} --> G{t_opt}",
    "A{opt} + O{opt} --> G{t_opt}",
    "E{opt} + O{opt} --> G{t_opt}",
    "A{opt}^ + E{opt}^ --> A{t_opt}^^ + E{t_opt}^^",
    "A{opt}^ + E{opt}^ + O{opt}^ --> A{t_opt}^^ + E{t_opt}^^ + O{t_opt}^^",
    "A{opt}^ + E{opt}^ --> A{t_opt}^^ + E{t_opt}^^ + G{t_opt}",
    "A{opt}^ + E{opt}^ + O{opt}^ --> A{t_opt}^^ + E{t_opt}^^ + O{t_opt}^^ + G{t_opt}"
    "A{opt} + E{opt} + O{opt} --> G{t_opt}"
]
PAPER_STRATS = {
    "sustainability_to_success": [
        "A{opt} --> G{t_opt}",
        "E{opt} --> G{t_opt}",
        "O{opt} --> G{t_opt}",
        \
        "A{opt} + E{opt} --> G{t_opt}",
        "A{opt} + O{opt} --> G{t_opt}",
        "E{opt} + O{opt} --> G{t_opt}",
        \
        "A{opt} + E{opt} + O{opt} --> G{t_opt}"
    ],
    "success_to_sustainability": [
        "G{opt} --> A{t_opt}",
        "G{opt} --> E{t_opt}",
        "G{opt} --> O{t_opt}"
    ],
    "in_incubator": [
        "A{opt}^ --> A{t_opt}^^",
        "E{opt}^ --> E{t_opt}^^",
        "O{opt}^ --> O{t_opt}^^",
        \
        "G{opt}^ --> G{t_opt}^^",
    ],
    "mix_incubator": [ # target present in train
        "A{opt}^ + E{opt} --> A{t_opt}^^",
        "A{opt}^ + O{opt} --> A{t_opt}^^",
        "A{opt}^ + E{opt} + O{opt} --> A{t_opt}^^",
        \
        "A{opt} + E{opt}^ --> E{t_opt}^^",
        "E{opt}^ + O{opt} --> E{t_opt}^^",
        "A{opt} + E{opt}^ + O{opt} --> E{t_opt}^^",
        \
        "O{opt}^ + A{opt} --> O{t_opt}^^",
        "O{opt}^ + E{opt} --> O{t_opt}^^",
        "O{opt}^ + A{opt} + E{opt} --> O{t_opt}^^",
        \
        "A{opt}^ + E{opt}^ --> A{t_opt}^^ + E{t_opt}^^",
        "A{opt}^ + O{opt}^ --> A{t_opt}^^ + O{t_opt}^^",
        "E{opt}^ + O{opt}^ --> E{t_opt}^^ + O{t_opt}^^",
        \
        "A{opt}^ + E{opt}^ + O{opt}^ --> A{t_opt}^^ + E{t_opt}^^ + O{t_opt}^^",
    ],
    "cross_incubator": [ # target not present in train
        "E{opt} --> A{t_opt}",
        "O{opt} --> A{t_opt}",
        "E{opt} + O{opt} --> A{t_opt}",
        \
        "A{opt} --> E{t_opt}",
        "O{opt} --> E{t_opt}",
        "A{opt} + O{opt} --> E{t_opt}",
        \
        "A{opt} --> O{t_opt}",
        "E{opt} --> O{t_opt}",
        "A{opt} + E{opt} --> O{t_opt}",
        \
        "A{opt} --> E{t_opt} + O{t_opt}",
        "E{opt} --> A{t_opt} + O{t_opt}",
        "O{opt} --> A{t_opt} + E{t_opt}"
    ]
}


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
    """Logs a different message depending on the type fed in. Default "log" now
    supports a timing scheme to dictate how long the PREVIOUS action took, i.e. 
    from the previous call to ANY log function.

    Args:
        msg (str, optional): output to print/store to console or log file. 
            Defaults to "".
        log_type (str, optional): str, one of {"log", "warning", "error", 
            "note", "debug", "summary", "report"}; "new" will default to a 
            custom type of log identifier. Defaults to "log".
        output (str, optional): buffer to print to, one of {"console", "file"}. 
            Defaults to "console".
        file_name (str, optional): filename to print to. Defaults to "logger".
        check_verbosity (bool, optional): condition to print or not, helpful for
            concise logging with verbosity checks. True indicates we do print, 
            False indicates we skip logging entirely. Defaults to True.
    """
    
    # timer
    delta_time = time() - log.prev_time
    nhrs = delta_time // 3600
    nmin, nsec = divmod(delta_time, 60)
    
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
            output_router[output](
                f"Log [{nhrs}h, {nmin}m, {nsec:.3f}s]> {msg.capitalize()}"
            )
        
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
            
            # automatically print the summary if the report exists
            if len(msg):
                log(msg, "report", output, file_name, check_verbosity)
        
        case "report":
            # skip print if we can't match the type
            if not isinstance(msg, dict):
                return
            
            # convert to strings then print the joined string
            stat_strs = [f"{k}:\t{v if isinstance(v, str) else pformat(v, indent=4)}" for k, v in msg.items()]
            output_router[output]("\n".join(stat_strs))
        
        case "none":
            output_router[output](msg)

        # default to new
        case _:
            output_router[output](f"\n<{msg.title()}>")
            
    # update time
    log.prev_time = time()
log.prev_time = time()


# Environment Setup
pandarallel.initialize(nb_workers=NUM_PROCESSES, progress_bar=True)
tqdm.pandas()
params_dict = load_params()

