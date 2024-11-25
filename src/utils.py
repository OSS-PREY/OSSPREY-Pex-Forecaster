"""
    @brief Defines basic utility for use across all programs.
    @author Arjun Ashok (arjun3.ashok@gmail.com)
    @creation-date February 2024
"""

# --- Environment Setup --- #
## built-in modules
import json
import os
import shutil
from typing import Any


# Utility
def _load_params() -> dict[str, Any]:
    """
        Wrapper for loading in the params dictionary for centralized params.
    """

    # load & return
    with open("./ref/params.json", "r") as f:
        params_dict = json.load(f)
    return params_dict


def _parse_input(argv: list[str], **kwargs) -> dict[str, Any]:
    """
        Centralized method for obtaining and parsing input into a readable 
        dictionary.

        Strategy Implemented: accept params as a dictionary string
    """

    return json.loads(argv[1])


def _check_dir(dir: str) -> None:
    """
        Checks if a dir exists, makes it if it doesn't
    """

    if not os.path.exists(dir):
        os.makedirs(dir)


def _check_path(path: str) -> None:
    """
        Checks if the dir for a file exists, otherwise creates it.
    """

    dir = os.path.dirname(path)
    _check_dir(dir)


def _clear_dir(dir: str, skip_input: bool=False) -> None:
    """
        Clears the contents of a directory if it exists. Main purpose is to 
        avoid accessing data from previous trials when generating networks.
    """

    # check existence
    if not os.path.exists(dir):
        return

    # failsafe
    if "_data" not in dir:
        print(f"<WARNING> attempting to delete potentially sensitive directory: {dir}")

        if not skip_input:
            resp = input("Continue [y/n]? ")
            if resp.lower() != "y":
                return
    
    # remove
    print("\t\t<clearing previous trials>")
    shutil.rmtree(dir)


def _log(msg: str="", log_type: str="log", output: str="console",
         file_name: str="logger") -> None:
    """
        Logs a different message depending on the type fed in.

        @param msg: output to print/store to console or log file
        @param log_type: str, one of {"log", "warning", "error", "note", "debug", 
            "summary"}; "new" will efault to a custom type of log identifier
    """

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
            output_router[output](f"<log> {msg.capitalize()}")
        
        case "warning":
            output_router[output](f"<WARNING> {msg}")
        
        case "error":
            output_router[output](f"<ERROR> {msg}")
        
        case "note":
            output_router[output](f"<Note> {msg}")
        
        case "debug":
            output_router[output](f"\n < DEBUGGING > {msg}")
        
        case "summary":
            output_router[output]("\n::: < SUMMARY > :::")
        
        case "none":
            output_router[output](msg)

        # default to new
        case _:
            output_router[output](f"\n<{msg.title()}>")

