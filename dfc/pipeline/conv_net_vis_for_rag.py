"""
    @brief Generates the monthly network information for each project for easy 
        visualization in an APEX-like tool.
    @creation-date April 2024
"""

# ------------- Environment Setup ------------- #
# external packages
import networkx as nx
import pandas as pd
from tqdm import tqdm

# built-in modules
import os
import json
import sys
from collections import Counter
from typing import Any
from pathlib import Path

# DECAL modules
from dfc.utils import *


# ---------------- processing utility ---------------- #
def conv_month(mdata: list[list[str | int]], month: int) -> pd.DataFrame:
    """Converts a month of information into a dataframe.
    """
    
    df = pd.DataFrame(
        mdata,
        columns=["sender", "recipient_or_file_type", "n_interactions"]
    )
    df["month"] = int(month)
    return df

def conv_proj(pdata: dict[str, list[list[str | int]]], pname: str) -> pd.DataFrame:
    """Converts a project's information into a dataframe.
    """
    
    pdata = [conv_month(mdata, int(m)) for m, mdata in pdata.items() if len(mdata) and len(mdata[0])]
    
    if not pdata:
        return pd.DataFrame()
    
    df = pd.concat(pdata, ignore_index=True)
    df["project"] = pname
    return df

def conv_netwise(ndata: dict[str, dict[str, list[list[str | int]]]], ntype: str="tech") -> pd.DataFrame:
    """
        Generates a JSON formatted file given the input net file for the 
        technical network edges (per month).
    """

    # process project by project
    ndata = [conv_proj(pdata, proj) for proj, pdata in tqdm(ndata.items()) if len(pdata)]
    df = pd.concat(ndata, ignore_index=True)

    # export
    df["interaction_type"] = ntype
    return df

# ---------------- script ---------------- #
def conv_to_csv(incubator: str) -> None:
    """
        Wraps the full utility for generating the necessary lookups for the tech 
        and social networks. Final columns are:
        
        - project: str
        - month: int
        - interaction_type (tech / social): str
        - sender: str
        - recipient_or_file_type (for social / tech, respectively): str
        - n_interactions: int | float
    """

    # setup
    print("\n<Converting Visualizations>")

    # execute input
    base_dir = Path(params_dict["network-visualization-dir"])
    vis_path = base_dir / f"{incubator}_network_visualizations.json"

    # load in
    with open(vis_path, "r") as f:
        vis = json.load(f)
    
    # conversion
    df = pd.concat(
        [
            conv_netwise(vis["tech"], "tech"),
            conv_netwise(vis["social"], "social")
        ],
        ignore_index=True
    )
    df = df[[
        "project", "month", "interaction_type", "sender",
        "recipient_or_file_type", "n_interactions"
    ]]

    # save
    df.sort_values(list(df.columns), ascending=True, inplace=True)
    df.to_csv(vis_path.with_suffix(".csv"), index=False)
    return df

if __name__ == "__main__":
    args_dict = parse_input(sys.argv)
    conv_to_csv(**args_dict)

