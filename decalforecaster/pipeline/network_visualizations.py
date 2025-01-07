"""
    @brief Generates the monthly network information for each project for easy 
        visualization in an APEX-like tool.
    @author Arjun Ashok (arjun3.ashok@gmail.com)
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
import decalforecaster.utils as util
from decalforecaster.utils import PARQUET_ENGINE, CSV_ENGINE


# ---------------- processing utility ---------------- #
def tech_net_info(t_path: Path) -> list[list[str | int]]:
    """
        Generates a JSON formatted file given the input net file for the 
        technical network edges (per month).
    """

    # check file
    if not t_path.exists() or t_path.stat().st_size == 0:
        return [[]]

    # read in file
    df = pd.read_csv(t_path, header=None, sep="##", engine=CSV_ENGINE)
    df.columns = ["file", "dev", "weight"]

    # file extensions
    df["file"] = df["file"].apply(lambda x: x.split(".")[-1])
    agg_df = df.groupby(["dev", "file"]).agg({"weight": "sum"}).reset_index()
    list_df = agg_df.values.tolist()

    # export
    return list_df

def social_net_info(s_path: Path) -> list[list[str | int]]:
    """
        Generates a JSON formatted file given the input net file for the social 
        network edges (per month).
    """

    # check file
    if not s_path.exists() or s_path.stat().st_size == 0:
        return [[]]

    # read in file
    df = pd.read_csv(s_path, header=None, sep="##", engine=CSV_ENGINE)
    df.columns = ["sender", "receiver", "weight"]

    # file extensions
    agg_df = df.groupby(["sender", "receiver"]).agg({"weight": "sum"}).reset_index()
    list_df = agg_df.values.tolist()

    # export
    return list_df


# ---------------- script ---------------- #
def net_vis_info(args_dict: dict[str, Any]) -> dict[str, list[list[str | int]]]:
    """
        Wraps the full utility for generating the necessary lookups for the tech 
        and social networks.
    """

    # setup
    print("\n<Generating Network Info for Visualization>")
    params_dict = util._load_params()

    # execute input
    social_type = params_dict["social-type"][args_dict["incubator"]]
    tech_type = params_dict["tech-type"][args_dict["incubator"]]
    network_dir = Path(params_dict["network-dir"])

    t_dir = network_dir / f"{args_dict['incubator']}_{tech_type}/"
    s_dir = network_dir / f"{args_dict['incubator']}_{social_type}/"
    proj_inc_path = params_dict["incubation-time"][args_dict["incubator"]]

    base_dir = Path(params_dict["network-visualization-dir"])
    t_output_dir = base_dir / f"{args_dict['incubator']}_{tech_type}/"
    s_output_dir = base_dir / f"{args_dict['incubator']}_{social_type}/"

    # setup & prepare (clear output dirs, get iteration list)
    util._check_dir(base_dir)
    util._clear_dir(t_output_dir, skip_input=True)
    util._check_dir(t_output_dir)
    util._clear_dir(s_output_dir, skip_input=True)
    util._check_dir(s_output_dir)

    s_nets = set(os.listdir(s_dir))
    t_nets = set(os.listdir(t_dir))
    nets = s_nets.union(t_nets)

    # load in
    projects = {}
    for net_file in nets:
        project_name, period = net_file.split("__")
        if project_name not in projects:
            projects[project_name] = set()
        projects[project_name].add(int(period.replace(".edgelist", "")))
    for project_name in projects:
        projects[project_name] = sorted(list(projects[project_name]))

    # incubation time
    with open(proj_inc_path, "r") as f:
        project_incubation_dict = json.load(f)

    # generate network visualization information & store into a json
    net_visuals = {
        "tech": dict(),
        "social": dict()
    }
    
    for project_name in tqdm(sorted(projects.keys())):
        # may not have the network data
        for month in range(project_incubation_dict.get(project_name, 0)):
            # unpack file directions
            net_file = "{}__{}.edgelist".format(project_name, month)
            tech_net_path = t_dir / net_file
            social_net_path = s_dir / net_file

            # grab necessary info & save
            net_visuals["tech"][month] = tech_net_info(tech_net_path)
            net_visuals["social"][month] = social_net_info(social_net_path)
            
    # export to memory
    return net_visuals


if __name__ == "__main__":
    args_dict = util._parse_input(sys.argv)
    net_vis_info(args_dict)

