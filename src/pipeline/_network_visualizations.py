"""
    @brief Generates the monthly network information for each project for easy 
           visualization in an APEX-like tool.
    @author Arjun Ashok (arjun3.ashok@gmail.com)
    @creation-date April 2024
"""

import os
import json
import sys
from collections import Counter
from typing import Any

import networkx as nx
import pandas as pd
from tqdm import tqdm

import general_utils as util


# ---------------- processing utility ---------------- #
def tech_net_info(t_path: str, output_path: str) -> None:
    """
        Generates a JSON formatted file given the input net file for the 
        technical network edges (per month).
    """

    # check file
    if not os.path.exists(t_path) or os.stat(t_path).st_size == 0:
        with open(output_path, "w") as f:
            json.dump([[]], f, indent=4)
        return

    # read in file
    df = pd.read_csv(t_path, header=None, sep="##", engine="python")
    df.columns = ["file", "dev", "weight"]

    # file extensions
    df["file"] = df["file"].apply(lambda x: x.split(".")[-1])
    agg_df = df.groupby(["dev", "file"]).agg({"weight": "sum"}).reset_index()
    list_df = agg_df.values.tolist()

    # export
    with open(output_path, "w") as f:
        json.dump(list_df, f, indent=4)


def social_net_info(s_path: str, output_path: str) -> None:
    """
        Generates a JSON formatted file given the input net file for the social 
        network edges (per month).
    """

    # check file
    if not os.path.exists(s_path) or os.stat(s_path).st_size == 0:
        with open(output_path, "w") as f:
            json.dump([[]], f, indent=4)
        return

    # read in file
    df = pd.read_csv(s_path, header=None, sep="##", engine="python")
    df.columns = ["sender", "receiver", "weight"]

    # file extensions
    agg_df = df.groupby(["sender", "receiver"]).agg({"weight": "sum"}).reset_index()
    list_df = agg_df.values.tolist()

    # export
    with open(output_path, "w") as f:
        json.dump(list_df, f, indent=4)


# ---------------- script ---------------- #
def net_vis_info(args_dict: dict[str, Any]) -> None:
    """
        Wraps the full utility for generating the necessary lookups for the tech 
        and social networks.
    """

    # setup
    print("\n<Generating Network Info for Visualization>")
    params_dict = util._load_params()

    # execute input
    versions = args_dict["versions"]
    social_type = params_dict["social-type"][args_dict["incubator"]]
    tech_type = params_dict["tech-type"][args_dict["incubator"]]

    mapping_path = f"../network_data/mappings/{args_dict['incubator']}-mapping-{versions['tech']}-{versions['social']}.csv"
    t_dir = f"../network_data/{args_dict['incubator']}_{tech_type}/"
    s_dir = f"../network_data/{args_dict['incubator']}_{social_type}/"
    proj_inc_path = params_dict["incubation-time"][args_dict["incubator"]]

    base_dir = f"../network_data/net-vis/"
    util._check_dir(base_dir)
    t_output_dir = f"{base_dir}{args_dict['incubator']}_{tech_type}/"
    s_output_dir = f"{base_dir}{args_dict['incubator']}_{social_type}/"

    # setup & prepare (clear output dirs, get iteration list)
    util._clear_dir(t_output_dir)
    util._check_dir(t_output_dir)
    util._clear_dir(s_output_dir)
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

    # generate network data
    for project_name in tqdm(sorted(projects.keys())):
        # may not have the network data
        for month in range(project_incubation_dict.get(project_name, 0)):
            # unpack
            net_file = "{}__{}.edgelist".format(project_name, month)
            tech_net_path = t_dir + net_file
            tech_net_out = t_output_dir + net_file

            social_net_path = s_dir + net_file
            social_net_out = s_output_dir + net_file

            # grab necessary info & save
            tech_net_info(tech_net_path, tech_net_out)
            social_net_info(social_net_path, social_net_out)


if __name__ == "__main__":
    args_dict = util._parse_input(sys.argv)
    net_vis_info(args_dict)

