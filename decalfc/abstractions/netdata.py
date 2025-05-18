"""
    @brief defines the network data class for easily interfacing with the 
        post-processed dataset. Primary utility comes from bundling all 
        relevant information in one object.
    @author Arjun Ashok (arjun3.ashok@gmail.com)
    @creation-date January 2024
    @version 0.1.0
"""

# Imports
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
from tqdm import tqdm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import os
import sys
import re
import json
import random
from dataclasses import dataclass, field
from math import floor, ceil
from typing import Any
from pathlib import Path

from decalfc.utils import *


# Constants
util_dir = Path(params_dict["ref-dir"])
network_dir = Path(params_dict["network-dir"])


# Class
@dataclass(slots=True)
class NetData:
    # user params
    incubator: str                                                              # incubator name
    versions: dict[str, str] = field(default=None)                              # { tech/social: version }
    options: dict[str, bool] = field(default_factory=lambda: {                  # options in ORDER OF OPERATIONS
        "clean": True,
        "feature-subset": False,
        "normalize-actdev": False,
        "normalize-minmax": False,
        "normalize-zscore": False,
        "smooth": False,
        "impute": False,
        "jitter": False,
        "upsample": False,
        "downsample": False,
        "diff": False,
        "aggregate": False,
        "interval": False,
        "proportion-interval": False,
        "lag-interval": False
    })
    transform_kwargs: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )                                                                           # specify params to pass into transform functions if netdata is to be generated
    is_train: str = field(default="none")                                       # specify if train or test; if None, generates both
    do_compute_tensors: bool = field(default=True)                              # whether to generate tensors or not
    soft_prob: bool = field(default=False)                                      # soft probabilities for training
    ignore_cache: bool = field(default=False)                                   # ignore cache if needed
    verbose: bool = field(default=True)                                         # verbose setup

    # internal
    generate: bool = field(default=True)                                        # generate data if not prepared already
    drop_cols: list[str] = field(default_factory=lambda: [                      # what to not include in training
        "proj_name", 
        "month"
    ])
    transform_router: dict[str, Any] = field(default_factory=lambda: {
        "normalize-minmax": (
            lambda self: NetData.normalize_netdata(self, strat="minmax")        # normalize via min-max
        ),
        "normalize-zscore": (
            lambda self: NetData.normalize_netdata(self, strat="zscore")        # normalize via zscore
        ),
        "normalize-actdev": NetData.normalize_netdata,                          # normalize via active developers
        "jitter": NetData.jitter_netdata,                                       # jitter data, doubles number projects  
        "upsample": NetData.upsample_netdata,                                   # up-sampling 
        "downsample": NetData.downsample_netdata,                               # down-sampling
        "smooth": NetData.smoothe_netdata,                                      # smoothing (default ETS)
        "imput": NetData.interpolate_netdata,                                   # interpolation (default spline)
        "diff": NetData.diff_netdata,                                           # first-order differentiation of the features
        "aggregate": NetData.aggregate_netdata,                                 # cumulative sum per column
        "interval": NetData.interval_netdata,                                   # interval subsets of data
        "proportion-interval": (
            lambda self: NetData.interval_netdata(self, proportion=0.125)       # interval subsets of data via proportion
        ),
        "lag-interval": (                                                       # lag intervals via proportion by default
            lambda self: NetData.interval_netdata(
                self, proportion=0.125, backwards=True
            )
        ),
        "feature-subset": NetData.subset_features                               # subsets features
    })
    ext: str = field(default="csv")                                             # extension for storage
    nd_path: str = field(init=False)                                            # path to load/save data
    is_intervaled: bool = field(init=False)                                     # indicates if intervals are generated
    data: pd.DataFrame = field(init=False)                                      # network data
    column_order: list[str] = field(default_factory=lambda: ([                  # ensure consistent feature orderings
        "s_num_nodes",
        "s_weighted_mean_degree",
        "s_num_component",
        "s_avg_clustering_coef",
        "s_largest_component",
        "s_graph_density",
        "t_num_dev_nodes",
        "t_num_file_nodes",
        "t_num_dev_per_file",
        "t_num_file_per_dev",
        "t_graph_density",
        "proj_name",
        "month",
        "st_num_dev",
        "t_net_overlap",
        "s_net_overlap"
    ]))

    # modeling
    rand_seed: int = field(default=17)                                          # for reproducability
    test_prop: float = field(default=0.30)                                      # proportion of data to be saved for testing
    base_projects: set[str] = field(default=None)                               # base projects to match to for end results association
    projects_set: set[str] = field(default=None)                                # set of projects in the data
    split_set: dict[str, set[str]] = field(default=None)                        # split projects by train, test
    data_dict: dict[str, Any] = field(default=None)                             # { project: list of data } for data lookup
    project_status: dict[str, set[str]] = field(default=None)                   # { status: projects set } for status lookup
    tensors: dict[str, dict[str, list[Any]]] = field(default=None)              # { train/test: { x/y: list[tensors] } }


    # internal utility
    def match_proj(self, proj: str) -> str:
        """
            Gets the base project for the corresponding pseudo-project.
        """
        # if base_proj
        if proj in self.base_projects:
            return proj

        # find match
        potential_base_proj = proj[proj.find('[') + 1: proj.find(']')]
        if potential_base_proj in self.base_projects:
            return potential_base_proj
        
        # failure means non-trainable project
        log(f"failed to associate project {proj} using {potential_base_proj}", "warning")
        return proj

    def combine_opts(self) -> str:
        """
            Generates a string representation for the network data augmentations
            for easy path lookup.
        """

        # gen_string
        if self.options == {}:
            return "clean"

        return "-".join([
            params_dict["network-augmentations"][opt] \
            for opt, selected in self.options.items() if selected
        ])
    
    def gen_netdata_path(self) -> None:
        """
            Generates the path for saving/loading the data.
        """

        # generate path
        PATH_FORMAT = params_dict["network-data-format"]
        options_str = self.combine_opts().replace("downsamp-", "")          # downsamp is post-loading
        self.nd_path = PATH_FORMAT.format(
            options_str,
            self.incubator,
            f"{self.versions['tech']}-{self.versions['social']}",
            self.ext
        )

    def gen_default_path(self) -> str:
        """
            Generates the default path for a given incubator.
        """

        # generate the default versions dict
        self.versions = (dict(zip(
            ["tech", "social"],
            params_dict["default-versions"][self.incubator]
        )))

        # make directory
        return params_dict["network-data-format"].format(
            "clean",
            self.incubator,
            f"{self.versions['tech']}-{self.versions['social']}",
            self.ext
        )

    def load(self) -> None:
        """
            Loads the network data, base projects, and projects set.

            @param generate: if True, automatically generates data if does not 
                yet exist
        """

        # base projects
        with open(util_dir / f"{self.incubator}_proj_incubation.json", "r") as f:
            self.base_projects = set(json.load(f).keys())

        # return network data
        if Path(self.nd_path).exists() and not self.ignore_cache:
            self.data = pd.read_csv(self.nd_path, engine="c")
            self.projects_set = set(self.data["proj_name"].unique())
            return
        
        # if doesn't exist, make it: 1. load default 2. apply options
        if not self.generate:
            log(f"Data was not available w/ versions {self.versions} for {self.incubator}", "warning")
            exit(1)
    
        log("Data was not available, automatically generating", "warning")
        log("using default params for transforms. . .", "note", check_verbosity=self.verbose)

        # load base
        self.data = pd.read_csv(self.gen_default_path())

        # applying options
        for option, selection in self.options.items():
            # skip unselected options
            if not selection:
                continue

            # skip if clean or downsamp
            if option == "clean":
                break
            if option == "downsample":
                continue
            
            # ensure option has an empty dict for accessing
            if option not in self.transform_kwargs:
                self.transform_kwargs[option] = dict()
            
            # flag if concat is necessary
            if "a" in self.options.keys() and "d" in self.options.keys():
                self.transform_kwargs[option]["concat"] = True
                log("BREAKING DEFAULT BEHAVIOR! Concatenation instead of overriding", "warning")

            # route the correct transforms
            if len(self.transform_kwargs[option]) == 0:
                log(f"Performing transformation [{option}] with default params", check_verbosity=self.verbose)
            else:
                log(f"Performing transformation [{option}] with the following params: {self.transform_kwargs[option]}", check_verbosity=self.verbose)
            
            self.transform_router[option](
                self,
                **self.transform_kwargs[option] # pass in kwargs if they exist
            )
        
        # automatically save
        self.projects_set = set(self.data["proj_name"].unique())
        self.save()
    
    def save(self) -> None:
        """
            Export network data.
        """

        # save
        self.data.to_csv(self.nd_path, index=False)
    
    def train_test_split(self, strat="stratified") -> None:
        """
            Generates the sets of projects for train and test. Defaults to 
            stratified sampling to test out performance more accurately.
        """

        # check if split is already specified
        if self.split_set is not None:
            # if the train is specified
            if len(self.split_set.get("train", list())) > 0:
                # if test is specified
                if len(self.split_set.get("test", list())) > 0:
                    return
                self.split_set["test"] = self.base_projects - self.split_set["train"]

            elif len(self.split_set.get("test", list())) > 0:
                self.split_set["train"] = self.base_projects - self.split_set["test"]
            
            # early exit
            return

        # figure out split proportions
        num_train = len(self.base_projects)
        num_test = int(num_train * self.test_prop)
        num_train -= num_test

        # ensure the split happens with projects we have
        # sort lists for consistency [sets are unordered]
        grad_list = list(self.projects_set & self.project_status["graduated"])
        ret_list = list(self.projects_set & self.project_status["retired"])
        grad_list.sort()
        ret_list.sort()

        # split mark
        num_grad_test = ceil(len(grad_list) / len(self.base_projects) * num_test)
        num_ret_test = ceil(len(ret_list) / len(self.base_projects) * num_test)

        # split into train/test set
        random.seed(self.rand_seed)
        random.shuffle(grad_list)
        random.shuffle(ret_list)
        
        # cut down if downsamp
        self.split_set = dict()
        
        if self.options.get("downsample", False):
            ## grab counts
            max_diff = 0.1                  # 10% leeway
            labels = ["retired", "graduated"]
            label_counts = [num_ret_test, num_grad_test]
            minority_test_count = np.min(label_counts)
            majority_test_count = ceil(minority_test_count * (1 + max_diff))
            
            ## choose as many projects as possible
            min_grad_test = min(majority_test_count, num_grad_test)
            self.split_set["test"] = set(grad_list[:min_grad_test])
            
            min_ret_test = min(majority_test_count, num_ret_test)
            self.split_set["test"] |= set(ret_list[:min_ret_test])
            
            ## repeat for train
            label_counts = [len(ret_list) - num_ret_test, len(grad_list) - num_grad_test]
            minority_train_count = np.min(label_counts)
            majority_train_count = ceil(minority_train_count * (1 + max_diff))
            
            ## choose as many projects as possible; we'll go from the original
            ## partition of num_{label}_test
            min_grad_train = min(majority_train_count, label_counts[1])
            self.split_set["train"] = set(grad_list[num_grad_test:num_grad_test + min_grad_train])
            
            min_ret_train = min(majority_train_count, label_counts[0])
            self.split_set["train"] |= set(ret_list[num_ret_test:num_ret_test + min_ret_train])
            
        else:
            ## normal split
            self.split_set["test"] = set(grad_list[:num_grad_test] + ret_list[:num_ret_test])
            self.split_set["train"] = set(grad_list[num_grad_test:] + ret_list[num_ret_test:])
        
        # empty set
        if self.is_train == "train":
            self.split_set["test"] = set()
        elif self.is_train == "test":
            self.split_set["train"] = set()
            
        # reporting
        if self.verbose:
            log("\n< :::: TRAIN SET :::: >", "new", "file", "temp_log")
            log(f"{self.split_set['train']}", "none", "file", "temp_log")
            log("\n< :::: TEST SET :::: >", "new", "file", "temp_log")
            log(f"{self.split_set['test']}", "none", "file", "temp_log")

    def gen_data_dict(self) -> None:
        """
            Generates data lookup for tensors.
        """

        # store data lookup + define utility
        project_subset = self.projects_set
        if self.is_train in ["train", "test"]:
            project_subset = ({
                p for p in project_subset 
                if self.match_proj(p) in self.split_set[self.is_train]
            })

        self.data_dict = {}
        
        def process_group(group):
            # truncate cols & generate list of data points
            return group.drop(self.drop_cols, axis=1).values.tolist()
        
        # group apply to every project for lookup generation
        grouped_data = self.data[self.data["proj_name"].isin(project_subset)].groupby("proj_name").apply(process_group)
        self.data_dict = grouped_data.to_dict()

    def load_proj_status(self) -> None:
        """
            Loads in the project status as given by the params dict.
        """

        with open(util_dir / f"{self.incubator}_project_status.json", "r") as f:
            project_status = json.load(f)
        self.project_status = {s: set(project_status[s]) for s in project_status}

    @staticmethod
    def split_data(dataset_dirs: dict[str, str]) -> dict[str, dict[str, set]]:
        """
            In the generic case, we'll always be using train, test, or both 
            sets. As such, it would be of great utility to have the split and 
            then combine as necessary.

            The return is a dictionary of the split datasets.
        """
        
        pass

    def interval_tensors(
        self, set_type: str=None, subset: set=None, 
        ignore_incubating: bool=False
    ) -> dict[str: dict[str, Any]]:
        """
            Testing for intervals creation; returns X, y, and info about the 
            selected projects for logging.
        """
        
        # setup
        X: dict[str, Any] = dict()
        y: dict[str, Any] = dict()
        log_info = dict(zip(["grad", "ret", "skip"], [0, 0, 0]))
        
        if subset is None:
            subset = self.split_set[set_type]

        # check every project (since we have to match to a base project)
        for proj in tqdm(self.projects_set):
            # check base
            base_proj = self.match_proj(proj)
            if base_proj not in subset:
                continue

            # skip non-present projects
            if proj not in self.data_dict:
                log_info["skip"] += 1
                continue

            # get month
            if proj == base_proj:
                temporal_num = "all"
            else:
                # either numbered or jittered/augmented, either way
                # distinguishes how to capture the month
                if any(temporal_str in proj for temporal_str in ["months", "lag-steps", "steps"]):
                    # match for a number
                    matched_str = re.search(r"\[.*\]-(\d+-.*)", proj)
                    temporal_num = str(matched_str.group(1))
                else:
                    # remove the augmentation
                    temporal_num = proj[proj.rfind("-") + 1:]

            # generate tensors
            if base_proj in self.project_status["graduated"]:
                if temporal_num not in X:
                    X[temporal_num] = []
                    y[temporal_num] = []
                X[temporal_num].append(torch.tensor(self.data_dict[proj]))
                y[temporal_num].append(torch.tensor([1]))
                log_info["grad"] += 1
            elif base_proj in self.project_status["retired"]:
                if temporal_num not in X:
                    X[temporal_num] = []
                    y[temporal_num] = []
                X[temporal_num].append(torch.tensor(self.data_dict[proj]))
                y[temporal_num].append(torch.tensor([0]))
                log_info["ret"] += 1
            else:
                if temporal_num not in X:
                    X[temporal_num] = []
                    y[temporal_num] = []
                X[temporal_num].append(torch.tensor(self.data_dict[proj]))
                y[temporal_num].append(torch.tensor([-1]))
                log_info["skip"] += 1
        
        # return the package + report
        if self.verbose:
            print(f"<{set_type}>")
            print(f"\tx: {len(subset)} base projects, {len(X)} total")
            print(f"\ty: {len(subset)} base projects, {len(y)} total")
            print(f"\tgraduated:          {len(self.project_status['graduated'] & (subset))} base, {log_info['grad']} total")
            print(f"\tretired:            {len(self.project_status['retired'] & (subset))} base, {log_info['ret']} total")
            print(f"\tincubating/skipped: {len(self.project_status['incubating'] & (subset))} base, {log_info['skip']} total")
        
        return {"x": X, "y": y, "log": log_info}

    def reg_tensors(self, set_type: str=None, subset: set=None) -> dict[str, Any]:
        """
            Testing for non-intervaled data creation; returns X, y, and info 
            about the selected projects for logging.
        """
        
        # setup
        X = []
        y = []
        log_info = dict(zip(["grad", "ret", "skip"], [0, 0, 0]))

        if subset is None:
            subset = self.split_set[set_type]

        # add all projects
        for proj in tqdm(self.projects_set):
            # check base
            base_proj = self.match_proj(proj)
            if base_proj not in subset:
                continue

            # skip non-present projects
            if proj not in self.data_dict:
                log_info["skip"] += 1
                continue

            # association
            base_proj = self.match_proj(proj)

            # apply status
            if base_proj in self.project_status["graduated"]:
                X.append(torch.tensor(self.data_dict[proj]))
                y.append(torch.tensor([1]))
                log_info["grad"] += 1
            elif base_proj in self.project_status["retired"]:
                X.append(torch.tensor(self.data_dict[proj]))
                y.append(torch.tensor([0]))
                log_info["ret"] += 1
            else:
                log_info["skip"] += 1

        # export + report
        if self.verbose:
            print(f"<{set_type}>")
            print(f"\tx: {len(X)}")
            print(f"\ty: {len(y)}")
            print(f"\tgraduated:          {log_info['grad']}")
            print(f"\tretired:            {log_info['ret']}")
            print(f"\tincubating/skipped: {log_info['skip']}")

        return {"x": X, "y": y, "log": log_info}

    def gen_tensors(self) -> None:
        """
            Generates the train and test tensors.
        """

        # generate tensors
        self.tensors = dict(zip(["train", "test"], [dict(), dict()]))
        self.is_intervaled = any((("interval" in opt) and sel) for opt, sel in self.options.items())

        log(f"Tensor Info for {self.incubator}", "new", check_verbosity=self.verbose)
        if self.is_train in {"train", "both"}:
            ## soft probabilities don't require a split, we can simply pretend
            ## they're pseudo projects again
            train_pkg = self.reg_tensors("train")
            self.tensors["train"]["x"] = train_pkg["x"]
            self.tensors["train"]["y"] = train_pkg["y"]
        if self.is_train in {"test", "both"}:
            test_pkg = self.interval_tensors("test") if self.is_intervaled \
                else self.reg_tensors("test")
            self.tensors["test"]["x"] = test_pkg["x"]
            self.tensors["test"]["y"] = test_pkg["y"]

    def __post_init__(self):
        # load default versions if needed
        if self.versions is None:
            self.gen_default_path()

        # ensure transform kwargs
        if self.transform_kwargs is None:
            self.transform_kwargs = dict()

        # generate NetData object
        log("setting up NetData", check_verbosity=self.verbose)
        self.gen_netdata_path()
        log("reading in/generating data", check_verbosity=self.verbose)
        self.load()
        
        # ensure column order
        if self.options.get("aggregate", False):
            self.column_order += [f"{f}_agg" for f in self.column_order if f not in self.drop_cols]
        if self.options.get("diff", False):
            self.column_order += [f"{f}_diff" for f in self.column_order if f not in self.drop_cols]
        
        self.column_order = [k for k in self.column_order if k in set(self.data.columns)]
        self.data = self.data[self.column_order]

        # generation
        log("generating project status, split", check_verbosity=self.verbose)
        self.load_proj_status()
        self.train_test_split()

        if self.do_compute_tensors:
            log("generating data lookup", check_verbosity=self.verbose)
            self.gen_data_dict()
            
            log("generating tensors", check_verbosity=self.verbose)
            self.gen_tensors()


    # external utility
    ## random utility
    def max_month(self) -> int:
        """
            Finds the maximum month for interval training and testing.
        """

        # load project incubation
        proj_inc_path = params_dict["incubation-time"][self.incubator]
        with open(proj_inc_path, "r") as f:
            proj_inc_dict = json.load(f)
        
        return max(proj_inc_dict.values())
    
    
    ## visualizations & statistics
    def visualize_synthetic(
        self, data: pd.DataFrame, base_proj: str, strategy: str
    ) -> None:
        """
            Visualizes the synthetic projects generated from a base project.

            @param data: synthetic + real data
            @param project: base project to observe
        """

        # setup
        stop_month = 50

        # get synthetic & base projects
        based_projects = [proj for proj in data["proj_name"].unique() 
                          if self.match_proj(proj) == base_proj]

        # generate activity over time (number developers in both)
        data = data[data["proj_name"].isin(based_projects)]
        data = data.sort_values(by=["proj_name", "month"])
        data["activity"] = data["s_num_nodes"] + data["t_num_dev_nodes"]
        data = data[data["month"] <= stop_month]

        # plotting activity per time
        check_dir("../model-reports/synthetic-data/")
        sns.set_theme(style="darkgrid")
        plt.figure(figsize=(20, 6))
        sns.lineplot(x="month", y="st_num_dev", data=data, hue="proj_name", 
                    palette="rocket", legend="auto")

        plt.xlabel("Month")
        plt.ylabel("Number of Social & Technical Devs [Overlap]")
        plt.title(f"Activity vs Time for {strategy.capitalize()} Synthetic Data ({base_proj} in {self.incubator})")
        plt.xticks(rotation=60)

        plt.tight_layout()
        plt.savefig(f"../model-reports/synthetic-data/{strategy}-{self.incubator}-comparison")
        plt.clf()
    
    def distributions(self, outlier_threshold: float=None) -> None:
        """
            Generates a summary of the distributions of a network by feature. 
            Does a project-wise aggregate (i.e. how is each project distributed) 
            as well as an incubator-wide aggregate (i.e. without grouping by 
            project, how does the incubator stack up) and generates a full 
            report.

            Notice that since it's hard to represent an entire distribution for 
            multiple time-series (projects), we'll take the statistics of each 
            feature for each project, then average across all projects for the 
            project-wise statistics.

            @param outlier_threshold: percentile to ignore

            # --- social-features --- #
            - s_num_nodes
            - s_weighted_mean_degree
            - s_num_component
            - s_avg_clustering_coef
            - s_largest_component
            - s_graph_density
            - s_net_overlap

            # --- technical-features --- #
            - t_num_dev_nodes
            - t_num_file_nodes
            - t_num_dev_per_file
            - t_num_file_per_dev
            - t_graph_density
            - t_net_overlap
            
            # --- overlap --- #
            - proj_name
            - month
            - st_num_dev
        """

        # setup report
        output_dir = network_dir / "statistics" / "distributions/"
        check_dir(output_dir)
        report_path = output_dir / f"{self.incubator}-{self.versions['tech']}-{self.versions['social']}"

        # project-wise metrics
        ## group by time-series
        project_wise_data = self.data.groupby("proj_name").agg(["describe"])

        ## aggregate each feature across all projects
        project_wise_data = project_wise_data.mean()
        project_wise_data = project_wise_data.reset_index()

        ## pivot to ensure one row per feature
        project_wise_data.drop(columns="level_1", inplace=True)
        project_wise_data.rename(columns={"level_0": "feature", "level_2": "statistic", 0: "measurement"}, inplace=True)
        project_wise_data = project_wise_data.pivot(index="feature", columns="statistic", values="measurement").reset_index()

        ## export to report
        with open(f"{report_path}.txt", "w") as f:
            f.write("<Project-Wise Aggregation>\n")
            f.write(project_wise_data.to_string(index=False))
            f.write("\n\n")
        project_wise_data.to_csv(f"{report_path}-project-wise.csv", index=False)


        # incubator-wide metrics
        ## select only numeric/useful columns
        incubator_wide_data = self.data.drop(columns=self.drop_cols)

        ## ignore outliers in the distributions
        if outlier_threshold is not None:
            threshold = incubator_wide_data.quantile(
                1 - outlier_threshold,
                numeric_only=True
            )
            incubator_wide_data = incubator_wide_data[incubator_wide_data <= threshold]

        ## describe & report
        incubator_wide_data = incubator_wide_data.describe()

        with open(f"{report_path}.txt", "a") as f:
            f.write("<Incubator-Wide Aggregation>\n")
            f.write(incubator_wide_data.to_string())
            f.write("\n")
        incubator_wide_data.to_csv(f"{report_path}-incubator-wide.csv")

    @staticmethod
    def project_length_distribution(incubators: list[str]=None) -> None:
        """
            Generates a distribution of the specified incubators and their 
            respective project lengths. Uses project incubation lookups for 
            this.
            
            @param incubators (list[str]): names of the incubators to include on 
                the same plot.
        """
        
        # check args
        if incubators is None:
            incubators = params_dict["datasets"]
        
        # load in all incubation timings
        incubation_paths = params_dict["incubation-time"]
        project_lengths = {
            incubator: list(json.load(open(incubation_paths[incubator], "r")).values())
            for incubator in incubators
        }

        # plotting
        ## color picking
        sns.set_style("darkgrid")
        num_colors = len(incubators)
        palette = sns.color_palette("mako", n_colors=num_colors)
        
        ## actual figure
        plt.figure(figsize=(4, 2))
        
        for i, incubator in enumerate(incubators):
            sns.kdeplot(
                project_lengths[incubator], label=incubator, fill=True,
                bw_adjust=0.5, color=palette[i]
            )
        
        ## metadata
        plt.xlabel("Project Length (months)", fontsize=8)
        plt.ylabel("Density", fontsize=8)
        plt.legend(loc="upper right", fontsize = 8)
        plt.grid(axis="y", linestyle="--", alpha=0.9)
        plt.tick_params(axis="both", which="major", labelsize=6)
        
        ## export
        incubator_names = "-".join(incubators)
        output_dir = network_dir / f"statistics" / "project_lengths/"
        check_dir(output_dir)
        
        plt.savefig(
            output_dir / f"{incubator_names}_incubation_distribution.png", 
            dpi=400, bbox_inches="tight"
        )
        plt.show()
        plt.close()
    
    def feature_correlations(self, save_dir: Path | str=None) -> None:
        """
            Generates a correlation matrix of all features. Note, the assumption 
            here is that every timestep's observation is independent, which may 
            lead to misleading correlations downstream. However, there is 
            
            @param save_dir: directory to save in, can be modified for custom 
                trials.
        """
        
        # save dir
        if save_dir is None:
            save_dir = network_dir / "statistics/"
        
        # copy & select out non-numeric features
        df = self.data.copy()
        df.drop(columns=self.drop_cols, inplace=True)
        
        # correlation matrix
        corr_matrix = df.corr()
        high_corr_pairs = list()
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.85:
                    high_corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_matrix.iloc[i, j]
                    ))

        # print report
        log("Highly Correlated Feature Pairs (corr > 0.85)", "new")
        for pair in high_corr_pairs:
            print(f"{pair[0]} - {pair[1]}: {pair[2]:.2f}")

        # heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0
        )
        plt.title("Network Feature Correlation Heatmap")
        plt.tight_layout()
        check_dir(save_dir)
        plt.savefig(Path(save_dir) / f"{self.incubator}_feature_correlations")
        plt.close()

    @staticmethod
    def compare_netdata_timelines(df1: pd.DataFrame, df2: pd.DataFrame, proj_name: str=None) -> None:
        """Plot a grid of lineplots to compare each column from two DataFrames.

        Both DataFrames must have the same columns. This function resets the
        index (assuming it represents time or sequential order) and adds a
        'Source' column to differentiate between the two DataFrames. The data is
        then transformed into long-format and a Seaborn FacetGrid is used to
        plot a separate lineplot for each variable.

        Args:
            df1 (pd.DataFrame): first DataFrame to compare.
            df2 (pd.DataFrame): second DataFrame to compare.
            proj_name (str, optional): project to compare. Defaults to comparing
                all projects as a distribution for each data point.
        """
        
        # ensure columns
        if set(df1.columns) != set(df2.columns):
            raise ValueError("Both DataFrames must have identical columns.")
        
        # divide the data
        if proj_name is not None:
            df1 = df1[df1.proj_name == proj_name]
            df2 = df2[df2.proj_name == proj_name]
        else:
            proj_name = "all projects"
        
        df1_reset = df1.reset_index()
        df2_reset = df2.reset_index()
        df1_reset["source"] = "DataFrame 1"
        df2_reset["source"] = "DataFrame 2"
        
        
        # Combine the DataFrames
        combined = pd.concat([df1_reset, df2_reset], ignore_index=True)
        
        # melt the combined DataFrame into long format each row now represents a
        # single observation for a given variable.
        value_vars = list(df1.columns)
        long_df = pd.melt(
            combined, id_vars=["month", "source"], value_vars=value_vars, 
            var_name="feature", value_name="feature value"
        )
        
        # grid of lineplots
        g = sns.FacetGrid(
            long_df, col="feature", hue="source", sharey=False, col_wrap=3,
            height=ceil(df1.shape[1] / 3), aspect=1.5, palette="viridis"
        )
        g.map(sns.lineplot, "month", "feature value").add_legend(title="source")
        plt.tight_layout()
        
        save_path = Path(params_dict["visuals-dir"]) / "strat-comparison-timelines" / f"{proj_name}.png"
        check_dir(save_path.parent)
        plt.savefig(save_path)


    ## Augmentations
    def jitter_netdata(
        self, entry_prop: float=0.4, delta_prop: float=0.05, 
        proj_subset: list[str]=None, num_cycles: float=1, inplace: bool=True,
        export: bool=False
    ) -> None | pd.DataFrame:
        """
            This program will jitter data for each project and create a duplicate
            for more consistent, and hopefully better, training. The idea is if 
            we change the data by an amount low enough to not deviate away from 
            the base project's label, but an amount high enough to create a 
            distinct example, we effecgively have a new project to train on.

            @param inplace: determines whether to run inplace of the dataset 
                            passed in
            @param export: determines whether to save the dataset generated
            @param entry_prop: percentage of entries in a project to augment by
            @param delta_prop: percentage of each entry's value to randomly shift by
            @param proj_subset: subset of projects to jitter, leaves the rest alone; 
                                defaults to all projects
            @param num_cycles: number of times to jitter the projects specified; 
                                if fractional, jitters that proportion of the proj 
                                subset, chosen at random
        """

        # setup
        log(f"Jittering Network Data", log_type="new")
        log(f"Performing {num_cycles} cycles for {len(proj_subset)} projects")
        jittered_df = self.data.copy()                                          # keep original projects
        ignore_cols = self.drop_cols                                            # don't jitter the months

        if proj_subset is None:
            proj_subset = self.data["proj_name"].unique()

        cols_to_augment = jittered_df.columns.difference(ignore_cols)
        num_cols = len(cols_to_augment)

        # auxiliary function for transform
        def jitter_group(group):
            # setup
            group_augment = group.drop(columns=ignore_cols)
            random.seed(self.rand_seed)
            indices_to_modify = np.random.choice(
                group.shape[0],
                floor(group.shape[0] * entry_prop), 
                replace=False
            )
            log(f"Augmenting {group.name} with :: {floor(group.shape[0] * entry_prop)} :: modded rows")

            # jitter group
            for index in indices_to_modify:
                # delta
                np.random.seed(self.rand_seed)
                rand_aug = np.random.uniform(-delta_prop, delta_prop, size=num_cols)
                group_augment.iloc[index] *= (1 + rand_aug)

            return pd.concat([group[ignore_cols], group_augment], axis=1)
        
        # only jitter necessary projects
        jittered_df = jittered_df[jittered_df["proj_name"].isin(proj_subset)]

        # ensure float type
        jittered_df[cols_to_augment] = jittered_df[cols_to_augment].astype(float)

        # repeat jitter as many times as necessary
        num_jitters = int(num_cycles)
        fractional_jitters = 0 if np.isclose(num_cycles, num_jitters) else num_cycles - num_jitters
        jittered_dfs: list[pd.DataFrame] = [None] * num_jitters

        for j in range(num_jitters):
            # jitter data after duplicating as needed
            jittered_data = jittered_df.groupby("proj_name").apply(jitter_group)
            jittered_data = jittered_data.reset_index(drop=True)

            # add augment tag to projects
            if num_jitters == 1:
                # don't number jitters if not needed
                jittered_data["proj_name"] = jittered_data["proj_name"].apply(lambda x: f"[{x}]-jit")
            else:
                jittered_data["proj_name"] = jittered_data["proj_name"].apply(lambda x: f"[{x}]-jit-{j}")

            # add to concat
            jittered_dfs[j] = jittered_data
        
        # add fractional part
        if fractional_jitters > 0:
            # randomly select projects
            random.seed(self.rand_seed)
            num_to_jitter = int(len(proj_subset) * fractional_jitters)
            sampled_subset = random.sample(proj_subset, num_to_jitter)

            # jitter again
            jittered_data = jittered_df[jittered_df["proj_name"].isin(sampled_subset)].groupby("proj_name").apply(jitter_group)
            jittered_data = jittered_data.reset_index(drop=True)

            # add augment tag to projects
            jittered_data["proj_name"] = jittered_data["proj_name"].apply(lambda x: f"[{x}]-jit-{num_jitters + 1}")

            # track
            jittered_dfs.append(jittered_data)

        # merge
        jittered_dfs = [self.data] + jittered_dfs
        merged = pd.concat(jittered_dfs)

        # export
        if inplace:
            self.data = merged
        else:
            return merged

        if export:
            pass

    def interpolate_netdata(
        self, entry_prop: float=0.15, num_cycles: float=1, inplace: bool=True,
        export: bool=False
    ) -> None | pd.DataFrame:
        """
            Interpolation for means of upsample/oversampling. The idea is we 
            can "remove" a proportion of rows within a project and then 
            interpolate to fix the entries. This preserves the notion of time 
            series within each project and generate some completely new entries. 

            The end result should be a project that has effectively smoothed 
            over the activity vs time for some months of the data, enough to 
            make a relatively unique project, but not enough to deviate from the 
            base target label.

            @param entry_prop: percentage of entries in a project to augment; if 
                set to 1.0, it effectively creates a smoothed curve for activity
                per time.
            @param proj_subset: subset of projects to jitter, leaves the rest alone; 
                                defaults to all projects
            @param num_cycles: number of times to jitter the projects specified; 
                                if fractional, jitters that proportion of the proj 
                                subset, chosen at random
            @param visualize: whether to visualize each base project and its 
                              interpolated augmentations
            @param inplace: determines whether to run inplace of the dataset 
                            passed in
            @param export: determines whether to save the dataset generated
        """

        # 

        pass

    def normalize_netdata(
        self, keep_nan: float=1e-10, strat: str="actdev", inplace: bool=True,
        export: bool=False
    ) -> None | pd.DataFrame:
        """
            This program will normalize network features, project-wise and then 
            feature-wise for better transfer.
            - ZScore normalization
            - Min-Max normalization
            - Active Developers normalization

            This program will normalize network features data by project and feature.
            Normalization will be used in the hopes of achieving a better transfer 
            from ASF to Github, Eclipse, etc. as each of these sources will have 
            their own nuances that can be addressed through relative measures
            instead of absolute ones.

            @param keep_nan: whether to force NaNs to zero or not; adds to every 
                division to avoid undefined numbers.
            @param strat: either active developers (actdev), min-max (minmax), 
                or z-score (zscore) normalized procedure.
            @param inplace: determines whether to run inplace of the dataset 
                passed in.
            @param export: determines whether to save the dataset generated.
        """

        # backwards compatibility
        incubator = self.incubator
        data = self.data
        
        # setup normalizing
        log(f"Normalizing Network Data [{strat}]", log_type="new")
        ignore_cols = self.drop_cols
        cols_to_normalize = data.columns.difference(ignore_cols)

        # routing scalers
        def zscore_normalize(group):
            non_nrm_data = group[ignore_cols]
            nrm_data = group.drop(columns=ignore_cols)

            # normalize
            # nrm_data = StandardScaler().fit_transform(nrm_data)
            # nrm_data = pd.DataFrame(nrm_data, columns=cols_to_normalize)
            nrm_data = (nrm_data - nrm_data.mean()) / (nrm_data.std() + keep_nan)

            # merge & return
            return pd.concat([non_nrm_data, nrm_data], axis=1)
        
        def minmax_normalize(group):
            # split
            non_nrm_data = group[ignore_cols]
            nrm_data = group.drop(columns=ignore_cols)

            # normalize
            nrm_data = (nrm_data - nrm_data.min()) / (nrm_data.max() + keep_nan)

            # merge & return
            return pd.concat([non_nrm_data, nrm_data], axis=1)
        
        def actdev_normalize(group):
            """
                Normalizes by the number of active developers per month.
            """

            # split
            non_nrm_data = group[ignore_cols]
            nrm_data = group.drop(columns=ignore_cols)

            # normalize; sum of both networks in conjunction per month
            # max_devs = (group["t_num_dev_nodes"] + group["s_num_nodes"])
            max_devs = group["st_num_dev"]
            max_devs = max_devs + (max_devs == 0)           # force to one
            nrm_data = nrm_data.div(max_devs, axis=0)

            # merge & return
            return pd.concat([non_nrm_data, nrm_data], axis=1)
        
        normalize_router = {
            "zscore": zscore_normalize,
            "minmax": minmax_normalize,
            "actdev": actdev_normalize
        }

        # normalize
        normalized_data = data.groupby("proj_name").apply(normalize_router[strat]).reset_index(drop=True)

        for col in cols_to_normalize:
            num_missing = normalized_data[col].isna().sum()
            if num_missing > 0:
                log(f"missing from {col}: {num_missing}", log_type="warning")
        
        num_before = normalized_data.shape[0]
        normalized_data.dropna(inplace=True)
        
        num_removed = normalized_data.shape[0] - num_before
        if num_removed > 0:
            log(f"Dropped {num_removed} from {num_before} to {normalized_data.shape[0]} rows", "warning")

        if inplace:
            self.data = normalized_data
        else:
            return normalized_data

        if export:
            pass

    def interval_netdata(
        self, start_month: int=0, spacing: int=1, end_month: int=None, 
        proportion: float=None, backwards: bool=False, inplace: bool=True, 
        export: bool=False
    ) -> None | pd.DataFrame:
        """
            Generates subsets of projects using the first x months of data, where x 
            is determined by the intervaling strategy.

            The end goal is a way to determine performance before the end of a 
            project and even improve performance (with soft probabilities).

            @param start_month: month to start intervaling from
            @param spacing: months in-between each interval
            @param end_month: month to stop at; defaults to project length
            @param proportion: proportion of a project to divide by for 
                proportion intervals; None by default.
            @param backwards: whether predictions are to be made backwards; i.e.
                generates the interval from the end to the start.
            @param inplace: determines whether to run inplace of the dataset 
                            passed in
            @param export: determines whether to save the dataset generated
        """

        # backwards compatability w/ prev versions
        incubator = self.incubator
        data = self.data

        # auxiliary fn
        def get_intervals(group):
            """
                Dispatching function.
            """
            
            # spaced intervals
            if proportion is None:
                return spaced_intervals(start_month, spacing, end_month, group, 
                                        backwards)
            return proportion_intervals(proportion, group, backwards)
            
        def gen_intervals(start_month, end_month, step_month):
            """
                Generate intervals, allowing for easy modularity.
            """

            intervals = list(range(start_month, end_month, step_month))    # stores end months (non-inclusive)
            return intervals

        def spaced_intervals(start_month, step_month, end_month, group, backwards):
            """
                Set of all months from [start_month, ..., all] with a step of step_month 
                size.
            """

            # get range; if the intervals stop before the last month, append the last
            # notice that we pick the first specified months if possible, else all months
            # notice also we end at one before the number of months to control 0-indexing
            max_month = group["month"].max()
            end_month = min(end_month + 1, max_month) if end_month is not None else max_month
            if (end_month - start_month) / step_month <= 1:                 # skip if too few months to interval
                return

            intervals = gen_intervals(start_month, end_month, step_month)  # skip end month

            # we could manually add the entire project data, or we can just duplicate the df and use that
            group_cleaned = group.copy()

            # generate data for each range
            intervals_dfs: list[pd.DataFrame] = []
            for end_month in intervals:
                # generate subset and name it
                subset_df = group_cleaned[group_cleaned["month"] < end_month]
                modified_project = f"[{group.name}]-{end_month}-months"
                subset_df["proj_name"] = modified_project

                # concatenate
                intervals_dfs.append(subset_df)

            intervals_data = pd.concat(intervals_dfs)
            return intervals_data

        def proportion_intervals(proportion, group, backwards):
            """
                Generates proportional intervals based on percentage of a 
                project completed.

                @param proportion: step for each proportion
                @param group: project data
            """
            
            # get divisions
            proj_length = group["month"].max() + 1
            exp_intervals = round(1 / proportion)
            
            # account for edge case with shorter projects
            num_intervals = min(proj_length, exp_intervals)
            step = proj_length / num_intervals
            
            if num_intervals <= 1:
                return pd.DataFrame(columns=group.columns)
            
            # generate intervals while ignoring the last one since we already 
            # have the full data
            intervals = [ceil(i * step) for i in range(1, num_intervals)]
            
            # we'll pad the intervals to ensure that we have eight, but also to 
            # ensure lag intervals at the end of a project appear as 8 steps 
            # rather than making them appear before, i.e. artificially boosting 
            # early interval performance.
            intervals = [0] * (exp_intervals - 1 - len(intervals)) + intervals
            
            # if lag predictions, we're going to need a start of 0 -> end of
            # proj 1
            if backwards:
                intervals = intervals + [proj_length]
                prev_step = 0

            # we could manually add the entire project data, or we can just
            # duplicate the df and use that
            group_cleaned = group.copy()

            # generate data for each range
            intervals_dfs: list[pd.DataFrame] = []
            for i, end_step in enumerate(intervals):
                # generate subset and name it
                subset_df = group_cleaned[group_cleaned["month"] <= end_step]
                modified_project = f"[{group.name}]"
                
                # set lower bound if lag predictions & name
                if backwards:
                    subset_df = subset_df[subset_df["month"] >= prev_step]
                    prev_step = end_step
                    modified_project += f"-{i}-lag-steps"
                else:
                    modified_project += f"-{i}-steps"
                
                subset_df["proj_name"] = modified_project

                # concatenate
                intervals_dfs.append(subset_df)

            intervals_data = pd.concat(intervals_dfs)
            return intervals_data

        def get_base_proj(base_projects: dict, proj: str) -> str:
            """
                Gets the base project given the set of base projects and a 
                string to match.
            """

            # find match
            if proj in base_projects:
                return proj

            potential_base_proj = proj[proj.find('[') + 1: proj.find(']')]
            if potential_base_proj in base_projects:
                return potential_base_proj
            
            # failure means error
            log(f"FAILED to find base project for {proj} using {potential_base_proj}", log_type="error")
            exit(1)

        def check_intervals(incubator: str, mod_data: pd.DataFrame, 
                            start_month: int, step: int, end_month: int=None, 
                            proportion: float=None) -> bool:
            """
                Checks all projects in a given intervaled dataset by comparing the counts
                and number of rows for each pseudo-project. True => passed checks.

                Rather than check all the modified projects, we simply 
            """
            
            # load data
            with open(params_dict["incubation-time"][incubator], "r") as f:
                base_projects = json.load(f)                                        # compare count for each base
            
            mod_unique_projects = set(mod_data["proj_name"].unique()) - set(base_projects.keys())
            mod_counter = dict(mod_data["proj_name"].value_counts())
            mod_projects = dict()               # store counts for each base
            flag = True

            # for every project, check its number of rows and track base
            for proj in tqdm(mod_unique_projects):
                # get expected rows
                base_proj = get_base_proj(base_projects, proj)
                expected_months = int(proj.split('-')[-2])

                # compare
                if expected_months != mod_counter[proj]:
                    log(f"FAILED CHECK @ {proj}, expected {expected_months} but got {(mod_data['proj_name'] == proj).sum()}", 
                            "error")
                    log(f"DEBUG INFO: step @ {step}", "error")
                    log(mod_data[mod_data["proj_name"] == proj], "error")
                    flag = False
                
                # base project counts
                mod_projects[base_proj] = mod_projects.get(base_proj, 1) + 1

            # check all projects appeared
            if set(mod_projects.keys()) != set(base_projects.keys()):
                log("missing the following projects from mod_projects: ", "warning")
                for p in (set(base_projects.keys()) - set(mod_projects.keys())):
                    # fill in projects with only 1 month
                    if mod_counter.get(p, 0) == 1:
                        mod_projects[p] = 1
                    else:
                        print(f"\t- {p}; has {mod_counter.get(p, 'NO')} months, expected {base_projects[p]}")

            # check that the two counts are equal
            for base_proj, base_count in base_projects.items():
                # skip projects with 1 month length
                if base_proj not in mod_projects:
                    continue

                # replace ending month if defined, else all months
                if end_month is not None:
                    base_count = min(end_month, base_count)

                expected_months = ceil((base_count - start_month - 1) / step)
                counted_months = mod_projects[base_proj]
                if expected_months != counted_months:
                    log(f"FAILED COUNTS CHECK, expected {expected_months}, got {counted_months} for {base_proj}", "error")
                    print(f"\t\tDEBUG COUNTS INFO, {base_count=}, {start_month=}, {step=}")
                    print(f"\t\tNOTE: if this prints negative / incorrect expected value, then the issue is related to base_count <= start_month")
                    flag = False
            
            # results
            return flag

        # read in
        log("Generating Intervals for Network Data", log_type="new")
        data_copy = data.copy()

        # setup augmentation
        ignore_cols = ["proj_name"]
        cols_to_interval = data_copy.columns.difference(ignore_cols)
        num_cols = len(cols_to_interval)

        intervaled_data = data_copy.groupby("proj_name").apply(get_intervals)
        intervaled_data = intervaled_data.reset_index(drop=True)

        # exporting
        intervaled_df = pd.concat([data, intervaled_data], ignore_index=True)

        # check intervals
        # if not check_intervals(incubator=incubator, mod_data=intervaled_df, start_month=start_month, step=spacing, end_month=end_month):
        #     _log("Intervals failed to pass tests", "error")
        #     exit(1)

        log(log_type="summary")
        print(f"Num Rows: {data.shape[0]} --> {intervaled_df.shape[0]}")
        bef_projects = data["proj_name"].unique().shape[0]
        aft_projects = intervaled_df["proj_name"].unique().shape[0]
        print(f"Num Projects: {bef_projects} --> {aft_projects}; {aft_projects / bef_projects:.2f}x increase")
        print("done!")

        # export
        if inplace:
            self.data = intervaled_df
        else:
            return intervaled_df
        if export:
            pass

    def aggregate_netdata(
        self, concat: bool=False, inplace: bool=True, export: bool=False
    ) -> None | pd.DataFrame:
        """
            Replaces the feature-set with a cumulative sum per month in hopes of 
            better communicating information about the proportion of commits to the 
            model better.

            NOTE: if running with diff data, will not override diff

            NOTE: if aggregated data fails to train (NaN loss), re-run; may be an 
            issue with saving and loading. :/

            @param concat: determines whether to add new features in addition to 
                           the original or simply override the original
            @param inplace: determines whether to run inplace of the dataset 
                            passed in
            @param export: determines whether to save the dataset generated
        """

        # backwards compatibility
        incubator = self.incubator
        data = self.data
        drop_cols = self.drop_cols

        # ignore diff columns
        drop_cols = drop_cols + [col for col in self.data.columns if "_diff" in col]

        # setup
        log("Aggregate NetData", "new")
        keep_cols = data.columns.difference(drop_cols)
        agg_column_order = list(data.columns)
        concat_column_order = agg_column_order + [f"{k}_agg" for k in keep_cols]

        # define transform utility
        def aggregate_project(project_data: pd.DataFrame) -> pd.DataFrame:
            """
                Utility function to aggregate a given project's data.
            """

            # don't apply to dropped cols
            project_data[keep_cols] = project_data[keep_cols].cumsum()
            return project_data


        # aggregation
        if concat:
            log("concatenating without overriding original features...")
            original = data.copy()
        
        agg_df = data.groupby("proj_name").apply(aggregate_project).reset_index(drop=True)[agg_column_order]

        if concat:
            agg_df.drop(columns=self.drop_cols, inplace=True)
            agg_df.columns = [f"{col}_agg" for col in agg_df.columns]
            agg_df = pd.concat([original, agg_df], axis=1)
            agg_df = agg_df[concat_column_order]

        if inplace:
            self.data = agg_df
        else:
            return agg_df
        if export:
            log("EXPORT FOR `aggregate_netdata` not supported yet", "error")

    def diff_netdata(
        self, concat: bool=False, inplace: bool=True, export: bool=False
    ) -> None | pd.DataFrame:
        """
            Generates a new numeric feature for every current network feature based 
            on the difference of the lagging mean (previous <= three months) and the 
            current month. The intended benefit is to suggest more explicitly to the 
            model a serious downturn.

            @param concat: determines whether to add new features in addition to 
                           the original or simply override the original
            @param inplace: determines whether to run inplace of the dataset 
                            passed in
            @param export: determines whether to save the dataset generated
        """

        # backwards compatability
        incubator = self.incubator
        data = self.data
        drop_cols = self.drop_cols
        
        # ignore agg columns
        drop_cols = drop_cols + [col for col in self.data.columns if "_agg" in col]

        # setup
        log("Diff NetData", "new")
        keep_cols = data.columns.difference(drop_cols)
        diff_column_order = list(data.columns)
        concat_column_order = diff_column_order + [f"{k}_diff" for k in keep_cols]

        # define transform utility
        def diff_project(project_data: pd.DataFrame, n_months: int=3) -> pd.DataFrame:
            """
                Utility function to differentiate a given project's data.
            """

            # to all numeric columns, generate the rolling mean for <= 3 steps behind
            # replace the rolling mean with a difference
            diff_cols = [f"{k}_diff" for k in keep_cols]
            project_data[diff_cols] = (project_data[keep_cols] - project_data[keep_cols].rolling(
                window=n_months,
                min_periods=1
            ).mean().shift(1, fill_value=0))
            project_data = project_data.iloc[n_months:, :]                # select only month 3 onwards

            # generate a difference
            return project_data

        # project-wise diff calculations (lagging strategy)
        if concat:
            log("concatenating without overriding original features...")
            original = data.copy()
        diff_df = data.groupby("proj_name").apply(diff_project).reset_index(drop=True)[diff_column_order]
        
        if concat:
            diff_df.drop(columns=self.drop_cols, inplace=True)
            diff_df.columns = [f"{col}_diff" for col in diff_df.columns]
            diff_df = pd.concat([original, diff_df], axis=1)
            diff_df = diff_df[concat_column_order]

        if inplace:
            self.data = diff_df
        else:
            return diff_df
        if export:
            log("EXPORT FOR `diff_netdata` not supported yet", "error")
            raise NotImplementedError

    def upsample_netdata(
        self, strat: str="jitter", inplace: bool=True, export: bool=False
    ) -> None | pd.DataFrame:
        """
            Upsamples using the specified strategies to ensure an equal balance 
            of both target variables. We'll only jitter the training set to 
            ensure we don't test on synthetic projects.

            @param strat: one of "smote" or "jitter"
            @param inplace: determines whether to run inplace of the dataset 
                            passed in
            @param export: determines whether to save the dataset generated
        """

        # setup
        log(f"upsample Data w/ [{strat}]", "new")
        strat_router = {
            "jitter": self.jitter_netdata,
            "interpolation": self.interpolate_netdata,
            "smote": None
        }

        # ensure completed args
        if self.project_status is None:
            self.load_proj_status()
        if self.split_set is None:
            self.train_test_split()
        if self.projects_set is None:
            self.projects_set = set(self.data["proj_name"].unique())

        # get target labels to gauge number of entries to create
        label_counts = [len(self.project_status["retired"] & self.split_set["train"] & self.projects_set), 
                        len(self.project_status["graduated"] & self.split_set["train"] & self.projects_set)]
        if label_counts[0] == label_counts[1]:
            log(f"no imbalance detected, {label_counts[0]} retired and {label_counts[1]} graduated", "warning")
            return

        labels = ["retired", "graduated"]
        imbalanced_label = np.argmin(label_counts)
        imbalance = label_counts[imbalanced_label - 1] - label_counts[imbalanced_label]
        imbalanced_label = labels[imbalanced_label]
        log(f"tt-split: {dict(zip(labels, label_counts))}")

        # strategy
        project_subset = list(self.split_set["train"] & self.project_status[imbalanced_label] & self.projects_set)
        num_subset = len(project_subset)
        log(f"Balancing for [{imbalanced_label}] projects w/ imbalance of {imbalance} projects and {num_subset} projects to sample from")

        upsampled_df = strat_router[strat](
            proj_subset=project_subset,
            num_cycles=(imbalance / num_subset),                                # number to fill in to balance both
            inplace=False
        )

        # visualize
        self.visualize_synthetic(upsampled_df, base_proj=project_subset[0], strategy=strat)

        # export
        if inplace:
            self.data = upsampled_df
        else:
            return upsampled_df
        
        if export:
            pass

    def downsample_netdata(
        self, max_diff: float=0.1, inplace: bool=True, export: bool=False,
    ) -> None | pd.DataFrame:
        """
            Downsamples the network data to ignore an imbalance. Allows for some 
            leeway if necessary. Notice that we can simply adjust the set of 
            projects we have and then continue as normal with the generation for 
            the netdata.

            @param max_diff: leeway to be granted; given as a proportion, i.e. 
                             the number of projects we can allow the imbalanced 
                             label to be over the undersampled label.
            @param inplace: determines whether to run inplace of the dataset 
                            passed in
            @param export: determines whether to save the dataset generated
        """

        # setup
        log(f"Downsampling Data [{max_diff * 100:.4f}% leeway]", "new")
        random.seed(self.rand_seed)

        # ensure completed args
        if self.project_status is None:
            self.load_proj_status()
        if self.projects_set is None:
            self.projects_set = set(self.data["proj_name"].unique())
        if self.base_projects is None:
            self.base_projects = self.projects_set & (
                set(self.project_status["graduated"]) |
                set(self.project_status["retired"])
            )
        if self.split_set is None:
            self.split_set = self.train_test_split()

        # get target labels to gauge number of entries to create
        label_counts = [len(self.project_status["retired"] & self.projects_set), 
                        len(self.project_status["graduated"] & self.projects_set)]
        if label_counts[0] == label_counts[1]:
            log(f"no imbalance detected, {label_counts[0]} retired and {label_counts[1]} graduated", "warning")
            return

        labels = ["retired", "graduated"]
        minority_label = np.argmin(label_counts)
        imbalance = label_counts[minority_label - 1] - label_counts[minority_label]
        
        majority_label = labels[minority_label - 1]
        minority_label = labels[minority_label]
        log(f"tt-split: {dict(zip(labels, label_counts))}")

        # strategy
        minority_subset = list(self.project_status[minority_label] & self.projects_set)

        majority_sample = ceil(len(minority_subset) * (1 + max_diff))
        majority_subset = list(self.project_status[majority_label] & self.projects_set)
        majority_subset = random.sample(majority_subset, majority_sample)

        downsampled_projects = set(majority_subset) | set(minority_subset)

        log(
            f"Balancing for [{minority_label}] projects w/ imbalance of " \
            f"{imbalance} projects; using ALL {len(minority_subset)} from " \
            f"{minority_label} and ONLY {majority_sample} from " \
            f"{majority_label}"
        )
        downsampled_df = self.data[self.data["proj_name"].isin(downsampled_projects)]

        # export
        if inplace:
            self.data = downsampled_df
        else:
            return downsampled_df
        
        if export:
            pass

    def subset_features(
        self, include: list=None, disclude: list=None, inplace: bool=True, 
        export: bool=False
    ) -> None | pd.DataFrame:
        """
            Takes a subset of the network features to use in training.

            @param include: features to include. Defaults to None.
            @param disclude: features to disclude. Defaults to None.
            @param inplace: determines whether to run inplace of the dataset 
                            passed in
            @param export: determines whether to save the dataset generated

            @returns None | pd.DataFrame: if not inplace, returns the df
        """
        
        # prepare args
        if (include is None) and (disclude is None):
            disclude = ["s_largest_component"]
        if include is None:
            include = list(set(self.data.columns) - set(disclude))
        if disclude is None:
            disclude = list(set(self.data.columns) - set(include))
            
        # filter data
        filtered_df = self.data[include]
            
        # export
        if inplace:
            self.data = filtered_df
        else:
            return filtered_df
        
        if export:
            pass

    def impute_netdata(
        self, impute_cols: list=None, strat: str="spline", threshold: int=3, 
        inplace: bool=True, export: bool=False, verbose: bool=True
    ) -> None | pd.DataFrame:
        """Imputes any holes in the network data by project and feature.
        
        Args:
            impute_cols (list, optional): columns to impute. Defaults to all but
                the ignore cols.
            strat (str, optional): strategy to use to impute the dataset; one of
                {MICE, spline, GAN}. Defaults to spline. Note, if you pass in 
                another strategy it defaults to the built-in pandas interpolate
                method with that strategy specified.
            threshold (int, optional): threshold for the number of consecutive 
                missing months of data to impute. Defaults to 3.
            inplace (bool, optional): determines whether to run inplace of the 
                dataset passed in. Defaults to True.
            export (bool, optional): determines whether to save the dataset 
                generated. Defaults to False.
            verbose (bool, optional): whether to print a summary of the 
                imputation. Defaults to True.

        Returns:
            None | pd.DataFrame: returns a df if not in-place
        """
        
        # aux imputation fns
        def spline_impute(data: pd.DataFrame, threshold: int, strat: str="spline") -> pd.DataFrame:
            """Impute missing values using a spline interpolation.

            Args:
                data (pd.DataFrame): dataset to impute.
                threshold (int, optional): threshold for the number of 
                    consecutive missing months of data to impute. Defaults to 3.

            Returns:
                pd.DataFrame: imputed dataset.
            """
            
            # infer 0's as NaNs
            data.replace(0, np.nan, inplace=True)
            
            # interpolate via spline (or custom fn)
            data.interpolate(method=strat, limit=threshold, axis=0, inplace=True)
            
            # re-infer NaNs as 0's
            data.fillna(0, inplace=True)
            
            # export
            return data
        
        def mice_impute(data: pd.DataFrame, threshold: int) -> pd.DataFrame:
            """Imputes the dataframe using the multi-variate chained equations 
            technique via decision tree based models.

            Args:
                data (pd.DataFrame): data to impute.

            Returns:
                pd.DataFrame: imputed dataframe.
            """
            
            # infer as a float array
            imputed_data = data.copy().astype(float)
            n_rows, n_cols = imputed_data.shape

            # mask for all chains of missing values of a valid length <= 
            # threshold
            mask = np.zeros_like(imputed_data, dtype=bool)

            # range each column and determine individually
            for j in range(n_cols):
                # unpack
                col = imputed_data[:, j]
                start = None
                
                # build chains of zeros
                for i in range(n_rows):
                    # start/continue chain
                    if col[i] == 0:
                        if start is None:
                            start = i
                    
                    # end chain if possible
                    else:
                        if start is not None:
                            chain_length = i - start
                            if chain_length <= threshold:
                                mask[start:i, j] = True
                            start = None

                # chain reaches the end of the col
                if start is not None:
                    chain_length = n_rows - start
                    if chain_length <= threshold:
                        mask[start:n_rows, j] = True

            # working copy with nan's over 0's
            impute_data = imputed_data.copy()
            impute_data[mask] = np.nan

            # iteratively impute via MICE
            imputer = IterativeImputer(self.rand_seed)
            imputed_result = imputer.fit_transform(impute_data)

            # only change the data points that don't exceed the threshold
            final_result = imputed_data.copy()
            final_result[mask] = imputed_result[mask]
            return final_result
        
        # routing
        impute_fn = {
            "spline": lambda x: spline_impute(x, threshold, "spline"),
            "mice": lambda x: mice_impute(x, threshold)
        }.get(strat, lambda x: spline_impute(x, threshold, strat))
        
        # impute on the copy and ensure column order
        data_copy = self.data.copy()
        data_copy[impute_cols] = impute_fn(data_copy[impute_cols])

        data_copy = data_copy[self.column_order]
        
        # determine the export strategy
        if export:
            raise NotImplementedError
        
        if inplace:
            self.data = data_copy
        return data_copy
    
    def smoothe_netdata(
        self, cols: list=None, strat: str="gauss", inplace: bool=True, 
        export: bool=False, verbose: bool=True
    ) -> None | pd.DataFrame:
        """Imputes any holes in the network data by project and feature.
        
        Args:
            cols (list, optional): columns to impute. Defaults to all but the 
                ignore.
            strat (str, optional): strategy to use to impute the dataset; one of
                {exp (exponential smoothing), wav (wavelet transform)}. Defaults
                to exp.
            inplace (bool, optional): determines whether to run inplace of the 
                dataset passed in. Defaults to True.
            export (bool, optional): determines whether to save the dataset 
                generated. Defaults to False.
            verbose (bool, optional): whether to print a summary of the 
                imputation. Defaults to True.

        Returns:
            None | pd.DataFrame: returns a df if not in-place
        """
        
        # smoothing aux fns
        def wavelet_smoothing(data: np.ndarray, wavelet: str="db4", level: int=1) -> np.ndarray:
            """Smoothing using the wavelet transform.

            Args:
                data (np.ndarray): column of data to smooth.
                wavelet (str, optional): wavelet type. Defaults to "db4".
                level (int, optional): level of detail coeffs to use for the 
                    de-noising protocol. Defaults to 1.

            Returns:
                np.ndarray: transformed data reconstructed from the core waves.
            """
            
            # constants
            median_abs_deviation_to_std = 1 / 0.6745
            
            # fit the wavelets onto the data
            coeff = pywt.wavedec(data, wavelet, mode="per")
            max_level = pywt.dwt_max_level(len(data), pywt.Wavelet("db4").dec_len)
            level = min(level, max_level)
            
            # estimate stddev
            sigma = median_abs_deviation_to_std * np.median(
                np.abs(coeff[-level] - np.median(coeff[-level]))
            )
            
            # universal threshold calculation
            universal_thresh = sigma * np.sqrt(2 * np.log(len(data)))
            universal_thresh = 1 if universal_thresh == 0.0 else universal_thresh
            coeff[1:] = (
                pywt.threshold(i, value=universal_thresh, mode="soft")
                for i in coeff[1:]
            )
            
            # apply the threshold and reconstruct the data
            return pywt.waverec(coeff, wavelet, mode="per")[:data.shape[0]]

        def exp_smoothing(data: np.ndarray, alpha: float=0.2) -> np.ndarray:
            """Smooth data using simple exponential smoothing.

            This function applies simple exponential smoothing to the input
            data. Computes a weighted average of past observations, w/ weights
            exponentially decaying with time. Higher alpha discounts older
            observations faster, giving more emphasis to recent data.

            Args:
                data (np.ndarray): col of data to smooth.
                alpha (float, optional): smoothing factor between 0 (exclusive)
                    and 1 (inclusive). A higher value gives more weight to
                    recent observations. Defaults to 0.2.

            Returns:
                np.ndarray: The exponentially smoothed data.
            """
            
            # validate alpha
            if not (0 < alpha <= 1):
                raise ValueError("alpha must be in the interval (0, 1].")

            # initialize smoothed col
            smoothed = np.empty_like(data)
            smoothed[0] = data[0]

            # recursively smooth via exp fn
            for i in range(1, len(data)):
                smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]

            # export
            return smoothed

        def gauss_kernel(data: np.ndarray, sigma: float=0.47) -> np.ndarray:
            """Gaussian kernel smoothing for time series.

            Args:
                data (np.ndarray): single column of data.
                sigma (float, optional): decay of the observations to consider.
                    Defaults to 0.47.

            Returns:
                np.ndarray: smoothed data.
            """
            
            # gen Gaussian kernel, cover the 99.7% conf interval for the kernel
            kernel_size = int(6 * sigma + 1)
            
            # ensure odd kernel size
            kernel_size += kernel_size % 2
            
            # if the kernel is larger than the data, we can't really smooth it
            if kernel_size > len(data):
                return data
            
            # create the kernel
            kernel_range = np.arange(-(kernel_size // 2), (kernel_size // 2) + 1)
            kernel = np.exp(-0.5 * (kernel_range / sigma) ** 2)
            kernel /= kernel.sum()
            
            # convolve across the time series
            smoothed = np.convolve(data, kernel, mode="same")
            
            # ensure equal size
            smoothed = smoothed[:data.shape[0]]
            
            # export
            return smoothed

        # infer args
        if cols is None:
            cols = self.data.columns.difference(self.drop_cols)
        smooth_fn = {
            "exp": exp_smoothing,
            "wav": wavelet_smoothing,
            "gauss": gauss_kernel
        }[strat]
        
        data_copy = self.data.copy()

        # smoothing
        for col in cols:
            data_copy[col] = data_copy.groupby("proj_name")[col].transform(
                lambda x: smooth_fn(x.values)
            )

        # exporting and finish
        if inplace:
            self.data = data_copy
        else:
            return data_copy

        if export:
            raise NotImplementedError

    def combine_options(self) -> None:
        """
            Allows for new column generation via concatenating two different 
            sets of options together.

            @param
        """

        raise NotImplementedError


# Scripting & Testing
def __nd_main(args_dict: dict[str, Any]) -> None:
    match args_dict.get("script", "tse"):
        case "tse":
            TSE_INCS = ["apache", "github", "eclipse", "osgeo"]
            
            NetData.project_length_distribution(
                incubators=TSE_INCS
            )
            
            for inc in TSE_INCS:
                nd = NetData(inc, do_compute_tensors=False)
                nd.distributions()
                nd.feature_correlations()
        
        case _:
            raise ValueError(f"Script {args_dict['script']} does not have an implementation / DNE")


if __name__ == "__main__":
    args_dict = parse_input(sys.argv)
    __nd_main(args_dict=args_dict)
    
    # # load all incubators
    # nd = NetData("apache", do_compute_tensors=False)
    # nd_s = NetData("apache", options={"smooth": True}, do_compute_tensors=False, ignore_cache=True)
    
    # NetData.compare_netdata_timelines(nd.data, nd_s.data, "spark")
    
    # print(nd_b.split_set["train"] & nd.split_set["test"])
    # print({len(s) for _, s in nd.split_set.items()})
    # print({len(s) for _, s in nd_b.split_set.items()})
        
    # nd.interval_netdata(proportion=1/8)
    # nd.aggregate_netdata()
    # print(nd.data)
    # print(nd.data.sort_values(by=["proj_name", "month"]))

    # nd = {s: NetData(s, test_prop=0.0, is_train="train") for s in ["apache", "github", "eclipse"]}
    # nd = {k: n.upsample_netdata(inplace=True) for k, n in nd.items()}
    # nd = {n.data.to_csv(f"../network-data/netdata/upsampled-{k}-network-data-{n.versions['tech']}-{n.versions['social']}.csv") for n, k in nd.items()}
