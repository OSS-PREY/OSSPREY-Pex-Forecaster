"""
    @brief defines the network data class for easily interfacing with the 
        post-processed dataset. Primary utility comes from bundling all 
        relevant information in one object.
    @author Arjun Ashok (arjun3.ashok@gmail.com)
    @acknowledgements Nafiz I. Khan, Dr. Likang Yin
    @creation-date January 2024
    @version 0.1.0
"""

# Imports
import pandas as pd
import numpy as np

import json
import re
from typing import Any, Iterator
from dataclasses import dataclass, field
from math import ceil

from decalfc.utils import *
from decalfc.abstractions.tsmodel import *
from decalfc.abstractions.netdata import *


# Class
@dataclass(slots=True)
class ModelData:
    # data
    transfer_strategy: str = field(default="A --> G")                           # how to train the model
    transform_kwargs: dict[str, Any] = field(default=None)                      # transform arguments to forward to NetData
    versions: dict[str, dict[str, dict[str, str]]] = field(                     # { train/test: { incubator: { tech/social: version } } }
        default_factory=dict
    )
    options: dict[str, dict[str, dict[str, bool]]] = field(                     # { train/test: { incbubator: { option: selection } } }
        default_factory=dict
    )
    is_interval: dict[str, bool] = field(
        default_factory=dict
    )                                                                           # { train/test: intervaled or not }
    soft_prob: bool = field(default=False)                           
    test_props: dict[str, float] = field(                                       # { incubator: test_prop }
        default_factory=dict
    )
    drop_cols: list[str] = field(default_factory=lambda: [                      # what to not include in training
        "proj_name", 
        "month"
    ])
    tensors: dict[str, dict[str, list[Any]]] = field(                           # { train/test: { x/y: list[tensors] } }
        init=False,
        repr=False
    )
    predict_project: dict[str, str] = field(default=None)                       # if a single project is to be predicted on, maps incubator: project-name
    rand_seed: int = field(default=42)                                          # seed for reproducability
    
    skip_tensors: bool = field(default=False)                                   # skip tensor generation

    # internal utility
    def _list_options_(self) -> None:
        """
            Prints the final interpretation of the options selected in the 
            transfer strategy in human-readable English.
        """

        # generate
        log("Decoded Transfer Strategy", "new")
        log(f"Original: `{self.transfer_strategy}`")
        for t, o_info in self.options.items():
            print(f"Options selected for {t} set:")
            
            for i, options in o_info.items():
                options_str = f"{', '.join(options)}"
                options_str = "no augmentations" if len(options_str) == 0 else options_str
                versions_str = (
                    f"tech: {self.versions[t][i]['tech']}, "
                    f"social: {self.versions[t][i]['social']}"
                )
                
                if t == "train":
                    log((
                        f"\t{i} dataset, version {versions_str} with {options_str} "
                        f"using {(1 - self.test_props[i]) * 100:.2f}% of the "
                        "data reserved for training"
                    ))
                else:
                    log((
                        f"\t{i} dataset, version {versions_str} with {options_str} "
                        f"using {self.test_props[i] * 100:.2f}% of the data "
                        "reserved for testing"
                    ))

    def _decode_strat_(self) -> None:
        """
            Decodes the strategy token into a lookup of data for train and 
            test sets in the form of the versions and options dictionaries.

            Decode strategy:
                - `[A/G/E]` refers to the incubator
                - `#-#` refers to the version number; defaults to 0-0
                - `^` means train // not needed
                - `^^` means test // not needed
                - `*` means intervaled
                - `-->` refers to the division between train and test
                - ` + ` separates datasets
                    # - `-` defines the start of the options ==> DEPRECATED FOR NOW
                === ex) `A-1-1 + G-1-1^ --> G-1-1^^*`
        """

        # setup
        params = params_dict
        possible_options = params["network-aug-shorthand"]
        decoder = dict(zip(
            params["abbreviations"].values(),
            params["abbreviations"].keys()
        ))
        self.versions = dict(zip(["train", "test"], [dict(), dict()]))
        self.options = dict(zip(["train", "test"], [dict(), dict()]))

        # tokenize
        processed_str = re.sub(r"\s*\+\s*", " ", self.transfer_strategy)    # remove adds
        train_prop_str, test_prop_str = processed_str.split("-->")
        processed_str = re.sub(r"\^", "", processed_str)                    # remove ticks
        train_str, test_str = processed_str.split("-->")                    # split into train and test
        
        train_prop_tokens = ([
            s for s in train_prop_str.split(" ")
            if any(c.upper() in decoder for c in s)
        ])  # generate tokens for train proportions
        test_prop_tokens = ([
            s for s in test_prop_str.split(" ") \
            if any(c.upper() in decoder for c in s)
        ])  # generate tokens for test proportions

        train_strat_tokens = train_str.split()
        test_strat_tokens = test_str.split()

        # generate test props using ticks (0 -- all, any -- 0.20)
        for token in train_prop_tokens:
            if "^" in token and token:
                self.test_props[decoder[token[0].upper()]] = float(params_dict["test-prop"])
            else:
                self.test_props[decoder[token[0].upper()]] = 0

        for token in test_prop_tokens:
            if "^" not in token and token:
                self.test_props[decoder[token[0].upper()]] = 1.00

        # generate options
        def gen_options(tokens: list[str], versions: dict[str, dict[str, str]], options: dict[str, dict[str, bool]]):
            """
                Given references to the final locations, decomposes the tokens 
                into code-readable format.
            """

            # parse tokens
            for token in tokens:
                # branch if versions specified
                if bool(re.search(r"\d", token)):
                    # get info
                    incubator, tech_num, pkg = token.split("-")
                    pkg = re.match(r"(\d+)(.*)", pkg).group
                    social_num = pkg(1)
                    sel_options = pkg(2)
                
                # otherwise default versions
                else:
                    incubator = token[0]
                    default_versions = params["default-versions"][
                        decoder[incubator]
                    ]
                    tech_num, social_num = default_versions[0], default_versions[1]
                    sel_options = token[1:]

                options_list = [possible_options[o] for o in sel_options]

                # implant info
                versions[decoder[incubator]] = {"tech": tech_num, "social": social_num}
                options[decoder[incubator]] = dict(zip(options_list, [True] * len(options_list)))
        
        gen_options(train_strat_tokens, self.versions["train"], self.options["train"])
        gen_options(test_strat_tokens, self.versions["test"], self.options["test"])

        # report
        self._list_options_()

    def _gen_tensors_(self) -> None:
        """
            Generate necessary tensors for strategy by picking NetData's, one at
            a time (avoids memory usage being *too* high).
        """

        # setup
        log("Generating Tensors for Model Data", "new")
        t_keys = ["train", "test"]
        d_keys = ["x", "y"]
        self.tensors = {t: {d: None for d in d_keys} for t in t_keys}
        
        # iterate all train/test
        for t, v_info in self.versions.items():
            # reference
            o_info = self.options[t]

            # iterate all incubators within train/test
            for i, versions in v_info.items():
                # reference
                options = o_info[i]

                # load NetData
                log(f"Tensor for {i} for {t}", "new")

                subset_project = None
                if self.predict_project is not None:
                    if i in self.predict_project:
                        subset_project = {"test": {self.predict_project[i]}}

                cur_nd = NetData(
                    incubator=i,
                    versions=versions,
                    options=options,
                    test_prop=self.test_props[i],
                    split_set=subset_project,
                    is_train=t,
                    rand_seed=self.rand_seed,
                    transform_kwargs=self.transform_kwargs,
                    soft_prob=self.soft_prob
                )

                # add tensors to existing tensor list notice for training
                # tensors, we don't care about month by month performance unless
                # a soft proba model is used and thus can treat them as normal
                # projects
                
                ## testing on intervals
                if cur_nd.is_intervaled and t == "test":
                    if self.tensors[t]["x"] is None:
                        self.tensors[t]["x"] = dict()
                        self.tensors[t]["y"] = dict()
                    
                    for m in cur_nd.tensors[t]["x"]:
                        if m not in self.tensors[t]["x"]:
                            self.tensors[t]["x"][m] = list()
                            self.tensors[t]["y"][m] = list()
                        self.tensors[t]["x"][m].extend(cur_nd.tensors[t]["x"][m])
                        self.tensors[t]["y"][m].extend(cur_nd.tensors[t]["y"][m])
                
                ## soft probability training needed
                elif cur_nd.is_intervaled and t == "train" and self.soft_prob:
                    ### grab training data
                    if self.tensors[t]["x"] is None:
                        self.tensors[t]["x"] = dict()
                    
                    for m in cur_nd.tensors[t]["x"]:
                        if m not in self.tensors[t]["x"]:
                            self.tensors[t]["x"][m] = list()
                        self.tensors[t]["x"][m].extend(cur_nd.tensors[t]["x"][m])
                    
                    ### grab soft probs
                    soft_probs = self.soft_proba(
                        train_strat=self.transfer_strategy,
                        X=self.tensors[t]["x"]
                    )
                    
                    ### fill in data
                    self.tensors[t]["y"] = soft_probs
                
                ## non-intervaled data or soft-probs not requested
                else:
                    if self.tensors[t]["x"] is None:
                        self.tensors[t]["x"] = list()
                        self.tensors[t]["y"] = list()
                    self.tensors[t]["x"].extend(cur_nd.tensors[t]["x"])
                    self.tensors[t]["y"].extend(cur_nd.tensors[t]["y"])
                
        ## adjust train intervals to be pseudo projects and not monthly 
        ## divisions
        if isinstance(self.tensors["train"]["x"], dict):
            ### store
            X_concat = list()
            y_concat = list()
            
            ### ensure the correct ordering for target and train pairs
            for timestep, step_data in self.tensors["train"]["x"].items():
                X_concat += step_data
                y_concat += self.tensors["train"]["y"][timestep]
            
            ## re-set
            self.tensors["train"]["x"] = X_concat
            self.tensors["train"]["y"] = y_concat
        
        # done
        return

    def __post_init__(self):
        # generate lookups
        self._decode_strat_()
        first_test_proj = list(self.options["test"].keys())[0]
        first_train_proj = list(self.options["train"].keys())[0]
        
        # interval inference
        self.is_interval["test"] = any(
            (("interval" in opt) and sel) 
            for opt, sel in self.options["test"][first_test_proj].items()
        )
        self.is_interval["train"] = any(
            (("interval" in opt) and sel) 
            for opt, sel in self.options["train"][first_train_proj].items()
        )

        # generate tensors
        if not self.skip_tensors:
            self._gen_tensors_()

    # external utility
    @staticmethod
    def soft_proba(train_strat: str, X: dict[str, list[torch.Tensor]]) -> dict[str, list[torch.Tensor]]:
        """
            Generates the soft probabilities for a given project across all of 
            its months.

            @param train_strat (str): how to train the model to do soft 
                probabilities on.
            @param X (dict[str, torch.tensor]): project data, month: 0...i month 
                tensor; to run on a single project, simply convert the 
                torch.tensor to a list of size 1 (done automatically as well)

            @returns dict[str, torch.tensor]: month: soft probability for that 
                month
        """
        
        # ensure args & setup
        print("\n\n< ::::::::::::::::::: SOFT PROBS ::::::::::::::::::: >")
        if not isinstance(next(iter(X.items()))[1], list):
            X = {timestep: [step_data] for timestep, step_data in X.items()}
        
        # train the appropriate model
        soft_prob_model = TimeSeriesModel(
            model_arch="BLSTM",
            strategy=train_strat
        )
        
        ## ensure we're not doing soft probs for the underlying soft prob model
        train_strat = train_strat.replace("*", "")
        
        ## attempt greedy load
        if not soft_prob_model.load(strategy=train_strat):
            soft_prob_model_data = ModelData(
                transfer_strategy=train_strat,
                soft_prob=0
            )
            soft_prob_model.train(
                md=soft_prob_model_data
            )
        
        # generate soft probs
        ## store the soft probailities
        soft_probs = dict()
        
        ## iterate each month's data
        for interval_step, step_data in X.items():
            ## setup tracker
            soft_probs[interval_step] = list()
            
            ## iterate each project's interval step data
            for proj_data in step_data:
                ### gen soft probs
                soft_prob_dict = {interval_step: proj_data}
                soft_prob_dict = soft_prob_model.soft_probs(
                    interval_data=soft_prob_dict
                )
                
                ### ensure integer soft probs; we simply want to know when the 
                ### project should be succeeding vs failing, i.e. even though 
                ### not ideal, we can train on hard probabilities but generated 
                ### in a soft-prob manner
                # soft_prob_dict[interval_step] = ([
                #     torch.round(prob) for prob in soft_prob_dict[interval_step]
                # ])

                ### ensure both labels accounted for; note the failure label is 
                ### 0 so it goes first
                # p_success = soft_prob_dict[interval_step].cpu().numpy()[0]
                # p_failure = 1 - p_success
                # soft_prob_dict[interval_step] = torch.tensor(np.array([
                #     p_failure,
                #     p_success
                # ]))
                
                ### update final results
                soft_probs[interval_step].append(soft_prob_dict[interval_step])
        
        # return the final result
        print("< ::::::::::::::::::: END SOFT PROBS ::::::::::::::::::: >\n\n")
        return soft_probs
    
    def soft_probabilities(
        self, train_strat: str, train_options: dict[str, dict[str, bool]]
    ) -> dict[str, list[torch.Tensor]]:
        """
            Generates soft-probabilities to every incubator used in the given 
            strategy selected. To ensure Kosher-ness, i.e. avoid generating 
            soft-probabilities on data a model has been trained on, we'll use 
            the following strategy:
                

            @param train_options (dict[str, dict[str, bool]]): dictionary lookup 
                of just the training options to determine how to train the soft 
                proba model for each incubator.

            @returns dict[str, list[Any]]: dictionary lookup of month: list of 
                tensor ground truths, now soft probabilities instead of hard 
                outcomes.
        """
        
        raise NotImplementedError

    @staticmethod
    def gen_k_folds(
        transfer_strategy: str, transform_kwargs: dict[str, Any]=None,
        nfolds: int=5
    ) -> Iterator:
        """Generates modeldata's to iterate through for model training.

        Args:
            transfer_strategy (str): transfer strategy to use.
            transform_kwargs (dict[str, Any], optional): augmentations to use on
                the data. Defaults to None.
            nfolds (int, optional): number of folds to generate. Defaults to 5
                to preserve the 80-20 TT-split.

        Yields:
            Iterator[ModelData]: iterator of model data objects to use for 
                training. Forgive me for how janky the following code will be.
        """
        
        # build the ModelData with all of the data aggregated; we specify no 
        # train/test split for the model data to ensure we load all of the data
        temp_transfer_strategy = transfer_strategy.replace("^", "")
        full_md = ModelData(
            transfer_strategy=temp_transfer_strategy,
            transform_kwargs=transform_kwargs,
            skip_tensors=True
        )
        train_incubators = set(full_md.options["train"].keys())
        test_incubators = set(full_md.options["test"].keys())
        
        # check we're not using kfolds on any banned augmentations
        if any("downsampled" in opts for opts in full_md.options["train"].values()):
            raise ValueError(
                "Cannot use downsampling/any augmentations to the base projects set with kfolds"
            )
        
        # find constant and dynamic incubators for the strategy
        const_incubators = train_incubators ^ test_incubators
        dynamic_incubators = train_incubators & test_incubators
        
        # create const dataset tensor lookup
        const_ds = {"train": {"x": list(), "y": list()}, "test": {"x": list(), "y": list()}}
        
        for incubator in const_incubators:
            ## unpack options
            cur_set = "train" if incubator in train_incubators else "test"
            
            ## load the netdata for this incubator
            const_nd = NetData(
                incubator=incubator,
                versions=full_md.versions[cur_set][incubator],
                options=full_md.options[cur_set][incubator],
                is_train="both",
                transform_kwargs=transform_kwargs,
                rand_seed=full_md.rand_seed,
                verbose=False
            )
            
            ## track the tensors into the constant ds
            const_ds[cur_set]["x"].extend(const_nd.tensors["train"]["x"])
            const_ds[cur_set]["y"].extend(const_nd.tensors["train"]["y"])
            
            const_ds[cur_set]["x"].extend(const_nd.tensors["test"]["x"])
            const_ds[cur_set]["y"].extend(const_nd.tensors["test"]["y"])
            
        # create dynamics sets
        dynamic_ds = {"train": {"x": dict(), "y": dict()}, "test": {"x": dict(), "y": dict()}}
        
        for incubator in dynamic_incubators:
            ## unpack options
            cur_set = "train"
            
            ## load the netdata for this incubator
            dynamic_nd = NetData(
                incubator=incubator,
                versions=full_md.versions[cur_set][incubator],
                options=full_md.options[cur_set][incubator],
                is_train="both",
                transform_kwargs=transform_kwargs,
                rand_seed=full_md.rand_seed,
                verbose=False
            )
            
            ## track the tensors into the dynamic ds
            dynamic_ds[cur_set]["x"][incubator] = list()
            dynamic_ds[cur_set]["y"][incubator] = list()
            
            dynamic_ds[cur_set]["x"][incubator].extend(dynamic_nd.tensors["train"]["x"])
            dynamic_ds[cur_set]["y"][incubator].extend(dynamic_nd.tensors["train"]["y"])
            
            dynamic_ds[cur_set]["x"][incubator].extend(dynamic_nd.tensors["test"]["x"])
            dynamic_ds[cur_set]["y"][incubator].extend(dynamic_nd.tensors["test"]["y"])
        
        # figure out the fold information
        nprojs_per_inc = {incubator: len(dynamic_ds["train"]["y"][incubator]) for incubator in dynamic_incubators}
        nprojs_per_fold = {incubator: int(nprojs / nfolds) for incubator, nprojs in nprojs_per_inc.items()}
        
        # iterate dynamic set folds, subset df each time for the new fold, add
        # const data, make model data, yield the result; we'll avoid trying
        # every combination of folds to avoid an O(x^n) problem with n
        # incubators and x projects; rather, we'll iterate each fold in parallel
        # which sacrifices on the folding structure but gains performance and
        # simplicity
        for f in range(nfolds):
            # unpack the fold ranges for this fold
            fold_ranges = {
                incubator: (
                    f * nprojs_per_fold[incubator],
                    (f + 1) * nprojs_per_fold[incubator] if f != nfolds - 1
                    else nprojs_per_inc[incubator]
                ) for incubator in dynamic_incubators
            }
            
            # dynamic sets for the current fold
            dynamic_tensors = {
                "train": {"x": list(), "y": list()},
                "test": {"x": list(), "y": list()}
            }
            
            for dyn_inc in dynamic_incubators:
                # unpack fold range
                fold_range = fold_ranges[dyn_inc]
                
                # grab the subset of data we'll use for this fold
                test_fold_x = dynamic_ds["train"]["x"][dyn_inc][fold_range[0]:fold_range[1]]
                test_fold_y = dynamic_ds["train"]["y"][dyn_inc][fold_range[0]:fold_range[1]]
                
                train_fold_x = (
                    dynamic_ds["train"]["x"][dyn_inc][0:fold_range[0]] 
                    + dynamic_ds["train"]["x"][dyn_inc][fold_range[1]:]
                )
                train_fold_y = (
                    dynamic_ds["train"]["y"][dyn_inc][0:fold_range[0]] 
                    + dynamic_ds["train"]["y"][dyn_inc][fold_range[1]:]
                )
                
                # append to the tensors for this fold
                dynamic_tensors["test"]["x"].extend(test_fold_x)
                dynamic_tensors["test"]["y"].extend(test_fold_y)
                
                dynamic_tensors["train"]["x"].extend(train_fold_x)
                dynamic_tensors["train"]["y"].extend(train_fold_y)
            
            # combine with the static set of tensors
            dynamic_tensors["train"]["x"].extend(const_ds["train"]["x"])
            dynamic_tensors["train"]["y"].extend(const_ds["train"]["y"])
            
            dynamic_tensors["test"]["x"].extend(const_ds["test"]["x"])
            dynamic_tensors["test"]["y"].extend(const_ds["test"]["y"])

            # create model data instance to yield
            cur_md = ModelData(
                transfer_strategy=transfer_strategy,
                transform_kwargs=transform_kwargs,
                skip_tensors=True
            )
            cur_md.tensors = dynamic_tensors
            
            # generator
            yield cur_md
            
        # done
        return

# Testing
if __name__ == "__main__":
    ss = "A --> G"
    # md = ModelData(transfer_strategy=ss)
    it = ModelData.gen_k_folds(ss)
    
