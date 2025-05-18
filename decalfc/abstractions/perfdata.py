"""
    @brief Abstraction for the storage of bulk performance data lookup.
    @author Arjun Ashok (arjun3.ashok@gmail.com)
    @creation-date February 2024
    @version 0.1.0
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

import os
import re
import sys
import datetime as dt
from pathlib import Path
from warnings import catch_warnings, filterwarnings
from typing import Any, Optional, Iterable
from dataclasses import dataclass, field
from functools import reduce
from itertools import permutations
from math import ceil

from decalfc.utils import *


# Auxiliary Functions
def icse_25_experiments():
    """
        Wraps the 3 primary experiments for the ICSE '25 paper.
    """
    
    # read in
    perf_db = PerfData("./model-reports/icse-trials/final_icse_perf_db")
    
    # subset breakdowns
    ## exp 1
    perf_db.subset_breakdown(
        options="cbn",
        metrics=[("accuracy", "accuracy"), 
        ("macro avg", "f1-score"), 
        ("macro avg", "precision"), 
        ("macro avg", "recall")], 
        strict=True
    )
    
    ## exp 2
    perf_db.subset_breakdown(
        options="cbna",
        metrics=[("accuracy", "accuracy"), 
        ("macro avg", "f1-score"), 
        ("macro avg", "precision"), 
        ("macro avg", "recall")], 
        strict=True
    )
    
    ## exp 3
    perf_db.subset_breakdown(
        options="cbnd",
        metrics=[("accuracy", "accuracy"), 
        ("macro avg", "f1-score"), 
        ("macro avg", "precision"), 
        ("macro avg", "recall")], 
        strict=True
    )
    
    ## all experiments
    perf_db.summary(metrics=["accuracy", "f1-score", "precision", "recall"], 
                    export=True, verbose=False) 


# Class
@dataclass(slots=True)
class PerfData:
    # data
    perf_source: str = field(
        default="./model-reports/databases/performance_db"
    )                                                                           # storage location
    data: pd.DataFrame = field(init=False, repr=False)                          # database
    
    # inferred members
    ext: str = field(default="parquet")                                         # type to use
    initialize_time: dt.datetime = field(default_factory=(                      # initialize timestamp for ensuring only most recent trials are considered for performance graphs
        lambda: dt.datetime.now() - dt.timedelta(seconds=1)
    ))

    # internal utility
    def _load_data(self) -> None:
        """
            Loads in the performance data in place.
        """

        # load
        reader = {
            "parquet": pd.read_parquet,
            "csv": pd.read_csv
        }

        try:
            self.data = reader[self.ext](f"{self.perf_source}.{self.ext}")
        except FileNotFoundError as fe:
            self.data = pd.DataFrame(columns=[
                "date",
                "transfer_strategy",
                "model_arch",
                "month",
                "label",
                "metric",
                "perf",
                "support"
            ])
            self.export()

    def _get_colors(self, n):
        """
            Custom color map with n colors.
        """

        # generate random colors
        cmap = plt.get_cmap("plasma", n * 2)
        colors = [cmap(i / n) for i in np.arange(0, n)]
        sns.set_palette(sns.color_palette("rocket", n))

        return colors
    
    def _order_intervals(self, interval: str | int) -> float:
        """
            Custom ordering for interval lengths.
        """

        # ensure order
        if interval == "all":
            return float("inf")  # "all" should be treated as the last entry
        if isinstance(interval, int):
            return float(interval)
        return float(interval.split("-")[0])
    
    def __post_init__(self):
        # load
        self._load_data()

    # external utility
    def export(self, path: Path | str="") -> None:
        """
            Exports the data (in the case that updates were made).

            CAUTION: overwrites unless otherwise specified
        """
        
        # ensure directory
        save_path = f"{path}.{self.ext}" if path != "" else f"{self.perf_source}.{self.ext}"
        check_dir(Path(save_path).parent)

        # save
        match self.ext:
            case "csv":
                self.data.to_csv(save_path, index=False)
            
            case "parquet":
                self.data.to_parquet(save_path, index=False)

            case _:
                log("failed to match extension to writer; defaulting to csv", "warning")
                self.data.to_csv("./model-reports/TEMP_SAVE_FOR_PERF_DB.csv", index=False)
    
    def perf_vs_time(
        self, transfer_strategy: str, model_arch: str="BLSTM",
        pred_labels: list[str]=["retired", "graduated"],
        metrics: list[str]=["f1-score", "precision", "recall"],
        stop_month: int=250
    ) -> None:
        """
            Generates a performance report with respect to each time period.

            @param transfer_strategy: strategy string in the specified format for 
                                      all functions
            @param model_arch: model_architecture to compare with; might be 
                               removed in the future
            @param pred_labels: labels to use when predicting
            @param metric: metric for each predicted label to use

            NOTE this assumes that the intervals are the measuring instrument, meaning 
                testing performance for each project with respect to each period of 
                time before a project's completion. This is similar to the deployment 
                context, since the alternative tests how well we can predict based on 
                the intervals of data.

            Performance is divided by each predicted class, then we measure the 
            performance using f1-score. NOTE we'll aggregate data for each interval of 
            time.
        """

        # setup
        report_name = f"./model-reports/perf-testing/perf-report-{transfer_strategy}-{model_arch}"
        df = self.data.copy()
        df["date"] = pd.to_datetime(df["date"])
        append_indicators = {
            "*": "-months",                                                     # what's appended for interval by months
            "%": "-steps",                                                      # what's appended for interval by proportion
            "l": "-lag-steps"                                                   # what's appended for interval by proportion, lag preds
        }
        label_indicators = {
            "*": "months",
            "%": "proportion-steps",
            "l": "backwards-steps"
        }

        # select only the required rows; note this also means dropping any
        # previous trials past the one we just conducted (done heuristically via
        # timestamps). In the event of post initialization graphing, defaults to 
        # another heuristic.
        df = df[(df["transfer_strategy"] == transfer_strategy) & \
                (df["model_arch"] == model_arch) & \
                (df["label"].isin(pred_labels)) & \
                (df["metric"].isin(metrics))]
        
        narrow_1 = df[df["date"] >= self.initialize_time]
        narrow_2 = df[df["date"] >= (
            df["date"].iloc[-1] - dt.timedelta(seconds=5)
        )]
        
        if narrow_1.shape[0] != 0:
            df = narrow_1
        elif narrow_2.shape[0] != 0:
            # choose the last entered time and create a delta that way
            df = narrow_2
        
        # output summary
        ## ignore label-wise differences, simply macro-avg them
        numeric_dict = {col: "first" for col in df.columns if col not in {"month", "metric", "perf"}}
        numeric_dict["perf"] = "mean"
        numeric_dict["support"] = "sum"
        
        summary_df = df.groupby(["month", "metric"]).agg(numeric_dict).reset_index()
        summary_df.drop(columns=["date", "label"], inplace=True)
        
        ## pivot so metrics are columns
        metrics_df = summary_df.pivot(
            index="month", columns="metric", values="perf"
        ).reset_index()
        metrics_df.reset_index(drop=True)
        
        ## sort
        metrics_df = metrics_df.sort_values(by="month", key=lambda x: x.apply(self._order_intervals))
        
        ## export summaries
        print(summary_df)
        print(metrics_df)
        with open(report_name + ".txt", "w") as f:
            f.write(summary_df.to_string())
            f.write("\n\n")
            f.write(metrics_df.to_string())
            f.write("\n\n")
            
        ## remove all other metrics, only use prioritized one
        metric = metrics[0]
        df = df[df["metric"] == metric]
        
        # ensure months column is numeric; assumes only one strategy is used
        flagged_indicator = [indicator for indicator in append_indicators.keys() if indicator in transfer_strategy]
        
        if len(flagged_indicator) == 0:
            flagged_indicator = ["*"]
        append_str = append_indicators[flagged_indicator[0]]
        
        df.loc[df["month"] != "all", "month"] = (                               # remove the append string
            df.loc[df["month"] != "all", "month"].str[0:-len(append_str)]
        )
        
        label_str = label_indicators[flagged_indicator[0]]

        # format metric as a new column
        df = df.drop("metric", axis=1)
        df.rename({"perf": metric}, inplace=True, axis=1)
            
        # combine strat & class
        if "transfer_strategy" in df.columns:
            df["label"] = [f"{t}-{l}" for t, l in zip(df['transfer_strategy'], 
                                                      df['label'].astype(str))]
            df.drop("transfer_strategy", inplace=True, axis=1)

        # drop all other columns
        df = df.filter(items=["label", "month", metric], axis=1)

        # aggregate data for plotting
        agg_df = df.groupby(["label", "month"])[metric].mean().reset_index()

        # ensure the months are numeric and ordered correctly
        agg_df["month"] = agg_df["month"].apply(self._order_intervals)
        numeric_all_month = agg_df[agg_df["month"] < float("inf")]["month"].max() + 1
        agg_df.loc[agg_df["month"] == float("inf"), "month"] = (                # make the all month come right after the prev max. interval
            numeric_all_month
        )
        
        agg_df = agg_df.sort_values(by=["month"])
        agg_df = agg_df[agg_df["month"] <= stop_month]
        
        # plotting
        plt.figure(figsize=(10, 6))
        sns.set_style("darkgrid")

        ## markers and lines
        sns.scatterplot(
            data=agg_df, x="month", y=metric, style="label", marker="4", 
            hue="label", palette="coolwarm"
        )
        
        ## smoothed plot
        for lbl in agg_df["label"].unique():
            label_df = agg_df[agg_df["label"] == lbl]
            sns.regplot(
                data=label_df, x="month", y=metric, lowess=True, scatter=False, 
                ci=99, robust=False
            )
        
        # setup plot metadata
        plt.xlabel(f"Interval Length ({label_str})")
        plt.ylabel(f"Accuracy: {metrics[0]}")
        plt.title("Performance vs Time Period")
        plt.tick_params(axis="x", rotation=0)
        plt.legend()

        # plot ticks
        every_nth = min(ceil(df.shape[0] / 10), 5)
        xticks = plt.gca().get_xticks()
        xtick_labels = [item.get_text() for item in plt.gca().get_xticklabels()]
        
        # xticks = xticks[::every_nth]
        # xtick_labels = xtick_labels[::every_nth]
        xticks[-1] = numeric_all_month
        xtick_labels[-1] = "all"

        plt.gca().set_xticks(xticks)
        plt.gca().set_xticklabels(xtick_labels)
        
        # finalize plot layout and export
        plt.ylim((0, 1.1))
        plt.xlim(left=0)
        plt.tight_layout()
        plt.savefig(report_name + "png", bbox_inches="tight")
        plt.close()

    def summary(self, verbose: bool=True, metrics: list[str]=None, full: bool=True,
                export: bool=False, save: bool=True, agg_fns: list[str]=None,
                output_path: str=None, average_type: str="macro") -> Optional[tuple[pd.DataFrame, str]]:
        """
            Aggregates the date from each strategy and returns the summary 
            dataframe and a string version.

            @param verbose: whether or not to decrypt every transfer strategy
            @param metrics: bool of metrics to use
            @param export: logical, whether to return or not
            @param save: logical, whether to save
            @param agg_fns: list defining aggregate functions for the df. 
                            Defaults to max and average.
        """

        # setup
        df = self.data.copy().drop(columns=["date"])
        group_cols = ["transfer_strategy", "model_arch", "month", "label", "metric"]
        
        if metrics == None:
            metrics = ["accuracy"]
        if agg_fns == None:
            agg_fns = ["max", "mean", "std", "count"]

        # human-readable format
        def translate_strat(strat: str) -> str:
            """
                Converts the strategy specified into a short summary of the data 
                and strategy used.
            """

            # setup
            translated_str = f"[{strat}] -- "
            decoder = params_dict["abbreviations"]
            decoder = {v: k for k, v in decoder.items()}
            lookup_augs = params_dict["augmentation-descriptions"]

            # tokenize
            train_str, test_str = strat.split("-->")
            train_tokens = train_str.split(r"\s*\+\s*")
            test_tokens = test_str.split(r"\s*\+\s*")

            def decode_incubator(incubator_str: str) -> dict[str, str]:
                """
                    Decompose incubator token into incubator, tech num, social 
                    num.
                """

                # template
                version_pattern = r"\s*([a-zA-Z])-(\d+)-(\d+)(.*)"
                default_pattern = r"\s*([a-zA-Z])(.*)"

                # return match
                matched_str = re.match(version_pattern, incubator_str)
                if matched_str is None:
                    matched_str = re.match(default_pattern, incubator_str)
                    inc = matched_str.group(1)
                    
                    default_version = params_dict["default-versions"][decoder[inc]]
                    tec = default_version[0]
                    soc = default_version[1]
                else:
                    inc = matched_str.group(1)
                    tec = matched_str.group(2)
                    soc = matched_str.group(3)
                return {
                    "incubator": decoder[inc],
                    "tech": str(tec),
                    "social": str(soc)
                }

            def incubator_descr(incubator_dict: dict[str, str]) -> str:
                """
                    Convert dictionary of incubator info into the formal 
                    description string.
                """

                # grab strings
                tech_info = lookup_augs[incubator_dict["incubator"]]["tech"] \
                    [incubator_dict["tech"]]
                social_info = lookup_augs[incubator_dict["incubator"]]["social"] \
                    [incubator_dict["social"]]
                
                # return formatted description
                return f"{incubator_dict['incubator']}, {tech_info}, {social_info}"
            
            def set_descr(tokens: list[str], prepend: str) -> str:
                """
                    Wraps the full process for all incubators in a train/test 
                    set.
                """

                # iterate
                full_descr = []
                for incubator_str in tokens:
                    incubator_dict = decode_incubator(incubator_str)
                    descr = incubator_descr(incubator_dict)
                    full_descr.append(descr)

                return f"{prepend}{','.join(full_descr)}"

            translated_str += f"{set_descr(train_tokens, 'trained w/ {')}{'}; '}"
            translated_str += f"{set_descr(test_tokens, 'tested w/ {')}{'}'}"

            return translated_str

        # truncate entries if required
        if not full:
            # remove monthly entries
            df = df[df["month"] == "all"]

            # not verbose anymore
            verbose = False

            # generate new output path
            if output_path is None:
                output_path = Path("./model-reports") / "summaries" / "summary_db_trunc"
                
        # clean & aggregate
        df = df[df["label"].isin([f"{average_type} avg", "accuracy"])]
        df = df[df["metric"].isin(metrics)]
        if verbose:
            df["transfer_strategy"] = df["transfer_strategy"].apply(translate_strat)
        df = df.groupby(group_cols).agg(agg_fns)

        # sorting
        df = df.sort_values(by=("perf", "mean"), ascending=False)

        # export
        if save:
            # generate output path
            if output_path is None:
                output_path = Path("./model-reports") / "summaries" / "summary_db"
            check_path(output_path)

            # save
            df.to_csv(f"{output_path}.csv")
            with open(f"{output_path}.txt", "w") as f:
                f.write(df.to_string())

        if export:
            print(df.to_string())
            return (df, df.to_string())

    def comparison(self, src_field: str="model_arch") -> pd.DataFrame:
        """
            Generates a comparison between every field, aggregating by the avg 
            of the other fields first. Generates a visualization and returns the 
            comparison data.

            ex) for model-arch, we would aggregate the mean per strategy, then 
                compare across model-archs for each strategy
        """

        # setup grouping
        group_fields = {
            "transfer_strategy": "categorical",
            "model_arch": "categorical",
            "month": "numeric"
        }

        if src_field not in group_fields:
            log("field is not currently supported", "warning")
            return
        
        # set accuracy field, group, & aggregate
        data = self.data.copy()
        data = data[data["metric"] == "accuracy"]
        data.drop("metric", inplace=True, axis=1)
        data.rename(columns={"perf": "accuracy"}, inplace=True)

        if src_field == "transfer_strategy":
            # remove all interval trials
            data = data[~data[src_field].str.contains("*")]

            # aggregate strategy, ignore version numbers (cleaner visualize)
            def clean_str(s):
                tokens = s.split("-->")
                tokens = [re.sub(r"[0-9-]", "", t) for t in tokens]
                return " --> ".join(tokens)
            data[src_field] = data[src_field].apply(clean_str)
        
        data = data.groupby(src_field)["accuracy"].agg(["mean", "std"])
        data = data.sort_values(by="mean")

        # visualize
        sns.set_theme(style="darkgrid")
        plt.figure(figsize=(10 + 2 * (len(data) // 5), 6))
        sns.barplot(x=data.index, y="mean", data=data, hue=data.index, 
                    palette="rocket")
        plt.errorbar(x=data.index, y=data["mean"], yerr=data["std"], fmt="none", 
                     color="white", capsize=5)

        plt.xlabel(f"{src_field.capitalize()}")
        plt.ylabel("Mean Accuracy")
        plt.title(f"Performance vs {src_field.capitalize()}")
        plt.xticks(rotation=60)
        plt.ylim((0, 1))

        plt.tight_layout()
        save_path = Path("./model-reports") / "comparisons" / f"{src_field}-comparison.png"
        check_dir(save_path.parent)
        plt.savefig(save_path)

        return data

    def strategy_comparison(self, strategy_options: list[str | dict[str, bool]],
                            comparison_metrics: list[str]=None) -> None:
        """
            Compare sets of options against each other. The idea is we can try 
            and narrow down what combinations of transfer strategy work well 
            with what augmentations and try and figure out an optimal set of 
            changes for each incubator.
            
            Always compares against the clean version.

            @param strategy_options: list of either strings (all options listed 
                                     out as specified by the transfer strat 
                                     grammar) or dictionaries of options. NOTE 
                                     have to be consistent, can't mix and match
            @param comparison_metrics: list of metrics to use; must be included 
                                       in the PerfDB to be valid
        """

        # subset data, group, and aggregate


        # numerical comparison


        # visual comparison

        raise NotImplementedError

    def schema(self) -> None:
        print("""Schema for Performance Database:
            \t- `date` (str - unique key for distinguishing trials)
            \t- `transfer_strategy` (str - following format defined in `ModelData`)
            \t- `model_arch` (str - indicates type of model used)
            \t- `month` (str - for intervals; for non-intervals, defaults to 'all')
            \t- `label` (str - graduation [1], retired [0], weighted-avg, accuracy, macro-avg)
            \t- `metric` (str - precision, recall, F1)
            \t- `perf` (float - measure of performance recorded)
            \t- `support` (int - num projects)""")

    def add_entry(self, transfer_strat: str, model_arch: str, preds: Iterable[Any], 
                  targets: Iterable[Any], month: str="all", intervaled: bool=False,
                  pred_labels: list[str]=["retired", "graduated"],
                  check_export: bool=False) -> None:
        """
            Adds an entry to the performance data db given the necessary info.
        """

        # dispatch
        if intervaled:
            # for each month add entries
            for m, pred in tqdm(preds.items()):
                self._add_entry(
                    transfer_strat=transfer_strat,
                    model_arch=model_arch,
                    preds=pred,
                    targets=targets[m],
                    month=m,
                    pred_labels=pred_labels,
                    check_export=check_export,
                    print_report=False
                )
            
        # no intervals
        else:
            self._add_entry(
                transfer_strat=transfer_strat,
                model_arch=model_arch,
                preds=preds,
                targets=targets,
                month=month,
                pred_labels=pred_labels,
                check_export=check_export,
                print_report=True
            )

        # generate summary
        self.summary()

    def _add_entry(self, transfer_strat: str, model_arch: str, preds: list[Any], 
                  targets: list[Any], month: str="all", 
                  pred_labels: list[str]=["retired", "graduated"],
                  check_export: bool=False, print_report: bool=False) -> None:
        """
            Adds an entry to the performance data db given the necessary info.
        """

        # setup
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_rows: list[pd.DataFrame] = []

        # convert labels
        preds = [pred_labels[int(p)] for p in preds]
        targets = [pred_labels[int(t)] for t in targets]

        # get performance data
        raw_report_dict = classification_report(
            y_true=targets,
            y_pred=preds,
            labels=pred_labels,
            output_dict=True,
            zero_division=0.0
        )

        # don't store trials w/ 0 support
        report_dict = dict()
        for label in raw_report_dict:
            if label == "accuracy" or raw_report_dict[label]["support"] != 0:
                report_dict[label] = raw_report_dict[label]

        # data tracking
        static_data = [                                             # data that doesn't change across metrics
            timestamp,
            transfer_strat,
            model_arch,
            month
        ]
        labels = pred_labels + ["weighted avg", "macro avg"]
        metrics = ["precision", "recall", "f1-score"]
        performance = {k: metrics for k in labels}                  # store the unique label: metric info

        # add all metrics first
        for label, metric_info in performance.items():
            # skip labels with no support (for interval testing)
            if label not in report_dict:
                continue
            
            # metrics
            for metric in metric_info:
                # add new row
                new_row = dict(zip(
                    self.data.columns,
                    static_data + [
                        label,
                        metric,
                        report_dict[label][metric],
                        report_dict[label]["support"]
                    ]
                ))
                new_row = pd.DataFrame({k: [v] for k, v in new_row.items()})
                new_row.fillna(0.0, inplace=True)
                new_rows.append(new_row)

        # add accuracy reporting
        new_row = dict(zip(
            self.data.columns,
            static_data + [
                "accuracy",
                "accuracy",
                accuracy_score(targets, preds),
                len(preds)                                          # total number of projects
            ]
        ))
        new_row = pd.DataFrame({k: [v] for k, v in new_row.items()})
        new_row.fillna(0.0, inplace=True)
        new_rows.append(new_row)
        
        # export
        new_entries = pd.concat(new_rows, ignore_index=True)
        self.data = pd.concat([self.data, new_entries], ignore_index=True)

        if print_report:
            print(new_entries)

        if check_export:
            resp = input("\n\ncontinue to export data [y/n]? ")
            if resp.lower() == "y":
                self.export()
        else:
            self.export()

    def remove_entries(self, columns: str | list[str], 
                       remove_values: str | float | int | list[str | float | int]) -> None:
        """
            Removes the entries matching the given format. If multiple values 
            are specified, can do more complicated removals. Otherwise, removes 
            only based on equality in one column.
        """

        # check types

        # craft conditions if multiple aren't specified
        if isinstance(columns, str):
            columns = [columns]
            remove_values = [remove_values]

        # generate default condition
        condition = (self.data[columns[0]] == remove_values[0])

        for i, col in enumerate(columns[1:]):
            condition = condition & (self.data[col] == remove_values[i])

        # remove values & save
        self.data = self.data[~condition]
        self.export()

    def subset_breakdown(
        self, trial_type: Any=None, options: dict[str, bool] | str=None,
        metrics: list[tuple[str, str]]=None, aggregate: str="mean",
        strict: bool=True, export: bool=True, print_summary: bool=True
    ) -> tuple[str, pd.DataFrame]:
        """
            Generates a breakdown for a subset of trials based on key features 
            in every trial. If `trial_type` is specified, it uses regex matching 
            to select the subset of trials, otherwise relies on the options fed 
            in.

            @param trial_type: regex str to match
            @param options: options the trial should have selected (at least one 
                            of the options should be present in at least one of 
                            the components); if provided a str, can generate the 
                            options dict automatically
            @param metrics: list of metrics to use, will default to accuracy and 
                            macro/weighted F1
            @param aggregate: how to aggregate the final results; for now only 
                one aggregation method at a time is supported
            @param strict: whether to enforce ONLY the options specified can be 
                           present
            @param export: whether to save to default file or not
        """
        
        # enforce args
        if metrics is None:
            metrics = [("accuracy", "accuracy"), ("macro avg", "f1-score"), 
                       ("weighted avg", "f1-score")]
        if isinstance(options, str):
            # generate options dict
            abbrevations = params_dict["network-aug-shorthand"]
            options = set(options)
            options = {k: (k in options) for k in abbrevations}

        # branch if regex matching
        if trial_type is not None:
            # select by regex matching
            name_modifier = "regex"
            match_locs = self.data["transfer_strategy"].str.match(re.escape(trial_type))
            subset_df = self.data[match_locs]
        elif options is not None:
            # name modifications
            name_modifier = "-".join([abbrevations.get(k, "ERR") for k in options if options.get(k, False)])

            # load abbrevations and only select specified ones
            abbrevations = params_dict["network-aug-shorthand"]
            shorthand_abbrv = [k for k, v in abbrevations.items() if options.get(k, False)]
            match_strs = "|".join(map(re.escape, shorthand_abbrv))       # escape special chars

            # rows with at least one
            subset_df = self.data[self.data["transfer_strategy"].str.contains(match_strs)]

            # strict match, remove rows where alternative abbrevations are present
            if strict:
                # remove other abbrevations
                non_abbrevations = [k for k, v in abbrevations.items() if not options.get(k, False)]
                non_match_strs = "|".join(map(re.escape, non_abbrevations))
                subset_df = subset_df[~subset_df["transfer_strategy"].str.contains(non_match_strs)]

                # remove any rows without our specific combo of abbreviations; 
                # this can be in any order, so we have to check every order of 
                # them
                shorthand_abbrv_orders = list(permutations(shorthand_abbrv))
                shorthand_abbrv_orders = list(map("".join, shorthand_abbrv_orders))
                group_strs = "|".join(map(re.escape, shorthand_abbrv_orders))
                subset_df = subset_df[subset_df["transfer_strategy"].str.contains(group_strs)]
        else:
            log("must specify either `trial_type` or `options` in `PerfData.subset_breakdown()`", "error")
            raise ValueError("must specify either `trial_type` or `options` in `PerfData.subset_breakdown()`")

        # narrow down the rows even further
        subset_df = subset_df[["transfer_strategy", "label", "metric", "perf", "support"]]
        conditions = [(subset_df["label"] == _label) & (subset_df["metric"] == _metric) \
                      for _label, _metric in metrics]
        conditions = reduce(lambda x, y: x | y, conditions)
        subset_df = subset_df[conditions]

        # combine label and metric fields into one
        subset_df["label"] = subset_df["label"].str.replace(" ", "-")
        subset_df["metric"] = subset_df["label"].str.cat(subset_df["metric"], sep="_")
        
        metrics = [f"{label.replace(' ', '-')}_{metric}" for (label, metric) in metrics]
        
        subset_df.drop(columns="label", inplace=True)

        # aggregate data to pivot correctly
        subset_df = subset_df.groupby(["transfer_strategy", "metric"])[["perf", "support"]].agg(aggregate).reset_index()
        subset_df.set_index("transfer_strategy")
        
        # pivot metrics to columns and sort
        breakdown = subset_df.pivot(
            index="transfer_strategy",
            columns=["metric"],
            values=["perf", "support"]
        )
        
        # exit prior to error
        if breakdown.shape[0] == 0:
            log("no matching entries found, escaping early", "warning")
            return "", pd.DataFrame()
        
        breakdown = breakdown.sort_values(
            by=[("perf", label_metric) for label_metric in metrics] + [("support", metrics[0])],
            ascending=False
        )

        # visualization
        renamer = {
            "macro-avg": "macro-",
            "weighted-avg": "weighted-",
            "f1-score": "f1",
            "accuracy_accuracy": "accuracy",
            "_": ""
        }
        
        ## create renaming for columns
        new_metric_names = dict()
        for metric in metrics:
            new_metric_names[metric] = metric
            for finder, replacer in renamer.items():
                new_metric_names[metric] = new_metric_names[metric].replace(finder, replacer)
        
        ## mapper for new names: column info
        metric_mapper = {
            new_metric_names[metric]: breakdown[("perf", metric)] for metric in metrics
        }
        
        performance_comp_df = pd.DataFrame({
            "strat": breakdown.index,
            **metric_mapper,
            "support": breakdown[("support", "accuracy_accuracy")]
        })
        export_df = performance_comp_df.reset_index(drop=True)
        performance_comp_df = performance_comp_df.head(10)

        performance_comp_df = pd.melt(
            performance_comp_df,
            id_vars=["strat"],
            value_vars=["accuracy"].extend([m for m in metrics if "f1" in m]),
            var_name="Metric",
            value_name="Value"
        )

        sns.set_style("darkgrid")
        plt.figure(figsize=(10, 8))
        sns.barplot(x="strat", y="Value", hue="Metric", palette="mako", 
                    data=performance_comp_df)

        plt.title(f"Perf vs Transfer for {name_modifier} Trials [First 10 Shown]")
        plt.xlabel("Strategy")
        plt.ylabel("Score")
        plt.xticks(rotation=60)
        plt.tight_layout()
        plt.legend()

        # export
        if print_summary:
            print(breakdown)

        if export:
            save_dir = "./model-reports/breakdowns/"
            check_dir(save_dir)

            plt.savefig(f"{save_dir}breakdown-{name_modifier}", bbox_inches="tight")
            breakdown.to_csv(f"{save_dir}breakdown-{name_modifier}.csv")
            with open(f"{save_dir}breakdown-{name_modifier}.txt", "w") as f:
                f.write(breakdown.to_string())
        
        # returns
        plt.close()
        return breakdown.to_string(), export_df

    def monthly_predictions(self, incubator: str=None, 
                            proj_subset: dict[str, list[str]]=dict(),
                            multi_incubator: bool=False, strategy: str="lowess") -> None:
        """
            Generates a graph of the Graduation Forecast versus the Incubation
            Month for each project.
            
            @param incubator: name of the incubator to graph; if None, defaults 
                              to all incubators
            @param proj_subset: subset of projects to graph on a single graph, 
                                incubator: list of projects
        """
        
        # auxiliary fn
        def graph_forecast(predictions: list[str], exp_dir: str, 
                           incubators: list[str], ext: str="png") -> None:
            """Helper tool for graphing the csvs listed.
            """
            
            # grab plot setters
            colors = self._get_colors(len(predictions))
            STEP_SIZE = 5
            incubators = [inc.lower() for inc in incubators]
            markers = {
                "apache": "o",
                "github": "X",
                "eclipse": "D"
            }
            
            # load data
            dfs = ([
                (
                    re.search(r"(.*?)_f_data", os.path.basename(path)).group(1),
                    pd.read_csv(path)
                )
                for path in predictions
            ])
            
            # setup plot
            plt.figure(figsize=(10, 6))
            sns.set_style("darkgrid")
            max_len = -1

            # graphing
            for i, (proj_name, df) in enumerate(dfs):
                # update max & months
                max_len = max(df.shape[0], max_len)
                df["month"] -= 1
                
                # markers and lines
                if strategy == "line":
                    ## skip after line plot
                    sns.lineplot(
                        data=df, x="month", y="close", color=colors[i], 
                        label=proj_name.title(), marker=markers[incubators[i]],
                        markevery=5, errorbar=("ci", 95), n_boot=10000,
                        markersize=8
                    )
                    continue
                    
                sns.scatterplot(
                    data=df, x="month", y="close", color=colors[i], 
                    label=proj_name.title(), marker=markers[incubators[i]]
                )
                
                # smoothed plot; in case of RankWarning for the polynomial fit,
                # we'll instead catch it as an error and attempt a default curve
                match strategy:
                    case "lowess":
                        ## please don't judge me for this code :(
                        import statsmodels.api as sm
                        from statsmodels.nonparametric.smoothers_lowess import lowess
                        
                        ## get lowess
                        lowess_fit = lowess(df["close"], df["month"], 0.2)
                        x_smooth = lowess_fit[:, 0]
                        y_smooth = lowess_fit[:, 1]
                        
                        ## error bars
                        residuals = df["close"] - np.interp(
                            df["month"], x_smooth, y_smooth
                        )
                        std_dev = np.std(residuals)
                        ci = 1.96 * std_dev
                        
                        ## plot
                        smooth_data = pd.DataFrame({"x": x_smooth, "y": y_smooth})
                        sns.lineplot(
                            data=smooth_data, x="x", y="y", color=colors[i],
                            marker=markers[incubators[i]], markevery=5
                        )
                        plt.fill_between(
                            x_smooth, y_smooth - ci, y_smooth + ci,
                            color=colors[i], alpha=0.2
                        )

                        ## depr lowess w/o error
                            # sns.regplot(
                            #     data=df, x="month", y="close", scatter=False, 
                            #     lowess=True, color=colors[i], ci=0.95
                            # )
                    
                    case "log":
                        sns.regplot(
                            data=df, x="month", y="close", scatter=False, 
                            lowess=False, color=colors[i], logistic=True, 
                            marker=markers[incubators[i]], markevery=5
                        )
                        
                    case "reg":
                        import warnings
                        warnings.filterwarnings("ignore")
                        sns.regplot(
                            data=df, x="month", y="close", scatter=False, 
                            lowess=False, color=colors[i], order=5,
                            marker=markers[incubators[i]], markevery=5
                        )

                    case _:
                        pass
                    
            # export
            projects = "-".join([pkg[0] for pkg in dfs])
            
            if len(predictions) > 1 and not multi_incubator:
                save_path = f"{exp_dir}[{incubator.capitalize()}-Forecast]_{projects}_f_graph.jpg"
            else:
                save_path = f"{exp_dir}{projects}_f_graph.{ext}"
            
            plt.xlabel("Incubation Month")
            plt.ylabel("Graduation Forecast")
            plt.ylim((0, 1.1))
            plt.title("Graduation Likelihood vs Time")
            plt.legend()
            plt.xticks(range(0, max_len, STEP_SIZE))
            
            plt.savefig(save_path)
            plt.close()
        
        # check incubator
        if incubator is None:
            incubators = params_dict["datasets"]
        else:
            incubators = [incubator]
            
        # check multi-incubator
        if multi_incubator:
            # setup saving
            exp_dir = "../predictions/comparisons/"
            check_dir(exp_dir)
            
            # generate multi-incubator paths
            paths = list()
            incubators = list()
            for inc, proj_set in proj_subset.items():
                paths += [f"../predictions/{inc}/{proj}_f_data.csv" for proj in proj_set]
                incubators.extend([inc] * len(proj_set))
                
            # graph & export
            graph_forecast(paths, exp_dir=exp_dir, incubators=incubators)
            return
            
        # for each incubator
        for incubator in tqdm(incubators):
            # check path
            predictions_dir = f"../predictions/{incubator}/"
            
            # check projects to use
            if proj_subset.get(incubator, None) is not None:
                # ensure they contain a full path to lookup
                subset = [f"{predictions_dir}{proj}_f_data.csv" for proj in proj_subset[incubator]]
                
                # graph
                graph_forecast(subset, exp_dir=predictions_dir, incubators=[incubator])
                continue
            
            # list all files, ensure they are a full relative path, then
            # ensure they're a forecast file
            subset = list(os.listdir(predictions_dir))
            subset = ([
                p for p in subset 
                if p.endswith(".csv")
            ])
            subset = [f"{predictions_dir}{fp}" for fp in subset]
            
            # iterate projects and graph each one
            for proj in tqdm(subset):
                graph_forecast([proj], exp_dir=predictions_dir, incubators=[incubator])

    def best_perfs(
        self, transfer_strats: list[str]=None, agg_strat: str="median",
        use_regex: bool=True, export: bool=True
    ) -> pd.DataFrame:
        """
            Breakdown of the best performances per transfer strategy.

            
            @param transfer_strats (list[str]): list of strings with format 
                brackets `{}` after each incubator. Note, this will be converted
                to regex formatting.
            @param agg_strat (str): determines how to condense the information 
                from all the trials, defaults to `max` i.e. best performance.
            @param use_regex (bool, optional): determines if you want to 
                consider every possible option or just a single option choice.
            @param export (bool, optional): whether to save the breakdown or 
                not. Defaults to True.
        """
        
        # args check
        if transfer_strats is None:
            transfer_strats = TRANSFER_STRATS
        
        # generate overall comparison across all architectures
        if use_regex:
            ## ensure all t_opts are replace
            transfer_strats = [s.replace("t_opt", "opt") for s in transfer_strats]
            
            ## escape for matching
            regex_matches = [re.sub(r"\\\{opt|\\}", r"[^+]*", re.escape(s)) for s in transfer_strats]
            
            ## ensure end of string after each strat so no duplicates occur
            regex_matches = [s + "$" for s in regex_matches]
        else:
            ## only consider base transform trials
            transfer_strats = [s.replace("{t_opt}", "cn") for s in transfer_strats]
            transfer_strats = [s.replace("{opt}", "cbn") for s in transfer_strats]
            
            ## convert to regex form
            regex_matches = ["^" + re.escape(s) + "$" for s in transfer_strats]
            
        ## compare every transfer strat
        best_dfs: list[pd.Dataframe] = list()
        
        for _, transfer_strat in enumerate(regex_matches):
            ## add the best rows from the results of the breakdown
            _, result_df = self.subset_breakdown(
                trial_type=transfer_strat, 
                metrics=[
                    ("accuracy", "accuracy"),
                    ("weighted avg", "f1-score"),
                    ("weighted avg", "precision"),
                    ("weighted avg", "recall")
                ],
                strict=True,
                aggregate=agg_strat,
                print_summary=False
            )
            
            ## skip if the result is empty
            if result_df.shape[0] > 0:
                result_df = result_df.iloc[0, :]
                best_dfs.append(result_df)
        
        ### concatenate all dfs and export
        if len(best_dfs) > 1:
            best_df = pd.concat(best_dfs, axis=1, ignore_index=True).transpose()
        elif len(best_dfs) == 0:
            log("no matching entries found, escaping early", "warning")
            return pd.DataFrame()
        else:
            best_df = best_dfs[0]
        
        if export:
            best_df.to_csv("./model-reports/icse-trials/icse_breakdown.csv", index=False)
            with open("./model-reports/icse-trials/icse_breakdown.txt", "w") as f:
                f.write(best_df.to_string(index=False))
        return best_df

    @staticmethod
    def remove_aug_chars(s: str) -> str:
        """Removes the augmentation chars from a transfer strategy.

        Args:
            s (str): transfer strat.

        Returns:
            str: cleaned str, i.e. the base level of the strategy.
        """
        
        # remove all lower case letters
        return "".join(filter(lambda c: not c.islower(), s))

    @staticmethod
    def extract_aug_chars(s: str) -> str:
        """Extracts just the augmentation string from a transfer strategy 
        string.
        """
        
        # regex extraction for the first group of lowercase chars
        m = re.search(r"[a-z]+", s)
        return m.group(0) if m else ""

    @staticmethod
    def gather_best_trial_augs(df: pd.DataFrame, acc_measure: str="mac-f1") -> tuple[dict[str, str], pd.DataFrame]:
        """Subsets a larger perfdb to only pick the unique transfer trials (i.e.
        augmentation agnostic grouping) that perform the best. Simultaneously 
        records the best performing augmentation for each unqiue transfer.

        Args:
            df (pd.DataFrame): performance db post summarization, i.e. with 
                columns {"transfer_strategy", "model_arg",
                "{acc_measure}_{median/mean/std}", "support"}. So, 6 columns 
                total.
            acc_measure (str, optional): accuracy measure to use when subsetting
                the db. Defaults to "mac-f1".

        Returns:
            tuple[dict[str, str], pd.DataFrame]:
                - lookup of strategy: best_aug_str
                - subsetted df with only the trials for a given strategy that 
                  performed best
        """
        
        # helper fn for picking the best augmentations for a given strategy
        def pick_best_augmentation(strat_group) -> pd.DataFrame:
            """Given a mono-strategic group of data, subsets the best trial aug
            to use.
            """
            
            # sort by the grouping we want (median, mean, -std, support) in 
            # decreasing order
            sorted_group = strat_group.sort_values(
                by=[
                    f"{acc_measure}_median",
                    f"{acc_measure}_mean",
                    f"{acc_measure}_std",
                    "support"
                ],
                ascending=[False, False, True, False]
            )
            
            # pick only the first entry for each model
            return sorted_group.head(1).reset_index(drop=True)
        
        # new field for the grouping
        df["strategy"] = df["transfer_strategy"].apply(PerfData.remove_aug_chars)
        df["augmentation"] = df["transfer_strategy"].apply(PerfData.extract_aug_chars)
        
        # modify based on the max performance
        df = pd.concat([
            pick_best_augmentation(strat_group[1])
            for strat_group in df.groupby(["strategy", "model_arch"])
        ])
        df.transfer_strategy = df.strategy
        df.drop(columns="strategy", inplace=True)
        
        # export result
        return df

    def paper_tables(self, save_path: Path | str, acc_measure: str="mic-f1", **kwargs):
        """Generates a full breakdown for a paper and exports to multiple 
        formats (csv, latex).
        
        Breakdown by transfer strategy and model architecture, uses the median 
        and reports the stddev (only in the csv, not in latex). 

        Args:
            save_path (Path | str, optional): directory to save to. Defaults to 
                TSE trial dirs.
            acc_measure (str, optional): accuracy measure to use, should be one 
                of {"acc", "mac-f1", "mic-f1"}. Defaults to "acc".
        """
        
        # we'll groupby transfer strategy and model archs, but we need to ignore
        # augmentations, we'll only consider the best transfer protocols. For 
        # now, compute them separately. We can aggregate by the best perf later.
        
        # groupby and breakdown setup
        group_cols = ["transfer_strategy", "model_arch"]
        measure_translation = {
            "acc": ("accuracy", "accuracy"),
            "mac-f1": ("macro avg", "f1-score"),
            "mic-f1": ("weighted avg", "f1-score")
        }[acc_measure]
        
        # only keep the measure we want
        measure_data = self.data[
            (self.data.label == measure_translation[0]) &
            (self.data.metric == measure_translation[1])
        ]
        measure_data = measure_data.drop(columns=["label", "metric", "month", "date"])
        measure_data.rename(columns={"perf": acc_measure}, inplace=True)
        
        # aggregate after grouping
        summary_df = measure_data.groupby(group_cols)[[acc_measure, "support"]].agg(["median", "mean", "std"])
        summary_df = summary_df.reset_index()
        summary_df.columns = ["_".join(col) if col[1] != "" else col[0] for col in summary_df.columns]
        
        summary_df.drop(columns=["support_mean", "support_std"], inplace=True)
        summary_df.rename(columns={"support_median": "support"}, inplace=True)
        summary_df["support"] = summary_df["support"].astype(int)
        
        # prior to removing the augmentation we used, let's pick the 
        # augmentation that worked best of the three by grouping by the strategy
        summary_df = PerfData.gather_best_trial_augs(summary_df, acc_measure)
        
        # after getting the best trials from each transfer, group the trials by 
        # the thing we're testing
        cleaned_strats = ({
            group: [s.replace(r"{opt}", "").replace(r"{t_opt}", "") for s in strats]
            for group, strats in PAPER_STRATS.items()
        })
        tables = ({
            group: summary_df[summary_df.transfer_strategy.isin(strats)]
            for group, strats in cleaned_strats.items()
        })
        
        # export csvs
        for group, table in tables.items():
            table.to_csv(Path(save_path) / f"paper_table_{group}.csv", index=False)
        
        # drop the useless columns and format in the paper style #
        # drop useless cols
        summary_df.drop(columns=[f"{acc_measure}_mean", f"{acc_measure}_std"], inplace=True)
        summary_df.rename(columns={f"{acc_measure}_median": "performance"}, inplace=True)
        
        # pivot to format in a model arch 
        summary_df = pd.pivot(
            summary_df,
            index="transfer_strategy",
            values="performance",
            columns="model_arch"
        ).reset_index()
        summary_df.set_index("transfer_strategy", inplace=True)
        
        # sort by performance and trial type
        summary_df["plus_count"] = summary_df.index.str.count(r"\+")
        summary_df.sort_values(
            by=["plus_count", "BLSTM", "Transformer"],
            ascending=[True, False, False],
            inplace=True
        )
        summary_df.drop(columns=["plus_count"], inplace=True)
        
        # group by table
        tables = ({
            group: summary_df[summary_df.index.isin(strats)]
            for group, strats in cleaned_strats.items()
        })
        
        # save to latex with bolded formats for each row
        def bold_max_row(data):
            return (data.style
                .highlight_max(axis=1, props="textbf:--rwrap;")
                .format(precision=4)
                .to_latex(
                    hrules=True
                )
            )
        
        tables = {group: bold_max_row(table) for group, table in tables.items()}
        
        # fix the formatting bugs (i.e. replace arrows, replace special escapes)
        def clean_latex_str(latex_str: str) -> str:
            return (latex_str
                .replace(r"-->", r"$\to$")
                .replace(r"^", "") #r"\textasciicircum")
                .replace(r"_", r" ")
                .replace(r"model arch", r"Model Architecture")
                .replace(r"transfer strategy", r"Strategy")
            )
        
        tables = {group: clean_latex_str(table) for group, table in tables.items()}
        
        # save the breakdowns for the paper
        for group, table_str in tables.items():
            with open(Path(save_path) / f"breakdown_{group}.tex", "w") as f:
                f.write(table_str)
        with open(Path(save_path) / "breakdown.tex", "w") as f:
            f.write("\n".join(tables.values()))

# Testing
def icse_wrapper():
    """Wrapper for icse experiments.
    """
    # MONTHLY PREDS #
    ## loading
    pfd = PerfData(perf_source="./model-reports/icse-trials/icse_monthly_preds")
    
    ## generate comparisons
    pfd.monthly_predictions(
        proj_subset={
            "apache": ["droids"],
            "github": ["ActionBarSherlock", "forem"],
            "eclipse": ["Concierge"]
        },
        multi_incubator=True,
        strategy="line"
    )
    pfd.monthly_predictions(incubator=None, strategy="line")
    
    # NORMAL TRIALS #
    ## loading
    pfd = PerfData(perf_source="./model-reports/icse-trials/final_icse_perf_db")
    
    ## breakdowns by options and transfer
    icse_25_experiments()
    
    ## breakdowns by model and transfer
    model_archs = ["BLSTM", "DLSTM", "BGRU"]
    best_model_perfs = dict()
    
    for model_arch in model_archs:
        pfd = PerfData(perf_source="./model-reports/icse-trials/final_icse_perf_db")
        pfd.data = pfd.data[pfd.data["model_arch"] == model_arch]
        best_model_perfs[model_arch] = pfd.best_perfs(use_regex=False)
        
    with open("./model-reports/icse-trials/icse_breakdown.txt", "w") as f:
        for ma, perf in best_model_perfs.items():
            f.write(f"< :::: {ma} :::: >\n")
            f.write(perf.to_string(index=False))
            f.write("\n\n\n")

def tse_wrapper(**kwargs):
    """Wraps the breakdowns for the TSE trials.
    """
    
    # load the perf db
    pfd = PerfData(perf_source="./model-reports/tse-trials/tse_perf_db")
    
    # breakdown
    pfd.paper_tables(save_path="./model-reports/tse-trials/", **kwargs["args_dict"])
    
    # load and re-save the paper tables for mic and macro
    df = pd.read_csv("./model-reports/tse-trials/paper_table.csv")
    df = df[~df.transfer_strategy.str.contains("+", regex=False)]
    df.transfer_strategy = df.transfer_strategy.str.replace("^", "")
    
    df.pivot()
    
    


# Script
def __pfd_main():
    # setup
    args_dict = parse_input(sys.argv)
    breakdown_type = args_dict.get("breakdown-type", "tse")
    
    # match trial
    match breakdown_type:
        case "icse":
            icse_wrapper()
        
        case "tse":
            tse_wrapper(
                args_dict=args_dict
            )
        
        case _:
            print(":(")

if __name__ == "__main__":
    """
    Schema for Performance Database:
        - `date` (str - unique key for distinguishing trials)
        - `transfer_strategy` (str - following format defined in `ModelData`)
        - `model_arch` (str - indicates type of model used)
        - `month` (str - for intervals; for non-intervals, defaults to 'all')
        - `label` (str - graduation [1], retired [0], weighted-avg, accuracy, macro-avg)
        - `metric` (str - precision, recall, F1)
        - `support` (int - num projects)
    """

    ############################################################################
    # ICSE EXPERIMENTS #
    # icse_wrapper()
    ############################################################################

    __pfd_main()
    
    # normal experiments
    # pfd = PerfData()
    # pfd.perf_vs_time(transfer_strategy="A-1-1*^ --> A-1-1*^^")
    # perf_db.summary()
    # perf_db.perf_vs_time("A-1-1a --> E-1-1a", stop_month=250)
    # perf_db.comparison()
    # perf_db.comparison(field="transfer_strategy")
    # report, ssbd = perf_db.subset_breakdown(trial_type=r".* --> E-1-1(?!.*\+).*", strict=True)
    # report, ssbd = perf_db.subset_breakdown(trial_type=r".* --> G-3-4(?!.*\+).*", strict=True)
    # print(ssbd.sort_values(by=["strat", "macro-f1", "accuracy"]))
    # perf_db.comparison(field="model_arch")
    # perf_db.summary(full=False)
    # perf_db.subset_breakdown(options="bda", strict=True)
    # perf_db.subset_breakdown(options="d", strict=True)
    # perf_db.subset_breakdown(options="da", strict=True)
    # perf_db.subset_breakdown(options="bd", strict=True)

