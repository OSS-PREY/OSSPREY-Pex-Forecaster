"""
    @brief [Temporary?] utility for running the full network pipeline.
    @author Arjun Ashok (arjun3.ashok@gmail.com)
    @creation-date February 2024
"""

# ------------- Environment Setup ------------- #
# external packages -- none for now

# built-in modules
import sys
from pathlib import Path

# DECAL modules
import decalforecaster.utils as util
import decalforecaster.abstractions.rawdata as rd
from decalforecaster.pipeline.monthly_data import *
from decalforecaster.pipeline.create_networks import *
from decalforecaster.pipeline.network_features import *
from decalforecaster.pipeline.network_visualizations import *


# dispatch utility
def process_data(args_dict: dict[str, Any]) -> None:
    """
        Dispatches the required processing steps to do.
    """

    # router; generate full only if processing isn't specified
    if len(args_dict["processing"]) == 0:
        raw = rd.RawData(
            incubator=args_dict["incubator"],
            versions=args_dict.get("source-versions", {"tech": 0, "social": 0}),
            gen_full=True
        )

        return

    raw = rd.RawData(
        incubator=args_dict["incubator"],
        versions=args_dict.get("source-versions", {"tech": 0, "social": 0}),
        gen_full=False
    )

    process_router = {
        "paths": rd.clean_file_paths,
        "names": rd.clean_sender_names,
        "months": rd.impute_months,
        "msg-id": rd.impute_messageid,
        "replies": rd.infer_replies,
        "is-coding": rd.clean_source_files,
        "bots": rd.infer_bots,
        "de-alias": rd.dealias_senders
    }

    # adjust data
    for process in args_dict["processing"]:
        if process in process_router:
            raw.data = process_router[process](raw.data, copy=False)
        else:
            util._log(f"Could not locate process-method {process}. Skipping.", "warning")
    
    # export
    new_versions = args_dict.get("new-versions", args_dict["versions"])
    rd._save_data(raw.data, raw.incubator, new_versions)


# Pipeline
if __name__ == "__main__":
    # params
    args_dict = util._parse_input(sys.argv)

    if "processing" in args_dict:
        process_data(args_dict)

    # monthly segment
    monthly_segmentation(args_dict)

    # create nets
    create_networks(args_dict)

    # network visualizations utility
    net_vis_info(args_dict)

    # extract net features
    extract_features(args_dict)

    # metrics
