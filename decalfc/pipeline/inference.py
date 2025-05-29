"""
    @brief Modeling framework w/ testing built-in for switching out model types, 
        testing accuracies with different methods, and augmenting data prior
        to testing. 
    @author Arjun Ashok (arjun3.ashok@gmail.com)
    @acknowledgements Nafiz I. Khan, Dr. Likang Yin
    @creation-date October 2023
"""


# ---------------- Environment Setup ---------------- #
# external modules
import pandas as pd
from tqdm import tqdm

# built-in modules
import sys
from time import time
from typing import Iterable, Any
from itertools import product, permutations, chain, combinations

# DECAL modules
from decalfc.utils import *
from decalfc.abstractions.modeldata import *
from decalfc.abstractions.perfdata import *
from decalfc.abstractions.tsmodel import *

# ---------------- auxiliary functions ---------------- #
def load_best_cached_weight(incubator: str, **kwargs) -> Any:
    """Loads the best model weights possible for the target incubator. BUILT FOR
    THE TSE PAPER, i.e. we're only considering all trials {A, E, G, O} -> target
    with any augmentation, no cross-/mix-incubators.

    Args:
        incubator (str): target incubator to load a model for.

    Returns:
        Any: model state dict from the `pt` file.
    """
    
    # helper functions
    def extraction(weight_str):
        # the only trials that we may want to keep are either 100 accuracy 
        # or the highest accuracy trial that exists; so let's unpack the 
        # values
        weight_items = weight_str.split("-")
        return weight_items[0], *[float(weight_items[i][2:-1]) for i in range(1, 5)]
    
    def update_weight_tracker(lookup: dict[str, list[str | float]], t_dir: Path) -> None:
        # organize weights
        weights = list(t_dir.iterdir())
        weights = [weight.stem for weight in weights]
        
        # update trial info
        for weight_str in weights:
            arch, a, f, p, r = extraction(weight_str)
            lookup["arch"].append(arch)
            lookup["acc"].append(a)
            lookup["f1"].append(f)
            lookup["prec"].append(p)
            lookup["rec"].append(r)
            lookup["strat"].append(t_dir.stem)
        
        # done
        return
    
    def prune_best_weights(lookup: dict[str, list[str | float]]) -> tuple[list, pd.DataFrame]:
        """Prunes all the non-best weights from the lookup.
        """
        
        # convert
        lookup: pd.DataFrame = pd.DataFrame(lookup)
        
        # sort and pick the best trials
        best_weights = lookup.sort_values(
            by=["acc", "f1", "prec", "rec"], ascending=False
        ).groupby("arch").first().reset_index()
        
        best_entries = [(
                row["strat"],
                f"{row['arch']}-[a{row['acc']:.2f}]-[f{row['f1']:.4f}]-[p{row['prec']:.4f}]-[r{row['rec']:.4f}]"
            )
            for _, row in best_weights.iterrows()
        ]
        
        # return the best entries
        return best_entries, best_weights
    
    # setup arguments
    model_weights_dir = Path().cwd() / params_dict["weights-dir"]
    lookup = {
        "arch": list(),
        "acc": list(),
        "f1": list(),
        "prec": list(),
        "rec": list(),
        "strat": list()
    }
    incubator = params_dict["abbreviations"][incubator]
    
    # walk through each directory
    for trial_dir in model_weights_dir.iterdir():
        ## skip any files; shouldn't exist but for safety
        if not trial_dir.is_dir():
            continue
        
        ## skip any directories where this is not the target incubator; we 
        ## empirically have shown `c`/`cn` is the optimal augmentation, so we 
        ## use this
        if not re.match(fr"^.*--> {re.escape(incubator)}c\^*$", trial_dir.stem):
            # print(f"skipping {trial_dir.stem} :: not target incubator or not `c`")
            continue
        
        ## TSE SPECIFIC: skip any directories with multiple incubators for
        ## the training; '_' is the char on UNIX systems for '+' since it's 
        ## illegal for filenames
        if ("+" in trial_dir.stem) or ("_" in trial_dir.stem):
            # print(f"skipping {trial_dir.stem} :: multi incubator")
            continue
        
        ## for every directory, aggregate the performances
        update_weight_tracker(lookup, trial_dir)
    
    # grab the best weights
    paths, weights = prune_best_weights(lookup)
    return paths, weights


# ---------------- inference scripts ---------------- #
def inference(incubator: str, model_arch: str=None, *args, **kwargs):
    """Runs inference on all projects in the given incubator using the best 
    weights possible for the given model arch.
    
    Hard-coded right now for `c` and TSE trials (i.e. no mix-incubator).

    Args:
        incubator (str): target incubator.
        model_arch (str, optional): if not one of {BLSTM, DLSTM, Transformer},
            will pick the best architecture available. Defaults to None.
    """
    
    # grab the best weights for the given architecture
    best_paths, best_weights = load_best_cached_weight(incubator=incubator)
    
    if model_arch is None:
        model_arch = best_weights.sort_values(
            by=["acc", "f1", "prec", "rec"],
            ascending=False
        )["arch"].iloc[0]
    
    optimal_weight_path = list(filter(
        lambda x: x[1].startswith(model_arch), best_paths
    ))[0]
    
    # initialize the best model and load its weights
    ## ensure some hyperparams
    hyperparams = {"input_size": 13}
    hyperparams.update(kwargs.get("hyperparams", dict()))
    
    ## initialize model
    model = TimeSeriesModel(
        model_arch=model_arch, hyperparams=hyperparams
    )
    
    ## load weights
    model.model.load_state_dict(torch.load(
        Path().cwd() / params_dict["weights-dir"] / optimal_weight_path[0] / f"{optimal_weight_path[1]}.pt",
        map_location=model.device
    ))
    
    # load the data for this incubator
    nd = NetData(incubator=incubator, options={"feature-subset": True})
    
    # trackers
    predictions = {
        "project": list(),
        "prediction": list(),
        "target": list(),
        "train_strategy": list(),
        "model_arch": list()
    }
    
    # inference on each project in the nd
    model.model.eval()
    
    with torch.no_grad():
        for proj_name, X in nd.data_dict.items():
            # convert to tensor
            X = torch.tensor(X).to(model.device)
            X = X.reshape(1, X.shape[0], -1)
            y_true = (
                "graduated" if proj_name in nd.project_status["graduated"] else
                ("retired" if proj_name in nd.project_status["retired"] else "incubating")
            )
            
            # predictions
            out = model.model(X)
            pred_label = torch.argmax(out, dim=1)
            y_pred = pred_label.cpu().numpy()[0]
            y_pred = "graduated" if y_pred == 1 else "retired"
            
            # update trackers
            predictions["project"].append(proj_name)
            predictions["prediction"].append(y_pred)
            predictions["target"].append(y_true)
            predictions["train_strategy"].append(optimal_weight_path[0])
            predictions["model_arch"].append(model_arch)

    # export the report
    pd.DataFrame(predictions).to_csv(
        Path().cwd() / params_dict["reports-dir"] / "tse-trials" / f"{incubator}_inferences.csv",
        index=False
    )

# ---------------- script ---------------- #
def __inference_main():
    # setup
    args_dict = parse_input(sys.argv)
    trial_type = args_dict.get("trial-type", "inference")
    
    match trial_type:
        case "inference":
            inference(**args_dict)
        
        case "proportion":
            raise NotImplementedError
        
        case "monthly":
            raise NotImplementedError
        
        case _:
            raise ValueError("invalid trial type")

if __name__ == "__main__":
    # forward parameters to main
    __inference_main()

