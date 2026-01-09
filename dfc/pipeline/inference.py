"""
    @brief Modeling framework w/ testing built-in for switching out model types, 
        testing accuracies with different methods, and augmenting data prior
        to testing.
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
from dfc.utils import *
from dfc.abstractions.modeldata import *
from dfc.abstractions.perfdata import *
from dfc.abstractions.tsmodel import *

INFERENCE_DIR = Path().cwd() / params_dict["reports-dir"] / "inferences"

# ---------------- auxiliary functions ---------------- #
def load_best_cached_weight(incubator: str, aug: str="") -> Any:
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
            by=["f1", "acc", "prec", "rec"], ascending=False
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
        if not re.match(fr"^.*--> {re.escape(incubator)}{aug}\^*$", trial_dir.stem):
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
        INFERENCE_DIR / f"{incubator}_inferences.csv",
        index=False
    )

def monthly_inference(incubator: str, model_arch: str=None, T: float=2, smooth: bool=True, *args, **kwargs):
    """Runs monthwise inference on all projects in the given incubator using the
    best weights possible for the given model arch. Uses a higher temperature in
    the final softmax to allow for more reasonable probabilities and capture the
    uncertainty.
    
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
        "month": list(),
        "prediction": list(),
        "target": list(),
        "train_strategy": list(),
        "model_arch": list()
    }
    
    # inference on each project in the nd
    def gauss_smoothing(data: np.ndarray, sigma: float=0.47) -> np.ndarray:
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
    
    def projectwise_monthly_inference(predictions: dict[str, list], proj: str, X: list):
        """Performs the monthwise inference for a given project.

        Args:
            predictions (dict[str, list]): tracker for the predictions.
            proj (str): project name.
            X (list): data_dict[proj].
            T (float, optional): temperature to use. Defaults to 2.
        """
        
        with torch.no_grad():
            # convert to tensor
            X = torch.tensor(X).to(model.device)
            nmonths = X.shape[0]
            X = X.reshape(1, nmonths, -1)
            
            y_true = (
                "graduated" if proj in nd.project_status["graduated"] else
                ("retired" if proj in nd.project_status["retired"] else "incubating")
            )
            preds = list()
            
            # monthwise predictions
            for i in range(1, nmonths):
                # extract the subset of months
                X_m = X[:, :i, :]
                
                # predictions
                out = model.model(X_m)
                y_pred = F.softmax(out / T, dim=1)[0].cpu().numpy()[1]
                
                # update trackers
                predictions["project"].append(proj_name)
                predictions["month"].append(i)
                preds.append(y_pred)
                predictions["target"].append(y_true)
                predictions["train_strategy"].append(optimal_weight_path[0])
                predictions["model_arch"].append(model_arch)
        
        # smooth and update preds as needed
        if smooth:
            preds = gauss_smoothing(data=np.array(preds))
        predictions["prediction"].extend(preds)
        
        # done
        return
    
    model.model.eval()
    for proj_name, X in tqdm(nd.data_dict.items()):
        projectwise_monthly_inference(predictions, proj_name, X)

    # export the report
    predictions = pd.DataFrame(predictions)
    predictions.to_csv(
        INFERENCE_DIR / f"{incubator}_monthwise_inferences.csv", index=False
    )
    return predictions

def monthwise_predictions_visual(incubator: str, model_arch: str=None, **kwargs):
    raise NotImplementedError

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

# ---------------- script ---------------- #
def __inference_main():
    # setup
    args_dict = parse_input(sys.argv)
    trial_type = args_dict.get("trial-type", "inference")
    check_dir(INFERENCE_DIR)
    
    match trial_type:
        case "inference":
            inference(**args_dict)
        
        case "proportion":
            raise NotImplementedError
        
        case "monthwise":
            monthly_inference(**args_dict)
        
        case _:
            raise ValueError("invalid trial type")

if __name__ == "__main__":
    # forward parameters to main
    __inference_main()

