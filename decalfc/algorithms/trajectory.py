"""
    @brief Trajectory algorithms for forecasting without network data available.
    @author Arjun Ashok (arjun3.ashok@gmail.com)
    @version 1.0.0
"""


# --- Environment Setup --- #
## external
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm
# from pmdarima import auto_arima

## built-in
import unittest
import json
import warnings
from pathlib import Path


# Helper Algorithms
def diminish_grad(grad: float, cur_pos: float, strat: str="diff") -> float:
    """Auxiliary algorithm to help control the diminishing gradient as we 
    approach the bounds of the probability output. For example, if we are 
    currently at P[success] = 0.7 w/ a trajectory of 0.2, within 3 months we'll
    have P[success] = 1.1. To avoid this, we'll diminish the probability growth
    as we approach the bounds 0 (neg gradient) or 1 (pos gradient).

    Args:
        grad (float): current gradient to step by
        cur_pos (float): current probability of success
        strat (str, optional): strategy to use:
            - "diff": multiplies grad by the difference between the bound and 
                the grad
            - "exp": exponentially decays the grad

    Returns:
        float: modified gradient
    """
    
    # infer projected bound
    bound = 0 if grad < 0 else 1
    
    # return a function of the difference between the position and projected 
    # bound
    match strat:
        case "diff":
            bound_diff = abs(bound - cur_pos)
            return grad * bound_diff

        case "exp":
            k = 0.5 # decay rate
            return (grad / k) * (1 - np.exp(-k * cur_pos))
        
        case _:
            return 0

def export_traj(trajectories: dict | list | np.ndarray, filename: Path=None) -> str:
    """Exports the trajectories into a JSON format for easy export.

    Args:
        trajectories (dict | list | np.ndarray): trajectories to use
        filename (Path, optional): saves to file if given. Defaults to None.

    Returns:
        str: returns a JSON string
    """
    
    # generate export
    if isinstance(trajectories, np.array):
        trajectories = list(trajectories)
        
    export_str = json.dumps(trajectories, indent=2)
    
    # check return type
    if filename is not None:
        with open(filename, "w") as f:
            f.write(export_str)
    return export_str

def plot_traj(forecast: np.ndarray | list, trajectories: np.ndarray, m: int=-1, ci: np.ndarray=None, **label_kwargs) -> None:
    """Plots the trajectory given a forecast.

    Args:
        forecast (np.array | list): actual forecast
        trajectories (np.array): trajectories for all months
        m (int, optional): month number to plot; if not specified, plot all
            months' trajectories. Defaults to -1.
        ci (np.ndarray, optional): confidence interval to plot; requires a 
            the same number of rows as trajectory, each with two elements 
            [lower, upper]. Defaults to None.
        **label_kwargs: specify the following:
            - "title" for the plot title
            - "path" for the save path

    Returns:
        None
    """
    
    # setup figure
    n = len(forecast)
    ntraj = len(trajectories)
    l = n - ntraj
    k = len(trajectories[0])
    colors = sns.color_palette("viridis", ntraj)
    
    plt.figure(figsize=(10, 6))
    
    # plot forecast
    plt.plot(
        range(n), forecast, color="black", label="Forecast", linewidth=2,
        marker="*"
    )
    
    # months to plot
    months = range(m, m + 1) if m >= 0 else range(len(trajectories))
    
    # plot the trajectories
    for i in months:
        # plot traj
        plt.plot(
            range(l + i, l + i + k), trajectories[i], color=colors[i],
            linestyle="dashed", marker="x"
        )
        
        # plot ci if possible
        if ci is not None:
            plt.fill_between(
                x=range(l + i, l + i + k), y1=ci[i, :, 0], y2=ci[i, :, 1],
                color=colors[i], alpha=0.3
            )
        
    # plot details
    plt.xlabel("Month")
    plt.ylabel("Forecasted Sustainbility")
    plt.ylim((0, 1))
    
    plt.title(label_kwargs.get("title", "Forecast & Trajectories"))
    plt.legend(["Forecast"], loc="upper left")
        
    # show the plot, save it to the specified path
    plt.savefig(label_kwargs.get(
        "path",
        Path().cwd() / "visuals" / "trajectories" / "experiment_traj"
    ))
    plt.show()
    plt.clf()

def bound_traj(traj_seq: np.ndarray | list) -> list:
    """Bounds a single sequence of trajectory predictions (i.e. next k months) 
    by probability bounds to prevent negative or > 1 predictions.

    Args:
        traj_seq (np.ndarray | list): iterable of trajectory predictions.

    Returns:
        list: list of bounded trajectories.
    """
    
    # simple wrapper
    return [max(min(t_pred, 1), 0) for t_pred in traj_seq]


# Primary Algorithms
def traj_simple(forecast: np.ndarray, lag: int=3, k: int=3, **grad_kwargs) -> np.ndarray:
    """Generates a simple, single trajectory for the next `k` months for every 
    month lag - 1 and onwards. Notice that if the trajectory is negative, we'll
    slow the rate of decrease as we approach closer to 0. The same is done for 
    the positive case, i.e. we diminish the gradient as we approach 1. The 
    result is a more curved projection.

    Args:
        forecast (np.array): original forecast information as an array.
        lag (int, optional): number of months to take into account when 
            generating a trajectory; this means we take lag - 1 slopes when 
            creating the projection
        k (int, optional): number of months to predict forward. Defaults to 3.

    Returns:
        np.ndarray: for every month lag - 1 and above, we generate the next k 
            forecasts. If our forecast has n months of data, we return an 
            ndarray of dimension (n - (lag + 1)) x k where the 0th row maps to
            the (lag + 1)th month forecasts, and so on.
    """
    
    # edge case
    n = len(forecast)
    if n < lag:
        raise ValueError(f"Length of forecast can't be less than lag: {n=} < {lag=}")
    if lag <= 1:
        raise ValueError(f"Lag must be greater than 1 (need at least one slope): {lag=}")
    
    # trackers
    trajectories = [[None] * (k + 1) for _ in range(lag, n)]
    
    # generation of the monthly trajectories
    for i in range(lag, n):
        # calculate the base gradient, get current position
        grads = np.diff(forecast[i - lag:i])
        grad = np.mean(grads)
        pos = forecast[i]
        
        # generation of each trajectory
        for j in range(k + 1):
            # clip position
            pos = max(0, pos)
            pos = min(1, pos)
            
            # update position
            trajectories[i - lag][j] = pos
            pos += grad
            
            # update grad
            grad = diminish_grad(grad, pos, **grad_kwargs)
        
    # export as ndarray
    return np.array(trajectories)

def traj_ar(forecast: np.ndarray, lag: int=3, k: int=3, arch: str="ARIMA", ci: int=95) -> tuple[np.ndarray, np.ndarray]:
    """Uses an autoregressive model (ARIMA, etc.) to generate the forecast more
    accurately.

    Args:
        forecast (np.array): original forecast information as an array.
        lag (int, optional): number of months to take into account when 
            generating a trajectory; this means we take lag - 1 slopes when 
            creating the projection
        k (int, optional): number of months to predict forward. Defaults to 3.
        arch (str, optional): model architecture to use. Defaults to "ARIMA".
        ci (int, optional): confidence interval for error bounds. Defaults to 
            95.

    Returns:
        np.ndarray: for every month lag - 1 and above, we generate the next k 
            forecasts. If our forecast has n months of data, we return an 
            ndarray of dimension (n - (lag + 1)) x k where the 0th row maps to
            the (lag + 1)th month forecasts, and so on.
        np.ndarray: for every month lag - 1 and above, we generate a lower and 
            upper bound on the trajectory for the forecast
    """
    
    # suppress MLE and convergence failure warnings
    warnings.filterwarnings("ignore")
    
    # edge case
    n = len(forecast)
    if n < lag:
        raise ValueError(f"Length of forecast can't be less than lag: {n=} < {lag=}")
    if lag <= 1:
        raise ValueError(f"Lag must be greater than 1 (need at least one slope): {lag=}")
    
    # trackers
    trajectories = [[None] for _ in range(lag, n)]
    ci = [[None] for _ in range(lag, n)]
    est_order = (
        lag,        # autoregressive, use all lag samples
        1,          # differencing
        lag         # moving average component, use all lag samples
    )
    
    # generation of the monthly trajectories
    for i in tqdm(range(lag, n)):
        # build simple AR model; we'll use basic lag terms
        match arch:
            case "ARIMA":
                model = ARIMA(forecast[:i + 1], order=est_order)
            
            case _:
                raise ValueError(f"Invalid architecture {arch=}")
        
        model = model.fit()
        
        # generate trajectory
        forecaster = model.get_forecast(k)
        cur_traj = forecaster.predicted_mean
        traj_ci = forecaster.conf_int()
        
        # clip trajectories to be a valid probability
        cur_traj = np.clip(cur_traj, 0, 1)
        
        # track
        trajectories[i - lag][0] = forecast[i]        # root the first prediction to cur value
        trajectories[i - lag].extend(cur_traj)
        ci[i - lag][0] = [forecast[i], forecast[i]]   # no variance in the current forecast
        ci[i - lag].extend([list(bounds) for bounds in traj_ci])
        
    # export as ndarray
    return np.array(trajectories), np.array(ci)

def traj_inf_ar(forecast: np.ndarray, lag: int=3, k: int=3, arch: str="ARIMA", ci: int=95) -> tuple[np.ndarray, np.ndarray]:
    """Autoregression for trajectories informed by the network features 
    evolution over time.

    Args:
        forecast (np.array): original forecast information as an array.
        lag (int, optional): number of months to take into account when 
            generating a trajectory; this means we take lag - 1 slopes when 
            creating the projection
        k (int, optional): number of months to predict forward. Defaults to 3.
        arch (str, optional): model architecture to use. Defaults to "ARIMA".
        ci (int, optional): confidence interval for error bounds. Defaults to 
            95.

    Returns:
        np.ndarray: for every month lag - 1 and above, we generate the next k 
            forecasts. If our forecast has n months of data, we return an 
            ndarray of dimension (n - (lag + 1)) x k where the 0th row maps to
            the (lag + 1)th month forecasts, and so on.
        np.ndarray: for every month lag - 1 and above, we generate a lower and 
            upper bound on the trajectory for the forecast
    """
    
    pass

def route_traj(forecast: np.ndarray, strat: str="AR", lag: int=3, k: int=3, **kwargs) -> dict[int, dict[str, list[float]]]:
    """Routes the forecast strategy requested. If the strategy doesn't produce
    upper and lower bounds by default, we return empty results.

    Args:
        forecast (np.ndarray): raw forecast for previous months.
        strat (str, optional): strategy to use when building the trajectory. 
            Should be one of {"AR", "INF_AR", "SIMPLE"}. Defaults to "AR".
        lag (int, optional): number of months to consider when making the next 
            step decision. Defaults to 3.
        k (int, optional): number of months to predict ahead. Defaults to 3.

    Returns:
        dict[int, dict[str, list[float]]]: lookup of month: 
            POSITIVE/NEGATIVE/NEUTRAL: list of trajectory results
    """
    
    # wrap call
    router = {
        "SIMPLE": traj_simple,
        "AR": traj_ar,
        "INF_AR": traj_inf_ar
    }
    trajectories = router[strat](forecast=forecast, lag=lag, k=k, **kwargs)
    
    # account for bounds information
    if isinstance(trajectories, tuple):
        trajectories, bounds = trajectories
    else:
        bounds = None

    # convert to dictionary in standard format
    traj_pkg = dict()
    
    for i in tqdm(range(trajectories.shape[0])):
        # current package; add the baseline results
        cur_pkg = dict()
        cur_pkg["NEUTRAL"] = bound_traj(list(trajectories[i]))
        
        # add the bound results if possible
        if bounds is not None:
            cur_pkg["NEGATIVE"] = bound_traj(list(bounds[i][:, 0]))
            cur_pkg["POSITIVE"] = bound_traj(list(bounds[i][:, 1]))
        else:
            cur_pkg["POSITIVE"] = None
            cur_pkg["NEGATIVE"] = None
        
        # track this month's result
        traj_pkg[i] = cur_pkg
    
    # export the standard format results
    return traj_pkg


# Unit Tests
class TrajTester(unittest.TestCase):
    def setUp(self):
        # arrays for testing
        self.data = [
            np.linspace(0.0, 0.7, 10),
            np.cumsum(np.random.random(size=(10)) / 10),
            np.sin(np.linspace(0, 2 * np.pi, 10)) / 2 + 0.5,
            1 / (1 + np.exp(-np.linspace(-5, 5, 10)))
        ]
        self.labels = [
            "linear",
            "random",
            "sinusoidal",
            "sigmoid"
        ]
    
    def diff_test(self):
        for i, data in enumerate(self.data):
            res = traj_simple(forecast=data)
            plot_traj(
                data, res,
                path=Path().cwd() / "visuals" / "trajectories" / (self.labels[i] + "-diff")
            )
            
    def exp_test(self):
        for i, data in enumerate(self.data):
            res = traj_simple(forecast=data, strat="exp")
            plot_traj(
                data, res,
                path=Path().cwd() / "visuals" / "trajectories" / (self.labels[i] + "-exp")
            )
            
    def ar_test(self):
        for i, data in enumerate(self.data):
            res, ci = traj_ar(forecast=data)
            plot_traj(
                data, res, ci=ci,
                path=Path().cwd() / "visuals" / "trajectories" / (self.labels[i] + "-arima")
            )


# Run Tests
if __name__ == "__main__":
    print(bound_traj([-1, 1.2, 0.8]))
    # tester = TrajTester()
    # tester.setUp()
    # tester.diff_test()
    # tester.exp_test()
    # tester.ar_test()
    
