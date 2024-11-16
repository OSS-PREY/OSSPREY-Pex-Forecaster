"""
    @brief Trajectory algorithms for forecasting without network data available.
    @author Arjun Ashok (arjun3.ashok@gmail.com)
    @version 1.0.0
"""


# Environment Setup
## external
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## built-in
import unittest
import json
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


def plot_traj(forecast: np.ndarray | list, trajectories: np.ndarray, m: int=-1, **label_kwargs) -> None:
    """Plots the trajectory given a forecast.

    Args:
        forecast (np.array | list): actual forecast
        trajectories (np.array): trajectories for all months
        m (int, optional): month number to plot; if not specified, plot all
            months' trajectories.
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
    
    # plot the trajectories
    if m < 0:
        for i, row in enumerate(trajectories):
            plt.plot(
                range(l + i, l + i + k), row, color=colors[i],
                linestyle="dashed", marker="x"
            )
    else:
        plt.plot(
            range(l + m, l + m + k), trajectories[m, :], color=colors[m],
            linestyle="dashed", marker="x"
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
    trajectories = [[None] * k for _ in range(lag, n)]
    
    # generation of the monthly trajectories
    for i in range(lag, n):
        # calculate the base gradient, get current position
        grads = np.diff(forecast[i - lag:i])
        grad = np.mean(grads)
        pos = forecast[i]
        
        # generation of each trajectory
        for j in range(k):
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


# Unit Tests
class TrajTesterSimple(unittest.TestCase):
    def setUp(self):
        # arrays for testing
        self.data = [
            np.linspace(0.0, 0.7, 10),
            np.random.random(size=(10))
        ]
        self.labels = [
            "linear",
            "random"
        ]
    
    def diff_test(self):
        for i, data in enumerate(self.data):
            res = traj_simple(forecast=data)
            print(res)
            plot_traj(
                data, res,
                path=Path().cwd() / "visuals" / "trajectories" / (self.labels[i] + "-diff")
            )
            
    def exp_test(self):
        for i, data in enumerate(self.data):
            res = traj_simple(forecast=data, strat="exp")
            print(res)
            plot_traj(
                data, res,
                path=Path().cwd() / "visuals" / "trajectories" / (self.labels[i] + "-exp")
            )
        

# Run Tests
if __name__ == "__main__":
    tester = TrajTesterSimple()
    tester.setUp()
    tester.diff_test()
    tester.exp_test()
    
