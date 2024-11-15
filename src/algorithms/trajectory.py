"""
    @brief Trajectory algorithms for forecasting without network data available.
    @author Arjun Ashok (arjun3.ashok@gmail.com)
    @version 1.0.0
"""


# Environment Setup
## external
import pandas as pd
import numpy as np

## built-in
import unittest
import json
from pathlib import Path


# Helper Algorithms
def diminish_grad(grad: float, cur_pos: float) -> float:
    """Auxiliary algorithm to help control the diminishing gradient as we 
    approach the bounds of the probability output. For example, if we are 
    currently at P[success] = 0.7 w/ a trajectory of 0.2, within 3 months we'll
    have P[success] = 1.1. To avoid this, we'll diminish the probability growth
    as we approach the bounds 0 (neg gradient) or 1 (pos gradient).

    Args:
        grad (float): current gradient to step by
        cur_pos (float): current probability of success

    Returns:
        float: modified gradient
    """
    
    # infer projected bound
    bound = 0 if grad < 0 else 1
    
    # return a function of the difference between the position and projected 
    # bound
    bound_diff = abs(bound - cur_pos)
    return grad * bound_diff


def export_traj(trajectories: dict | list | np.array, filename: Path=None) -> str:
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


def plot_traj(forecast: np.array | list, trajectories: np.array) -> None:
    pass


# Primary Algorithms
def traj_simple(forecast: np.array, lag: int=3, k: int=3) -> np.array:
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
            # update position
            pos += grad
            trajectories[i - lag][j] = pos
            
            # update grad
            grad = diminish_grad(grad, pos)
        
    # export as ndarray
    return np.array(trajectories)
    
# Unit Tests
class TrajTesterSimple(unittest.TestCase):
    def setUp(self):
        # arrays for testing
        self.data = np.array(range(0, 60, 12)) / 100
    
    def initial_test(self):
        res = traj_simple(forecast=self.data)
        print(res)
        

# Run Tests
if __name__ == "__main__":
    tester = TrajTesterSimple()
    tester.setUp()
    tester.initial_test()
