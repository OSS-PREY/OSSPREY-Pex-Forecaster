"""
@brief End-to-End tests for the pex-forecaster tool deployed in OSSPREY. Uses 
    default parquets from real projects to ensure accurate computation.
@author Arjun Ashok (arjun3.ashok@gmail.com)
"""


# --- Environment Setup --- #
from decalfc.app.server import *
from tests.__init__ import INPUT_PATH, RESULT_PATH, OUTPUT_PATH

from pathlib import Path

# --- Utility --- #
def save_result(path: Path, res_dict: dict) -> None:
    """Saves the result of the pipeline for comparison."""
    
    with open(path, "w") as f:
        dump(res_dict, f, indent=2)

def load_result(path: Path) -> dict:
    """Saves the result of the pipeline for comparison."""
    
    with open(path, "r") as f:
        return load(f)

def compare_result(output: dict, target: dict, eps: float=1e-2) -> bool:
    """Compares two result dictionaries to see if they're close enough (i.e. 
    with some margin of error to account for float precision).

    Args:
        output (dict): output dictionary with all output keys present.
        target (dict): target dictionary with all output keys present.
        eps (float, optional): margin of error allowed. Defaults to 0.01.

    Returns:
        bool: True if the results are similar enough (pass), otherwise False 
            (fail)
    """
    
    # compare basic info
    assert output[""]

def run_pipeline_with_data(ds_name: str):
    """Wrapper for the dispatch to the pex-forecaster module, essentially 
    mimicking the call the OSSPREY backend should be making to the module on 
    requests.

    Args:
        ds_name (str): name of the dataset to use.
    """
    
    # unpack args
    tech_ds_path = INPUT_PATH / f"{ds_name}_tech.parquet"
    social_ds_path = INPUT_PATH / f"{ds_name}_social.parquet"
    compare_path = RESULT_PATH / f"{ds_name}.parquet"
    out_path = OUTPUT_PATH / f"{ds_name}.parquet"
    
    # data package
    test_pkg = {
        "project_name": ds_name,
        "tech_data": pd.read_parquet(tech_ds_path),
        "social_data": pd.read_parquet(social_ds_path),
        "tasks": ["ALL"],
        "month_range": [0, -1],
        "ignore_cache": True
    }
    
    # dispatch to forecaster, catch any errors
    try:
        res = compute_forecast(data_pkg)
    except Exception as e:
        log(f"errored in testing :: {e}", "error")
    
    # load in expected outputs, save current results
    save_result(out_path, res)
    exp_res = load_result(compare_path)

    # compare result if it exists
    assert compare_result(res, exp_res)

# --- Testing Protocols --- #
def test_end_to_end():
    # sample datasets
    sample_ds = [
        
    ]
    
    # assertion
    for ds in sample_ds:
        run_pipeline_with_data(ds)

